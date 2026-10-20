"""Compare Dory decode distributions on identical histories across the SWA window.

The reference uses the pinned upstream model's projection and nGPT math with
ordinary contiguous PyTorch attention instead of vLLM's paged attention. Prefill
chunks are recorded from actual engine forwards and replayed verbatim. A second
reference deliberately prefills one token at a time to test the reported
chunk-boundary confound. These are numerical checks, not task-accuracy scores.

Run ``.venv/bin/python -I bench/window_agreement.py run --out-dir PATH``.
Large logits and checkpoints stay in the ignored logs directory; the final
report and per-position metrics can be published separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import run_baseline as rb


def write_json(path: Path, value) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def upstream(out: Path):
    """Import the unchanged, pinned local Git source without editing the checkout."""
    source = subprocess.check_output(
        ["git", "-C", str(ROOT / "src/vllm"), "show",
         f"{rb.VLLM_PR_SHA}:vllm/model_executor/models/dory.py"])
    path = out / "upstream_dory.py"
    if path.exists():
        assert path.read_bytes() == source, "Pinned model source changed"
    else:
        path.write_bytes(source)
    name = "vllm.model_executor.models.dory"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure(variant: str) -> None:
    for key in ("DORY_FUSED_DECODE", "DORY_WAVEFRONT_FORCE_CATCH_UP"):
        os.environ.pop(key, None)
    os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0",
                      VLLM_USE_FLASHINFER_SAMPLER="0",
                      TOKENIZERS_PARALLELISM="false",
                      DORY_LM_HEAD_8BIT="int8", DORY_FUSED_ATTENTION="1",
                      DORY_FUSED_KV_WRITE="1", DORY_FOLD_RESIDUAL="mlp,attention")
    os.environ["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + os.environ["PATH"]
    if variant in ("fused", "wavefront", "catchup"):
        os.environ["DORY_FUSED_DECODE"] = str(ROOT / "kernels")
    if variant == "catchup":
        os.environ["DORY_WAVEFRONT_FORCE_CATCH_UP"] = "1"


def cases() -> list[dict]:
    import numpy as np
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(ROOT / "models/bf16"))
    short = tokenizer.encode("Explain why the sky is blue in clear, simple language.",
                             add_special_tokens=False)
    natural = tokenizer.encode(
        ("The observatory recorded the weather every morning. The researchers "
         "compared the blue daytime sky with the red light at sunset. " * 100),
        add_special_tokens=False)
    random = np.random.default_rng(7).integers(1000, 131072, 1100).tolist()
    return [dict(name="short", prompt=short, tokens=256),
            dict(name="random_cross", prompt=random[:1000], tokens=128),
            dict(name="random_long", prompt=random, tokens=64),
            dict(name="natural_cross", prompt=natural[:1000], tokens=128),
            dict(name="natural_long", prompt=natural[:1100], tokens=64)]


def label(args) -> str:
    return f"{args.model}-{args.dtype}-L{args.loops}-{args.mode}-chunk{args.chunk}"


def engine(args) -> None:
    configure(args.variant)
    if args.variant == "upstream":
        upstream(args.out_dir)
    import numpy as np
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    group = label(args)
    destination = args.out_dir / f"{group}-{args.variant}"
    destination.mkdir(exist_ok=True)
    assert not (destination / "complete.json").exists(), "Engine run already completed"
    controls = args.out_dir / f"{group}-upstream"
    overrides = {"n_recurrent_loops": args.loops,
                 "recurrent_kv_cache_mode": args.mode}
    if args.variant in ("wavefront", "catchup"):
        overrides["wavefront_decode"] = True
    # The FP32/head-256 Triton prefill kernel exceeds the 4090's shared memory.
    backend = "FLEX_ATTENTION" if args.dtype == "float32" else "TRITON_ATTN"
    llm = LLM(model=str(ROOT / "models" / args.model), dtype=args.dtype,
              hf_overrides=overrides, max_model_len=2048, max_num_seqs=1,
              max_num_batched_tokens=args.chunk, gpu_memory_utilization=0.90,
              enable_prefix_caching=False, enable_chunked_prefill=True,
              enforce_eager=True, async_scheduling=True, seed=0,
              attention_backend=backend)

    def install(model):
        original = model.compute_logits
        model._validation = None

        def record_forward(module, values, keywords):
            if module._validation is not None:
                positions = (values[1] if len(values) > 1 else keywords["positions"])
                positions = positions.detach().cpu().tolist()
                module._validation["chunks"].append(positions)

        model.register_forward_pre_hook(record_forward, with_kwargs=True)

        def logits(hidden):
            raw = original(hidden)
            state = model._validation
            if state is None:
                return raw
            # The runner can compute a dummy logit after an unfinished prefill
            # chunk. It is discarded by the engine and produces no output token.
            if state["chunks"][-1][-1] < state["prompt_len"] - 1:
                return raw
            assert raw.shape[0] == 1, raw.shape
            index = len(state["logits"])
            state["logits"].append(raw.detach().float().cpu().numpy().copy()[0])
            if state["forced"] is not None:
                forced = torch.full_like(raw, -float("inf"))
                forced[0, state["forced"][index]] = 0
                return forced
            return raw

        model.compute_logits = logits

    llm.apply_model(install)
    for case in json.loads((args.out_dir / "cases.json").read_text()):
        if (destination / f"{case['name']}.json").exists():
            assert (destination / f"{case['name']}.npy").exists()
            continue
        forced = None
        if args.variant != "upstream":
            forced = json.loads((controls / f"{case['name']}.json").read_text())["completion"]

        def begin(model):
            model._validation = {"logits": [], "chunks": [], "forced": forced,
                                 "prompt_len": len(case["prompt"])}
            wf = getattr(model, "_wavefront", None)
            return None if wf is None else {
                key: getattr(wf, key) for key in ("steps", "catch_ups", "matched", "unallocated")}

        before = llm.apply_model(begin)[0]
        start = time.monotonic()
        result = llm.generate(
            [TokensPrompt(prompt_token_ids=case["prompt"])],
            SamplingParams(temperature=0, max_tokens=case["tokens"], ignore_eos=True),
            use_tqdm=False)[0]
        completion = list(result.outputs[0].token_ids)
        assert len(completion) == case["tokens"]
        if forced is not None:
            assert completion == forced, "Teacher forcing did not retain the baseline history"

        def finish(model):
            state, model._validation = model._validation, None
            assert len(state["logits"]) == case["tokens"], (
                len(state["logits"]), case["tokens"], state["chunks"])
            np.save(destination / f"{case['name']}.npy", np.stack(state["logits"]))
            wf = getattr(model, "_wavefront", None)
            counters = None if wf is None else {
                key: getattr(wf, key) - before[key] for key in before}
            return {"chunks": state["chunks"], "counters": counters}

        metadata = llm.apply_model(finish)[0]
        counters = metadata["counters"]
        if args.variant in ("wavefront", "catchup"):
            assert counters["steps"] == case["tokens"] - 1, counters
            assert counters["unallocated"] == 0, counters
            if args.variant == "catchup":
                assert counters["catch_ups"] == counters["steps"], counters
        # Every actual input position must be represented once, without padding.
        positions = sum(metadata["chunks"], [])
        assert positions == list(range(len(case["prompt"]) + len(completion) - 1)), positions
        metadata.update(completion=completion, elapsed_seconds=time.monotonic() - start,
                        variant=args.variant, group=group, attention_backend=backend)
        write_json(destination / f"{case['name']}.json", metadata)
        print(f"SAVED {group} {args.variant} {case['name']} {metadata['counters']}", flush=True)
    write_json(destination / "complete.json", {"exit_code": 0})
    llm.llm_engine.engine_core.shutdown()


def build_reference(args):
    import torch
    from torch import nn
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.transformers_utils.configs.dory import DoryConfig
    dory = upstream(args.out_dir)
    parallel_state._TP = SimpleNamespace(world_size=1, rank_in_group=0)

    class ContiguousAttention(nn.Module):
        """Causal GQA with absolute positions and no paging or cache eviction."""

        def __init__(self, num_heads, head_size, scale, num_kv_heads, **kwargs):
            super().__init__()
            self.heads, self.kv_heads = num_heads, num_kv_heads
            self.dim, self.scale = head_size, scale
            self.window = kwargs["per_layer_sliding_window"]
            self.keys = self.values = None
            self.start = 0

        def forward(self, q, k, v):
            q = q.reshape(-1, self.heads, self.dim).transpose(0, 1)
            k = k.reshape(-1, self.kv_heads, self.dim).transpose(0, 1)
            v = v.reshape(-1, self.kv_heads, self.dim).transpose(0, 1)
            if self.keys is None:
                self.keys = torch.empty(self.kv_heads, 2048, self.dim, device=k.device, dtype=k.dtype)
                self.values = torch.empty_like(self.keys)
            end = self.start + q.shape[1]
            self.keys[:, self.start:end] = k
            self.values[:, self.start:end] = v
            kp = torch.arange(end, device=q.device)
            qp = torch.arange(self.start, end, device=q.device)[:, None]
            mask = kp <= qp
            if self.window is not None:
                mask &= kp > qp - self.window
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                output = torch.nn.functional.scaled_dot_product_attention(
                    q, self.keys[:, :end], self.values[:, :end], attn_mask=mask,
                    scale=self.scale, enable_gqa=True)
            return output.transpose(0, 1).reshape(-1, self.heads * self.dim)

    dory.Attention = ContiguousAttention
    cfg = DoryConfig(**json.loads((ROOT / "models/bf16/config.json").read_text()))
    cfg.update({"n_recurrent_loops": args.loops, "recurrent_kv_cache_mode": args.mode})
    vc = VllmConfig(device_config=DeviceConfig(device="cuda"),
                   compilation_config=CompilationConfig(mode=0, custom_ops=["none"]))
    dtype = getattr(torch, args.reference_dtype)
    vc.model_config = SimpleNamespace(hf_config=cfg, head_dtype=None, dtype=dtype, max_model_len=2048)
    vc.quant_config = None
    torch.set_default_dtype(dtype)
    with set_current_vllm_config(vc), torch.device("cuda"):
        model = dory.DoryForCausalLM(vllm_config=vc)
    from safetensors import safe_open
    with safe_open(ROOT / "models/bf16/model.safetensors", framework="pt", device="cpu") as f:
        loaded = model.load_weights((name, f.get_tensor(name)) for name in f.keys())
    assert loaded == set(dict(model.named_parameters())), "Some reference weights were not loaded"
    model.eval()
    model.logits_processor._gather_logits = lambda x: x
    caches = [m for m in model.modules() if isinstance(m, ContiguousAttention)]

    def position_caches(module, inputs):
        for cache in caches:
            cache.start = int(inputs[1][0])

    model.register_forward_pre_hook(position_caches)
    return model, caches, vc


def reference(args) -> None:
    configure("upstream")
    import numpy as np
    import torch
    from vllm.config import set_current_vllm_config
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, caches, vc = build_reference(args)
    group = label(args)
    control = args.out_dir / f"{group}-upstream"
    destination = args.out_dir / f"{group}-reference-{args.reference_dtype}-{args.prefill}"
    destination.mkdir(exist_ok=True)
    assert not (destination / "complete.json").exists(), "Reference run already completed"
    with torch.inference_mode(), set_current_vllm_config(vc):
        for case in json.loads((args.out_dir / "cases.json").read_text()):
            if args.prefill == "token" and case["name"] != "random_long":
                continue
            if (destination / f"{case['name']}.json").exists():
                assert (destination / f"{case['name']}.npy").exists()
                continue
            start = time.monotonic()
            meta = json.loads((control / f"{case['name']}.json").read_text())
            tokens = case["prompt"] + meta["completion"][:-1]
            chunks = meta["chunks"]
            if args.prefill == "token":
                chunks = [[position] for position in range(len(tokens))]
            for cache in caches:
                cache.keys = cache.values = None
            logits = []
            for positions in chunks:
                ids = torch.tensor([tokens[p] for p in positions], device="cuda", dtype=torch.long)
                hidden = model(ids, torch.tensor(positions, device="cuda", dtype=torch.long))
                if positions[-1] >= len(case["prompt"]) - 1:
                    raw = model.compute_logits(hidden[-1:])
                    logits.append(raw[0].float().cpu().numpy().copy())
            assert len(logits) == case["tokens"]
            np.save(destination / f"{case['name']}.npy", np.stack(logits))
            write_json(destination / f"{case['name']}.json",
                       {"seconds": time.monotonic() - start, "chunks": chunks})
            print(f"SAVED {group} reference {args.reference_dtype} {args.prefill} {case['name']}", flush=True)
    write_json(destination / "complete.json", {"exit_code": 0})


def metrics(expected, actual, prompt_len: int) -> dict:
    import numpy as np
    import torch
    assert expected.shape == actual.shape
    a = torch.from_numpy(expected).double().log_softmax(-1)
    b = torch.from_numpy(actual).double().log_softmax(-1)
    delta = a - b
    rows = []
    for i in range(len(a)):
        pa, pb = a[i].exp(), b[i].exp()
        rows.append({"position": prompt_len - 1 + i,
                     "logprob_rms": float(delta[i].square().mean().sqrt()),
                     "kl_baseline_to_candidate": float((pa * (a[i] - b[i])).sum()),
                     "total_variation": float((pa - pb).abs().sum() / 2),
                     "same_argmax": bool(a[i].argmax() == b[i].argmax())})
    # Exclude the final prefill logit to match the report's decode-only comparison.
    def aggregate(indices):
        if not indices:
            return None
        return {"rows": len(indices),
                "logprob_rms": float(delta[indices].square().mean().sqrt()),
                "argmax_agreement": float(np.mean([rows[i]["same_argmax"] for i in indices])),
                "mean_total_variation": float(np.mean([rows[i]["total_variation"] for i in indices]))}
    return {"decode": aggregate(list(range(1, len(rows)))),
            "before_window": aggregate([i for i in range(1, len(rows)) if rows[i]["position"] < 1024]),
            "after_window": aggregate([i for i in range(1, len(rows)) if rows[i]["position"] >= 1024]),
            "positions": rows}


def report(out: Path) -> None:
    import numpy as np
    import torch
    torch.set_num_threads(4)
    all_cases = json.loads((out / "cases.json").read_text())
    comparisons = []
    for control in sorted(out.glob("*-upstream")):
        group = control.name.removesuffix("-upstream")
        for candidate in sorted(out.glob(f"{group}-*")):
            if (candidate == control or candidate.name.endswith("-attention-audit")
                    or not (candidate / "complete.json").exists()):
                continue
            for case in all_cases:
                path = candidate / f"{case['name']}.npy"
                if not path.exists():
                    continue
                expected = np.load(control / path.name)
                actual = np.load(path)
                comparisons.append({"group": group, "baseline": "upstream",
                                    "candidate": candidate.name[len(group) + 1:],
                                    "case": case["name"],
                                    "counters": json.loads((candidate / f"{case['name']}.json").read_text()).get("counters"),
                                    **metrics(expected, actual, len(case["prompt"]))})
        matched = out / f"{group}-reference-float32-matched"
        token = out / f"{group}-reference-float32-token"
        if (matched / "complete.json").exists() and (token / "complete.json").exists():
            for case in all_cases:
                name = f"{case['name']}.npy"
                if not (token / name).exists():
                    continue
                comparisons.append({"group": group, "baseline": "reference-float32-matched",
                    "candidate": "reference-float32-token",
                    "case": case["name"], "counters": None,
                    **metrics(np.load(matched / name), np.load(token / name), len(case["prompt"]))})
        for baseline, candidate in (("catchup", "wavefront"), ("fused", "catchup")):
            a, b = out / f"{group}-{baseline}", out / f"{group}-{candidate}"
            if not all((path / "complete.json").exists() for path in (a, b)):
                continue
            for case in all_cases:
                name = f"{case['name']}.npy"
                comparisons.append({"group": group, "baseline": baseline, "candidate": candidate,
                    "case": case["name"],
                    "counters": json.loads((b / f"{case['name']}.json").read_text())["counters"],
                    **metrics(np.load(a / name), np.load(b / name), len(case["prompt"]))})
    write_json(out / "metrics.json", comparisons)
    lines = ["# Dory agreement across the sliding-window boundary", "",
             "All comparisons use identical prompt and completion tokens. Prefix caching is off. "
             "The upstream control loads the unchanged pinned Dory model file on the installed "
             "vLLM runtime. BF16 engine attention: TRITON_ATTN; FP32 controls: FLEX_ATTENTION "
             "(the FP32 Triton prefill kernel exceeds this GPU's shared memory). Outer execution "
             "is eager with a single request. The contiguous "
             "reference uses PyTorch math attention and the pinned upstream model math. FP32 "
             "reference results are not a reproduction of the comment's FP64 reference. "
             "The first (prefill) logit is excluded from aggregates. Full per-position metrics "
             "are in metrics.json; raw logits are retained in the private local run directory.", "",
             "| Engine group | Baseline | Candidate | Prompt | Decode RMS Δ log p | Before 1024 | From 1024 | Top-token agreement | Mean TV |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        dec = row["decode"]
        before = row["before_window"]
        after = row["after_window"]
        fmt = lambda x: "—" if x is None else f"{x['logprob_rms']:.5f}"
        lines.append(f"| {row['group']} | {row['baseline']} | {row['candidate']} | {row['case']} | "
                     f"{dec['logprob_rms']:.5f} | {fmt(before)} | {fmt(after)} | "
                     f"{dec['argmax_agreement']:.1%} | {dec['mean_total_variation']:.5f} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run(args) -> None:
    rb.require_idle_gpu()
    out = args.out_dir.absolute()
    out.mkdir(mode=0o700, parents=True, exist_ok=args.resume)
    if not args.resume:
        write_json(out / "cases.json", cases())
    source = subprocess.check_output(["git", "-C", str(ROOT / "src/vllm"), "show",
        f"{rb.VLLM_PR_SHA}:vllm/model_executor/models/dory.py"])
    provenance = {"vllm_pr_sha": rb.VLLM_PR_SHA,
        "model_revisions": rb.MODEL_REVISIONS, "gpu": rb.gpu_snapshot(),
        "upstream_dory_sha256": hashlib.sha256(source).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.resume:
        previous = json.loads((out / "provenance.json").read_text())
        for key in ("vllm_pr_sha", "model_revisions", "upstream_dory_sha256"):
            assert previous[key] == provenance[key], f"Cannot resume changed input: {key}"
        previous.setdefault("resumes", []).append({"harness_sha256": provenance["harness_sha256"],
                                                   "gpu": provenance["gpu"]})
        write_json(out / "provenance.json", previous)
    else:
        write_json(out / "provenance.json", provenance)
    jobs = []
    for loops, mode in ((1, "per_loop"), (1, "last_loop"), (2, "per_loop"),
                        (2, "last_loop"), (4, "per_loop"), (4, "last_loop")):
        base = ["--model", "bf16", "--loops", str(loops), "--mode", mode]
        jobs += [["engine", *base, "--variant", "upstream"],
                 ["reference", *base, "--reference-dtype", "bfloat16"],
                 ["reference", *base, "--reference-dtype", "float32"]]
        if loops == 2:
            jobs += [["reference", *base, "--reference-dtype", "float32", "--prefill", "token"],
                     ["engine", *base, "--chunk", "512", "--variant", "upstream"],
                     ["reference", *base, "--chunk", "512", "--reference-dtype", "float32"]]
            jobs += [["engine", *base, "--dtype", "float32", "--variant", "upstream"],
                     ["reference", *base, "--dtype", "float32", "--reference-dtype", "float32"]]
    for loops in (2, 4):
        for mode in ("per_loop", "last_loop"):
            base = ["--model", "nvfp4", "--loops", str(loops), "--mode", mode]
            jobs += [["engine", *base, "--variant", variant]
                     for variant in ("upstream", "fused", "catchup", "wavefront")]
    for index, job in enumerate(jobs):
        status = out / f"job-{index + 1:02d}.status.json"
        if status.exists():
            previous = json.loads(status.read_text())
            assert previous["argv"] == job
            if previous.get("exit_code") == 0:
                continue
        print(f"JOB {index + 1}/{len(jobs)} {' '.join(job)}", flush=True)
        log = out / f"job-{index + 1:02d}.log"
        write_json(status, {"state": "running", "argv": job})
        with log.open("a") as stream:
            result = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()),
                                     *job, "--out-dir", str(out)], stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        completion = {"state": "finished", "job": index + 1, "total": len(jobs),
                      "argv": job, "exit_code": result.returncode,
                      "harness_sha256": provenance["harness_sha256"]}
        write_json(status, completion)
        write_json(out / "progress.json", completion)
        assert result.returncode == 0, f"Failed job {index + 1}: {log} EXIT={result.returncode}"
        report(out)
    write_json(out / "complete.json", {"exit_code": 0, "jobs": len(jobs)})
    print("EXIT=0", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("run", "engine", "reference", "report"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("bf16", "nvfp4"), default="bf16")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--loops", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--mode", choices=("per_loop", "last_loop"), default="last_loop")
    parser.add_argument("--variant", choices=("upstream", "fused", "wavefront", "catchup"), default="upstream")
    parser.add_argument("--chunk", type=int, choices=(512, 2048), default=2048)
    parser.add_argument("--reference-dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--prefill", choices=("matched", "token"), default="matched")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.step == "report":
        report(args.out_dir)
    else:
        globals()[args.step](args)


if __name__ == "__main__":
    main()
