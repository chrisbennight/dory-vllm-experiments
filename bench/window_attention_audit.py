"""Check Dory's paged cache and attention against the exact inputs to each call.

Unlike an independently evolving model, the attention reference receives the
engine's own query/key/value tensors. A contiguous mirror checks every retained
cache entry, and FP64 PyTorch attention checks the last query of each forward.
The audit returns the original engine output and teacher-forces saved histories.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import window_agreement as wa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("per_loop", "last_loop"), default="last_loop")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    wa.configure("upstream")
    wa.upstream(args.out_dir)
    import numpy as np
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.attention.attention import get_attention_context

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    group = f"bf16-{args.dtype}-L2-{args.mode}-chunk2048"
    destination = args.out_dir / f"{group}-attention-audit"
    destination.mkdir(exist_ok=False)
    llm = LLM(model=str(ROOT / "models/bf16"), dtype=args.dtype,
              hf_overrides={"n_recurrent_loops": 2, "recurrent_kv_cache_mode": args.mode},
              max_model_len=2048, max_num_seqs=1, max_num_batched_tokens=2048,
              gpu_memory_utilization=0.80, enable_prefix_caching=False,
              enforce_eager=True, async_scheduling=True, seed=0,
              attention_backend="FLEX_ATTENTION" if args.dtype == "float32" else "TRITON_ATTN")

    def install(model):
        model._audit = None
        original_logits = model.compute_logits

        def forward_positions(module, values, keywords):
            if model._audit is not None:
                model._audit["positions"] = keywords["positions"] if not values else values[1]
                model._audit["calls"] = {}

        model.register_forward_pre_hook(forward_positions, with_kwargs=True)

        def logits(hidden):
            raw = original_logits(hidden)
            state = model._audit
            if state is None:
                return raw
            state["logits"].append(raw.detach().float().cpu().numpy().copy()[0])
            result = torch.full_like(raw, -float("inf"))
            result[0, state["forced"][len(state["logits"]) - 1]] = 0
            return result

        model.compute_logits = logits

        def wrap(attention, window):
            original = attention.forward

            def audited(query, key, value, *values, **keywords):
                state = model._audit
                if state is None:
                    return original(query, key, value, *values, **keywords)
                positions = state["positions"]
                end = int(positions[-1]) + 1
                name = attention.layer_name
                if name not in state["mirror"]:
                    shape = (2048, attention.num_kv_heads, attention.head_size)
                    state["mirror"][name] = [torch.empty(shape, device=key.device, dtype=key.dtype)
                                              for _ in range(2)]
                keys, vals = state["mirror"][name]
                keys[positions] = key.reshape(-1, attention.num_kv_heads, attention.head_size)
                vals[positions] = value.reshape(-1, attention.num_kv_heads, attention.head_size)
                output = original(query, key, value, *values, **keywords)
                metadata, _, cache, _ = get_attention_context(name)
                first = 0 if window is None else max(0, end - window)
                indices = torch.arange(first, end, device=query.device)
                kc, vc = cache.transpose(1, 2).split(attention.head_size, dim=-1)
                size = kc.shape[1]
                physical = metadata.block_table[0, indices // size].long()
                saved_k, saved_v = kc[physical, indices % size], vc[physical, indices % size]
                cache_error = torch.maximum((saved_k - keys[first:end]).abs().max(),
                                            (saved_v - vals[first:end]).abs().max()).item()
                q = query.reshape(-1, attention.num_heads, attention.head_size)[-1:]
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    expected = torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(0, 1).double(), keys[first:end].transpose(0, 1).double(),
                        vals[first:end].transpose(0, 1).double(),
                        scale=attention.impl.scale, enable_gqa=True)
                actual = output[-1].reshape(attention.num_heads, 1, attention.head_size).double()
                error = actual - expected
                call = state["calls"].get(name, 0)
                state["calls"][name] = call + 1
                state["rows"].append({"layer": name, "call": call, "position": end - 1,
                    "window": window, "cache_max_abs": cache_error,
                    "attention_max_abs": error.abs().max().item(),
                    "attention_relative_l2": (error.norm() / expected.norm().clamp_min(1e-30)).item()})
                return output

            attention.forward = audited

        for index, layer in enumerate(model.backbone.layers):
            if layer.is_attention:
                window = model.config.sliding_window if model.config.layer_types[index] == "sliding_attention" else None
                for attention in layer.mixer.attn:
                    wrap(attention, window)

    llm.apply_model(install)
    for case in json.loads((args.out_dir / "cases.json").read_text()):
        if case["name"] not in ("random_cross", "random_long"):
            continue
        meta = json.loads((args.out_dir / f"{group}-upstream" / f"{case['name']}.json").read_text())

        def begin(model):
            model._audit = {"forced": meta["completion"], "rows": [], "logits": [], "mirror": {}}

        llm.apply_model(begin)
        result = llm.generate([TokensPrompt(prompt_token_ids=case["prompt"])],
                             SamplingParams(temperature=0, max_tokens=case["tokens"], ignore_eos=True),
                             use_tqdm=False)[0]
        assert list(result.outputs[0].token_ids) == meta["completion"]

        def finish(model):
            state, model._audit = model._audit, None
            np.save(destination / f"{case['name']}.npy", np.stack(state["logits"]))
            wa.write_json(destination / f"{case['name']}.json", state["rows"])
            return {"calls": len(state["rows"]),
                    "max_cache_error": max(r["cache_max_abs"] for r in state["rows"]),
                    "max_attention_error": max(r["attention_max_abs"] for r in state["rows"]),
                    "max_attention_relative_l2": max(r["attention_relative_l2"] for r in state["rows"])}

        summary = llm.apply_model(finish)[0]
        assert summary["max_cache_error"] == 0, summary
        print(case["name"], summary, flush=True)
    llm.llm_engine.engine_core.shutdown()
    wa.write_json(destination / "complete.json", {"exit_code": 0})
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
