"""In-process check of Dory's wavefront decoding inside vLLM.

Runs chat prompts (bench/compare_outputs.py's) one at a time through vLLM with
the fused decode path, with or without the config's wavefront_decode, and
writes each prompt's generated token ids, its decode speed (the time for N
tokens minus the time for one) and, with the wavefront, the engine's step
counters to a JSON file. Two such files (with and without the wavefront)
compare with --compare.

Usage:
  .venv/bin/python bench/wavefront_smoke.py --wavefront --out results/x.json
  .venv/bin/python bench/wavefront_smoke.py --compare a.json b.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("DORY_FUSED_DECODE", str(ROOT / "kernels"))
# JIT builds run tools installed in the environment, found on PATH only when
# it is activated.
os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"

from compare_outputs import PROMPTS  # noqa: E402
from run_baseline import MODELS  # noqa: E402


def shared_prefix(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def compare(paths: list[str]) -> int:
    runs = [json.loads(Path(p).read_text()) for p in paths]
    a, b = runs
    print(f"{paths[0]}: wavefront={a['wavefront']}; {paths[1]}: wavefront={b['wavefront']}")
    for index, (x, y) in enumerate(zip(a["prompts"], b["prompts"])):
        print(f"prompt {index}: shared {shared_prefix(x['tokens'], y['tokens'])} of "
              f"{len(x['tokens'])}; {x['decode_tok_s']:.1f} vs {y['decode_tok_s']:.1f} tok/s")
    for run, path in zip(runs, paths):
        rate = sum(p["tokens_timed"] for p in run["prompts"]) / sum(
            p["decode_s"] for p in run["prompts"])
        print(f"{path}: overall {rate:.1f} tok/s; engine {run.get('engine')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavefront", action="store_true")
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2, metavar="JSON")
    args = parser.parse_args()
    if args.compare:
        return compare(args.compare)
    if not args.out:
        parser.error("--out is required")

    from profile_decode import gpu_is_idle
    if not gpu_is_idle():
        raise RuntimeError("the GPU is not idle; refusing to run")
    from transformers import PreTrainedTokenizerFast
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model = MODELS / "nvfp4"
    # AutoTokenizer validates Dory's model config and rejects its "mlp" layer type.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model))
    overrides = {"n_recurrent_loops": args.loops, "recurrent_kv_cache_mode": "per_loop"}
    if args.wavefront:
        overrides["wavefront_decode"] = True
    llm = LLM(model=str(model), hf_overrides=overrides, max_model_len=8192, max_num_seqs=64,
              gpu_memory_utilization=0.90, async_scheduling=True, seed=0,
              attention_backend="TRITON_ATTN")

    def run(prompt, tokens):
        sampling = SamplingParams(max_tokens=tokens, ignore_eos=True,
                                  temperature=args.temperature, top_p=args.top_p, seed=0)
        start = time.perf_counter()
        out = llm.generate([prompt], sampling, use_tqdm=False)[0]
        return list(out.outputs[0].token_ids), time.perf_counter() - start

    results = []
    # The first generation compiles and captures; it is not timed.
    warm = TokensPrompt(prompt_token_ids=tokenizer.encode("Hello", add_special_tokens=False))
    run(warm, 64)
    for text in PROMPTS[:args.prompts]:
        prompt = TokensPrompt(prompt_token_ids=tokenizer.encode(
            tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                          add_generation_prompt=True),
            add_special_tokens=False))
        # The first call caches the prompt's KV (prefix caching), so both timed
        # calls prefill the same way.
        run(prompt, 1)
        _, first_s = run(prompt, 1)
        tokens, total_s = run(prompt, args.tokens)
        decode_s = total_s - first_s
        results.append({"tokens": tokens, "tokens_timed": args.tokens - 1, "decode_s": decode_s,
                        "decode_tok_s": (args.tokens - 1) / decode_s})
        print(f"prompt {len(results) - 1}: {(args.tokens - 1) / decode_s:.1f} tok/s", flush=True)

    def engine_counters(m):
        engine = getattr(m, "_wavefront", None)
        if engine is None:
            return None
        return {"steps": engine.steps, "catch_ups": engine.catch_ups,
                "verified": engine.verified, "matched": engine.matched,
                "unallocated": engine.unallocated}

    engine = llm.apply_model(engine_counters)[0]
    report = {"wavefront": args.wavefront, "loops": args.loops,
              "temperature": args.temperature, "top_p": args.top_p, "engine": engine,
              "prompts": results}
    Path(args.out).write_text(json.dumps(report))
    rate = sum(r["tokens_timed"] for r in results) / sum(r["decode_s"] for r in results)
    print(f"overall {rate:.1f} tok/s; engine {engine}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
