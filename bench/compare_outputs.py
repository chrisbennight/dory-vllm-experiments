"""Measures how closely server configurations agree on greedy generations.

Runs a fixed set of prompts with greedy decoding (256 tokens, top-5
log-probabilities) against each requested configuration, then compares every
configuration with the first one:

- shared prefix: generated tokens identical to the reference before the first
  difference;
- log-probability difference: mean and largest |difference| in the chosen
  token's log-probability while the two generations agree;
- divergence margin: at the first difference, how much more likely the
  reference's token was than the other configuration's token, according to the
  reference's own top-5 log-probabilities. A margin near zero means the two
  tokens were nearly tied, so the divergence is numerical noise rather than a
  wrong prediction.

Configurations: nvfp4-triton (unmodified, Triton attention), nvfp4-flashinfer
(unmodified, FlashInfer attention; needs the project CUDA toolchain),
nvfp4-triton-fused (the fused decode path with vLLM's BF16 LM head; needs the
patched vLLM checkout), nvfp4-triton-fused-int8-head (the fused path's default
INT8 LM head), nvfp4-triton-fused-ngram (the default fused path plus vLLM's
prompt-lookup speculation) and nvfp4-triton-fused-wavefront (the default fused
path plus Dory's wavefront decoding for a request decoding alone, which these
one-at-a-time prompts always are).

With --reference, the configurations are compared with generations saved by
an earlier run (its outputs_<configuration>.json) instead of with the first
configuration, so a fixed baseline can be reused.

Usage: .venv/bin/python -I bench/compare_outputs.py nvfp4-triton nvfp4-flashinfer nvfp4-triton-fused
       .venv/bin/python -I bench/compare_outputs.py nvfp4-triton-fused \
           --reference results/output_agreement_<stamp>/outputs_nvfp4-triton.json
"""

import argparse
import contextlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# Run with -I, which leaves this directory off the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_baseline as rb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = (
    "What is 17 * 23? Answer with just the number.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain in two sentences why the sky is blue.",
    "List the first five prime numbers greater than 100.",
    "A train leaves at 3:15 pm and arrives at 6:40 pm. How long is the trip?",
    "Translate 'The weather is lovely today' into French and Spanish.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Write a haiku about GPUs.",
    "Solve for x: 3x + 7 = 25.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Write a SQL query that returns the ten most recent rows from a table named orders.",
    "What are the main differences between TCP and UDP?",
    "A rectangle has a perimeter of 30 and a width of 5. What is its area?",
    "Give three tips for writing clear technical documentation.",
    "Write a bash one-liner that counts the lines in all .py files in a directory.",
    "Explain what a hash table is to a beginner.",
    "What is the derivative of x^3 * sin(x)?",
    "Write a Rust function that reverses a string.",
    "Why do leaves change color in the autumn?",
    "Convert 72 degrees Fahrenheit to Celsius and show the steps.",
    "Name three causes of the French Revolution.",
    "Write a short email asking a colleague to review a pull request.",
    "What is the difference between a list and a tuple in Python?",
    "If a shirt costs $40 after a 20% discount, what was the original price?",
    "Describe how a refrigerator keeps food cold.",
    "Write a JavaScript function that debounces another function.",
    "What are the three branches of the United States government?",
    "Give a one-paragraph summary of the theory of evolution.",
    "How many seconds are there in a week? Show the calculation.",
    "Write a limerick about a cat who loves coffee.",
    "What does the HTTP status code 404 mean?",
    "Explain recursion using a simple example.",
    "Translate 'Where is the train station?' into German and Italian.",
    "List four renewable sources of energy and one drawback of each.",
    "What is the time complexity of binary search, and why?",
    "Write a regular expression that matches a US ZIP code.",
    "Why is the ocean salty?",
    "A car travels 150 km in 2.5 hours. What is its average speed?",
    "Write a haiku about the first snow of winter.",
    "What is the difference between weather and climate?",
    "Explain what a database index is and when to use one.",
    "Write a C function that computes the length of a null-terminated string.",
    "Who wrote 'Pride and Prejudice', and what is it about?",
    "What is the Pythagorean theorem? Give an example.",
    "Suggest three names for a bakery that specializes in sourdough.",
    "How does a vaccine train the immune system?",
    "Write a Python list comprehension that squares the even numbers from 1 to 20.",
    "What is the capital of Canada, and what river is it on?",
    "Explain the difference between RAM and storage.",
    "Write two sentences persuading someone to drink more water.",
    "What is 15% of 260?",
    "Describe the water cycle in four steps.",
    "Write a SQL query that counts orders per customer.",
    "What is the main function of mitochondria?",
    "Explain what an API is using a restaurant analogy.",
    "Write a bash command that finds files larger than 100 MB.",
    "Why does the Moon have phases?",
    "Solve: 2x - 5 = 3x + 4.",
    "Give three tips for a good night's sleep.",
    "What is the difference between 'affect' and 'effect'?",
    "Write a Go function that returns the maximum of a slice of integers.",
    "Explain what inflation is in simple terms.",
    "What are prime numbers? List the primes below 30.",
    "Write a polite message declining a meeting invitation.",
)
MAX_TOKENS = 256
CUDA_HOME = ROOT / "cuda" / "home"
CONFIGS = {
    "nvfp4-triton": {"attention": "TRITON_ATTN", "env": {}},
    "nvfp4-flashinfer": {"attention": "FLASHINFER",
                         "env": {"CUDA_HOME": str(CUDA_HOME),
                                 "PATH": f"{CUDA_HOME / 'bin'}{os.pathsep}{os.environ['PATH']}"}},
    "nvfp4-triton-fused": {"attention": "TRITON_ATTN",
                           "env": {"DORY_FUSED_DECODE": str(ROOT / "kernels"),
                                   "DORY_LM_HEAD_8BIT": "none"}},
    "nvfp4-triton-fused-int8-head": {"attention": "TRITON_ATTN",
                                     "env": {"DORY_FUSED_DECODE": str(ROOT / "kernels"),
                                             "DORY_LM_HEAD_8BIT": "int8"}},
    # vLLM's prompt-lookup speculation (its GPU version, which supports async
    # scheduling): up to 4 draft tokens copied from the context, verified in one
    # multi-token step (which runs vLLM's own kernels).
    "nvfp4-triton-fused-ngram": {
        "attention": "TRITON_ATTN", "env": {"DORY_FUSED_DECODE": str(ROOT / "kernels"),
                                            "DORY_LM_HEAD_8BIT": "int8"},
        "args": ["--speculative-config", json.dumps({
            "method": "ngram_gpu", "num_speculative_tokens": 4, "prompt_lookup_max": 4,
            "prompt_lookup_min": 2})]},
    "nvfp4-triton-fused-wavefront": {
        "attention": "TRITON_ATTN", "env": {"DORY_FUSED_DECODE": str(ROOT / "kernels"),
                                            "DORY_LM_HEAD_8BIT": "int8"},
        "overrides": {"wavefront_decode": True}},
}
# Variables a configuration may set; cleared for configurations that do not.
CONFIG_ENV = ("CUDA_HOME", "PATH", "DORY_FUSED_DECODE", "DORY_LM_HEAD_8BIT")


@contextlib.contextmanager
def serving(name: str, loops: int, server_log: Path):
    """Runs the server for configuration `name` with its environment until the
    block exits; the block runs once the server is ready."""
    config = CONFIGS[name]
    saved = {key: os.environ.get(key) for key in CONFIG_ENV}
    os.environ.pop("DORY_FUSED_DECODE", None)
    os.environ.pop("DORY_LM_HEAD_8BIT", None)
    os.environ.update(config["env"])
    try:
        rb.require_idle_gpu()
        server = rb.start_server("nvfp4", loops, config["attention"], server_log,
                                 config.get("args"), extra_overrides=config.get("overrides"))
        try:
            rb.wait_until_ready(server)
            yield
        finally:
            rb.stop_server(server)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def generate(name: str, out_dir: Path, loops: int) -> list[dict]:
    with serving(name, loops, out_dir / f"server_{name}.log"):
        results = []
        for prompt in PROMPTS:
            response = rb.http_json(f"http://127.0.0.1:{rb.PORT}/v1/chat/completions", {
                "model": rb.SERVED_NAME, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": MAX_TOKENS, "logprobs": True,
                "top_logprobs": 5})
            results.append({"prompt": prompt,
                            "tokens": response["choices"][0]["logprobs"]["content"]})
    (out_dir / f"outputs_{name}.json").write_text(json.dumps(results))
    return results


def compare(reference: list[dict], other: list[dict]) -> dict:
    shared, deltas, margins, identical = [], [], [], 0
    for ref, oth in zip(reference, other, strict=True):
        ref_tokens, oth_tokens = ref["tokens"], oth["tokens"]
        limit = min(len(ref_tokens), len(oth_tokens))
        n = next((i for i in range(limit)
                  if ref_tokens[i]["token"] != oth_tokens[i]["token"]), limit)
        shared.append(n)
        deltas += [abs(ref_tokens[i]["logprob"] - oth_tokens[i]["logprob"]) for i in range(n)]
        if n == limit and len(ref_tokens) == len(oth_tokens):
            identical += 1
            continue
        if n < limit:
            alternatives = {t["token"]: t["logprob"] for t in ref_tokens[n]["top_logprobs"]}
            other_choice = oth_tokens[n]["token"]
            margins.append(ref_tokens[n]["logprob"] - alternatives[other_choice]
                           if other_choice in alternatives else float("inf"))
    finite = [m for m in margins if m != float("inf")]
    return {
        "identical": identical, "prompts": len(reference),
        "mean_shared": statistics.mean(shared), "min_shared": min(shared),
        "mean_delta": statistics.mean(deltas) if deltas else 0.0,
        "max_delta": max(deltas) if deltas else 0.0,
        "divergences": len(margins), "outside_top5": len(margins) - len(finite),
        "median_margin": statistics.median(finite) if finite else float("nan"),
        "max_margin": max(finite) if finite else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", choices=list(CONFIGS))
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--reference", type=Path,
                        help="saved outputs_<configuration>.json to compare against")
    args = parser.parse_args()
    saved_reference = json.loads(args.reference.read_text()) if args.reference else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "results" / f"output_agreement_{stamp}"
    out_dir.mkdir(parents=True)
    outputs = {name: generate(name, out_dir, args.loops) for name in args.configs}

    if saved_reference is None:
        reference, compared = args.configs[0], args.configs[1:]
        reference_outputs = outputs[reference]
    else:
        reference, compared = f"saved {args.reference.resolve().relative_to(ROOT)}", args.configs
        reference_outputs = saved_reference
    lines = [f"# Greedy output agreement with {reference} ({len(PROMPTS)} prompts, "
             f"{MAX_TOKENS} tokens, {args.loops} loops)", "",
             "| Configuration | Identical outputs | Mean shared tokens | Fewest shared tokens "
             "| Mean abs logprob difference | Largest abs logprob difference | Divergences "
             "| Median divergence margin | Largest divergence margin | Divergent token outside top 5 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in compared:
        r = compare(reference_outputs, outputs[name])
        lines.append(f"| {name} | {r['identical']}/{r['prompts']} | {r['mean_shared']:.0f} "
                     f"| {r['min_shared']} | {r['mean_delta']:.4f} | {r['max_delta']:.4f} "
                     f"| {r['divergences']} | {r['median_margin']:.3f} | {r['max_margin']:.3f} "
                     f"| {r['outside_top5']} |")
    report = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(report)
    print(report)
    print(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
