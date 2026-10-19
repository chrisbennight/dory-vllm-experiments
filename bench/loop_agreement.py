"""Measures how often Dory's shallow predictions agree with its full-depth ones.

Speculative decoding with Dory's own loops drafts each token from a shallow
pass (fewer recurrent loops) and verifies it at full depth. Its payoff depends
on how often the two agree on the same history. With per-loop KV caches, a
token's loop-1 state does not depend on whether later loops ran, so a 1-loop
model reading a history makes exactly the shallow prediction a mixed-depth
scheme with separate shallow output-block caches would make.

Steps, each in its own process so that GPU memory is released in between:

1. generate: 2-loop greedy completions (256 tokens, thinking on, as in normal
   chat use) for the 64 prompts of bench/compare_outputs.py;
2. score --loops L: the top-20 next-token log-probabilities of an L-loop model
   at every completion position of those histories (prompt log-probabilities,
   one prefill per history);
3. report: for each pair of loop counts, greedy agreement of the top token, the
   overlap sum(min(p, q)) of the two distributions (a lower bound computed from
   the top-20 entries; for sampling at temperature 1 it is the acceptance rate of
   speculative sampling), and for greedy drafts of k tokens the mean number
   accepted. The 4-loop comparisons reuse the 2-loop histories.

Usage: .venv/bin/python -I bench/loop_agreement.py run
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Run with -I, which leaves this directory off the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_outputs as co  # noqa: E402
import run_baseline as rb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "nvfp4"
COMPLETION_TOKENS = 256
TOP = 20
LOOPS = (1, 2, 4)
PAIRS = ((1, 2), (1, 4), (2, 4))
DRAFT_LENGTHS = (1, 2, 3, 4)


def engine(loops: int):
    from vllm import LLM
    # Prompt log-probabilities materialize logits for every prefill token, so
    # prefill chunks stay small and memory headroom is left for them.
    return LLM(model=str(MODEL),
               hf_overrides={"n_recurrent_loops": loops, "recurrent_kv_cache_mode": "per_loop"},
               max_model_len=8192, max_num_batched_tokens=2048, gpu_memory_utilization=0.80,
               seed=0, max_logprobs=TOP, attention_backend="TRITON_ATTN")


def generate(out_dir: Path) -> None:
    from vllm import SamplingParams
    llm = engine(2)
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.encode(tokenizer.apply_chat_template(
                   [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True),
                   add_special_tokens=False) for p in co.PROMPTS]
    from vllm.inputs import TokensPrompt
    outputs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                           SamplingParams(temperature=0, max_tokens=COMPLETION_TOKENS,
                                          ignore_eos=True), use_tqdm=False)
    histories = [{"prompt": list(p), "completion": list(o.outputs[0].token_ids)}
                 for p, o in zip(prompts, outputs)]
    (out_dir / "histories.json").write_text(json.dumps(histories))


def score(out_dir: Path, loops: int) -> None:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    histories = json.loads((out_dir / "histories.json").read_text())
    llm = engine(loops)
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=h["prompt"] + h["completion"]) for h in histories],
        SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=TOP), use_tqdm=False)
    scores = []
    for history, output in zip(histories, outputs):
        start = len(history["prompt"])
        positions = []
        for entry in output.prompt_logprobs[start:]:
            ranked = sorted(entry.items(), key=lambda kv: kv[1].rank)[:TOP]
            positions.append([[token, lp.logprob] for token, lp in ranked])
        scores.append(positions)
    (out_dir / f"scores_loops{loops}.json").write_text(json.dumps(scores))


def overlap(a: list, b: list) -> float:
    pa = {token: math.exp(lp) for token, lp in a}
    pb = {token: math.exp(lp) for token, lp in b}
    return sum(min(pa.get(t, 0.0), pb.get(t, 0.0)) for t in pa.keys() | pb.keys())


def report(out_dir: Path) -> str:
    histories = json.loads((out_dir / "histories.json").read_text())
    scores = {loops: json.loads((out_dir / f"scores_loops{loops}.json").read_text())
              for loops in LOOPS}
    lines = ["# Agreement of shallow and deep Dory predictions on the same histories", "",
             f"{len(histories)} chat prompts (bench/compare_outputs.py), 2-loop greedy "
             f"completions of {COMPLETION_TOKENS} tokens with thinking on; every completion "
             "position scored by 1-, 2- and 4-loop models (prompt log-probabilities, top "
             f"{TOP}).", ""]
    teacher = [s[i][0][0] == history["completion"][i]
               for s, history in zip(scores[2], histories) for i in range(len(s))]
    lines += [f"Sanity check: the 2-loop model's top token equals the 2-loop greedy history at "
              f"{statistics.mean(teacher):.1%} of positions (prefill and decode use different "
              "kernels, so near-ties can flip).", "",
              "| Draft loops | Verify loops | Top-token agreement | Mean distribution overlap "
              "(lower bound) | "
              + " | ".join(f"Mean accepted, {k}-token greedy draft" for k in DRAFT_LENGTHS) + " |",
              "|---:|---:|---:|---:|" + "---:|" * len(DRAFT_LENGTHS)]
    for draft, verify in PAIRS:
        agree, overlaps, accepted = [], [], {k: [] for k in DRAFT_LENGTHS}
        for shallow, deep in zip(scores[draft], scores[verify]):
            matches = [s[0][0] == d[0][0] for s, d in zip(shallow, deep)]
            agree += matches
            overlaps += [overlap(s, d) for s, d in zip(shallow, deep)]
            for start in range(len(matches)):
                run = 0
                while start + run < len(matches) and matches[start + run]:
                    run += 1
                for k in DRAFT_LENGTHS:
                    accepted[k].append(min(run, k))
        lines.append(f"| {draft} | {verify} | {statistics.mean(agree):.1%} "
                     f"| {statistics.mean(overlaps):.3f} | "
                     + " | ".join(f"{statistics.mean(accepted[k]):.2f}" for k in DRAFT_LENGTHS)
                     + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("run", "generate", "score", "report"))
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.step == "run":
        rb.require_idle_gpu()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT / "results" / f"loop_agreement_{stamp}"
        out_dir.mkdir(parents=True)
        env = {key: value for key, value in os.environ.items()
               if key not in ("DORY_FUSED_DECODE", "DORY_LM_HEAD_8BIT")}
        # Same engine settings as the benchmark servers (no FlashInfer sampler JIT).
        env.update(rb.SERVER_ENV)
        steps = [(["generate"], "histories.json")] + [
            (["score", "--loops", str(loops)], f"scores_loops{loops}.json") for loops in LOOPS]
        for step, output in steps:
            result = subprocess.run(
                [sys.executable, "-I", __file__, *step, "--out-dir", str(out_dir)], env=env)
            if result.returncode != 0:
                # vLLM can abort while tearing down its engine after the step is
                # done. Each step writes its output file last, so the file shows
                # whether the step itself finished.
                if not (out_dir / output).exists():
                    raise subprocess.CalledProcessError(result.returncode, result.args)
                print(f"warning: step {step[0]} exited with {result.returncode} after writing "
                      f"{output}", file=sys.stderr)
        text = report(out_dir)
        (out_dir / "report.md").write_text(text)
        print(text)
        print(out_dir)
    elif args.step == "generate":
        generate(args.out_dir)
    elif args.step == "score":
        score(args.out_dir, args.loops)
    else:
        print(report(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
