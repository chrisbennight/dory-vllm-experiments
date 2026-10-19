"""Evaluates wavefront self-speculative decoding on the stand-alone Dory decoder.

For each chat prompt (bench/compare_outputs.py's prompts, chat template with
thinking on), greedy decoding of the same number of tokens three ways
(see kernels/dory_wavefront.py):

- A: sequential decoding on the single-token fused kernels (the speed to beat);
- B: sequential decoding on the multi-token kernels (the exactness reference);
- W: wavefront decoding, drafting at loop 1.

Checks that W commits exactly B's tokens, and reports decode speed (prefill
excluded), the draft acceptance rate and how many tokens were in flight. With
--temperature > 0 all three sample (W with speculative acceptance, which keeps
the sampling distribution but not the tokens), so B and the token comparison
are skipped.

Usage: .venv/bin/python kernels/bench_wavefront.py --prompts 8 --tokens 256
"""

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from bench_gemv import CHECKPOINT, ROOT, require_idle_gpu
from dory_wavefront import DoryPrototype

sys.path.insert(0, str(ROOT / "bench"))
from compare_outputs import PROMPTS  # noqa: E402


def shared_prefix(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 decodes greedily; Dory's default is 1.0")
    parser.add_argument("--top-p", type=float, default=1.0, help="Dory's default is 0.95")
    parser.add_argument("--context", type=int, default=0,
                        help="prepend about this many tokens of EXPERIMENTS.md to each prompt, "
                             "to measure at a longer context")
    args = parser.parse_args()
    require_idle_gpu()
    device = torch.device("cuda")
    # AutoTokenizer validates Dory's model config and rejects its "mlp" layer type.
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(CHECKPOINT))
    model = DoryPrototype(CHECKPOINT, args.loops, max_len=2048, device=device)

    # Compile and capture every wavefront graph shape (1 to `loops` tokens in
    # flight, one or two readouts) before timing; the sequential graphs are
    # captured during each prompt's untimed prefill.
    warm_prompt = tokenizer.encode(PROMPTS[0], add_special_tokens=False)
    sampling = dict(temperature=args.temperature, top_p=args.top_p)
    model.generate_wavefront(warm_prompt, 4 * args.loops, **sampling)
    greedy = args.temperature == 0

    rows, totals = [], {"A": [0, 0.0], "B": [0, 0.0], "W": [0, 0.0]}
    accepted = commits = 0
    in_flight = []
    document = ""
    if args.context:
        notes = tokenizer.encode((ROOT / "EXPERIMENTS.md").read_text(), add_special_tokens=False)
        document = ("Here are some engineering notes:\n\n" + tokenizer.decode(notes[:args.context])
                    + "\n\nSet the notes aside and answer this: ")
    for index, text in enumerate(PROMPTS[:args.prompts]):
        text = document + text
        prompt = tokenizer.encode(tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True),
            add_special_tokens=False)
        a, seconds_a = model.generate_sequential(prompt, args.tokens, multi=False, **sampling)
        if greedy:
            b, seconds_b = model.generate_sequential(prompt, args.tokens, multi=True)
        else:
            b, seconds_b = None, float("nan")
        w, seconds_w, stats = model.generate_wavefront(prompt, args.tokens, seed=index,
                                                       **sampling)
        for name, seconds in (("A", seconds_a), ("B", seconds_b), ("W", seconds_w)):
            totals[name][0] += args.tokens - 1
            totals[name][1] += seconds
        accepted += stats.accepted
        commits += stats.commits
        in_flight += stats.tokens_per_step
        b_rate = f"{(args.tokens - 1) / seconds_b:.1f}" if greedy else "n/a"
        row = (f"| {index} | {len(prompt)} | {(args.tokens - 1) / seconds_a:.1f} "
               f"| {b_rate} | {(args.tokens - 1) / seconds_w:.1f} "
               f"| {stats.accepted / max(stats.commits, 1):.1%} "
               + (f"| {w == b} | {shared_prefix(w, b)} | {shared_prefix(a, b)} |" if greedy
                  else "| n/a | n/a | n/a |"))
        rows.append(row)
        print(row, flush=True)

    rate = {name: count / seconds for name, (count, seconds) in totals.items()}
    lines = [f"# Wavefront self-speculative decoding, stand-alone prototype "
             f"({torch.cuda.get_device_name()}, {args.loops} loops, drafts at loop 1, "
             + ("greedy" if greedy else f"temperature {args.temperature}, top-p {args.top_p}")
             + ")", "",
             f"{args.prompts} chat prompts"
             + (f" after about {args.context} tokens of notes" if args.context else "")
             + f", {args.tokens} tokens each, decode time only. "
             "A: sequential, single-token kernels. B: sequential, multi-token kernels. "
             "W: wavefront.", "",
             "| Prompt | Prompt tokens | A tok/s | B tok/s | W tok/s | Drafts accepted "
             "| W equals B | Shared W/B | Shared A/B |",
             "|---:|---:|---:|---:|---:|---:|---|---:|---:|"] + rows + [
             "", f"- Overall: A {rate['A']:.1f} tok/s ({1000 / rate['A']:.2f} ms per token), "
             + (f"B {rate['B']:.1f} tok/s, " if greedy else "")
             + f"W {rate['W']:.1f} tok/s ({1000 / rate['W']:.2f} ms per token); "
             f"W is {rate['W'] / rate['A']:.2f}x A.",
             f"- Drafts accepted: {accepted / max(commits, 1):.1%} of {commits} verified drafts; "
             f"mean tokens in flight per step {statistics.mean(in_flight):.2f}."]
    report = "\n".join(lines) + "\n"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / "results" / f"wavefront_bench_{stamp}.md"
    out_path.write_text(report)
    print(report)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
