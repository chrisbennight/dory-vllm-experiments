"""Statistical check that speculative_accept commits tokens distributed as the target.

For fixed target and draft distributions p and q (random, top-p filtered, and
some cases where q misses tokens p allows or p misses tokens q allows), draws N
drafts from q, applies speculative_accept, and compares the committed tokens'
histogram with p: total variation distance and a chi-square statistic against
its degrees of freedom. A correct rule gives a total variation distance near
sqrt(V / N) and a chi-square near V - 1.

Usage: .venv/bin/python kernels/check_speculative_sampling.py
"""

import sys

import torch

from dory_sampling import filtered_probs, sample, speculative_accept

N = 400_000


def check(name: str, p: torch.Tensor, q: torch.Tensor, generator: torch.Generator) -> bool:
    drafts = sample(q.expand(N, -1).contiguous(), generator)
    tokens, accepted = speculative_accept(p.expand(N, -1), q.expand(N, -1), drafts,
                                          generator=generator)
    counts = torch.bincount(tokens, minlength=p.numel()).double()
    expected = p.double() * N
    support = expected > 0
    tv = 0.5 * (counts / N - p.double()).abs().sum().item()
    chi2 = (((counts - expected) ** 2)[support] / expected[support]).sum().item()
    dof = int(support.sum().item()) - 1
    outside = counts[~support].sum().item()
    overlap = torch.minimum(p, q).sum().item()
    ok = outside == 0 and chi2 < dof + 6 * (2 * dof) ** 0.5
    print(f"{name}: acceptance {accepted.double().mean().item():.4f} (overlap {overlap:.4f}), "
          f"total variation {tv:.4f}, chi-square {chi2:.1f} for {dof} degrees of freedom, "
          f"tokens outside p's support {outside:.0f} -> {'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    torch.manual_seed(0)
    results = []
    for vocab, top_p in ((50, 1.0), (50, 0.9), (200, 0.95)):
        logits_p = torch.randn(1, vocab, device=device) * 2
        logits_q = logits_p + torch.randn(1, vocab, device=device)  # a close but different draft
        p = filtered_probs(logits_p, 1.0, top_p)[0]
        q = filtered_probs(logits_q, 1.0, top_p)[0]
        results.append(check(f"vocab {vocab}, top-p {top_p}", p, q, generator))
    # Draft and target nearly disjoint; and identical.
    p = filtered_probs(torch.randn(1, 50, device=device) * 3, 1.0, 0.8)[0]
    q = filtered_probs(torch.randn(1, 50, device=device) * 3, 1.0, 0.8)[0]
    results.append(check("vocab 50, unrelated draft", p, q, generator))
    results.append(check("vocab 50, identical draft", p, p.clone(), generator))
    print("all ok" if all(results) else "FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
