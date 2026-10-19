"""Sampling for the stand-alone Dory decoder, with lossless speculative acceptance.

filtered_probs applies temperature and top-p (nucleus) filtering, as the
model's default generation settings (temperature 1, top-p 0.95) require.

speculative_accept is the standard speculative-sampling rule (Leviathan et
al. 2023, Chen et al. 2023): a draft token x drawn from a draft distribution q
is accepted with probability min(1, p(x) / q(x)), and otherwise replaced by a
token drawn from the normalized residual max(0, p - q). The committed token is
then distributed exactly as the target distribution p, whatever q is. For the
wavefront, q is the shallow (loop-1) prediction and p the full-depth one, both
filtered the same way.

Draws use the Gumbel-max trick (argmax of log-probability plus Gumbel noise),
which needs no host synchronization, so all of this can run inside a CUDA
graph with the default random generator.
"""

import torch


def filtered_probs(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Probabilities [R, V] from FP32 logits [R, V]: softmax at `temperature`, then
    the smallest set of most likely tokens whose mass reaches top_p, renormalized
    (top_p = 1 keeps every token)."""
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p >= 1.0:
        return probs
    sorted_probs, order = probs.sort(dim=-1, descending=True)
    # A token is dropped when the mass before it already reaches top_p, so the
    # most likely token always stays.
    drop = sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p
    kept = torch.zeros_like(probs).scatter_(-1, order, sorted_probs.masked_fill(drop, 0.0))
    return kept / kept.sum(dim=-1, keepdim=True)


def sample(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """One token [R] from each row of probs [R, V], by the Gumbel-max trick."""
    uniform = torch.rand(probs.shape, device=probs.device, generator=generator)
    gumbel = -torch.log(-torch.log(uniform.clamp_(min=1e-20)))
    return torch.argmax(torch.log(probs) + gumbel, dim=-1)


def speculative_accept(p: torch.Tensor, q: torch.Tensor, drafts: torch.Tensor,
                       valid: torch.Tensor | None = None,
                       generator: torch.Generator | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Speculative acceptance of drafts [R] (int64, each drawn from its row of q)
    against targets p [R, V]. Rows where `valid` [R] is false have no draft: they
    sample from p directly. Returns the committed tokens [R] and whether each
    draft was accepted [R]."""
    rows = torch.arange(p.shape[0], device=p.device)
    ratio = p[rows, drafts] / q[rows, drafts]
    accepted = torch.rand(p.shape[0], device=p.device, generator=generator) < ratio
    residual = (p - q).clamp_min(0.0)
    mass = residual.sum(dim=-1, keepdim=True)
    # A zero residual means p == q, where every draft is accepted; p then only
    # stands in so that the replacement draw is well defined.
    replacement = torch.where(mass > 0, residual / mass.clamp_min(1e-30), p)
    if valid is not None:
        accepted &= valid
        replacement = torch.where(valid[:, None], replacement, p)
    return torch.where(accepted, drafts, sample(replacement, generator)), accepted
