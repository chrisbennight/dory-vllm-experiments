"""Pieces shared by the fused Dory sublayers: the residual update and constants.

Every Dory sublayer ends with the same normalized interpolation between its
input a and its output b,

    h = n(n(a) + lr * (n(b) - n(a))),  n(v) = v / sqrt(sum(v^2) + EPS),
    lr = |alpha * alpha_scale|,

which vLLM implements as slerp_alpha. The kernel below computes it in FP32 and
writes h both as BF16 (the residual stream) and in nibble-major FP32 (the input
layout of the next GEMV).
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

EPS = 1e-6  # as in vLLM's Dory justnorm


@dataclass
class LaunchConfig:
    """Launch parameters of a GEMV-based kernel."""
    block_n: int
    block_w: int
    num_warps: int


@triton.jit
def hidden_tile(H: tl.constexpr, W_BLOCK: tl.constexpr):
    """Natural-order offsets [W_BLOCK, 8] of a hidden vector, and their mask.

    Element (w, j) is index 8w + j, so tl.sum over the tile is a sum over the
    vector and store_hidden can write the nibble-major copy (index j * H/8 + w)
    with coalesced stores: Triton transposes the tile through shared memory
    instead of issuing one scattered store per element.
    """
    w = tl.arange(0, W_BLOCK)[:, None]
    j = tl.arange(0, 8)[None, :]
    return w * 8 + j, (w < H // 8) & (j < 8)


@triton.jit
def store_hidden(h_ptr, h_nibble_major_ptr, h, H: tl.constexpr, W_BLOCK: tl.constexpr):
    """Stores the [W_BLOCK, 8] tile h as BF16 (natural order) and nibble-major FP32."""
    w = tl.arange(0, W_BLOCK)[:, None]
    j = tl.arange(0, 8)[None, :]
    mask = (w < H // 8) & (j < 8)
    tl.store(h_ptr + w * 8 + j, h.to(tl.bfloat16), mask=mask)
    tl.store(h_nibble_major_ptr + j * (H // 8) + w, h, mask=mask)


@triton.jit
def _residual_update_kernel(
    a_ptr, b_ptr, alpha_ptr, h_ptr, h_nibble_major_ptr, alpha_scale, eps,
    H: tl.constexpr, W_BLOCK: tl.constexpr,
):
    offsets, mask = hidden_tile(H, W_BLOCK)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    lr = tl.abs(tl.load(alpha_ptr + offsets, mask=mask, other=0.0).to(tl.float32) * alpha_scale)
    ya = a * tl.rsqrt(tl.sum(a * a) + eps)
    yb = b * tl.rsqrt(tl.sum(b * b) + eps)
    z = ya + lr * (yb - ya)
    h = z * tl.rsqrt(tl.sum(z * z) + eps)
    store_hidden(h_ptr, h_nibble_major_ptr, h, H, W_BLOCK)


def residual_update(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor, alpha_scale: float,
                    h_out: torch.Tensor, h_out_nibble_major: torch.Tensor,
                    num_warps: int = 4) -> None:
    """a: BF16 [H] sublayer input; b: FP32 [H] sublayer output; alpha: [H]."""
    hidden = a.numel()
    if h_out.data_ptr() == a.data_ptr():
        raise ValueError("output must not alias the sublayer input")
    _residual_update_kernel[(1,)](
        a, b, alpha, h_out, h_out_nibble_major, alpha_scale, EPS,
        H=hidden, W_BLOCK=triton.next_power_of_2(hidden // 8), num_warps=num_warps)


@triton.jit
def _to_residual_kernel(a_ptr, b_ptr, h_ptr, h_nibble_major_ptr, eps, H: tl.constexpr,
                        W_BLOCK: tl.constexpr, HAS_B: tl.constexpr, NORMALIZE: tl.constexpr):
    offsets, mask = hidden_tile(H, W_BLOCK)
    x = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_B:
        x += tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if NORMALIZE:
        x = x * tl.rsqrt(tl.sum(x * x) + eps)
    store_hidden(h_ptr, h_nibble_major_ptr, x, H, W_BLOCK)


def to_residual(a: torch.Tensor, b: torch.Tensor | None, h_out: torch.Tensor,
                h_out_nibble_major: torch.Tensor, normalize: bool, num_warps: int = 4) -> None:
    """Writes n(a + b) (or a + b, or a) as BF16 and nibble-major FP32.

    Produces a sublayer input where the model forms the hidden state outside a
    sublayer: from the embedding (no normalization), and at the entry (n(state +
    input)) and exit (n(h)) of the recurrent loop.
    """
    hidden = a.numel()
    _to_residual_kernel[(1,)](
        a, b if b is not None else a, h_out, h_out_nibble_major, EPS, H=hidden,
        W_BLOCK=triton.next_power_of_2(hidden // 8), HAS_B=b is not None, NORMALIZE=normalize,
        num_warps=num_warps)


@triton.jit
def residual_inputs(a_ptr, alpha_ptr, rows, row_mask, alpha_scale):
    """This program's rows of the sublayer input a and of lr, for residual_epilogue.

    Called before the GEMV loop so that the loads overlap it.
    """
    a = tl.load(a_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    lr = tl.abs(tl.load(alpha_ptr + rows, mask=row_mask, other=0.0).to(tl.float32) * alpha_scale)
    return a, lr


@triton.jit
def residual_epilogue(b, a, lr, rows, row_mask, a_ptr, alpha_ptr, b_ptr, partials_ptr,
                      counter_ptr, h_ptr, h_nibble_major_ptr, alpha_scale, eps,
                      H: tl.constexpr, NUM_PROGRAMS: tl.constexpr, P_BLOCK: tl.constexpr,
                      W_BLOCK: tl.constexpr):
    """Residual update folded into the kernel that produces the sublayer output b.

    a and lr are this program's rows from residual_inputs. Every program stores
    its rows of b and five partial sums over them; the last program to finish
    combines the partial sums (in program order, so results are deterministic)
    and writes h = n(n(a) + lr * (n(b) - n(a))) as BF16 and nibble-major FP32.
    With z = (1 - lr) a / |a| + lr b / |b|,

        |z|^2 = S1 / |a|^2 + 2 S2 / (|a| |b|) + S3 / |b|^2,
        S1 = sum((1 - lr)^2 a^2), S2 = sum((1 - lr) lr a b), S3 = sum(lr^2 b^2),

    so no pass over H is needed before the final one. partials_ptr is FP32
    [5, NUM_PROGRAMS]; the counter is a zeroed int32 that the last program resets
    for the next launch.
    """
    tl.store(b_ptr + rows, b, mask=row_mask)
    keep = 1.0 - lr
    pid = tl.program_id(0)
    tl.store(partials_ptr + pid, tl.sum(a * a))
    tl.store(partials_ptr + NUM_PROGRAMS + pid, tl.sum(keep * keep * a * a))
    tl.store(partials_ptr + 2 * NUM_PROGRAMS + pid, tl.sum(b * b))
    tl.store(partials_ptr + 3 * NUM_PROGRAMS + pid, tl.sum(keep * lr * a * b))
    tl.store(partials_ptr + 4 * NUM_PROGRAMS + pid, tl.sum(lr * lr * b * b))
    # Every thread's stores precede this program's arrival, which releases them.
    tl.debug_barrier()
    arrived = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    if arrived == NUM_PROGRAMS - 1:
        # Issue every load before the first use so that they share one round trip.
        programs = tl.arange(0, P_BLOCK)
        present = programs < NUM_PROGRAMS
        p_a = tl.load(partials_ptr + programs, mask=present, other=0.0, cache_modifier=".cg")
        p_s1 = tl.load(partials_ptr + NUM_PROGRAMS + programs, mask=present, other=0.0,
                       cache_modifier=".cg")
        p_b = tl.load(partials_ptr + 2 * NUM_PROGRAMS + programs, mask=present, other=0.0,
                      cache_modifier=".cg")
        p_s2 = tl.load(partials_ptr + 3 * NUM_PROGRAMS + programs, mask=present, other=0.0,
                       cache_modifier=".cg")
        p_s3 = tl.load(partials_ptr + 4 * NUM_PROGRAMS + programs, mask=present, other=0.0,
                       cache_modifier=".cg")
        # One-dimensional on purpose: a tile transpose here would reserve shared
        # memory in every program of the GEMV and lower its occupancy.
        offsets = tl.arange(0, W_BLOCK * 8)
        mask = offsets < H
        a_all = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_all = tl.load(b_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
        alpha_all = tl.load(alpha_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        ra = tl.rsqrt(tl.sum(p_a) + eps)
        rb = tl.rsqrt(tl.sum(p_b) + eps)
        rz = tl.rsqrt(tl.sum(p_s1) * ra * ra + 2.0 * tl.sum(p_s2) * ra * rb
                      + tl.sum(p_s3) * rb * rb + eps)
        lr_all = tl.abs(alpha_all * alpha_scale)
        h = ((1.0 - lr_all) * a_all * ra + lr_all * b_all * rb) * rz
        tl.store(h_ptr + offsets, h.to(tl.bfloat16), mask=mask)
        tl.store(h_nibble_major_ptr + (offsets % 8) * (H // 8) + offsets // 8, h, mask=mask)
        tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")


class ResidualScratch:
    """Partial-sum and arrival-counter buffers for residual_epilogue."""

    def __init__(self, num_programs: int, device: torch.device):
        self.partials = torch.empty(5, num_programs, dtype=torch.float32, device=device)
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
