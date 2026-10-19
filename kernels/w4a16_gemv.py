"""Single-token W4A16 NVFP4 matrix-vector product in Triton.

Computes y = W x for one activation vector x. W uses the ModelOpt NVFP4 layout
found in the Dory checkpoint: two E2M1 values per byte (even input index in the
low nibble), one FP8 E4M3 scale per 16 consecutive input elements, and a global
FP32 scale. The global scale is passed per output row so that weights merged
from shards with different global scales stay exact.

The kernel reads the weight as 32-bit words (eight FP4 values each) and takes x
as FP32 in nibble-major order (see to_nibble_major), so the inputs paired with
the same nibble position of consecutive words are contiguous. In a fused
pipeline the kernel that produces x can write this layout directly.
"""

import torch
import triton
import triton.language as tl

GROUP_SIZE = 16  # input elements per FP8 block scale
_GROUP = tl.constexpr(GROUP_SIZE)  # the same value, readable inside kernels


@triton.jit
def _e2m1_to_f32_scaled(nibble):
    """Decodes E2M1 codes (low 4 bits of `nibble`) to FP32 values times 2^-14.

    Placing the magnitude bits at FP16 bits 9-11 and the sign at bit 15 yields
    an FP16 number equal to the E2M1 value times 2^-14, with no selects; the
    E2M1 subnormal 0.5 becomes the FP16 subnormal 2^-15, which FP16 represents
    exactly (an FP32 version of this trick could be flushed to zero).
    """
    bits = ((nibble & 0x7) << 9) | ((nibble & 0x8) << 12)
    return bits.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def gemv_rows(x_ptr, weight_ptr, scale_ptr, rows, row_mask, K: tl.constexpr,
              BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr, X_SUMSQ: tl.constexpr,
              X_GATED_NATURAL: tl.constexpr, gate_ptr):
    """Dot products of weight rows `rows` with x, before the global scale.

    Returns (acc, x_sumsq): acc[i] is 2^-14 times row i of W x without its
    global scale; x_sumsq is sum(x^2) over all K inputs when X_SUMSQ is set
    (every program reads all of x, so this costs no extra loads), else 0.
    x_ptr points at x in nibble-major FP32; weight_ptr at the weight as int32.

    With X_GATED_NATURAL, the input is instead x * gate elementwise, with x in
    natural order (any float type) and gate natural-order FP32 at gate_ptr. It
    serves inputs written by kernels whose output layout cannot change; the
    strided loads cost a little bandwidth efficiency.
    """
    words = tl.arange(0, BLOCK_W)
    groups = tl.arange(0, BLOCK_W // 2)  # two words (16 inputs) per block scale
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    x_squares = tl.zeros([BLOCK_W], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_W * 8):
        packed = tl.load(weight_ptr + rows[:, None] * (K // 8) + (k0 // 8 + words)[None, :],
                         mask=row_mask[:, None], other=0)
        word_sums = tl.zeros([BLOCK_N, BLOCK_W], dtype=tl.float32)
        for j in tl.static_range(8):
            if X_GATED_NATURAL:
                natural = k0 + 8 * words + j
                xj = tl.load(x_ptr + natural).to(tl.float32) * tl.load(gate_ptr + natural)
            else:
                xj = tl.load(x_ptr + j * (K // 8) + k0 // 8 + words)
            word_sums += _e2m1_to_f32_scaled(packed >> (4 * j)) * xj[None, :]
            if X_SUMSQ:
                x_squares += xj * xj
        partial = tl.sum(tl.reshape(word_sums, [BLOCK_N, BLOCK_W // 2, 2]), axis=2)
        scales = tl.load(scale_ptr + rows[:, None] * (K // _GROUP)
                         + (k0 // _GROUP + groups)[None, :],
                         mask=row_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(partial * scales, axis=1)
    return acc, tl.sum(x_squares, axis=0)


@triton.jit
def _w4a16_gemv_kernel(
    x_ptr, weight_ptr, scale_ptr, global_scale_ptr, y_ptr, N,
    K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    acc, _ = gemv_rows(x_ptr, weight_ptr, scale_ptr, rows, row_mask, K, BLOCK_N, BLOCK_W, False,
                       False, x_ptr)
    # 16384 undoes the 2^-14 factor of the decode.
    global_scale = tl.load(global_scale_ptr + rows, mask=row_mask, other=0.0) * 16384.0
    tl.store(y_ptr + rows, (acc * global_scale).to(tl.bfloat16), mask=row_mask)


def to_nibble_major(x: torch.Tensor) -> torch.Tensor:
    """Returns x as FP32 [8, K/8] with element [j, w] = x[8w + j]."""
    return x.float().view(-1, 8).T.contiguous()


def w4a16_gemv(x_nibble_major: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
               global_scale: torch.Tensor, out: torch.Tensor, *,
               block_n: int, block_w: int, num_warps: int) -> torch.Tensor:
    """Writes W x into `out` (BF16, [N]).

    x_nibble_major: FP32 [8, K/8] from to_nibble_major; weight: uint8 [N, K/2];
    scale: float8_e4m3fn [N, K/16]; global_scale: float32 [N]. K must be a
    multiple of 8 * block_w, and block_w must be even.
    """
    n, half_k = weight.shape
    k = 2 * half_k
    if k % (8 * block_w) or block_w % 2:
        raise ValueError(f"K={k} must be a multiple of 8 * block_w={8 * block_w}")
    if (x_nibble_major.shape != (8, k // 8) or x_nibble_major.dtype != torch.float32
            or scale.shape != (n, k // GROUP_SIZE) or global_scale.shape != (n,)):
        raise ValueError("tensor shapes or dtypes do not match the weight")
    _w4a16_gemv_kernel[(triton.cdiv(n, block_n),)](
        x_nibble_major, weight.view(torch.int32), scale, global_scale, out, n,
        K=k, BLOCK_N=block_n, BLOCK_W=block_w, num_warps=num_warps)
    return out


@triton.jit
def gemv_rows_tokens(x_ptr, weight_ptr, scale_ptr, rows, row_mask, K: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr, M: tl.constexpr,
                     M_PAD: tl.constexpr):
    """gemv_rows for M tokens: [BLOCK_N, M_PAD] dot products, before the global scale.

    x_ptr points at M nibble-major FP32 inputs, [M, 8, K/8]. Each weight word is
    decoded once and multiplied with every token's inputs, so the weight bytes
    read are those of a single token; columns M..M_PAD-1 of the result are 0.
    """
    words = tl.arange(0, BLOCK_W)
    groups = tl.arange(0, BLOCK_W // 2)
    tokens = tl.arange(0, M_PAD)
    token_mask = tokens < M
    acc = tl.zeros([BLOCK_N, M_PAD], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_W * 8):
        packed = tl.load(weight_ptr + rows[:, None] * (K // 8) + (k0 // 8 + words)[None, :],
                         mask=row_mask[:, None], other=0)
        word_sums = tl.zeros([BLOCK_N, M_PAD, BLOCK_W], dtype=tl.float32)
        for j in tl.static_range(8):
            xj = tl.load(x_ptr + tokens[:, None] * K + j * (K // 8) + k0 // 8 + words[None, :],
                         mask=token_mask[:, None], other=0.0)
            w = _e2m1_to_f32_scaled(packed >> (4 * j))
            word_sums += w[:, None, :] * xj[None, :, :]
        partial = tl.sum(tl.reshape(word_sums, [BLOCK_N, M_PAD, BLOCK_W // 2, 2]), axis=3)
        scales = tl.load(scale_ptr + rows[:, None] * (K // _GROUP)
                         + (k0 // _GROUP + groups)[None, :],
                         mask=row_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(partial * scales[:, None, :], axis=2)
    return acc


@triton.jit
def _w4a16_gemv_tokens_kernel(
    x_ptr, weight_ptr, scale_ptr, global_scale_ptr, y_ptr, N,
    K: tl.constexpr, M: tl.constexpr, M_PAD: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    acc = gemv_rows_tokens(x_ptr, weight_ptr, scale_ptr, rows, row_mask, K, BLOCK_N, BLOCK_W,
                           M, M_PAD)
    global_scale = tl.load(global_scale_ptr + rows, mask=row_mask, other=0.0) * 16384.0
    tokens = tl.arange(0, M_PAD)
    tl.store(y_ptr + tokens[None, :] * N + rows[:, None],
             (acc * global_scale[:, None]).to(tl.bfloat16),
             mask=row_mask[:, None] & (tokens < M)[None, :])


def w4a16_gemv_tokens(x_nibble_major: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                      global_scale: torch.Tensor, out: torch.Tensor, *,
                      block_n: int, block_w: int, num_warps: int) -> torch.Tensor:
    """Writes W x_m for each of M tokens into `out` (BF16, [M, N]).

    x_nibble_major: FP32 [M, 8, K/8], each token as from to_nibble_major; the
    other arguments as for w4a16_gemv.
    """
    n, half_k = weight.shape
    k = 2 * half_k
    m = x_nibble_major.shape[0]
    if k % (8 * block_w) or block_w % 2:
        raise ValueError(f"K={k} must be a multiple of 8 * block_w={8 * block_w}")
    if (x_nibble_major.shape != (m, 8, k // 8) or x_nibble_major.dtype != torch.float32
            or not x_nibble_major.is_contiguous() or out.shape != (m, n)):
        raise ValueError("tensor shapes or dtypes do not match the weight")
    _w4a16_gemv_tokens_kernel[(triton.cdiv(n, block_n),)](
        x_nibble_major, weight.view(torch.int32), scale, global_scale, out, n,
        K=k, M=m, M_PAD=triton.next_power_of_2(m), BLOCK_N=block_n, BLOCK_W=block_w,
        num_warps=num_warps)
    return out


@triton.jit
def gemm_rows_tc(x_ptr, weight_ptr, scale_ptr, rows, row_mask, K: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr, M: tl.constexpr,
                 M_PAD: tl.constexpr, k_begin, k_end):
    """gemv_rows_tokens on tensor cores: [BLOCK_N, M_PAD] dot products of
    BLOCK_N weight rows with M nibble-major inputs ([M, 8, K/8] FP32), before
    the global scale.

    Each weight word is decoded once, multiplied by its block scale (exact in
    FP16: an E2M1 value times an E4M3 scale needs at most 5 significant bits
    and stays within 2688), and the eight nibble streams of a tile go through
    tl.dot against the matching input slices in FP16. BLOCK_N, BLOCK_W and
    M_PAD must be at least 16. Only inputs k_begin <= k < k_end contribute
    (multiples of 8 * BLOCK_W), so several programs can split one row's K.
    """
    words = tl.arange(0, BLOCK_W)
    groups = tl.arange(0, BLOCK_W // 2)
    tokens = tl.arange(0, M_PAD)
    token_mask = tokens < M
    acc = tl.zeros([BLOCK_N, M_PAD], dtype=tl.float32)
    for k0 in tl.range(k_begin, k_end, BLOCK_W * 8):
        packed = tl.load(weight_ptr + rows[:, None] * (K // 8) + (k0 // 8 + words)[None, :],
                         mask=row_mask[:, None], other=0)
        scales = tl.load(scale_ptr + rows[:, None] * (K // _GROUP)
                         + (k0 // _GROUP + groups)[None, :],
                         mask=row_mask[:, None], other=0.0).to(tl.float32) * 16384.0
        # Each block scale covers two consecutive words (16 inputs).
        word_scales = tl.reshape(tl.broadcast_to(scales[:, :, None], [BLOCK_N, BLOCK_W // 2, 2]),
                                 [BLOCK_N, BLOCK_W])
        for j in tl.static_range(8):
            w = (_e2m1_to_f32_scaled(packed >> (4 * j)) * word_scales).to(tl.float16)
            xj = tl.load(x_ptr + tokens[None, :] * K + j * (K // 8) + k0 // 8 + words[:, None],
                         mask=token_mask[None, :], other=0.0).to(tl.float16)
            acc = tl.dot(w, xj, acc)
    return acc


@triton.jit
def _w4a16_gemm_tc_kernel(
    x_ptr, weight_ptr, scale_ptr, global_scale_ptr, y_ptr, N,
    K: tl.constexpr, M: tl.constexpr, M_PAD: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr, SPLIT_K: tl.constexpr,
):
    # With SPLIT_K > 1, y_ptr is FP32 [SPLIT_K, M, N] partial sums (already
    # globally scaled) that the consumer adds up; otherwise BF16 [M, N].
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    split = tl.program_id(1)
    k_per_split: tl.constexpr = K // SPLIT_K
    acc = gemm_rows_tc(x_ptr, weight_ptr, scale_ptr, rows, row_mask, K, BLOCK_N, BLOCK_W, M,
                       M_PAD, split * k_per_split, (split + 1) * k_per_split)
    global_scale = tl.load(global_scale_ptr + rows, mask=row_mask, other=0.0)
    tokens = tl.arange(0, M_PAD)
    mask = row_mask[:, None] & (tokens < M)[None, :]
    if SPLIT_K == 1:
        tl.store(y_ptr + tokens[None, :] * N + rows[:, None],
                 (acc * global_scale[:, None]).to(tl.bfloat16), mask=mask)
    else:
        tl.store(y_ptr + (split * M + tokens[None, :]) * N + rows[:, None],
                 acc * global_scale[:, None], mask=mask)


def w4a16_gemm_tc(x_nibble_major: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                  global_scale: torch.Tensor, out: torch.Tensor, *,
                  block_n: int, block_w: int, num_warps: int, split_k: int = 1,
                  num_stages: int = 3) -> torch.Tensor:
    """Tensor-core version of w4a16_gemv_tokens for 1 to 16 tokens; same arguments.

    With split_k > 1, `out` is FP32 [split_k, M, N] and receives partial sums
    over split_k slices of K, which the caller adds up.
    """
    n, half_k = weight.shape
    k = 2 * half_k
    m = x_nibble_major.shape[0]
    if k % (8 * block_w * split_k) or block_w % 2 or min(block_n, block_w) < 16 or m > 16:
        raise ValueError("needs block_n, block_w >= 16, K a multiple of 8 * block_w * split_k, "
                         "M <= 16")
    expected = (m, n) if split_k == 1 else (split_k, m, n)
    if (x_nibble_major.shape != (m, 8, k // 8) or x_nibble_major.dtype != torch.float32
            or not x_nibble_major.is_contiguous() or out.shape != expected):
        raise ValueError("tensor shapes or dtypes do not match the weight")
    _w4a16_gemm_tc_kernel[(triton.cdiv(n, block_n), split_k)](
        x_nibble_major, weight.view(torch.int32), scale, global_scale, out, n,
        K=k, M=m, M_PAD=16, BLOCK_N=block_n, BLOCK_W=block_w, SPLIT_K=split_k,
        num_warps=num_warps, num_stages=num_stages)
    return out
