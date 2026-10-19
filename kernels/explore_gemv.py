"""Diagnoses where the single-token NVFP4 GEMV loses time and tries variants.

Times, in the same decode-order CUDA-graph setup as bench_gemv.py:

- stream_read: the fastest plain read of each projection's bytes (weights and
  block scales) by a single kernel, i.e. the practical ceiling for one launch;
- load_only: the prototype's access pattern without FP4 decoding or math;
- v1: the prototype kernel (best configuration per shape from a sweep);
- pipelined: v1 with software pipelining of the K loop;
- wide: 32-bit weight loads, eight FP4 values decoded per word;
- whole_rows: each program loads its rows' full K extent at once.

Every GEMV variant is checked against the dequantized FP32 reference.

Usage: .venv/bin/python kernels/explore_gemv.py
"""

import argparse
import itertools
import sys
from datetime import datetime, timezone

import torch
import triton
import triton.language as tl

from bench_gemv import (BYTES_PER_PARAM, PEAK_BYTES_PER_S, ROOT, decode_order, dequantize,
                        kinds_for_layer, load_projections, relative_error, require_idle_gpu,
                        time_graph)
from w4a16_gemv import _GROUP, _e2m1_to_f32_scaled, gemv_rows

KINDS = ("gate", "attn_out", "gate_up", "down")
VARIANTS = ("stream_read", "load_only", "v1", "pipelined", "wide", "wide_f16", "wide_f16_xperm",
            "whole_rows", "natural_f32", "natural_bf16", "natural_bf16_gated", "gated_strided",
            "segmented_f32", "segmented_bf16", "segmented_bf16_gated", "gate_nibble_major")
REPLAYS = 50


@triton.jit
def _e2m1_to_f32(nibble):
    """Decodes 4-bit E2M1 codes to FP32 with selects (the first prototype's decode)."""
    magnitude = nibble & 0x07
    value = (0x3F000000 + (magnitude.to(tl.int32) << 22)).to(tl.float32, bitcast=True)
    value = tl.where(magnitude == 0, 0.0, value)
    value = tl.where(magnitude == 1, 0.5, value)
    return tl.where((nibble & 0x08) != 0, -value, value)


@triton.jit
def _stream_read_kernel(w_ptr, n_words, s_ptr, n_scale_words, out_ptr,
                        W_BLOCK: tl.constexpr, S_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * W_BLOCK + tl.arange(0, W_BLOCK)
    total = tl.sum(tl.load(w_ptr + offs, mask=offs < n_words, other=0))
    scale_offs = pid * S_BLOCK + tl.arange(0, S_BLOCK)
    total += tl.sum(tl.load(s_ptr + scale_offs, mask=scale_offs < n_scale_words, other=0))
    tl.store(out_ptr + pid, total)


@triton.jit
def _load_only_kernel(x_ptr, w_ptr, s_ptr, g_ptr, y_ptr, N, K: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    pairs = tl.arange(0, BLOCK_K // 2)
    groups = tl.arange(0, BLOCK_K // _GROUP)
    acc = tl.zeros([BLOCK_N], dtype=tl.int32)
    for k0 in tl.range(0, K, BLOCK_K):
        packed = tl.load(w_ptr + rows[:, None] * (K // 2) + (k0 // 2 + pairs)[None, :],
                         mask=row_mask[:, None], other=0)
        scales = tl.load(s_ptr + rows[:, None] * (K // _GROUP) + (k0 // _GROUP + groups)[None, :],
                         mask=row_mask[:, None], other=0)
        acc += tl.sum(packed.to(tl.int32), axis=1) + tl.sum(scales.to(tl.int32), axis=1)
    tl.store(y_ptr + rows, acc.to(tl.bfloat16), mask=row_mask)


@triton.jit
def _pipelined_kernel(x_ptr, w_ptr, s_ptr, g_ptr, y_ptr, N, K: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, STAGES: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    pairs = tl.arange(0, BLOCK_K // 2)
    groups = tl.arange(0, BLOCK_K // _GROUP)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K, num_stages=STAGES):
        packed = tl.load(w_ptr + rows[:, None] * (K // 2) + (k0 // 2 + pairs)[None, :],
                         mask=row_mask[:, None], other=0)
        x_even = tl.load(x_ptr + k0 + 2 * pairs).to(tl.float32)
        x_odd = tl.load(x_ptr + k0 + 2 * pairs + 1).to(tl.float32)
        products = (_e2m1_to_f32(packed & 0x0F) * x_even[None, :]
                    + _e2m1_to_f32(packed >> 4) * x_odd[None, :])
        partial = tl.sum(tl.reshape(products, [BLOCK_N, BLOCK_K // _GROUP, 8]), axis=2)
        scales = tl.load(s_ptr + rows[:, None] * (K // _GROUP) + (k0 // _GROUP + groups)[None, :],
                         mask=row_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(partial * scales, axis=1)
    g = tl.load(g_ptr + rows, mask=row_mask, other=0.0)
    tl.store(y_ptr + rows, (acc * g).to(tl.bfloat16), mask=row_mask)


@triton.jit
def _wide_kernel(x_ptr, w_ptr, s_ptr, g_ptr, y_ptr, N, K: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """w_ptr is the weight viewed as int32 [N, K/8]: word w holds inputs 8w..8w+7."""
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    words = tl.arange(0, BLOCK_W)
    groups = tl.arange(0, BLOCK_W // 2)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_W * 8):
        packed = tl.load(w_ptr + rows[:, None] * (K // 8) + (k0 // 8 + words)[None, :],
                         mask=row_mask[:, None], other=0)
        word_sums = tl.zeros([BLOCK_N, BLOCK_W], dtype=tl.float32)
        for j in tl.static_range(8):
            xj = tl.load(x_ptr + k0 + 8 * words + j).to(tl.float32)
            word_sums += _e2m1_to_f32((packed >> (4 * j)) & 0x0F) * xj[None, :]
        partial = tl.sum(tl.reshape(word_sums, [BLOCK_N, BLOCK_W // 2, 2]), axis=2)
        scales = tl.load(s_ptr + rows[:, None] * (K // _GROUP) + (k0 // _GROUP + groups)[None, :],
                         mask=row_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(partial * scales, axis=1)
    g = tl.load(g_ptr + rows, mask=row_mask, other=0.0)
    tl.store(y_ptr + rows, (acc * g).to(tl.bfloat16), mask=row_mask)


@triton.jit
def _wide_f16_kernel(x_ptr, w_ptr, s_ptr, g_ptr, y_ptr, N, K: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """Like _wide_kernel, with the select-free FP16 decode."""
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    words = tl.arange(0, BLOCK_W)
    groups = tl.arange(0, BLOCK_W // 2)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_W * 8):
        packed = tl.load(w_ptr + rows[:, None] * (K // 8) + (k0 // 8 + words)[None, :],
                         mask=row_mask[:, None], other=0)
        word_sums = tl.zeros([BLOCK_N, BLOCK_W], dtype=tl.float32)
        for j in tl.static_range(8):
            xj = tl.load(x_ptr + k0 + 8 * words + j).to(tl.float32)
            word_sums += _e2m1_to_f32_scaled(packed >> (4 * j)) * xj[None, :]
        partial = tl.sum(tl.reshape(word_sums, [BLOCK_N, BLOCK_W // 2, 2]), axis=2)
        scales = tl.load(s_ptr + rows[:, None] * (K // _GROUP) + (k0 // _GROUP + groups)[None, :],
                         mask=row_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(partial * scales, axis=1)
    g = tl.load(g_ptr + rows, mask=row_mask, other=0.0) * 16384.0  # undoes the 2^-14 decode factor
    tl.store(y_ptr + rows, (acc * g).to(tl.bfloat16), mask=row_mask)


@triton.jit
def _wide_f16_xperm_kernel(xp_ptr, w_ptr, s_ptr, g_ptr, y_ptr, N, K: tl.constexpr,
                           BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """Like _wide_f16_kernel, reading x as FP32 in nibble-major order.

    xp_ptr points at x_perm [8, K/8] with x_perm[j, w] = x[8w + j], so the
    inputs paired with nibble j of consecutive words are contiguous. In a fused
    pipeline the previous kernel writes this layout directly.
    """
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    words = tl.arange(0, BLOCK_W)
    groups = tl.arange(0, BLOCK_W // 2)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_W * 8):
        packed = tl.load(w_ptr + rows[:, None] * (K // 8) + (k0 // 8 + words)[None, :],
                         mask=row_mask[:, None], other=0)
        word_sums = tl.zeros([BLOCK_N, BLOCK_W], dtype=tl.float32)
        for j in tl.static_range(8):
            xj = tl.load(xp_ptr + j * (K // 8) + k0 // 8 + words)
            word_sums += _e2m1_to_f32_scaled(packed >> (4 * j)) * xj[None, :]
        partial = tl.sum(tl.reshape(word_sums, [BLOCK_N, BLOCK_W // 2, 2]), axis=2)
        scales = tl.load(s_ptr + rows[:, None] * (K // _GROUP) + (k0 // _GROUP + groups)[None, :],
                         mask=row_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(partial * scales, axis=1)
    g = tl.load(g_ptr + rows, mask=row_mask, other=0.0) * 16384.0  # undoes the 2^-14 decode factor
    tl.store(y_ptr + rows, (acc * g).to(tl.bfloat16), mask=row_mask)


@triton.jit
def _natural_3d_kernel(x_ptr, gate_ptr, w_ptr, s_ptr, g_ptr, y_ptr, N, K: tl.constexpr,
                       BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr, GATED: tl.constexpr):
    """Reads x in natural order as contiguous [BLOCK_W, 8] tiles (any float type).

    Each weight word is decoded into its eight values at once ([BLOCK_N, BLOCK_W,
    8]) and multiplied by the matching input tile, so no nibble-major copy of x
    is needed. With GATED, the input is x * gate (gate natural-order FP32).
    """
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    words = tl.arange(0, BLOCK_W)
    nibbles = tl.arange(0, 8)
    groups = tl.arange(0, BLOCK_W // 2)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_W * 8):
        packed = tl.load(w_ptr + rows[:, None] * (K // 8) + (k0 // 8 + words)[None, :],
                         mask=row_mask[:, None], other=0)
        offsets = k0 + words[:, None] * 8 + nibbles[None, :]
        x = tl.load(x_ptr + offsets).to(tl.float32)
        if GATED:
            x = x * tl.load(gate_ptr + offsets)
        values = _e2m1_to_f32_scaled(packed[:, :, None] >> (4 * nibbles)[None, None, :])
        word_sums = tl.sum(values * x[None, :, :], axis=2)
        partial = tl.sum(tl.reshape(word_sums, [BLOCK_N, BLOCK_W // 2, 2]), axis=2)
        scales = tl.load(s_ptr + rows[:, None] * (K // _GROUP) + (k0 // _GROUP + groups)[None, :],
                         mask=row_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(partial * scales, axis=1)
    g = tl.load(g_ptr + rows, mask=row_mask, other=0.0) * 16384.0
    tl.store(y_ptr + rows, (acc * g).to(tl.bfloat16), mask=row_mask)


@triton.jit
def _gated_strided_kernel(x_ptr, gate_ptr, w_ptr, s_ptr, g_ptr, y_ptr, N, K: tl.constexpr,
                          BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """The production attention-output read: strided natural-order x times gate."""
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    acc, _ = gemv_rows(x_ptr, w_ptr, s_ptr, rows, row_mask, K, BLOCK_N, BLOCK_W, False, True,
                       gate_ptr)
    g = tl.load(g_ptr + rows, mask=row_mask, other=0.0) * 16384.0
    tl.store(y_ptr + rows, (acc * g).to(tl.bfloat16), mask=row_mask)


@triton.jit
def _segmented_kernel(x_ptr, gate_ptr, w_ptr, s_ptr, g_ptr, y_ptr, N, K: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr, GATED: tl.constexpr):
    """Segment-major weights: nibble j of word w holds input j * (K/8) + w.

    The input is read in natural order (any float type): nibble stream j is the
    contiguous segment x[j * K/8 : (j + 1) * K/8]. Sixteen consecutive words of
    one stream come from one original 16-element block, so the original scale
    array, viewed as [N, 8, K/128], applies unchanged.
    """
    SEGMENT: tl.constexpr = K // 8
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    words = tl.arange(0, BLOCK_W)
    blocks = tl.arange(0, BLOCK_W // 16)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for w0 in tl.range(0, SEGMENT, BLOCK_W):
        packed = tl.load(w_ptr + rows[:, None] * SEGMENT + (w0 + words)[None, :],
                         mask=row_mask[:, None], other=0)
        for j in tl.static_range(8):
            xj = tl.load(x_ptr + j * SEGMENT + w0 + words).to(tl.float32)
            if GATED:
                xj = xj * tl.load(gate_ptr + j * SEGMENT + w0 + words)
            values = _e2m1_to_f32_scaled(packed >> (4 * j)) * xj[None, :]
            partial = tl.sum(tl.reshape(values, [BLOCK_N, BLOCK_W // 16, 16]), axis=2)
            scales = tl.load(s_ptr + rows[:, None] * (K // _GROUP) + j * (SEGMENT // 16)
                             + (w0 // 16 + blocks)[None, :],
                             mask=row_mask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(partial * scales, axis=1)
    g = tl.load(g_ptr + rows, mask=row_mask, other=0.0) * 16384.0
    tl.store(y_ptr + rows, (acc * g).to(tl.bfloat16), mask=row_mask)


@triton.jit
def _gate_nm_kernel(x_ptr, gate_ptr, w_ptr, s_ptr, g_ptr, y_ptr, N, K: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """x in natural order (strided reads), gate in nibble-major FP32 (contiguous)."""
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    words = tl.arange(0, BLOCK_W)
    groups = tl.arange(0, BLOCK_W // 2)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_W * 8):
        packed = tl.load(w_ptr + rows[:, None] * (K // 8) + (k0 // 8 + words)[None, :],
                         mask=row_mask[:, None], other=0)
        word_sums = tl.zeros([BLOCK_N, BLOCK_W], dtype=tl.float32)
        for j in tl.static_range(8):
            xj = (tl.load(x_ptr + k0 + 8 * words + j).to(tl.float32)
                  * tl.load(gate_ptr + j * (K // 8) + k0 // 8 + words))
            word_sums += _e2m1_to_f32_scaled(packed >> (4 * j)) * xj[None, :]
        partial = tl.sum(tl.reshape(word_sums, [BLOCK_N, BLOCK_W // 2, 2]), axis=2)
        scales = tl.load(s_ptr + rows[:, None] * (K // _GROUP) + (k0 // _GROUP + groups)[None, :],
                         mask=row_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(partial * scales, axis=1)
    g = tl.load(g_ptr + rows, mask=row_mask, other=0.0) * 16384.0
    tl.store(y_ptr + rows, (acc * g).to(tl.bfloat16), mask=row_mask)


@triton.jit
def _whole_rows_kernel(x_ptr, w_ptr, s_ptr, g_ptr, y_ptr, N, K: tl.constexpr,
                       BLOCK_N: tl.constexpr, K_POW2: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    pairs = tl.arange(0, K_POW2 // 2)
    pair_mask = pairs < K // 2
    groups = tl.arange(0, K_POW2 // _GROUP)
    group_mask = groups < K // _GROUP
    packed = tl.load(w_ptr + rows[:, None] * (K // 2) + pairs[None, :],
                     mask=row_mask[:, None] & pair_mask[None, :], other=0)
    scales = tl.load(s_ptr + rows[:, None] * (K // _GROUP) + groups[None, :],
                     mask=row_mask[:, None] & group_mask[None, :], other=0.0).to(tl.float32)
    x_even = tl.load(x_ptr + 2 * pairs, mask=pair_mask, other=0.0).to(tl.float32)
    x_odd = tl.load(x_ptr + 2 * pairs + 1, mask=pair_mask, other=0.0).to(tl.float32)
    products = (_e2m1_to_f32(packed & 0x0F) * x_even[None, :]
                + _e2m1_to_f32(packed >> 4) * x_odd[None, :])
    partial = tl.sum(tl.reshape(products, [BLOCK_N, K_POW2 // _GROUP, 8]), axis=2)
    acc = tl.sum(partial * scales, axis=1)
    g = tl.load(g_ptr + rows, mask=row_mask, other=0.0)
    tl.store(y_ptr + rows, (acc * g).to(tl.bfloat16), mask=row_mask)


_X_PERM_CACHE: dict[tuple, torch.Tensor] = {}
_ONES: dict[int, torch.Tensor] = {}


def _ones(k: int, device) -> torch.Tensor:
    if k not in _ONES:
        _ONES[k] = torch.ones(k, dtype=torch.float32, device=device)
    return _ONES[k]


_SEGMENTED: dict[int, torch.Tensor] = {}


def _segment_major(weight: torch.Tensor) -> torch.Tensor:
    """Repacks uint8 [N, K/2] so nibble j of word w holds input j * (K/8) + w.

    Made once per weight (on the untimed warm-up call) and cached.
    """
    key = weight.data_ptr()
    if key not in _SEGMENTED:
        n, half_k = weight.shape
        k = 2 * half_k
        codes = torch.stack((weight & 0xF, weight >> 4), dim=-1).reshape(n, k)
        codes = codes.view(n, 8, k // 8).transpose(1, 2).reshape(n, k)
        _SEGMENTED[key] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return _SEGMENTED[key]


def _x_f32(x: torch.Tensor) -> torch.Tensor:
    """FP32 copy of x, made once per input outside timed graphs."""
    key = (x.data_ptr(), x.numel(), "f32")
    if key not in _X_PERM_CACHE:
        _X_PERM_CACHE[key] = x.float()
    return _X_PERM_CACHE[key]


def x_perm(x: torch.Tensor) -> torch.Tensor:
    """FP32 copy of x in nibble-major order, made once per input outside timed graphs."""
    key = (x.data_ptr(), x.numel())
    if key not in _X_PERM_CACHE:
        _X_PERM_CACHE[key] = x.float().view(-1, 8).T.contiguous()
    return _X_PERM_CACHE[key]


def launch(variant: str, cfg: dict, p, x, out):
    k, n = p.shape
    if variant in ("v1", "load_only", "pipelined"):
        kernel = {"v1": _pipelined_kernel, "load_only": _load_only_kernel,
                  "pipelined": _pipelined_kernel}[variant]
        extra = {} if variant == "load_only" else {"STAGES": cfg.get("stages", 1)}
        # The load-only variant sums raw scale bytes instead of decoding them.
        scale = p.scale.view(torch.uint8) if variant == "load_only" else p.scale
        kernel[(triton.cdiv(n, cfg["block_n"]),)](
            x, p.weight, scale, p.global_scale, out, n, K=k, BLOCK_N=cfg["block_n"],
            BLOCK_K=cfg["block_k"], num_warps=cfg["num_warps"], **extra)
    elif variant in ("wide", "wide_f16", "wide_f16_xperm"):
        kernel = {"wide": _wide_kernel, "wide_f16": _wide_f16_kernel,
                  "wide_f16_xperm": _wide_f16_xperm_kernel}[variant]
        if variant == "wide_f16_xperm":
            x = x_perm(x)
        kernel[(triton.cdiv(n, cfg["block_n"]),)](
            x, p.weight.view(torch.int32), p.scale, p.global_scale, out, n, K=k,
            BLOCK_N=cfg["block_n"], BLOCK_W=cfg["block_w"], num_warps=cfg["num_warps"])
    elif variant in ("natural_f32", "natural_bf16", "natural_bf16_gated"):
        # Gate of ones: the timing includes reading it, the result stays W x.
        gate = _ones(k, x.device)
        xin = _x_f32(x) if variant == "natural_f32" else x
        _natural_3d_kernel[(triton.cdiv(n, cfg["block_n"]),)](
            xin, gate, p.weight.view(torch.int32), p.scale, p.global_scale, out, n, K=k,
            BLOCK_N=cfg["block_n"], BLOCK_W=cfg["block_w"],
            GATED=variant == "natural_bf16_gated", num_warps=cfg["num_warps"])
    elif variant in ("segmented_f32", "segmented_bf16", "segmented_bf16_gated"):
        _segmented_kernel[(triton.cdiv(n, cfg["block_n"]),)](
            _x_f32(x) if variant == "segmented_f32" else x, _ones(k, x.device),
            _segment_major(p.weight).view(torch.int32), p.scale, p.global_scale, out, n, K=k,
            BLOCK_N=cfg["block_n"], BLOCK_W=cfg["block_w"],
            GATED=variant == "segmented_bf16_gated", num_warps=cfg["num_warps"])
    elif variant == "gate_nibble_major":
        # A gate of ones is the same in any order.
        _gate_nm_kernel[(triton.cdiv(n, cfg["block_n"]),)](
            x, _ones(k, x.device), p.weight.view(torch.int32), p.scale, p.global_scale, out,
            n, K=k, BLOCK_N=cfg["block_n"], BLOCK_W=cfg["block_w"],
            num_warps=cfg["num_warps"])
    elif variant == "gated_strided":
        _gated_strided_kernel[(triton.cdiv(n, cfg["block_n"]),)](
            x, _ones(k, x.device), p.weight.view(torch.int32), p.scale, p.global_scale, out,
            n, K=k, BLOCK_N=cfg["block_n"], BLOCK_W=cfg["block_w"],
            num_warps=cfg["num_warps"])
    elif variant == "whole_rows":
        _whole_rows_kernel[(triton.cdiv(n, cfg["block_n"]),)](
            x, p.weight, p.scale, p.global_scale, out, n, K=k, BLOCK_N=cfg["block_n"],
            K_POW2=triton.next_power_of_2(k), num_warps=cfg["num_warps"])
    else:
        raise ValueError(variant)
    return out


def configs(variant: str, k: int) -> list[dict]:
    warps = (2, 4, 8)
    if variant in ("v1", "load_only"):
        return [{"block_n": n, "block_k": bk, "num_warps": w}
                for n, bk, w in itertools.product((2, 4, 8, 16), (256, 512), warps) if k % bk == 0]
    if variant == "pipelined":
        return [{"block_n": n, "block_k": bk, "num_warps": w, "stages": s}
                for n, bk, w, s in itertools.product((2, 4, 8), (256, 512), warps, (2, 3, 4))
                if k % bk == 0]
    if variant in ("wide", "wide_f16", "wide_f16_xperm", "natural_f32", "natural_bf16",
                   "natural_bf16_gated", "gated_strided", "gate_nibble_major"):
        return [{"block_n": n, "block_w": bw, "num_warps": w}
                for n, bw, w in itertools.product((2, 4, 8, 16), (32, 64, 128), warps)
                if k % (bw * 8) == 0]
    if variant.startswith("segmented"):
        # Each loop step covers BLOCK_W words of every nibble stream, so the
        # stream length K/8 must be a multiple of BLOCK_W.
        return [{"block_n": n, "block_w": bw, "num_warps": w}
                for n, bw, w in itertools.product((2, 4, 8, 16), (32, 64, 128), warps)
                if (k // 8) % bw == 0]
    if variant == "whole_rows":
        return [{"block_n": n, "num_warps": w} for n, w in itertools.product((1, 2, 4), (4, 8))]
    raise ValueError(variant)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    variants = parser.parse_args().variants
    require_idle_gpu()
    device = torch.device("cuda")
    torch.manual_seed(0)
    projections, config = load_projections(device)
    order = decode_order(config, loops=2)
    lines = [f"# GEMV variant exploration ({torch.cuda.get_device_name()}, Triton {triton.__version__})",
             "", "Microseconds per call, decode order in a CUDA graph (2 loops); best configuration per "
             "variant. Efficiency is the bandwidth minimum divided by the time.", "",
             "| Projection | Minimum us | " + " | ".join(variants)
             + " | Best variant (config) | Max relative error |",
             "|---|---:|" + "---:|" * len(variants) + "---|---:|"]
    for kind in KINDS:
        keys = [(kind, layer) for layer in order if kind in kinds_for_layer(config, layer, False)]
        k, n = projections[keys[0]].shape
        _X_PERM_CACHE.clear()  # a new input may reuse a freed address
        x = torch.randn(k, dtype=torch.bfloat16, device=device)
        outs = [torch.empty(n, dtype=torch.bfloat16, device=device) for _ in keys]
        ideal_us = k * n * BYTES_PER_PARAM / PEAK_BYTES_PER_S * 1e6
        cells, best = {}, (float("inf"), None, None)

        # Streaming-read ceiling over the same bytes.
        if "stream_read" in variants:
            stream = []
            for w_block, warps in itertools.product((1024, 2048, 4096), (4, 8)):
                def call(p, o, w_block=w_block, warps=warps):
                    words = p.weight.view(torch.int32)
                    scale_words = p.scale.view(torch.int32)
                    programs = triton.cdiv(words.numel(), w_block)
                    s_block = triton.next_power_of_2(triton.cdiv(scale_words.numel(), programs))
                    _stream_read_kernel[(programs,)](words, words.numel(), scale_words,
                                                     scale_words.numel(), o, W_BLOCK=w_block,
                                                     S_BLOCK=s_block, num_warps=warps)
                sinks = [torch.empty(n, dtype=torch.int32, device=device) for _ in keys]
                calls = [lambda p=projections[key], o=o: call(p, o) for key, o in zip(keys, sinks)]
                stream.append(time_graph(calls, REPLAYS) / len(keys) * 1000)
            cells["stream_read"] = min(stream)

        max_error = 0.0
        for variant in [v for v in variants if v != "stream_read"]:
            timings = []
            for cfg in configs(variant, k):
                calls = [lambda p=projections[key], o=o, cfg=cfg: launch(variant, cfg, p, x, o)
                         for key, o in zip(keys, outs)]
                timings.append((time_graph(calls, REPLAYS) / len(keys) * 1000, cfg))
            us, cfg = min(timings, key=lambda t: t[0])
            cells[variant] = us
            if variant != "load_only":
                p = projections[keys[0]]
                reference = dequantize(p) @ x.float()
                out = launch(variant, cfg, p, x, torch.empty(n, dtype=torch.bfloat16, device=device))
                max_error = max(max_error, relative_error(out, reference))
                if us < best[0]:
                    best = (us, variant, cfg)
        lines.append(f"| {kind} ({k} x {n}) | {ideal_us:.1f} | "
                     + " | ".join(f"{cells[v]:.1f} ({ideal_us / cells[v]:.0%})" for v in
                                  variants)
                     + f" | {best[1]} ({best[2]}) | {max_error:.2e} |")
        print(lines[-1], flush=True)

    report = "\n".join(lines) + "\n"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / "results" / f"gemv_explore_{stamp}.md"
    out_path.write_text(report)
    print(report)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
