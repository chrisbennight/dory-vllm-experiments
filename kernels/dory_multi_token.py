"""Dory attention and MLP sublayers for several tokens of one sequence at once.

The wavefront schedule advances up to four tokens at different loop depths in
one call of the recurrent block, so every weight matrix is read once for all of
them. These sublayers do the work of FusedAttentionSublayer and FusedMlpSublayer
for T tokens: the weights go through the tensor-core GEMV (w4a16_gemm_tc, whose
cost is nearly flat in the token count), and each token keeps its own position,
KV cache slot and cache, so tokens at different loop depths can share a call.
They read the same weight layouts as the single-token sublayers.

Hidden states are BF16 [T, H] plus nibble-major FP32 [T, 8, H/8], as in the
single-token path. The math follows the single-token kernels: rotary embedding
on adjacent row pairs, the sigmoid output gate, SwiGLU on interleaved gate and
up rows, and the normalized residual update
h = n(n(a) + lr * (n(b) - n(a))) with b normalized by its input's norm.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from dory_attention import AttentionWeights
from dory_common import EPS, hidden_tile, store_hidden
from dory_decode_attention import DecodeAttention
from dory_mlp import MlpWeights
from w4a16_gemv import gemm_rows_tc, w4a16_gemm_tc


@dataclass
class TcConfig:
    """Launch parameters of the tensor-core GEMV."""
    block_n: int
    block_w: int
    num_warps: int
    split_k: int
    num_stages: int


# Fastest configurations for 4 tokens (kernels/bench_gemv_tokens.py --wide).
QKVG = TcConfig(64, 32, 4, 2, 2)
ATTENTION_OUT = TcConfig(64, 32, 4, 4, 2)
GATE_UP = TcConfig(64, 32, 4, 1, 3)
DOWN = TcConfig(128, 32, 8, 4, 3)


@triton.jit
def _qkvg_epilogue_kernel(
    partials_ptr, positions_ptr, cos_sin_ptr, s_gate_ptr, qk_ptr, v_ptr, gate_ptr,
    s_gate_factor, Q_ROWS: tl.constexpr, KV_ROWS: tl.constexpr, HEAD_DIM: tl.constexpr,
    SPLIT_K: tl.constexpr, T: tl.constexpr,
):
    # One program per (head-sized block of 256 rows, token). Rows are q, k, v,
    # output gate, with q and k rows ordered so that rows 2i and 2i + 1 of a
    # head hold rotary dimensions i and i + HEAD_DIM / 2.
    ROWS: tl.constexpr = 2 * (Q_ROWS + KV_ROWS)
    row0 = tl.program_id(0) * HEAD_DIM
    token = tl.program_id(1)
    rows = row0 + tl.arange(0, HEAD_DIM)
    y = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for split in tl.static_range(SPLIT_K):
        y += tl.load(partials_ptr + (split * T + token) * ROWS + rows)
    HALF: tl.constexpr = HEAD_DIM // 2
    if row0 < Q_ROWS + KV_ROWS:
        first, second = tl.split(tl.reshape(y, [HALF, 2]))
        pair = row0 // 2 + tl.arange(0, HALF)
        dim = pair % HALF
        position = tl.load(positions_ptr + token)
        cos = tl.load(cos_sin_ptr + position * HEAD_DIM + dim).to(tl.float32)
        sin = tl.load(cos_sin_ptr + position * HEAD_DIM + HALF + dim).to(tl.float32)
        out = token * (Q_ROWS + KV_ROWS) + (pair // HALF) * HEAD_DIM + dim
        tl.store(qk_ptr + out, first * cos - second * sin)
        tl.store(qk_ptr + out + HALF, second * cos + first * sin)
    elif row0 < Q_ROWS + 2 * KV_ROWS:
        tl.store(v_ptr + token * KV_ROWS + rows - (Q_ROWS + KV_ROWS), y)
    else:
        index = rows - (Q_ROWS + 2 * KV_ROWS)
        s = tl.load(s_gate_ptr + index).to(tl.float32) * s_gate_factor
        tl.store(gate_ptr + token * Q_ROWS + index, tl.sigmoid(y * s * s))


@triton.jit
def _gate_up_swiglu_tc_kernel(
    x_ptr, weight_ptr, scale_ptr, global_scale_ptr, suv_gate_ptr, suv_up_ptr, m_ptr,
    suv_factor, F: tl.constexpr, K: tl.constexpr, T: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Row 2i of the weight is gate row i and row 2i + 1 is up row i.
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < 2 * F
    acc = gemm_rows_tc(x_ptr, weight_ptr, scale_ptr, rows, row_mask, K, BLOCK_N, BLOCK_W, T, 16,
                       0, K)
    y = acc * tl.load(global_scale_ptr + rows, mask=row_mask, other=0.0)[:, None]
    gate, up = tl.split(tl.permute(tl.reshape(y, [BLOCK_N // 2, 2, 16]), (0, 2, 1)))
    index = tl.program_id(0) * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    mask = index < F
    gate *= (tl.load(suv_gate_ptr + index, mask=mask, other=0.0).to(tl.float32)
             * suv_factor)[:, None]
    up *= (tl.load(suv_up_ptr + index, mask=mask, other=0.0).to(tl.float32) * suv_factor)[:, None]
    m = gate * tl.sigmoid(gate) * up
    tokens = tl.arange(0, 16)
    # Nibble-major position of intermediate index i of token t is [t, i % 8, i // 8].
    tl.store(m_ptr + tokens[None, :] * F + ((index % 8) * (F // 8) + index // 8)[:, None], m,
             mask=mask[:, None] & (tokens < T)[None, :])


@triton.jit
def _residual_tokens_kernel(
    partials_ptr, x_ptr, a_ptr, alpha_ptr, h_ptr, h_nibble_major_ptr, alpha_scale, eps,
    H: tl.constexpr, K_IN: tl.constexpr, SPLIT_K: tl.constexpr, T: tl.constexpr,
    W_BLOCK: tl.constexpr, X_BLOCK: tl.constexpr,
):
    # One program per token. b is the sum of the GEMV's split-K partials over
    # the norm of its input x; then h = n(n(a) + lr * (n(b) - n(a))).
    token = tl.program_id(0)
    x_offsets = tl.arange(0, X_BLOCK)
    x = tl.load(x_ptr + token * K_IN + x_offsets, mask=x_offsets < K_IN, other=0.0)
    inverse_x_norm = tl.rsqrt(tl.sum(x * x, axis=0) + eps)
    offsets, mask = hidden_tile(H, W_BLOCK)
    b = tl.zeros(offsets.shape, dtype=tl.float32)
    for split in tl.static_range(SPLIT_K):
        b += tl.load(partials_ptr + (split * T + token) * H + offsets, mask=mask, other=0.0)
    b = b * inverse_x_norm
    a = tl.load(a_ptr + token * H + offsets, mask=mask, other=0.0).to(tl.float32)
    lr = tl.abs(tl.load(alpha_ptr + offsets, mask=mask, other=0.0).to(tl.float32) * alpha_scale)
    ya = a * tl.rsqrt(tl.sum(tl.sum(a * a, axis=1), axis=0) + eps)
    yb = b * tl.rsqrt(tl.sum(tl.sum(b * b, axis=1), axis=0) + eps)
    z = ya + lr * (yb - ya)
    h = z * tl.rsqrt(tl.sum(tl.sum(z * z, axis=1), axis=0) + eps)
    store_hidden(h_ptr + token * H, h_nibble_major_ptr + token * H, h, H, W_BLOCK)


class MultiTokenSublayers:
    """Attention and MLP sublayers for up to max_tokens tokens of one sequence.

    Scratch buffers are owned by the instance, so one instance serves every
    layer of a sequential schedule.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int,
                 num_kv_heads: int, head_dim: int, initializer_range: float, alpha_scale: float,
                 gate_logit_scale_init: float | None, device: torch.device, max_tokens: int = 4):
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.q_rows, self.kv_rows = num_heads * head_dim, num_kv_heads * head_dim
        self.alpha_scale, self.max_tokens = alpha_scale, max_tokens
        self.sqk_factor = head_dim ** 0.5 / initializer_range
        g0 = gate_logit_scale_init if gate_logit_scale_init is not None else hidden_size ** 0.5
        self.s_gate_factor = g0 ** 0.5 / initializer_range
        self.suv_factor = hidden_size ** 0.5
        f32 = dict(dtype=torch.float32, device=device)
        T, H, F = max_tokens, hidden_size, intermediate_size
        self.qkvg_rows = 2 * (self.q_rows + self.kv_rows)
        # Split-K partial sums, flat so that a [split_k, tokens, N] view for fewer
        # than max_tokens tokens stays contiguous.
        self.qkvg_partials = torch.empty(QKVG.split_k * T * self.qkvg_rows, **f32)
        self.qk = torch.empty(T, self.q_rows + self.kv_rows, **f32)
        self.v = torch.empty(T, self.kv_rows, **f32)
        self.gate = torch.empty(T, self.q_rows, **f32)
        self.attention_out = torch.empty(T, 8, self.q_rows // 8, **f32)
        self.out_partials = torch.empty(ATTENTION_OUT.split_k * T * H, **f32)
        self.m = torch.empty(T, 8, F // 8, **f32)
        self.down_partials = torch.empty(DOWN.split_k * T * H, **f32)
        self.decode_attention = DecodeAttention(num_heads, num_kv_heads, head_dim, device,
                                                max_tokens=max_tokens)

    def _gemm(self, x_nm: torch.Tensor, weight, scale, global_scale, out: torch.Tensor,
              cfg: TcConfig) -> None:
        w4a16_gemm_tc(x_nm, weight, scale, global_scale, out, block_n=cfg.block_n,
                      block_w=cfg.block_w, num_warps=cfg.num_warps, split_k=cfg.split_k,
                      num_stages=cfg.num_stages)

    def _residual(self, partials: torch.Tensor, x: torch.Tensor, a: torch.Tensor,
                  alpha: torch.Tensor, h_out: torch.Tensor, h_out_nm: torch.Tensor) -> None:
        tokens, k_in = a.shape[0], x[0].numel()
        _residual_tokens_kernel[(tokens,)](
            partials, x, a, alpha, h_out, h_out_nm, self.alpha_scale, EPS, H=self.hidden_size,
            K_IN=k_in, SPLIT_K=partials.shape[0], T=tokens,
            W_BLOCK=triton.next_power_of_2(self.hidden_size // 8),
            X_BLOCK=triton.next_power_of_2(k_in), num_warps=8)

    def attention(self, w: AttentionWeights, h: torch.Tensor, h_nm: torch.Tensor,
                  positions: torch.Tensor, cos_sin_cache: torch.Tensor,
                  cache_offsets: torch.Tensor, key_cache: torch.Tensor,
                  value_cache: torch.Tensor, block_tables: torch.Tensor,
                  block_table_offsets: torch.Tensor, seq_lens: torch.Tensor, window: int,
                  softmax_scale: float, h_out: torch.Tensor, h_out_nm: torch.Tensor) -> None:
        """One attention sublayer for T = h.shape[0] tokens. positions, seq_lens,
        cache_offsets and block_table_offsets are per token (see
        DecodeAttention.multi_from_raw); h_out must not alias h."""
        tokens = h.shape[0]
        partials = self.qkvg_partials[:QKVG.split_k * tokens * self.qkvg_rows].view(
            QKVG.split_k, tokens, self.qkvg_rows)
        self._gemm(h_nm, w.qkvg_weight, w.qkvg_scale, w.qkvg_global_scale, partials, QKVG)
        _qkvg_epilogue_kernel[(partials.shape[-1] // self.head_dim, tokens)](
            partials, positions, cos_sin_cache, w.s_gate, self.qk, self.v, self.gate,
            self.s_gate_factor, Q_ROWS=self.q_rows, KV_ROWS=self.kv_rows,
            HEAD_DIM=self.head_dim, SPLIT_K=QKVG.split_k, T=tokens, num_warps=4)
        attention_out = self.attention_out[:tokens]
        self.decode_attention.multi_from_raw(
            self.qk[:tokens], self.v[:tokens], w.sqk, self.sqk_factor, EPS, cache_offsets,
            key_cache, value_cache, block_tables, block_table_offsets, seq_lens, window,
            softmax_scale, self.gate[:tokens], attention_out)
        out_partials = self.out_partials[:ATTENTION_OUT.split_k * tokens * self.hidden_size].view(
            ATTENTION_OUT.split_k, tokens, self.hidden_size)
        self._gemm(attention_out, w.o_weight, w.o_scale, w.o_global_scale, out_partials,
                   ATTENTION_OUT)
        self._residual(out_partials, attention_out, h, w.attn_alpha, h_out, h_out_nm)

    def mlp(self, w: MlpWeights, h: torch.Tensor, h_nm: torch.Tensor, h_out: torch.Tensor,
            h_out_nm: torch.Tensor) -> None:
        """One MLP sublayer for T = h.shape[0] tokens; h_out must not alias h."""
        tokens, F = h.shape[0], self.intermediate_size
        _gate_up_swiglu_tc_kernel[(triton.cdiv(2 * F, GATE_UP.block_n),)](
            h_nm, w.gate_up_weight.view(torch.int32), w.gate_up_scale, w.gate_up_global_scale,
            w.suv_gate, w.suv_up, self.m, self.suv_factor, F=F, K=self.hidden_size, T=tokens,
            BLOCK_N=GATE_UP.block_n, BLOCK_W=GATE_UP.block_w, num_warps=GATE_UP.num_warps,
            num_stages=GATE_UP.num_stages)
        m = self.m[:tokens]
        down_partials = self.down_partials[:DOWN.split_k * tokens * self.hidden_size].view(
            DOWN.split_k, tokens, self.hidden_size)
        self._gemm(m, w.down_weight, w.down_scale, w.down_global_scale, down_partials, DOWN)
        self._residual(down_partials, m, h, w.mlp_alpha, h_out, h_out_nm)
