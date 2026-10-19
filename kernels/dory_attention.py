"""Fused Dory attention sublayer for single-token decode.

vLLM runs a Dory attention sublayer as ten kernels: the q/k/v GEMM, the
output-gate GEMM, rotary embedding with q/k normalization (two kernels), the KV
cache write, attention with its segment merge (two kernels), the output gate
with normalization, the output GEMM, and the residual update. Here, around the
unchanged attention kernels:

1. qkvg_rope: one GEMV over the q, k, v and output-gate rows. Rows of each q
   and k head are reordered so that rotary partners (dimensions d and d + 128)
   are adjacent, which lets the epilogue apply the rotary embedding. The
   epilogue also applies the output gate's sigmoid.
2. qk_norm_cache: one program per head normalizes q (and scales it by sqk^2)
   or k, and writes k and v into the paged KV cache.
3. o_gated_normalized: the output GEMV. It multiplies the attention output by
   the gate while reading it and divides by the norm of that product at the
   end, since W (x / |x|) = (W x) / |x|.
4. the residual update from dory_common.py.

Intermediate values stay in FP32 where vLLM rounds them to BF16.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from dory_common import (EPS, LaunchConfig, ResidualScratch, residual_epilogue,
                         residual_inputs, residual_update)
from w4a16_gemv import gemv_rows


@triton.jit
def _qkvg_rope_kernel(
    x_ptr, weight_ptr, scale_ptr, global_scale_ptr, cos_sin_ptr, positions_ptr, s_gate_ptr,
    qk_ptr, v_ptr, gate_ptr, s_gate_factor,
    Q_ROWS: tl.constexpr, KV_ROWS: tl.constexpr, HEAD_DIM: tl.constexpr, K: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Row layout: q (Q_ROWS), k (KV_ROWS), v (KV_ROWS), output gate (Q_ROWS). Block
    # boundaries never straddle these regions because BLOCK_N divides HEAD_DIM.
    row0 = tl.program_id(0) * BLOCK_N
    rows = row0 + tl.arange(0, BLOCK_N)
    row_mask = rows < 2 * (Q_ROWS + KV_ROWS)
    acc, _ = gemv_rows(x_ptr, weight_ptr, scale_ptr, rows, row_mask, K, BLOCK_N, BLOCK_W,
                       False, False, x_ptr)
    y = acc * tl.load(global_scale_ptr + rows, mask=row_mask, other=0.0) * 16384.0
    HALF: tl.constexpr = HEAD_DIM // 2
    if row0 < Q_ROWS + KV_ROWS:
        # Rows 2i and 2i + 1 of a head hold rotary dimensions i and i + HALF.
        first, second = tl.split(tl.reshape(y, [BLOCK_N // 2, 2]))
        pair = row0 // 2 + tl.arange(0, BLOCK_N // 2)
        dim = pair % HALF
        position = tl.load(positions_ptr)
        cos = tl.load(cos_sin_ptr + position * HEAD_DIM + dim).to(tl.float32)
        sin = tl.load(cos_sin_ptr + position * HEAD_DIM + HALF + dim).to(tl.float32)
        out = (pair // HALF) * HEAD_DIM + dim  # natural index within q followed by k
        tl.store(qk_ptr + out, first * cos - second * sin)
        tl.store(qk_ptr + out + HALF, second * cos + first * sin)
    elif row0 < Q_ROWS + 2 * KV_ROWS:
        tl.store(v_ptr + rows - (Q_ROWS + KV_ROWS), y)
    else:
        index = rows - (Q_ROWS + 2 * KV_ROWS)
        s = tl.load(s_gate_ptr + index).to(tl.float32) * s_gate_factor
        tl.store(gate_ptr + index, tl.sigmoid(y * s * s))


@triton.jit
def _qk_norm_cache_kernel(
    qk_ptr, v_ptr, sqk_ptr, q_out_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
    k_out_ptr, v_out_ptr, sqk_factor, eps, stride_block, stride_offset, stride_head,
    NUM_Q_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, CACHE_BLOCK: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
):
    head = tl.program_id(0)
    dims = tl.arange(0, HEAD_DIM)
    x = tl.load(qk_ptr + head * HEAD_DIM + dims)
    x = x * tl.rsqrt(tl.sum(x * x) + eps)
    if head < NUM_Q_HEADS:
        s = tl.load(sqk_ptr + head * HEAD_DIM + dims).to(tl.float32) * sqk_factor
        tl.store(q_out_ptr + head * HEAD_DIM + dims, (s * s * x).to(tl.bfloat16))
    else:
        kv_head = head - NUM_Q_HEADS
        v = tl.load(v_ptr + kv_head * HEAD_DIM + dims)
        if WRITE_CACHE:
            slot = tl.load(slot_mapping_ptr)
            # A negative slot marks a padding token, which vLLM does not cache.
            if slot >= 0:
                base = ((slot // CACHE_BLOCK) * stride_block + (slot % CACHE_BLOCK) * stride_offset
                        + kv_head * stride_head)
                tl.store(key_cache_ptr + base + dims, x.to(tl.bfloat16))
                tl.store(value_cache_ptr + base + dims, v.to(tl.bfloat16))
        else:
            tl.store(k_out_ptr + kv_head * HEAD_DIM + dims, x.to(tl.bfloat16))
            tl.store(v_out_ptr + kv_head * HEAD_DIM + dims, v.to(tl.bfloat16))


@triton.jit
def _o_gated_normalized_kernel(
    attn_ptr, gate_ptr, weight_ptr, scale_ptr, global_scale_ptr, b_ptr, eps,
    a_ptr, alpha_ptr, partials_ptr, counter_ptr, h_ptr, h_nibble_major_ptr, alpha_scale,
    N: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
    FUSE_RESIDUAL: tl.constexpr, NUM_PROGRAMS: tl.constexpr, P_BLOCK: tl.constexpr,
    W_BLOCK: tl.constexpr, GATED_NATURAL: tl.constexpr,
):
    # GATED_NATURAL: attn_ptr holds the attention output in natural order and the
    # gate is applied here; otherwise attn_ptr holds the gated attention output in
    # nibble-major FP32 and gate_ptr is unused.
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    if FUSE_RESIDUAL:
        a, lr = residual_inputs(a_ptr, alpha_ptr, rows, row_mask, alpha_scale)
    acc, x_sumsq = gemv_rows(attn_ptr, weight_ptr, scale_ptr, rows, row_mask, K, BLOCK_N,
                             BLOCK_W, True, GATED_NATURAL, gate_ptr)
    global_scale = tl.load(global_scale_ptr + rows, mask=row_mask, other=0.0) * 16384.0
    b = acc * global_scale * tl.rsqrt(x_sumsq + eps)
    if FUSE_RESIDUAL:
        residual_epilogue(b, a, lr, rows, row_mask, a_ptr, alpha_ptr, b_ptr, partials_ptr,
                          counter_ptr, h_ptr, h_nibble_major_ptr, alpha_scale, eps, N,
                          NUM_PROGRAMS, P_BLOCK, W_BLOCK)
    else:
        tl.store(b_ptr + rows, b, mask=row_mask)


def rotary_pair_order(num_heads: int, head_dim: int) -> torch.Tensor:
    """Row order placing rotary partners d and d + head_dim/2 of each head side by side."""
    half = head_dim // 2
    within = torch.stack((torch.arange(half), torch.arange(half) + half), dim=1).flatten()
    return (torch.arange(num_heads)[:, None] * head_dim + within[None, :]).flatten()


@dataclass
class AttentionWeights:
    """One Dory attention layer, laid out for the fused kernels."""
    qkvg_weight: torch.Tensor  # uint8 [2Q + 2KV, H/2]: q, k (rotary-paired rows), v, gate
    qkvg_scale: torch.Tensor  # float8_e4m3fn [2Q + 2KV, H/16]
    qkvg_global_scale: torch.Tensor  # float32 [2Q + 2KV]
    o_weight: torch.Tensor  # uint8 [H, Q/2]
    o_scale: torch.Tensor  # float8_e4m3fn [H, Q/16]
    o_global_scale: torch.Tensor  # float32 [H]
    sqk: torch.Tensor  # [Q]
    s_gate: torch.Tensor  # [Q]
    attn_alpha: torch.Tensor  # [H]


def load_attention_weights(checkpoint, layer: int, num_heads: int, num_kv_heads: int,
                           head_dim: int) -> AttentionWeights:
    """Reads one attention layer from an open safetensors checkpoint (ModelOpt NVFP4)."""
    mixer = f"backbone.layers.{layer}.mixer"

    def projection(name, order=None):
        weight = checkpoint.get_tensor(f"{mixer}.{name}.weight")
        scale = checkpoint.get_tensor(f"{mixer}.{name}.weight_scale")
        global_scale = (checkpoint.get_tensor(f"{mixer}.{name}.weight_scale_2")
                        .float().expand(weight.shape[0]))
        if order is not None:
            order = order.to(weight.device)
            weight, scale, global_scale = weight[order], scale[order], global_scale[order]
        return weight, scale, global_scale

    parts = [projection("q_proj", rotary_pair_order(num_heads, head_dim)),
             projection("k_proj", rotary_pair_order(num_kv_heads, head_dim)),
             projection("v_proj"), projection("g_proj")]
    o = projection("o_proj")
    return AttentionWeights(
        qkvg_weight=torch.cat([p[0] for p in parts]).contiguous(),
        qkvg_scale=torch.cat([p[1] for p in parts]).contiguous(),
        qkvg_global_scale=torch.cat([p[2] for p in parts]).contiguous(),
        o_weight=o[0], o_scale=o[1], o_global_scale=o[2].contiguous(),
        sqk=checkpoint.get_tensor(f"{mixer}.sqk"),
        s_gate=checkpoint.get_tensor(f"{mixer}.s_gate"),
        attn_alpha=checkpoint.get_tensor(f"backbone.layers.{layer}.attn_alpha"),
    )


class FusedAttentionSublayer:
    """Runs a Dory attention sublayer around an external attention kernel.

    pre_attention produces the normalized query and writes this token's k and v
    into the paged KV cache; the caller runs attention; post_attention applies
    the output gate, the output projection and the residual update. Scratch
    buffers are owned by the instance, so one instance serves every layer of a
    sequential decode step.
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 initializer_range: float, alpha_scale: float, gate_logit_scale_init: float | None,
                 device: torch.device, qkvg: LaunchConfig, o: LaunchConfig,
                 residual_warps: int = 4, fuse_residual: bool = True,
                 o_nibble_major: LaunchConfig | None = None):
        if qkvg.block_n % 2 or head_dim % qkvg.block_n:
            raise ValueError("qkvg block_n must be even and divide head_dim")
        self.hidden_size, self.head_dim = hidden_size, head_dim
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.q_rows, self.kv_rows = num_heads * head_dim, num_kv_heads * head_dim
        self.alpha_scale = alpha_scale
        self.sqk_factor = head_dim ** 0.5 / initializer_range
        g0 = gate_logit_scale_init if gate_logit_scale_init is not None else hidden_size ** 0.5
        self.s_gate_factor = g0 ** 0.5 / initializer_range
        self.qkvg, self.o, self.residual_warps = qkvg, o, residual_warps
        f32 = dict(dtype=torch.float32, device=device)
        self.qk = torch.empty(self.q_rows + self.kv_rows, **f32)
        self.v = torch.empty(self.kv_rows, **f32)
        self.gate = torch.empty(self.q_rows, **f32)
        self.b = torch.empty(hidden_size, **f32)
        self.q = torch.empty(self.q_rows, dtype=torch.bfloat16, device=device)
        self.k = torch.empty(self.kv_rows, dtype=torch.bfloat16, device=device)
        self.v_out = torch.empty(self.kv_rows, dtype=torch.bfloat16, device=device)
        # With fuse_residual, the output projection's last program performs the
        # residual update instead of a separate kernel.
        self.fuse_residual = fuse_residual
        # Launch configuration for a gated nibble-major attention output.
        self.o_nibble_major = o_nibble_major or o
        self.scratch = ResidualScratch(
            triton.cdiv(hidden_size, min(o.block_n, self.o_nibble_major.block_n)), device)

    def pre_attention(self, w: AttentionWeights, h_nibble_major: torch.Tensor,
                      positions: torch.Tensor, cos_sin_cache: torch.Tensor,
                      slot_mapping: torch.Tensor, key_cache: torch.Tensor,
                      value_cache: torch.Tensor) -> torch.Tensor:
        """Returns the normalized query (BF16 [Q]); writes k and v at slot_mapping[0].

        cos_sin_cache: [max_position, head_dim] with cos then sin halves, as vLLM's
        rotary embedding stores it. key_cache, value_cache: [blocks, block_size,
        kv_heads, head_dim] views with unit stride in the last dimension.
        """
        if key_cache.stride() != value_cache.stride() or key_cache.stride(-1) != 1:
            raise ValueError("key and value caches need equal strides and unit last stride")
        self.project_qkvg(w, h_nibble_major, positions, cos_sin_cache)
        _qk_norm_cache_kernel[(self.num_heads + self.num_kv_heads,)](
            self.qk, self.v, w.sqk, self.q, key_cache, value_cache, slot_mapping,
            self.k, self.v_out, self.sqk_factor, EPS, key_cache.stride(0), key_cache.stride(1),
            key_cache.stride(2), NUM_Q_HEADS=self.num_heads, HEAD_DIM=self.head_dim,
            CACHE_BLOCK=key_cache.shape[1], WRITE_CACHE=True, num_warps=2)
        return self.q

    def pre_attention_qkv(self, w: AttentionWeights, h_nibble_major: torch.Tensor,
                          positions: torch.Tensor, cos_sin_cache: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like pre_attention, but returns (q, k, v) in BF16 for the caller's
        attention layer to cache, instead of writing the KV cache itself."""
        self.project_qkvg(w, h_nibble_major, positions, cos_sin_cache)
        _qk_norm_cache_kernel[(self.num_heads + self.num_kv_heads,)](
            self.qk, self.v, w.sqk, self.q, self.k, self.v_out, positions,
            self.k, self.v_out, self.sqk_factor, EPS, 0, 0, 0,
            NUM_Q_HEADS=self.num_heads, HEAD_DIM=self.head_dim, CACHE_BLOCK=1,
            WRITE_CACHE=False, num_warps=2)
        return self.q, self.k, self.v_out

    def project_qkvg(self, w: AttentionWeights, h_nibble_major: torch.Tensor,
                     positions: torch.Tensor, cos_sin_cache: torch.Tensor) -> None:
        """Writes the rotated, unnormalized q and k heads to self.qk (FP32, q heads
        then k heads), v to self.v and the output gate to self.gate."""
        if cos_sin_cache.shape[-1] != self.head_dim or not cos_sin_cache.is_contiguous():
            raise ValueError("expected a contiguous full-dimension rotary cache")
        cfg = self.qkvg
        rows = 2 * self.q_rows + 2 * self.kv_rows
        _qkvg_rope_kernel[(rows // cfg.block_n,)](
            h_nibble_major, w.qkvg_weight.view(torch.int32), w.qkvg_scale, w.qkvg_global_scale,
            cos_sin_cache, positions, w.s_gate, self.qk, self.v, self.gate, self.s_gate_factor,
            Q_ROWS=self.q_rows, KV_ROWS=self.kv_rows, HEAD_DIM=self.head_dim, K=self.hidden_size,
            BLOCK_N=cfg.block_n, BLOCK_W=cfg.block_w, num_warps=cfg.num_warps)

    def post_attention(self, w: AttentionWeights, h: torch.Tensor, attention_out: torch.Tensor,
                       h_out: torch.Tensor, h_out_nibble_major: torch.Tensor,
                       gated_nibble_major: bool = False) -> None:
        """h: BF16 sublayer input; attention_out: [Q] in natural order (any float type),
        or with gated_nibble_major, the gated attention output in nibble-major FP32
        (from DecodeAttention, given this sublayer's gate)."""
        cfg = self.o_nibble_major if gated_nibble_major else self.o
        programs = triton.cdiv(self.hidden_size, cfg.block_n)
        _o_gated_normalized_kernel[(programs,)](
            attention_out, self.gate, w.o_weight.view(torch.int32), w.o_scale, w.o_global_scale,
            self.b, EPS, h, w.attn_alpha, self.scratch.partials, self.scratch.counter, h_out,
            h_out_nibble_major, self.alpha_scale, N=self.hidden_size, K=self.q_rows,
            BLOCK_N=cfg.block_n, BLOCK_W=cfg.block_w, FUSE_RESIDUAL=self.fuse_residual,
            NUM_PROGRAMS=programs, P_BLOCK=triton.next_power_of_2(programs),
            W_BLOCK=triton.next_power_of_2(self.hidden_size // 8),
            GATED_NATURAL=not gated_nibble_major, num_warps=cfg.num_warps)
        if not self.fuse_residual:
            residual_update(h, self.b, w.attn_alpha, self.alpha_scale, h_out,
                            h_out_nibble_major, num_warps=self.residual_warps)
