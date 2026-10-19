"""Fused Dory MLP sublayer for single-token decode: three kernels instead of four.

vLLM runs a Dory MLP sublayer as four kernels: the gate+up GEMM, a kernel for
SwiGLU with its scaling and normalization, the down GEMM, and the residual
update. Here:

1. gate_up_swiglu: one GEMV over the gate and up rows, interleaved per
   intermediate index, whose epilogue computes
   m_i = silu(g_i * s_gate_i * sqrt(H)) * (u_i * s_up_i * sqrt(H)) and writes m
   in nibble-major FP32 for the next GEMV.
2. down_normalized: the down GEMV. vLLM normalizes m before it; because
   W (m / |m|) = (W m) / |m|, this kernel divides its result by |m| instead,
   with sum(m^2) accumulated from the input every program already reads.
3. the residual update from dory_common.py, which also writes the next
   sublayer's input in nibble-major FP32.

Intermediate values stay in FP32 where vLLM rounds them to BF16.
"""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from dory_common import (EPS, LaunchConfig, ResidualScratch, residual_epilogue,
                         residual_inputs, residual_update)
from w4a16_gemv import gemv_rows


@triton.jit
def _gate_up_swiglu_kernel(
    x_ptr, weight_ptr, scale_ptr, global_scale_ptr, suv_gate_ptr, suv_up_ptr, m_ptr,
    suv_factor, F: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Row 2i of the weight is gate row i and row 2i + 1 is up row i, so every
    # program holds matching gate and up outputs.
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < 2 * F
    acc, _ = gemv_rows(x_ptr, weight_ptr, scale_ptr, rows, row_mask, K, BLOCK_N, BLOCK_W, False,
                       False, x_ptr)
    y = acc * tl.load(global_scale_ptr + rows, mask=row_mask, other=0.0) * 16384.0
    gate, up = tl.split(tl.reshape(y, [BLOCK_N // 2, 2]))
    index = tl.program_id(0) * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    mask = index < F
    gate *= tl.load(suv_gate_ptr + index, mask=mask, other=0.0).to(tl.float32) * suv_factor
    up *= tl.load(suv_up_ptr + index, mask=mask, other=0.0).to(tl.float32) * suv_factor
    m = gate * tl.sigmoid(gate) * up
    # Nibble-major position of intermediate index i is [i % 8, i // 8].
    tl.store(m_ptr + (index % 8) * (F // 8) + index // 8, m, mask=mask)


@triton.jit
def _down_normalized_kernel(
    x_ptr, weight_ptr, scale_ptr, global_scale_ptr, b_ptr, eps,
    a_ptr, alpha_ptr, partials_ptr, counter_ptr, h_ptr, h_nibble_major_ptr, alpha_scale,
    N: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
    FUSE_RESIDUAL: tl.constexpr, NUM_PROGRAMS: tl.constexpr, P_BLOCK: tl.constexpr,
    W_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    if FUSE_RESIDUAL:
        a, lr = residual_inputs(a_ptr, alpha_ptr, rows, row_mask, alpha_scale)
    acc, x_sumsq = gemv_rows(x_ptr, weight_ptr, scale_ptr, rows, row_mask, K, BLOCK_N, BLOCK_W,
                             True, False, x_ptr)
    global_scale = tl.load(global_scale_ptr + rows, mask=row_mask, other=0.0) * 16384.0
    b = acc * global_scale * tl.rsqrt(x_sumsq + eps)
    if FUSE_RESIDUAL:
        residual_epilogue(b, a, lr, rows, row_mask, a_ptr, alpha_ptr, b_ptr, partials_ptr,
                          counter_ptr, h_ptr, h_nibble_major_ptr, alpha_scale, eps, N,
                          NUM_PROGRAMS, P_BLOCK, W_BLOCK)
    else:
        tl.store(b_ptr + rows, b, mask=row_mask)


@dataclass
class MlpWeights:
    """One Dory MLP layer, laid out for the fused kernels."""
    gate_up_weight: torch.Tensor  # uint8 [2F, H/2], gate and up rows interleaved
    gate_up_scale: torch.Tensor  # float8_e4m3fn [2F, H/16]
    gate_up_global_scale: torch.Tensor  # float32 [2F]
    suv_gate: torch.Tensor  # [F]
    suv_up: torch.Tensor  # [F]
    down_weight: torch.Tensor  # uint8 [H, F/2]
    down_scale: torch.Tensor  # float8_e4m3fn [H, F/16]
    down_global_scale: torch.Tensor  # float32 [H]
    mlp_alpha: torch.Tensor  # [H]

    @property
    def hidden_size(self) -> int:
        return self.down_weight.shape[0]

    @property
    def intermediate_size(self) -> int:
        return self.gate_up_weight.shape[0] // 2


def load_mlp_weights(checkpoint, layer: int) -> MlpWeights:
    """Reads one MLP layer from an open safetensors checkpoint (ModelOpt NVFP4)."""
    mixer = f"backbone.layers.{layer}.mixer"

    def projection(name):
        weight = checkpoint.get_tensor(f"{mixer}.{name}.weight")
        scale = checkpoint.get_tensor(f"{mixer}.{name}.weight_scale")
        global_scale = (checkpoint.get_tensor(f"{mixer}.{name}.weight_scale_2")
                        .float().expand(weight.shape[0]))
        return weight, scale, global_scale

    def interleave(gate_part, up_part):
        return torch.stack((gate_part, up_part), dim=1).flatten(0, 1).contiguous()

    gate, up, down = projection("gate_proj"), projection("up_proj"), projection("down_proj")
    return MlpWeights(
        gate_up_weight=interleave(gate[0], up[0]),
        gate_up_scale=interleave(gate[1], up[1]),
        gate_up_global_scale=interleave(gate[2], up[2]),
        suv_gate=checkpoint.get_tensor(f"{mixer}.suv_gate"),
        suv_up=checkpoint.get_tensor(f"{mixer}.suv_up"),
        down_weight=down[0], down_scale=down[1], down_global_scale=down[2].contiguous(),
        mlp_alpha=checkpoint.get_tensor(f"backbone.layers.{layer}.mlp_alpha"),
    )


class FusedMlpSublayer:
    """Runs one Dory MLP sublayer and its residual update with the fused kernels.

    Scratch buffers are owned by the instance, so one instance can serve every
    layer of a sequential decode step.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, alpha_scale: float,
                 device: torch.device, gate_up: LaunchConfig, down: LaunchConfig,
                 residual_warps: int = 8, fuse_residual: bool = True):
        if gate_up.block_n % 2:
            raise ValueError("gate_up block_n must be even to pair gate and up rows")
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.alpha_scale = alpha_scale
        self.suv_factor = math.sqrt(hidden_size)
        self.gate_up, self.down, self.residual_warps = gate_up, down, residual_warps
        self.m = torch.empty(8, intermediate_size // 8, dtype=torch.float32, device=device)
        self.b = torch.empty(hidden_size, dtype=torch.float32, device=device)
        # With fuse_residual, the down kernel's last program performs the residual
        # update instead of a separate kernel.
        self.fuse_residual = fuse_residual
        self.scratch = ResidualScratch(triton.cdiv(hidden_size, down.block_n), device)

    def __call__(self, w: MlpWeights, h: torch.Tensor, h_nibble_major: torch.Tensor,
                 h_out: torch.Tensor, h_out_nibble_major: torch.Tensor) -> None:
        """Reads the sublayer input (BF16 and nibble-major FP32), writes the output."""
        H, F = self.hidden_size, self.intermediate_size
        cfg = self.gate_up
        _gate_up_swiglu_kernel[(triton.cdiv(2 * F, cfg.block_n),)](
            h_nibble_major, w.gate_up_weight.view(torch.int32), w.gate_up_scale,
            w.gate_up_global_scale, w.suv_gate, w.suv_up, self.m, self.suv_factor,
            F=F, K=H, BLOCK_N=cfg.block_n, BLOCK_W=cfg.block_w, num_warps=cfg.num_warps)
        cfg = self.down
        programs = triton.cdiv(H, cfg.block_n)
        _down_normalized_kernel[(programs,)](
            self.m, w.down_weight.view(torch.int32), w.down_scale, w.down_global_scale,
            self.b, EPS, h, w.mlp_alpha, self.scratch.partials, self.scratch.counter, h_out,
            h_out_nibble_major, self.alpha_scale, N=H, K=F, BLOCK_N=cfg.block_n,
            BLOCK_W=cfg.block_w, FUSE_RESIDUAL=self.fuse_residual, NUM_PROGRAMS=programs,
            P_BLOCK=triton.next_power_of_2(programs), W_BLOCK=triton.next_power_of_2(H // 8),
            num_warps=cfg.num_warps)
        if not self.fuse_residual:
            residual_update(h, self.b, w.mlp_alpha, self.alpha_scale, h_out,
                            h_out_nibble_major, num_warps=self.residual_warps)

