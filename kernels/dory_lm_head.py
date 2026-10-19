"""Output projection (LM head) with 8-bit weights for a single token.

Dory's logits are logit_scale / initializer_range * (W h) with a BF16 W of
vocab_size x hidden_size, read once per decoded token. This module stores W
with one byte per weight and one scale per row, into which the per-vocabulary
logit scale is folded, and computes FP32 logits for one hidden state: half
the bytes of the BF16 head. Two formats: INT8 (uniform steps of amax / 127)
and FP8 E4M3 (3 mantissa bits). Either changes the model's numerics and is
therefore opt-in.
"""

import torch
import triton
import triton.language as tl

# Largest representable magnitude per storage format.
FORMATS = {"int8": (torch.int8, 127.0), "fp8": (torch.float8_e4m3fn, 448.0)}


@triton.jit
def _quantized_logits_kernel(h_ptr, weight_ptr, row_scale_ptr, out_ptr, V,
                             K: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = rows < V
    acc = tl.zeros([BLOCK_V], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(weight_ptr + rows[:, None].to(tl.int64) * K + cols[None, :],
                    mask=row_mask[:, None], other=0.0).to(tl.float32)
        x = tl.load(h_ptr + cols).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0)
    tl.store(out_ptr + rows, acc * scale, mask=row_mask)


@triton.jit
def _quantized_logits_rows_kernel(h_ptr, weight_ptr, row_scale_ptr, out_ptr, V, T,
                                  K: tl.constexpr, BLOCK_V: tl.constexpr,
                                  BLOCK_K: tl.constexpr):
    # Logits of up to 16 hidden states ([T, K]) with one read of the weights, on
    # tensor cores: INT8 weights are exact in FP16, and the code path does not
    # depend on T (always 16 columns), so each row's result is the same
    # whatever T is.
    rows = tl.program_id(0) * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = rows < V
    tokens = tl.arange(0, 16)
    acc = tl.zeros([BLOCK_V, 16], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(weight_ptr + rows[:, None].to(tl.int64) * K + cols[None, :],
                    mask=row_mask[:, None], other=0).to(tl.float16)
        x = tl.load(h_ptr + tokens[None, :] * K + cols[:, None], mask=(tokens < T)[None, :],
                    other=0.0).to(tl.float16)
        acc = tl.dot(w, x, acc)
    scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0)
    tl.store(out_ptr + tokens[None, :] * V + rows[:, None], acc * scale[:, None],
             mask=row_mask[:, None] & (tokens < T)[None, :])


class QuantizedLmHead:
    """8-bit copy of the LM head with the logit scale folded into its row scales."""

    def __init__(self, weight: torch.Tensor, logit_scale: torch.Tensor,
                 initializer_range: float, storage: str = "int8", block_v: int = 4,
                 block_k: int = 512, num_warps: int = 4, chunk_rows: int = 8192):
        vocab, hidden = weight.shape
        if hidden % block_k:
            raise ValueError("hidden size must be a multiple of block_k")
        dtype, largest = FORMATS[storage]
        self.weight = torch.empty(vocab, hidden, dtype=dtype, device=weight.device)
        self.row_scale = torch.empty(vocab, dtype=torch.float32, device=weight.device)
        # Chunks bound the FP32 temporary to chunk_rows x hidden.
        for start in range(0, vocab, chunk_rows):
            rows = weight[start:start + chunk_rows].float()
            scale = rows.abs().amax(dim=1).clamp(min=1e-12) / largest
            scaled = rows / scale[:, None]
            if storage == "int8":
                scaled = scaled.round()
            self.weight[start:start + chunk_rows] = scaled.to(dtype)
            self.row_scale[start:start + chunk_rows] = (
                scale * logit_scale[start:start + chunk_rows].float() / initializer_range)
        self.vocab, self.hidden = vocab, hidden
        self.block_v, self.block_k, self.num_warps = block_v, block_k, num_warps

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: [1, hidden]; returns FP32 logits [1, vocab]."""
        out = torch.empty(1, self.vocab, dtype=torch.float32, device=hidden_states.device)
        _quantized_logits_kernel[(triton.cdiv(self.vocab, self.block_v),)](
            hidden_states, self.weight, self.row_scale, out, self.vocab, K=self.hidden,
            BLOCK_V=self.block_v, BLOCK_K=self.block_k, num_warps=self.num_warps)
        return out

    def rows(self, hidden_states: torch.Tensor, out: torch.Tensor, block_v: int = 64,
             block_k: int = 256, num_warps: int = 4) -> None:
        """FP32 logits [T, vocab] into `out` for up to 16 hidden states [T, hidden]."""
        tokens = hidden_states.shape[0]
        if tokens > 16 or out.shape != (tokens, self.vocab) or self.hidden % block_k:
            raise ValueError("too many rows, an output of the wrong shape, or a bad block_k")
        _quantized_logits_rows_kernel[(triton.cdiv(self.vocab, block_v),)](
            hidden_states, self.weight, self.row_scale, out, self.vocab, tokens, K=self.hidden,
            BLOCK_V=block_v, BLOCK_K=block_k, num_warps=num_warps)
