"""A single-token Dory decode step built from the fused sublayers, run inside vLLM.

FusedDoryDecoder mirrors vLLM's DoryBackbone.forward for one token: the input
layers, the recurrent layers once per loop with the loop entry
n(state + input) and exit n(h), then the output layers. Each sublayer is a
fused kernel sequence from dory_attention.py or dory_mlp.py, and the hidden
state travels between them as BF16 plus its nibble-major FP32 copy.

The paged KV cache, its attention metadata and the sliding windows stay
vLLM's. With vLLM's Triton attention backend and a BF16 cache, the fused q/k
normalization kernel writes k and v into the paged cache itself, and
dory_decode_attention computes the gated attention output straight into the
input layout of the attention-output GEMV. With any other backend, or attention
features that kernel does not implement, vLLM's attention layer runs as usual.
The decoder builds its weight layouts from the raw checkpoint tensors captured
while vLLM loads the model; vLLM's own copies remain in use for prefill and
multi-token batches.
"""

import os

import torch
from vllm.model_executor.layers.attention.attention import (get_attention_context,
                                                            unified_attention_with_output)

from dory_attention import FusedAttentionSublayer, load_attention_weights
from dory_common import EPS, LaunchConfig, to_residual
from dory_decode_attention import DecodeAttention
from dory_lm_head import QuantizedLmHead
from dory_mlp import FusedMlpSublayer, load_mlp_weights

# Launch configurations chosen by kernels/bench_attention.py and kernels/bench_mlp.py.
QKVG = LaunchConfig(block_n=4, block_w=64, num_warps=4)
ATTENTION_OUT = LaunchConfig(block_n=4, block_w=128, num_warps=4)
ATTENTION_OUT_NIBBLE_MAJOR = LaunchConfig(block_n=4, block_w=64, num_warps=4)
GATE_UP = LaunchConfig(block_n=4, block_w=64, num_warps=4)
DOWN = LaunchConfig(block_n=4, block_w=64, num_warps=8)
# With the residual update folded into the down kernel, fewer warps per program
# finish sooner: the last program's final pass is latency-bound.
DOWN_FOLDED = LaunchConfig(block_n=4, block_w=64, num_warps=4)
# Sublayers whose last GEMV also performs the residual update, comma-separated
# ("mlp", "attention"); anything else keeps the separate residual kernel.
FOLD_RESIDUAL = set(os.environ.get("DORY_FOLD_RESIDUAL", "mlp,attention").split(","))
# "0" leaves the KV cache write to vLLM's attention layer on every backend.
FUSED_KV_WRITE = os.environ.get("DORY_FUSED_KV_WRITE", "1") != "0"
# "0" keeps vLLM's attention kernel even where dory_decode_attention applies.
FUSED_ATTENTION = os.environ.get("DORY_FUSED_ATTENTION", "1") != "0"
# Single-token logits come from an 8-bit copy of the LM head: "int8" (default)
# or "fp8"; "none" leaves them to vLLM's BF16 head. INT8 keeps task accuracy and
# greedy agreement within the noise of switching attention kernels; FP8 does not.
LM_HEAD_STORAGE = os.environ.get("DORY_LM_HEAD_8BIT", "int8")
if LM_HEAD_STORAGE not in ("int8", "fp8", "none"):
    raise ValueError(f"DORY_LM_HEAD_8BIT must be int8, fp8 or none, not {LM_HEAD_STORAGE!r}")


class TensorSource:
    """Exposes captured checkpoint tensors through safetensors' get_tensor()."""

    def __init__(self, tensors: dict[str, torch.Tensor], device: torch.device):
        self.tensors, self.device = tensors, device

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name].to(self.device)


class _HiddenState:
    """A hidden state as BF16 [H] plus its nibble-major FP32 copy [8, H/8]."""

    def __init__(self, hidden_size: int, device: torch.device):
        self.bf16 = torch.empty(hidden_size, dtype=torch.bfloat16, device=device)
        self.nibble_major = torch.empty(8, hidden_size // 8, dtype=torch.float32, device=device)


class FusedDoryDecoder:
    def __init__(self, config, tensors: dict[str, torch.Tensor], device: torch.device,
                 lm_head_weight: torch.Tensor | None = None,
                 logit_scale: torch.Tensor | None = None):
        """config: vLLM's Dory model config; tensors: raw checkpoint tensors by name;
        lm_head_weight, logit_scale: vLLM's loaded LM head, for the FP8 copy."""
        self.config = config
        H = config.hidden_size
        source = TensorSource(tensors, device)
        self.attention_weights, self.mlp_weights = {}, {}
        for index, layer_type in enumerate(config.layer_types):
            if layer_type == "mlp":
                self.mlp_weights[index] = load_mlp_weights(source, index)
            else:
                self.attention_weights[index] = load_attention_weights(
                    source, index, config.num_attention_heads, config.num_key_value_heads,
                    config.head_dim)
        alpha_scale = config.ngpt_alpha_init / config.initializer_range
        self.attention = FusedAttentionSublayer(
            H, config.num_attention_heads, config.num_key_value_heads, config.head_dim,
            config.initializer_range, alpha_scale, config.attention_output_gate_logit_scale_init,
            device, QKVG, ATTENTION_OUT, fuse_residual="attention" in FOLD_RESIDUAL,
            o_nibble_major=ATTENTION_OUT_NIBBLE_MAJOR)
        self.decode_attention = DecodeAttention(
            config.num_attention_heads, config.num_key_value_heads, config.head_dim, device)
        self.gated_attention = torch.empty(
            8, config.num_attention_heads * config.head_dim // 8, dtype=torch.float32,
            device=device)
        fold_mlp = "mlp" in FOLD_RESIDUAL
        self.mlp = FusedMlpSublayer(H, config.intermediate_size, alpha_scale, device, GATE_UP,
                                    DOWN_FOLDED if fold_mlp else DOWN, residual_warps=4,
                                    fuse_residual=fold_mlp)
        # Two alternating sublayer outputs, plus the loop input and loop state,
        # which must survive the sublayers in between.
        self.outputs = [_HiddenState(H, device) for _ in range(2)]
        self.loop_input = _HiddenState(H, device)
        self.loop_state = _HiddenState(H, device)
        self.attention_out = torch.empty(1, config.num_attention_heads, config.head_dim,
                                         dtype=torch.bfloat16, device=device)
        # Single-token logits; None leaves them to vLLM.
        self.lm_head = (QuantizedLmHead(lm_head_weight, logit_scale, config.initializer_range,
                                        LM_HEAD_STORAGE)
                        if LM_HEAD_STORAGE != "none" and lm_head_weight is not None else None)

    def _paged_cache(self, attention):
        """(key cache, value cache, slot mapping, metadata, impl) when the fused
        kernels can write this layer's KV cache, else None.

        Only for vLLM's Triton attention backend with a BF16 cache, whose cache
        is [blocks, kv_heads, block_size, 2 * head_dim] with keys in the first
        half of the last dimension (see TritonAttentionImpl.do_kv_cache_update).
        """
        if not FUSED_KV_WRITE:
            return None
        metadata, layer, kv_cache, slot_mapping = get_attention_context(attention.layer_name)
        # Warm-up and profiling runs (the V1 model runner's CUDA graph warm-up,
        # for one) carry no attention metadata. vLLM's attention then skips its
        # work, so those runs take the vLLM attention path instead of writing
        # the cache through slots that do not belong to a request.
        if (metadata is None or slot_mapping is None
                or type(layer.impl).__name__ != "TritonAttentionImpl"
                or kv_cache.dtype != torch.bfloat16):
            return None
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.config.head_dim, dim=-1)
        return key_cache, value_cache, slot_mapping, metadata, layer.impl

    @staticmethod
    def _decode_attention_window(metadata, impl) -> int | None:
        """The sliding window (0 for full attention) when dory_decode_attention
        implements this layer's attention for a single decode token, else None."""
        plain = (impl.alibi_slopes is None and impl.sinks is None and not impl.logits_soft_cap
                 and (impl.chunk_lookback is None or impl.chunk_lookback < 0)
                 and getattr(metadata, "rswa_window", None) is None
                 and getattr(metadata, "mm_prefix_range_tensor", None) is None
                 and metadata.num_actual_tokens == 1 and metadata.max_query_len == 1)
        if not (FUSED_ATTENTION and plain and metadata.causal is True):
            return None
        left, right = impl.sliding_window
        if right not in (-1, 0):
            return None
        return left + 1 if left >= 0 else 0

    def __call__(self, backbone, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Runs the backbone for one token; returns the final hidden state [1, H]."""
        config = self.config
        first, recurrent = config.n_input_layers, config.n_recurrent_layers
        after = first + recurrent
        next_output = 0

        def sublayer(index: int, loop: int, h: _HiddenState) -> _HiddenState:
            nonlocal next_output
            out = self.outputs[next_output]
            next_output = 1 - next_output
            layer = backbone.layers[index]
            if index in self.mlp_weights:
                self.mlp(self.mlp_weights[index], h.bf16, h.nibble_major, out.bf16,
                         out.nibble_major)
                return out
            mixer = layer.mixer
            weights = self.attention_weights[index]
            attention = mixer.attn[loop if mixer.per_loop_kv else 0]
            cos_sin_cache = mixer.rotary_emb.cos_sin_cache
            paged_cache = self._paged_cache(attention)
            if paged_cache is None:
                q, k, v = self.attention.pre_attention_qkv(weights, h.nibble_major, positions,
                                                           cos_sin_cache)
                attention_out = attention(q[None], k[None], v[None])
                self.attention.post_attention(weights, h.bf16, attention_out.view(-1), out.bf16,
                                              out.nibble_major)
                return out
            key_cache, value_cache, slot_mapping, metadata, impl = paged_cache
            window = self._decode_attention_window(metadata, impl)
            fused = self.attention
            if window is not None:
                # The attention kernel normalizes q and k and writes the cache.
                fused.project_qkvg(weights, h.nibble_major, positions, cos_sin_cache)
                self.decode_attention.from_raw(
                    fused.qk, fused.v, weights.sqk, fused.sqk_factor, EPS, slot_mapping,
                    key_cache, value_cache, metadata.block_table[0], metadata.seq_lens, window,
                    impl.scale, fused.gate, self.gated_attention)
                fused.post_attention(weights, h.bf16, self.gated_attention, out.bf16,
                                     out.nibble_major, gated_nibble_major=True)
                return out
            q = fused.pre_attention(weights, h.nibble_major, positions, cos_sin_cache,
                                    slot_mapping, key_cache, value_cache)
            # The backend reads k and v from the cache; these arguments are unused.
            k = fused.k.view(1, config.num_key_value_heads, config.head_dim)
            unified_attention_with_output(q.view(1, config.num_attention_heads, config.head_dim),
                                          k, k, self.attention_out, attention.layer_name)
            fused.post_attention(weights, h.bf16, self.attention_out.view(-1), out.bf16,
                                 out.nibble_major)
            return out

        h = self.outputs[next_output]
        next_output = 1 - next_output
        to_residual(backbone.embeddings(input_ids).view(-1), None, h.bf16, h.nibble_major,
                    normalize=False)
        for index in range(first):
            h = sublayer(index, 0, h)
        to_residual(h.bf16, None, self.loop_input.bf16, self.loop_input.nibble_major,
                    normalize=False)
        state = self.loop_input
        for loop in range(backbone.num_loops):
            h = self.outputs[next_output]
            next_output = 1 - next_output
            to_residual(state.bf16, self.loop_input.bf16, h.bf16, h.nibble_major, normalize=True)
            for index in range(first, after):
                h = sublayer(index, loop, h)
            to_residual(h.bf16, None, self.loop_state.bf16, self.loop_state.nibble_major,
                        normalize=True)
            state = self.loop_state
        h = state
        for index in range(after, config.num_hidden_layers):
            h = sublayer(index, 0, h)
        return h.bf16[None]
