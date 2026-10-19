"""Validates and times the fused Dory attention sublayer against vLLM's implementation.

For every attention layer, the KV cache is filled with CONTEXT tokens computed
from random unit-norm hidden states through the layer's real projections, and
one decode token at position CONTEXT is run three ways:

- reference: dequantized FP32 weights, FP32 rotary tables and FP32 math;
- vllm: vLLM's path, Marlin GEMMs, vLLM's rotary embedding and Dory functions
  (compiled with torch.compile as vLLM compiles them) and vLLM's Triton cache
  write;
- fused: the kernels in dory_attention.py.

All three use the same attention computation (PyTorch, over the paged cache);
the attention kernel is validated and timed separately by
bench_decode_attention.py. Accuracy compares the query,
the k and v written to the cache, and the sublayer output. Timing replays the
decode-order sequence of attention sublayers (2 loops) in CUDA graphs with the
attention kernel left out of both implementations.

Usage: .venv/bin/python kernels/bench_attention.py
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
from safetensors import safe_open

from bench_gemv import (CHECKPOINT, ROOT, MarlinProjection, Projection, decode_order, dequantize,
                        require_idle_gpu, time_graph)
from bench_mlp import kernel_breakdown, relative_error, short_name
from dory_attention import AttentionWeights, FusedAttentionSublayer, load_attention_weights
from dory_common import EPS, LaunchConfig
from w4a16_gemv import to_nibble_major

CONTEXT = 640  # tokens already cached; the decoded token sits at this position
CACHE_BLOCK = 16  # vLLM's default KV cache block size
QKVG_CONFIGS = [LaunchConfig(*c) for c in
                ((4, 64, 4), (2, 64, 4), (8, 32, 8), (4, 32, 4), (8, 64, 4), (4, 128, 4))]
O_CONFIGS = [LaunchConfig(*c) for c in
             ((4, 64, 8), (4, 32, 8), (8, 64, 8), (2, 64, 8), (4, 128, 4), (8, 32, 8))]


@dataclass
class AttentionLayer:
    index: int
    sliding: bool
    fused: AttentionWeights
    qkv: Projection  # vLLM order: q, k, v rows
    gate: Projection
    out: Projection
    marlin_qkv: MarlinProjection
    marlin_gate: MarlinProjection
    marlin_out: MarlinProjection
    caches: dict  # implementation name -> (key_cache, value_cache)


def new_cache(kv_heads: int, head_dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Key and value views of one layer's cache in the Triton backend's layout."""
    blocks = (CONTEXT + CACHE_BLOCK) // CACHE_BLOCK + 1
    kv = torch.zeros(blocks, CACHE_BLOCK, kv_heads, 2 * head_dim, dtype=torch.bfloat16,
                     device=device)
    return kv.split(head_dim, dim=-1)


def slots(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(count)
    return index // CACHE_BLOCK, index % CACHE_BLOCK


def apply_rotary(x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """NeoX-style rotary embedding of x [T, heads, head_dim] in x's dtype."""
    half = x.shape[-1] // 2
    cos = cos_sin[positions, :half][:, None, :].to(x.dtype)
    sin = cos_sin[positions, half:][:, None, :].to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + EPS)


def attend(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, count: int,
           num_heads: int) -> torch.Tensor:
    """Softmax attention of q [Q] over the first `count` cached tokens, in FP32.

    count <= 1024, so sliding-window and full-attention layers see the same keys.
    """
    blocks, offsets = slots(count)
    keys, values = key_cache[blocks, offsets].float(), value_cache[blocks, offsets].float()
    group = num_heads // keys.shape[1]
    keys, values = keys.repeat_interleave(group, dim=1), values.repeat_interleave(group, dim=1)
    q = q.float().view(num_heads, -1)
    scores = torch.einsum("hd,thd->ht", q, keys) / math.sqrt(q.shape[-1])
    return torch.einsum("ht,thd->hd", scores.softmax(dim=-1), values).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--replays", type=int, default=50)
    args = parser.parse_args()
    require_idle_gpu()
    device = torch.device("cuda")
    torch.manual_seed(0)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.models.dory import justnorm, slerp_alpha
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash)

    config = json.loads((CHECKPOINT / "config.json").read_text())
    H, NH, NKV, HD = (config["hidden_size"], config["num_attention_heads"],
                      config["num_key_value_heads"], config["head_dim"])
    Q, KV = NH * HD, NKV * HD
    std = config["initializer_range"]
    alpha_scale = config["ngpt_alpha_init"] / std
    sqk_factor = HD ** 0.5 / std
    g0 = config["attention_output_gate_logit_scale_init"] or H ** 0.5
    s_gate_factor = g0 ** 0.5 / std

    # vLLM's rotary embeddings (BF16 tables, as in the model) and exact FP32 tables.
    ropes, tables32 = {}, {}
    with set_current_vllm_config(VllmConfig()):
        for sliding, kind in ((True, "sliding_attention"), (False, "full_attention")):
            rope = get_rope(HD, max_position=config["max_position_embeddings"], is_neox_style=True,
                            rope_parameters=config["rope_parameters"][kind],
                            dtype=torch.bfloat16).to(device)
            ropes[sliding] = rope
            tables32[sliding] = rope._compute_cos_sin_cache().to(device)

    # Load layers and fill each layer's context into identical caches.
    layers = {}
    with safe_open(str(CHECKPOINT / "model.safetensors"), framework="pt", device=str(device)) as f:
        for index, layer_type in enumerate(config["layer_types"]):
            if layer_type == "mlp":
                continue
            mixer = f"backbone.layers.{index}.mixer"

            def projection(names, kind):
                parts = [(f.get_tensor(f"{mixer}.{n}.weight"),
                          f.get_tensor(f"{mixer}.{n}.weight_scale"),
                          f.get_tensor(f"{mixer}.{n}.weight_scale_2").float()) for n in names]
                return Projection(kind, index, torch.cat([p[0] for p in parts]).contiguous(),
                                  torch.cat([p[1] for p in parts]).contiguous(),
                                  torch.cat([p[2].expand(p[0].shape[0]) for p in parts]).contiguous())

            qkv = projection(("q_proj", "k_proj", "v_proj"), "qkv")
            gate, out = projection(("g_proj",), "gate"), projection(("o_proj",), "attn_out")
            sliding = layer_type == "sliding_attention"
            layer = AttentionLayer(
                index, sliding, load_attention_weights(f, index, NH, NKV, HD), qkv, gate, out,
                MarlinProjection(qkv), MarlinProjection(gate), MarlinProjection(out),
                {name: new_cache(NKV, HD, device) for name in ("vllm", "fused")})
            hidden = torch.randn(CONTEXT, H, device=device)
            projected = (hidden / hidden.norm(dim=1, keepdim=True)) @ dequantize(qkv).T
            k = normalize(apply_rotary(projected[:, Q:Q + KV].view(CONTEXT, NKV, HD),
                                       torch.arange(CONTEXT, device=device), tables32[sliding]))
            v = projected[:, Q + KV:].view(CONTEXT, NKV, HD)
            blocks, offsets = slots(CONTEXT)
            for key_cache, value_cache in layer.caches.values():
                key_cache[blocks, offsets] = k.to(torch.bfloat16)
                value_cache[blocks, offsets] = v.to(torch.bfloat16)
            layers[index] = layer
    order = [i for i in decode_order(config, args.loops) if i in layers]

    positions = torch.tensor([CONTEXT], device=device)
    slot_mapping = torch.tensor([CONTEXT], device=device)
    unit_scale = torch.ones((), device=device)

    # vLLM's path, with its elementwise steps compiled as vLLM compiles them.
    def make_rope_qk_norm(rope):
        def rope_qk_norm(pos, q, k, sqk):
            q, k = rope.forward_native(pos, q, k)
            s = (sqk.float() * sqk_factor).view(1, NH, HD)
            q = (s.square() * justnorm(q.view(1, NH, HD)).float()).to(q.dtype).view(1, Q)
            return q, justnorm(k.view(1, NKV, HD)).view(1, KV)
        return torch.compile(rope_qk_norm, fullgraph=True)

    rope_qk_norm = {sliding: make_rope_qk_norm(rope) for sliding, rope in ropes.items()}

    def gate_norm(attn_out, gate, s_gate):
        sg = (s_gate.float() * s_gate_factor).square()
        return justnorm((attn_out.float() * torch.sigmoid(gate.float() * sg)).to(attn_out.dtype))

    gate_norm = torch.compile(gate_norm, fullgraph=True)
    residual = torch.compile(lambda a, b, alpha: slerp_alpha(a, b, alpha, alpha_scale),
                             fullgraph=True)

    def vllm_pre(layer, h):
        qkv = layer.marlin_qkv(h)[None]
        q, k, v = qkv.split([Q, KV, KV], dim=-1)
        gate = layer.marlin_gate(h)[None]
        q, k = rope_qk_norm[layer.sliding](positions, q, k, layer.fused.sqk)
        key_cache, value_cache = layer.caches["vllm"]
        triton_reshape_and_cache_flash(k.view(1, NKV, HD), v.reshape(1, NKV, HD), key_cache,
                                       value_cache, slot_mapping, "auto", unit_scale, unit_scale)
        return q[0], gate

    def vllm_post(layer, h, attn_out, gate):
        x = gate_norm(attn_out[None], gate, layer.fused.s_gate)
        return residual(h[None], layer.marlin_out(x[0])[None], layer.fused.attn_alpha)[0]

    def reference(layer, h):
        """FP32 sublayer; returns (output, q, k, v) for the decoded token."""
        qkv = dequantize(layer.qkv) @ h
        q = apply_rotary(qkv[:Q].view(1, NH, HD), positions, tables32[layer.sliding])[0]
        k = apply_rotary(qkv[Q:Q + KV].view(1, NKV, HD), positions, tables32[layer.sliding])[0]
        q = (normalize(q) * (layer.fused.sqk.float() * sqk_factor).view(NH, HD).square()).reshape(-1)
        k = normalize(k).reshape(-1)
        v = qkv[Q + KV:]
        # Context from the cache (shared input), plus this token's exact k and v.
        key_cache, value_cache = layer.caches["vllm"]
        blocks, offsets = slots(CONTEXT)
        keys = torch.cat((key_cache[blocks, offsets].float(), k.view(1, NKV, HD)))
        values = torch.cat((value_cache[blocks, offsets].float(), v.view(1, NKV, HD)))
        group = NH // NKV
        scores = torch.einsum("hd,thd->ht", q.view(NH, HD),
                              keys.repeat_interleave(group, dim=1)) / math.sqrt(HD)
        attn = torch.einsum("ht,thd->hd", scores.softmax(dim=-1),
                            values.repeat_interleave(group, dim=1)).reshape(-1)
        g = dequantize(layer.gate) @ h
        x = normalize(attn * torch.sigmoid(g * (layer.fused.s_gate.float() * s_gate_factor).square()))
        b = dequantize(layer.out) @ x
        ya, yb = normalize(h), normalize(b)
        z = ya + (layer.fused.attn_alpha.float() * alpha_scale).abs() * (yb - ya)
        return normalize(z), q, k, v

    def make_fused(qkvg: LaunchConfig, o: LaunchConfig, residual_warps: int = 4,
                   fuse_residual: bool = True):
        return FusedAttentionSublayer(H, NH, NKV, HD, std, alpha_scale,
                                      config["attention_output_gate_logit_scale_init"], device,
                                      qkvg, o, residual_warps, fuse_residual)

    lines = [f"# Fused Dory attention sublayer ({torch.cuda.get_device_name()}, {args.loops} loops, "
             f"{len(order)} attention sublayers per token, {CONTEXT} cached tokens)", ""]

    # Timing sequences: attention itself is left out, using a fixed attention output.
    fixed_attention = torch.randn(Q, device=device).to(torch.bfloat16)
    h_start = torch.randn(H, device=device)
    h_start = (h_start / h_start.norm()).to(torch.bfloat16)
    buffers = [(torch.empty(H, dtype=torch.bfloat16, device=device),
                torch.empty(8, H // 8, dtype=torch.float32, device=device)) for _ in range(2)]

    def fused_sequence(fused):
        h0_nm = to_nibble_major(h_start.float())

        def run():
            h, h_nm = h_start, h0_nm
            for step, index in enumerate(order):
                layer = layers[index]
                key_cache, value_cache = layer.caches["fused"]
                fused.pre_attention(layer.fused, h_nm, positions, ropes[layer.sliding].cos_sin_cache,
                                    slot_mapping, key_cache, value_cache)
                out, out_nm = buffers[step % 2]
                fused.post_attention(layer.fused, h, fixed_attention, out, out_nm)
                h, h_nm = out, out_nm
        return run

    def vllm_sequence():
        def run():
            h = h_start
            for index in order:
                _, gate = vllm_pre(layers[index], h)
                h = vllm_post(layers[index], h, fixed_attention, gate)
        return run

    best_qkvg, best_o, tuning = QKVG_CONFIGS[0], O_CONFIGS[0], []
    for stage, candidates in (("qkvg", QKVG_CONFIGS), ("o", O_CONFIGS)):
        timings = []
        for candidate in candidates:
            cfg = {"qkvg": best_qkvg, "o": best_o, stage: candidate}
            ms = time_graph([fused_sequence(make_fused(cfg["qkvg"], cfg["o"]))], args.replays)
            timings.append((ms, candidate))
        ms, chosen = min(timings, key=lambda t: t[0])
        tuning.append(f"- {stage}: {chosen} ({ms * 1000 / len(order):.2f} us per sublayer)")
        best_qkvg, best_o = (chosen, best_o) if stage == "qkvg" else (best_qkvg, chosen)
    fused = make_fused(best_qkvg, best_o)
    # The same kernels with the residual update as its own kernel.
    separate = make_fused(best_qkvg, best_o, fuse_residual=False)
    lines += ["## Chosen launch configurations", ""] + tuning

    # Accuracy per attention layer, one decode token after CONTEXT cached tokens.
    lines += ["", "## Accuracy against the FP32 reference, one decode token per layer", "",
              "Relative errors. q: normalized query; k, v: entries written to the KV cache; "
              "output: the sublayer output after the residual update.", "",
              "| Layer | Window | q vLLM | q fused | k vLLM | k fused | v vLLM | v fused "
              "| output vLLM | output fused | output fused vs vLLM "
              "| output fused vs separate residual kernel |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    worst = defaultdict(float)
    for index in sorted(layers):
        layer = layers[index]
        h = torch.randn(H, device=device)
        h = (h / h.norm()).to(torch.bfloat16)
        ref_out, ref_q, ref_k, ref_v = reference(layer, h.float())
        block, offset = CONTEXT // CACHE_BLOCK, CONTEXT % CACHE_BLOCK

        q_vllm, gate = vllm_pre(layer, h)
        key_cache, value_cache = layer.caches["vllm"]
        attn = attend(q_vllm, key_cache, value_cache, CONTEXT + 1, NH).to(torch.bfloat16)
        out_vllm = vllm_post(layer, h, attn, gate)
        k_vllm, v_vllm = key_cache[block, offset].reshape(-1), value_cache[block, offset].reshape(-1)

        key_cache, value_cache = layer.caches["fused"]
        q_fused = fused.pre_attention(layer.fused, to_nibble_major(h.float()), positions,
                                      ropes[layer.sliding].cos_sin_cache, slot_mapping,
                                      key_cache, value_cache).clone()
        attn = attend(q_fused, key_cache, value_cache, CONTEXT + 1, NH).to(torch.bfloat16)
        out_fused, out_nm = buffers[0]
        fused.post_attention(layer.fused, h, attn, out_fused, out_nm)
        k_fused, v_fused = key_cache[block, offset].reshape(-1), value_cache[block, offset].reshape(-1)
        separate.pre_attention(layer.fused, to_nibble_major(h.float()), positions,
                               ropes[layer.sliding].cos_sin_cache, slot_mapping,
                               key_cache, value_cache)
        out_separate, out_separate_nm = buffers[1]
        separate.post_attention(layer.fused, h, attn, out_separate, out_separate_nm)

        cells = {
            "q vLLM": relative_error(q_vllm, ref_q), "q fused": relative_error(q_fused, ref_q),
            "k vLLM": relative_error(k_vllm, ref_k), "k fused": relative_error(k_fused, ref_k),
            "v vLLM": relative_error(v_vllm, ref_v), "v fused": relative_error(v_fused, ref_v),
            "output vLLM": relative_error(out_vllm, ref_out),
            "output fused": relative_error(out_fused, ref_out),
            "output fused vs vLLM": relative_error(out_fused, out_vllm.float()),
            "output fused vs separate": relative_error(out_fused, out_separate.float()),
        }
        for name, value in cells.items():
            worst[name] = max(worst[name], value)
        window = "sliding" if layer.sliding else "full"
        lines.append(f"| {index} | {window} | " + " | ".join(f"{v:.2e}" for v in cells.values()) + " |")
    lines.append("| worst | | " + " | ".join(f"{v:.2e}" for v in worst.values()) + " |")

    vllm_ms = time_graph([vllm_sequence()], args.replays)
    fused_ms = time_graph([fused_sequence(fused)], args.replays)
    separate_ms = time_graph([fused_sequence(separate)], args.replays)
    per = 1000 / len(order)
    lines += ["", "## Time per token for all attention sublayers, attention kernel excluded", "",
              "| Implementation | Kernels per sublayer (without attention) | ms per token "
              "| us per sublayer |", "|---|---:|---:|---:|",
              f"| vLLM path | 7 | {vllm_ms:.3f} | {vllm_ms * per:.1f} |",
              f"| Fused, separate residual kernel | 4 | {separate_ms:.3f} "
              f"| {separate_ms * per:.1f} |",
              f"| Fused | 3 | {fused_ms:.3f} | {fused_ms * per:.1f} |",
              f"| Saving | | {vllm_ms - fused_ms:.3f} | {(vllm_ms - fused_ms) * per:.1f} |"]

    lines += ["", "## Kernel time per sublayer (profiler, us)", "",
              "| Implementation | Kernel | us per sublayer |", "|---|---|---:|"]
    for name, run in (("vLLM path", vllm_sequence()), ("Fused", fused_sequence(fused))):
        totals = defaultdict(float)
        for kernel, us in kernel_breakdown(run).items():
            totals[short_name(kernel)] += us
        for kernel, us in sorted(totals.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name} | `{kernel}` | {us / len(order):.1f} |")

    report = "\n".join(lines) + "\n"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / "results" / f"attention_bench_{stamp}.md"
    out_path.write_text(report)
    print(report)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
