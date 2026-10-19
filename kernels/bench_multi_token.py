"""Validates and times the multi-token sublayers against the single-token ones.

The wavefront schedule runs Dory's recurrent block for up to four tokens at
different loop depths at once. This benchmark loads the real recurrent-block
weights and gives each token its own KV cache (as a token at its own loop
depth has) and its own position.

- Accuracy with one shared cache (loops sharing a cache): MultiTokenSublayers
  with shared_cache against the single-token sublayers run in position order.
- Accuracy: MultiTokenSublayers for 4 tokens against FusedAttentionSublayer
  (with DecodeAttention) and FusedMlpSublayer run once per token, for a
  sliding-window and a full-attention layer and an MLP layer: relative
  difference of each token's output and of the key and value it caches.
- Timing: the recurrent block's 20 attention+MLP layer pairs in a CUDA graph,
  for the single-token sublayers (one token) and the multi-token sublayers
  for 1, 2 and 4 tokens.

Usage: .venv/bin/python kernels/bench_multi_token.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import torch
from safetensors import safe_open

from bench_gemv import CHECKPOINT, ROOT, require_idle_gpu, time_graph
from dory_attention import FusedAttentionSublayer, load_attention_weights
from dory_common import EPS, LaunchConfig
from dory_decode_attention import DecodeAttention
from dory_mlp import FusedMlpSublayer, load_mlp_weights
from dory_multi_token import MultiTokenSublayers
from w4a16_gemv import to_nibble_major

CONTEXT = 640
CACHE_BLOCK = 16
TOKENS = 4
# The single-token launch configurations the fused decoder uses.
QKVG = LaunchConfig(block_n=4, block_w=64, num_warps=4)
ATTENTION_OUT = LaunchConfig(block_n=4, block_w=128, num_warps=4)
ATTENTION_OUT_NIBBLE_MAJOR = LaunchConfig(block_n=4, block_w=64, num_warps=4)
GATE_UP = LaunchConfig(block_n=4, block_w=64, num_warps=4)
DOWN_FOLDED = LaunchConfig(block_n=4, block_w=64, num_warps=4)


def relative(x: torch.Tensor, y: torch.Tensor) -> float:
    return ((x.float() - y.float()).norm() / y.float().norm()).item()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()
    require_idle_gpu()
    device = torch.device("cuda")
    torch.manual_seed(0)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope

    config = json.loads((CHECKPOINT / "config.json").read_text())
    H, F = config["hidden_size"], config["intermediate_size"]
    NH, NKV, HD = (config["num_attention_heads"], config["num_key_value_heads"],
                   config["head_dim"])
    std = config["initializer_range"]
    alpha_scale = config["ngpt_alpha_init"] / std
    gate_init = config["attention_output_gate_logit_scale_init"]
    window_size = config["sliding_window"]
    with set_current_vllm_config(VllmConfig()):
        ropes = {kind: get_rope(HD, max_position=config["max_position_embeddings"],
                                is_neox_style=True,
                                rope_parameters=config["rope_parameters"][kind],
                                dtype=torch.bfloat16).to(device).cos_sin_cache
                 for kind in ("sliding_attention", "full_attention")}

    first = config["n_input_layers"]
    recurrent = range(first, first + config["n_recurrent_layers"])
    attention_layers = [i for i in recurrent if config["layer_types"][i] != "mlp"]
    mlp_layers = [i for i in recurrent if config["layer_types"][i] == "mlp"]
    with safe_open(str(CHECKPOINT / "model.safetensors"), framework="pt",
                   device=str(device)) as f:
        attention = {i: load_attention_weights(f, i, NH, NKV, HD) for i in attention_layers}
        mlp = {i: load_mlp_weights(f, i) for i in mlp_layers}

    multi = MultiTokenSublayers(H, F, NH, NKV, HD, std, alpha_scale, gate_init, device,
                                max_tokens=TOKENS)
    single_attention = FusedAttentionSublayer(H, NH, NKV, HD, std, alpha_scale, gate_init, device,
                                              QKVG, ATTENTION_OUT,
                                              o_nibble_major=ATTENTION_OUT_NIBBLE_MAJOR)
    single_decode = DecodeAttention(NH, NKV, HD, device)
    single_mlp = FusedMlpSublayer(H, F, alpha_scale, device, GATE_UP, DOWN_FOLDED)
    scale = HD ** -0.5

    # Each layer has one cache per token (per loop depth); the newest token has
    # the highest position, as in the wavefront. Each token also has its own
    # shuffled block table, as when vLLM puts a layer's loop caches in KV cache
    # groups with different block tables.
    blocks = (CONTEXT + TOKENS) // CACHE_BLOCK + 2
    caches = {i: torch.randn(TOKENS, blocks, CACHE_BLOCK, NKV, 2 * HD, device=device)
              .to(torch.bfloat16) for i in attention_layers}
    block_tables = torch.stack([torch.randperm(blocks, device=device)
                                for _ in range(TOKENS)]).to(torch.int32)
    table_offsets = torch.arange(TOKENS, device=device, dtype=torch.int64) * blocks
    positions = CONTEXT + TOKENS - 1 - torch.arange(TOKENS, device=device)
    seq_lens = (positions + 1).to(torch.int32)
    physical = block_tables[torch.arange(TOKENS, device=device), positions // CACHE_BLOCK]
    slots = physical.to(torch.int64) * CACHE_BLOCK + positions % CACHE_BLOCK
    cache_offsets = torch.arange(TOKENS, device=device, dtype=torch.int64) * caches[
        attention_layers[0]][0].numel()
    h = torch.randn(TOKENS, H, device=device)
    h = (h / h.norm(dim=-1, keepdim=True)).to(torch.bfloat16)
    h_nm = torch.stack([to_nibble_major(row.float()) for row in h])
    out = torch.empty(TOKENS, H, dtype=torch.bfloat16, device=device)
    out_nm = torch.empty(TOKENS, 8, H // 8, device=device)
    gated = torch.empty(8, NH * HD // 8, device=device)

    def window_of(index: int) -> int:
        return window_size if config["layer_types"][index] == "sliding_attention" else 0

    def rope_of(index: int) -> torch.Tensor:
        kind = "sliding_attention" if window_of(index) else "full_attention"
        return ropes[kind]

    lines = [f"# Multi-token Dory sublayers ({torch.cuda.get_device_name()}, {CONTEXT}-token "
             f"context, {TOKENS} tokens in their own caches)", "",
             "## Accuracy against the single-token sublayers run once per token", "",
             "Relative difference per token (BF16 outputs).", "",
             "| Sublayer | Token | Output | Cached key | Cached value |", "|---|---:|---:|---:|---:|"]
    sliding = next(i for i in attention_layers if window_of(i))
    full = next(i for i in attention_layers if not window_of(i))
    for index in (sliding, full):
        w, kv = attention[index], caches[index]
        start = kv.clone()
        key, value = kv[0].split(HD, dim=-1)
        multi.attention(w, h, h_nm, positions, rope_of(index), cache_offsets, key, value,
                        block_tables, table_offsets, seq_lens, window_of(index), scale, out,
                        out_nm)
        multi_out, multi_kv = out.clone(), kv.clone()
        for t in range(TOKENS):
            kv_t = start[t].clone()
            key_t, value_t = kv_t.split(HD, dim=-1)
            single_attention.project_qkvg(w, h_nm[t], positions[t:t + 1], rope_of(index))
            single_decode.from_raw(single_attention.qk, single_attention.v, w.sqk,
                                   single_attention.sqk_factor, EPS, slots[t:t + 1], key_t,
                                   value_t, block_tables[t], seq_lens[t:t + 1],
                                   window_of(index), scale, single_attention.gate, gated)
            single_out, single_out_nm = out[0].clone(), out_nm[0].clone()
            single_attention.post_attention(w, h[t], gated, single_out, single_out_nm,
                                            gated_nibble_major=True)
            block, offset = physical[t], positions[t] % CACHE_BLOCK
            lines.append(
                f"| attention layer {index} ({'sliding' if window_of(index) else 'full'}) | {t} "
                f"| {relative(multi_out[t], single_out):.2e} "
                f"| {relative(multi_kv[t, block, offset, :, :HD], kv_t[block, offset, :, :HD]):.2e} "
                f"| {relative(multi_kv[t, block, offset, :, HD:], kv_t[block, offset, :, HD:]):.2e} |")
        kv.copy_(start)

    # One cache for all tokens, as when the loops share a cache: each token reads
    # the new entries of the tokens at lower positions. The reference runs the
    # single-token sublayer on the same cache one token at a time, lowest
    # position first, so each token sees exactly those entries.
    shared_lines = ["", "## One shared cache, against the single-token sublayers in position "
                    "order", "", "| Sublayer | Token | Output | Cached key | Cached value |",
                    "|---|---:|---:|---:|---:|"]
    same = torch.zeros(TOKENS, dtype=torch.int64, device=device)
    table = block_tables[0]
    shared_physical = table[positions // CACHE_BLOCK]
    shared_slots = shared_physical.to(torch.int64) * CACHE_BLOCK + positions % CACHE_BLOCK
    for index in (sliding, full):
        w, start = attention[index], caches[index][0].clone()
        multi_kv = start.clone()
        key, value = multi_kv.split(HD, dim=-1)
        multi.attention(w, h, h_nm, positions, rope_of(index), same, key, value, table, same,
                        seq_lens, window_of(index), scale, out, out_nm, shared_cache=True)
        multi_out = out.clone()
        single_kv = start.clone()
        key_s, value_s = single_kv.split(HD, dim=-1)
        single_outs = {}
        for t in sorted(range(TOKENS), key=lambda t: positions[t].item()):
            single_attention.project_qkvg(w, h_nm[t], positions[t:t + 1], rope_of(index))
            single_decode.from_raw(single_attention.qk, single_attention.v, w.sqk,
                                   single_attention.sqk_factor, EPS, shared_slots[t:t + 1],
                                   key_s, value_s, table, seq_lens[t:t + 1], window_of(index),
                                   scale, single_attention.gate, gated)
            single_out, single_out_nm = out[0].clone(), out_nm[0].clone()
            single_attention.post_attention(w, h[t], gated, single_out, single_out_nm,
                                            gated_nibble_major=True)
            single_outs[t] = single_out
        for t in range(TOKENS):
            block, offset = shared_physical[t], positions[t] % CACHE_BLOCK
            shared_lines.append(
                f"| attention layer {index} ({'sliding' if window_of(index) else 'full'}) | {t} "
                f"| {relative(multi_out[t], single_outs[t]):.2e} "
                f"| {relative(multi_kv[block, offset, :, :HD], single_kv[block, offset, :, :HD]):.2e} "
                f"| {relative(multi_kv[block, offset, :, HD:], single_kv[block, offset, :, HD:]):.2e} |")
    index = mlp_layers[0]
    multi.mlp(mlp[index], h, h_nm, out, out_nm)
    multi_out = out.clone()
    for t in range(TOKENS):
        single_out, single_out_nm = out[0].clone(), out_nm[0].clone()
        single_mlp(mlp[index], h[t], h_nm[t], single_out, single_out_nm)
        lines.append(f"| MLP layer {index} | {t} | {relative(multi_out[t], single_out):.2e} | | |")
    lines += shared_lines

    # Timing: the recurrent block's layer pairs, reading every layer's weights.
    pairs = list(zip(attention_layers, mlp_layers))
    buffers = [(torch.empty(TOKENS, H, dtype=torch.bfloat16, device=device),
                torch.empty(TOKENS, 8, H // 8, device=device)) for _ in range(2)]

    def multi_run(tokens: int):
        def run():
            x, x_nm = h[:tokens], h_nm[:tokens]
            for attention_index, mlp_index in pairs:
                key, value = caches[attention_index][0].split(HD, dim=-1)
                a, a_nm = buffers[0][0][:tokens], buffers[0][1][:tokens]
                multi.attention(attention[attention_index], x, x_nm, positions[:tokens],
                                rope_of(attention_index), cache_offsets[:tokens], key, value,
                                block_tables, table_offsets[:tokens], seq_lens[:tokens],
                                window_of(attention_index), scale, a, a_nm)
                m, m_nm = buffers[1][0][:tokens], buffers[1][1][:tokens]
                multi.mlp(mlp[mlp_index], a, a_nm, m, m_nm)
                x, x_nm = m, m_nm
        return run

    def single_run():
        x, x_nm = h[0], h_nm[0]
        for attention_index, mlp_index in pairs:
            key, value = caches[attention_index][0].split(HD, dim=-1)
            w = attention[attention_index]
            single_attention.project_qkvg(w, x_nm, positions[:1], rope_of(attention_index))
            single_decode.from_raw(single_attention.qk, single_attention.v, w.sqk,
                                   single_attention.sqk_factor, EPS, slots[:1], key, value,
                                   block_tables[0], seq_lens[:1], window_of(attention_index),
                                   scale, single_attention.gate, gated)
            a, a_nm = buffers[0][0][0], buffers[0][1][0]
            single_attention.post_attention(w, x, gated, a, a_nm, gated_nibble_major=True)
            m, m_nm = buffers[1][0][0], buffers[1][1][0]
            single_mlp(mlp[mlp_index], a, a_nm, m, m_nm)
            x, x_nm = m, m_nm

    per_pair = 1000 / len(pairs)
    single_us = time_graph([single_run], args.replays) * per_pair
    lines += ["", f"## Time per attention+MLP layer pair ({len(pairs)} recurrent pairs, CUDA "
              "graph)", "", "| Path | Tokens | us per pair | Relative to single-token, 1 token |",
              "|---|---:|---:|---:|",
              f"| single-token sublayers | 1 | {single_us:.1f} | 1.00 |"]
    for tokens in (1, 2, TOKENS):
        us = time_graph([multi_run(tokens)], args.replays) * per_pair
        lines.append(f"| multi-token sublayers | {tokens} | {us:.1f} | {us / single_us:.2f} |")
        print(lines[-1], flush=True)

    report = "\n".join(lines) + "\n"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / "results" / f"multi_token_bench_{stamp}.md"
    out_path.write_text(report)
    print(report)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
