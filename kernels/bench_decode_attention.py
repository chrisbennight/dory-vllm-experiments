"""Validates and times the gated decode attention kernel against vLLM's Triton attention.

Each attention layer application of a decode step (2 loops, decode order) gets
its own paged KV cache, with blocks placed in shuffled order through the block
table, so the timed sequence reads the cache from DRAM as a real step does.
Two implementations of single-token attention plus the output gate:

- vllm: vLLM's unified_attention as its Triton backend calls it for a decode
  token (split over 16 softmax segments, then a reduction kernel), followed by
  the gate multiplication the fused attention-output GEMV applies on its input;
- fused: dory_decode_attention.DecodeAttention, whose combine kernel applies
  the gate and writes the nibble-major input of the attention-output GEMV.

Accuracy is checked against FP32 softmax attention for full and sliding-window
layers at several context lengths, including contexts longer than the window.
Timing replays the decode-order sequence in a CUDA graph for every launch
configuration of the fused kernel. A second part checks the raw-input mode
(DecodeAttention.from_raw), which also normalizes q and k and writes the new
token into the cache, against the q/k normalization kernel followed by the
attention kernel.

Usage: .venv/bin/python kernels/bench_decode_attention.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import torch

from bench_gemv import CHECKPOINT, ROOT, decode_order, require_idle_gpu, time_graph
from bench_mlp import kernel_breakdown, relative_error, short_name
from dory_attention import _qk_norm_cache_kernel
from dory_common import EPS
from dory_decode_attention import DecodeAttention

CACHE_BLOCK = 16
CONTEXTS = (640, 2048, 8192)  # cached tokens before the decoded one
SPLITS = (32, 64)
BLOCKS_PER_TILE = (1, 2)
STAGES = (1, 2)
VLLM_SEGMENTS = 16  # vLLM's NUM_PAR_SOFTMAX_SEGMENTS on this GPU


class PagedCache:
    """One layer's KV cache in the Triton backend's layout with a shuffled block table."""

    def __init__(self, tokens: int, kv_heads: int, head_dim: int, device: torch.device):
        blocks = tokens // CACHE_BLOCK + 1
        kv = torch.randn(blocks, CACHE_BLOCK, kv_heads, 2 * head_dim, device=device)
        kv = kv / kv.norm(dim=-1, keepdim=True) * 8
        self.kv = kv.to(torch.bfloat16)
        self.key, self.value = self.kv.split(head_dim, dim=-1)
        self.block_table = torch.randperm(blocks, device=device).to(torch.int32)


def reference_attention(q: torch.Tensor, cache: PagedCache, seq_len: int, window: int,
                        num_heads: int, scale: float) -> torch.Tensor:
    first = max(seq_len - window, 0) if window else 0
    positions = torch.arange(first, seq_len, device=q.device)
    blocks = cache.block_table[positions // CACHE_BLOCK].long()
    keys = cache.key[blocks, positions % CACHE_BLOCK].float()
    values = cache.value[blocks, positions % CACHE_BLOCK].float()
    group = num_heads // keys.shape[1]
    keys, values = keys.repeat_interleave(group, dim=1), values.repeat_interleave(group, dim=1)
    scores = torch.einsum("hd,thd->ht", q.float().view(num_heads, -1), keys) * scale
    return torch.einsum("ht,thd->hd", scores.softmax(dim=-1), values).reshape(-1)


class VllmAttention:
    """vLLM's Triton unified attention for one decode token, as its backend calls it."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, device: torch.device):
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention
        self.unified_attention = unified_attention
        self.num_heads, self.head_dim = num_heads, head_dim
        f32 = dict(dtype=torch.float32, device=device)
        self.segm_output = torch.empty(1, num_heads, VLLM_SEGMENTS, head_dim, **f32)
        self.segm_max = torch.empty(1, num_heads, VLLM_SEGMENTS, **f32)
        self.segm_expsum = torch.empty(1, num_heads, VLLM_SEGMENTS, **f32)
        self.descale = torch.ones(1, num_kv_heads, **f32)
        self.cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
        self.out = torch.empty(1, num_heads, head_dim, dtype=torch.bfloat16, device=device)

    def __call__(self, q, cache: PagedCache, seq_lens, max_seq_len: int, window: int,
                 scale: float, gate: torch.Tensor, gated_out: torch.Tensor) -> None:
        self.unified_attention(
            q=q.view(1, self.num_heads, self.head_dim), k=cache.key, v=cache.value,
            out=self.out, cu_seqlens_q=self.cu_seqlens_q, max_seqlen_q=1, seqused_k=seq_lens,
            max_seqlen_k=max_seq_len, softmax_scale=scale, causal=True,
            window_size=(window - 1, 0) if window else (-1, -1),
            block_table=cache.block_table[None], softcap=0, q_descale=None,
            k_descale=self.descale, v_descale=self.descale, seq_threshold_3D=32,
            num_par_softmax_segments=VLLM_SEGMENTS, softmax_segm_output=self.segm_output,
            softmax_segm_max=self.segm_max, softmax_segm_expsum=self.segm_expsum)
        torch.mul(self.out.view(-1), gate, out=gated_out)


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + EPS)


def raw_mode_section(NH: int, NKV: int, HD: int, window_size: int, scale: float,
                     order: list, windows: list, gate: torch.Tensor, replays: int,
                     device: torch.device) -> list[str]:
    """Accuracy and timing of DecodeAttention.from_raw."""
    sqk = (torch.rand(NH * HD, device=device) + 0.5).to(torch.bfloat16)
    sqk_factor = 0.7
    qk = torch.randn((NH + NKV) * HD, device=device)
    v = torch.randn(NKV * HD, device=device)
    q_expected = ((sqk.float() * sqk_factor).square().view(NH, HD)
                  * normalize_rows(qk[:NH * HD].view(NH, HD))).to(torch.bfloat16).view(-1)
    k_expected = normalize_rows(qk[NH * HD:].view(NKV, HD)).to(torch.bfloat16)
    v_expected = v.view(NKV, HD).to(torch.bfloat16)
    fused = DecodeAttention(NH, NKV, HD, device)
    out = torch.empty(8, NH * HD // 8, device=device)
    lines = ["", "## Raw-input mode: q/k normalization and cache write inside the kernel", "",
             "| Cached tokens | Layer | New k and v cached exactly | Relative error |",
             "|---:|---|---|---:|"]
    for context in CONTEXTS:
        for window in (0, window_size):
            cache = PagedCache(context + 1, NKV, HD, device)
            physical = cache.block_table[context // CACHE_BLOCK].long()
            slot = torch.tensor([physical * CACHE_BLOCK + context % CACHE_BLOCK], device=device)
            seq_lens = torch.tensor([context + 1], dtype=torch.int32, device=device)
            fused.from_raw(qk, v, sqk, sqk_factor, EPS, slot, cache.key, cache.value,
                           cache.block_table, seq_lens, window, scale, gate, out)
            exact = (torch.equal(cache.key[physical, context % CACHE_BLOCK], k_expected)
                     and torch.equal(cache.value[physical, context % CACHE_BLOCK], v_expected))
            # With the new token verified in the cache, it is the reference cache.
            reference = reference_attention(q_expected, cache, context + 1, window, NH,
                                            scale) * gate
            lines.append(f"| {context} | {'sliding' if window else 'full'} | {exact} "
                         f"| {relative_error(out.t().reshape(-1), reference):.2e} |")

    context = CONTEXTS[0]
    caches = [PagedCache(context + 1, NKV, HD, device) for _ in order]
    seq_lens = torch.tensor([context + 1], dtype=torch.int32, device=device)
    slot = torch.tensor([context], device=device)  # any valid slot; timing only
    q_out = torch.empty(NH * HD, dtype=torch.bfloat16, device=device)
    k_out = torch.empty(NKV * HD, dtype=torch.bfloat16, device=device)

    def separate():
        for cache, window in zip(caches, windows):
            _qk_norm_cache_kernel[(NH + NKV,)](
                qk, v, sqk, q_out, cache.key, cache.value, slot, k_out, k_out, sqk_factor, EPS,
                cache.key.stride(0), cache.key.stride(1), cache.key.stride(2),
                NUM_Q_HEADS=NH, HEAD_DIM=HD, CACHE_BLOCK=CACHE_BLOCK, WRITE_CACHE=True,
                num_warps=2)
            fused(q_out, cache.key, cache.value, cache.block_table, seq_lens, window, scale,
                  gate, out)

    def combined():
        for cache, window in zip(caches, windows):
            fused.from_raw(qk, v, sqk, sqk_factor, EPS, slot, cache.key, cache.value,
                           cache.block_table, seq_lens, window, scale, gate, out)
    per = 1000 / len(order)
    lines += ["", f"Time per attention layer application at {context} cached tokens (CUDA "
              "graph): q/k normalization kernel + attention "
              f"{time_graph([separate], replays) * per:.2f} us; raw-input mode "
              f"{time_graph([combined], replays) * per:.2f} us."]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--replays", type=int, default=30)
    args = parser.parse_args()
    require_idle_gpu()
    device = torch.device("cuda")
    torch.manual_seed(0)

    config = json.loads((CHECKPOINT / "config.json").read_text())
    NH, NKV, HD = (config["num_attention_heads"], config["num_key_value_heads"],
                   config["head_dim"])
    window_size = config["sliding_window"]
    scale = HD ** -0.5
    order = [i for i in decode_order(config, args.loops) if config["layer_types"][i] != "mlp"]
    windows = [window_size if config["layer_types"][i] == "sliding_attention" else 0
               for i in order]
    vllm = VllmAttention(NH, NKV, HD, device)
    gate = torch.rand(NH * HD, device=device)
    gated_vllm = torch.empty(NH * HD, device=device)
    out_fused = torch.empty(8, NH * HD // 8, device=device)
    q = torch.randn(NH, HD, device=device)
    q = (q / q.norm(dim=-1, keepdim=True) * 8).to(torch.bfloat16).view(-1)

    lines = [f"# Gated decode attention ({torch.cuda.get_device_name()}, {args.loops} loops, "
             f"{len(order)} attention layer applications per token, {NH} query and {NKV} KV "
             f"heads of {HD}, window {window_size})", "",
             "## Accuracy against FP32 attention", "",
             "Relative error of the gated attention output; the fused kernel uses 32 splits.",
             "", "| Cached tokens | Layer | vLLM | Fused | Fused vs vLLM |",
             "|---:|---|---:|---:|---:|"]
    fused = DecodeAttention(NH, NKV, HD, device, splits=32)
    for context in CONTEXTS:
        cache = PagedCache(context + 1, NKV, HD, device)
        seq_lens = torch.tensor([context + 1], dtype=torch.int32, device=device)
        for window in (0, window_size):
            reference = reference_attention(q, cache, context + 1, window, NH, scale) * gate
            vllm(q, cache, seq_lens, context + 1, window, scale, gate, gated_vllm)
            fused(q, cache.key, cache.value, cache.block_table, seq_lens, window, scale, gate,
                  out_fused)
            got = out_fused.t().reshape(-1)
            lines.append(f"| {context} | {'sliding' if window else 'full'} "
                         f"| {relative_error(gated_vllm, reference):.2e} "
                         f"| {relative_error(got, reference):.2e} "
                         f"| {relative_error(got, gated_vllm):.2e} |")
        del cache

    lines += ["", "## Time per token for all attention layer applications (CUDA graph, us)", "",
              "Each configuration is splits x cache blocks per tile x pipeline stages "
              "(4 warps).", "",
              "| Implementation | " + " | ".join(f"{c} tokens" for c in CONTEXTS) + " |",
              "|---|" + "---:|" * len(CONTEXTS)]
    timings = {}
    for column, context in enumerate(CONTEXTS):
        caches = [PagedCache(context + 1, NKV, HD, device) for _ in order]
        seq_lens = torch.tensor([context + 1], dtype=torch.int32, device=device)

        def vllm_run():
            for cache, window in zip(caches, windows):
                vllm(q, cache, seq_lens, context + 1, window, scale, gate, gated_vllm)
        timings.setdefault("vLLM (16 segments) + gate", {})[context] = (
            time_graph([vllm_run], args.replays) * 1000)
        for splits in SPLITS:
            for blocks_per_tile in BLOCKS_PER_TILE:
                for stages in STAGES:
                    fused = DecodeAttention(NH, NKV, HD, device, splits, blocks_per_tile,
                                            num_stages=stages)

                    def fused_run():
                        for cache, window in zip(caches, windows):
                            fused(q, cache.key, cache.value, cache.block_table, seq_lens,
                                  window, scale, gate, out_fused)
                    name = f"Fused {splits} x {blocks_per_tile} x {stages}"
                    timings.setdefault(name, {})[context] = (
                        time_graph([fused_run], args.replays) * 1000)
        if column == 0:
            breakdown = {"vLLM": kernel_breakdown(vllm_run)}
            best = min((n for n in timings if n.startswith("Fused")),
                       key=lambda n: timings[n][context])
            splits, blocks_per_tile, stages = (int(v) for v in best.split()[1::2])
            fused = DecodeAttention(NH, NKV, HD, device, splits, blocks_per_tile,
                                    num_stages=stages)
            breakdown[best] = kernel_breakdown(fused_run)
        del caches
        torch.cuda.empty_cache()
    per = 1 / len(order)
    for name, by_context in timings.items():
        lines.append(f"| {name} | " + " | ".join(
            f"{by_context[c] * per:.2f}" for c in CONTEXTS) + " |")
    lines += ["", f"Microseconds per attention layer application; {len(order)} per token.",
              "", f"## Kernel time per application at {CONTEXTS[0]} cached tokens "
              "(profiler, us)", "", "| Implementation | Kernel | us |", "|---|---|---:|"]
    for name, totals in breakdown.items():
        for kernel, us in sorted(totals.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name} | `{short_name(kernel)}` | {us * per:.2f} |")

    lines += raw_mode_section(NH, NKV, HD, window_size, scale, order, windows, gate,
                              args.replays, device)
    report = "\n".join(lines) + "\n"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / "results" / f"decode_attention_bench_{stamp}.md"
    out_path.write_text(report)
    print(report)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
