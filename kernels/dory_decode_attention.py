"""Single-token decode attention over vLLM's paged KV cache, with the output gate.

The attention of one query token is split over the cache (flash decoding): a
program per (KV head, split) runs online softmax over a contiguous run of cache
blocks for the KV head's group of query heads and stores its partial result;
a combine kernel merges the splits per query head, multiplies by the output
gate and writes the gated attention output in nibble-major FP32, the input
layout of the attention-output GEMV.

With raw inputs (DecodeAttention.from_raw), the kernel also does the work of
the q/k normalization kernel: it normalizes the query heads and applies the
query scale, and the split holding the new token normalizes its key, writes
the key and value into the cache at vLLM's slot and uses them directly.

Semantics follow vLLM's Triton attention backend for a causal decode token at
position seq_len - 1: keys at positions [seq_len - window, seq_len) for a
sliding-window layer (window > 0), [0, seq_len) otherwise, scaled dot products
with softmax_scale. The cache is the backend's BF16 layout, viewed as key and
value tensors [blocks, block_size, kv_heads, head_dim] with unit stride in the
last dimension.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _split_layout(seq_lens_ptr, WINDOW: tl.constexpr, CACHE_BLOCK: tl.constexpr,
                  SPLITS: tl.constexpr, BLOCKS_PER_TILE: tl.constexpr):
    """(seq_len, first attended position, its cache block, tiles, tiles per split).

    A padding request (seq_len 0) has no tiles and one tile per split.
    """
    seq_len = tl.load(seq_lens_ptr)
    if WINDOW > 0:
        first = tl.maximum(seq_len - WINDOW, 0)
    else:
        first = seq_len * 0
    first_block = first // CACHE_BLOCK
    num_blocks = tl.cdiv(seq_len, CACHE_BLOCK) - first_block
    # Whole tiles per split, so every tile but the last of the range is full.
    tiles = tl.cdiv(num_blocks, BLOCKS_PER_TILE)
    return seq_len, first, first_block, tiles, tl.maximum(tl.cdiv(tiles, SPLITS), 1)


@triton.jit
def _decode_attention_kernel(
    q_ptr, key_cache_ptr, value_cache_ptr, block_table_ptr, seq_lens_ptr,
    partial_out_ptr, partial_max_ptr, partial_sum_ptr, softmax_scale,
    stride_block, stride_offset, stride_head,
    v_ptr, sqk_ptr, sqk_factor, eps, slot_mapping_ptr, cache_offsets_ptr,
    block_table_offsets_ptr,
    WINDOW: tl.constexpr, CACHE_BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr,
    GROUP: tl.constexpr, GROUP_PAD: tl.constexpr, NUM_HEADS: tl.constexpr,
    SPLITS: tl.constexpr, BLOCKS_PER_TILE: tl.constexpr, RAW_QKV: tl.constexpr,
    Q_STRIDE: tl.constexpr, V_STRIDE: tl.constexpr, MULTI_TOKEN: tl.constexpr,
    WRITE_KV: tl.constexpr,
):
    # RAW_QKV: q_ptr holds FP32 [heads + kv_heads, head_dim], rotated query then
    # key heads before normalization, and v_ptr FP32 [kv_heads, head_dim];
    # otherwise q_ptr holds the normalized BF16 query.
    kv_head = tl.program_id(0)
    split = tl.program_id(1)
    # One query token per program_id(2), each with its own inputs, sequence
    # length, cache (an element offset from key_cache_ptr and value_cache_ptr,
    # so tokens at different loop depths use their own loop's cache) and block
    # table (an element offset from block_table_ptr, since vLLM can place a
    # layer's loop caches in KV cache groups with different block tables). The
    # token's cache slot comes from its block table at position seq_len - 1.
    # With one token this is the single-token layout. MULTI_TOKEN is off for the
    # single-token entry points, which compiles this out.
    if MULTI_TOKEN:
        token = tl.program_id(2)
        q_ptr += token * Q_STRIDE
        v_ptr += token * V_STRIDE
        seq_lens_ptr += token
        cache_offset = tl.load(cache_offsets_ptr + token)
        key_cache_ptr += cache_offset
        value_cache_ptr += cache_offset
        block_table_ptr += tl.load(block_table_offsets_ptr + token)
        partial_out_ptr += token * SPLITS * NUM_HEADS * HEAD_DIM
        partial_max_ptr += token * SPLITS * NUM_HEADS
        partial_sum_ptr += token * SPLITS * NUM_HEADS
    rows = tl.arange(0, GROUP_PAD)
    dims = tl.arange(0, HEAD_DIM)
    heads = kv_head * GROUP + rows
    # Loads that do not depend on the sequence length come first, so that their
    # latency overlaps the chain sequence length -> block table -> cache.
    q = tl.load(q_ptr + heads[:, None] * HEAD_DIM + dims[None, :],
                mask=rows[:, None] < GROUP, other=0.0)
    if RAW_QKV:
        s = tl.load(sqk_ptr + heads[:, None] * HEAD_DIM + dims[None, :],
                    mask=rows[:, None] < GROUP, other=0.0).to(tl.float32) * sqk_factor
        k_new = tl.load(q_ptr + (NUM_HEADS + kv_head) * HEAD_DIM + dims)
        v_new = tl.load(v_ptr + kv_head * HEAD_DIM + dims).to(tl.bfloat16)
        if not MULTI_TOKEN:
            slot = tl.load(slot_mapping_ptr)
    seq_len, first, first_block, tiles, tiles_per_split = _split_layout(
        seq_lens_ptr, WINDOW, CACHE_BLOCK, SPLITS, BLOCKS_PER_TILE)
    if RAW_QKV and MULTI_TOKEN:
        last = seq_len - 1
        slot = (tl.load(block_table_ptr + last // CACHE_BLOCK).to(tl.int64) * CACHE_BLOCK
                + last % CACHE_BLOCK)
    tile_begin = split * tiles_per_split
    if tile_begin >= tiles:
        return  # the combine kernel reads only splits that hold tiles
    tile_end = tl.minimum(tile_begin + tiles_per_split, tiles)

    if RAW_QKV:
        # As _qk_norm_cache_kernel: q = (sqk * sqk_factor)^2 * q / |q|, k = k / |k|.
        q = q * tl.rsqrt(tl.sum(q * q, axis=1) + eps)[:, None]
        q = (s * s * q).to(tl.bfloat16)
        k_new = (k_new * tl.rsqrt(tl.sum(k_new * k_new, axis=0) + eps)).to(tl.bfloat16)
        # The split whose last tile holds the new token caches it; a negative
        # slot marks a padding token, which vLLM does not cache. Without
        # WRITE_KV, _write_new_kv_kernel has cached every token already.
        if WRITE_KV & (split == (tiles - 1) // tiles_per_split) & (slot >= 0):
            base = ((slot // CACHE_BLOCK) * stride_block + (slot % CACHE_BLOCK) * stride_offset
                    + kv_head * stride_head)
            tl.store(key_cache_ptr + base + dims, k_new)
            tl.store(value_cache_ptr + base + dims, v_new)
    q = (q.to(tl.float32) * softmax_scale).to(tl.bfloat16)

    TILE: tl.constexpr = BLOCKS_PER_TILE * CACHE_BLOCK
    tokens = tl.arange(0, TILE)
    running_max = tl.full([GROUP_PAD], float("-inf"), tl.float32)
    running_sum = tl.zeros([GROUP_PAD], tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], tl.float32)
    for tile in range(tile_begin, tile_end):
        block = first_block + tile * BLOCKS_PER_TILE + tokens // CACHE_BLOCK
        position = block * CACHE_BLOCK + tokens % CACHE_BLOCK
        valid = (position >= first) & (position < seq_len)
        physical = tl.load(block_table_ptr + block, mask=valid, other=0)
        offsets = (physical.to(tl.int64) * stride_block + (tokens % CACHE_BLOCK) * stride_offset
                   + kv_head * stride_head)
        k = tl.load(key_cache_ptr + offsets[:, None] + dims[None, :], mask=valid[:, None],
                    other=0.0)
        v = tl.load(value_cache_ptr + offsets[:, None] + dims[None, :], mask=valid[:, None],
                    other=0.0)
        if RAW_QKV and WRITE_KV:
            # This program may have just written the new token; use the registers.
            new = (position == seq_len - 1)[:, None]
            k = tl.where(new, k_new[None, :], k)
            v = tl.where(new, v_new[None, :], v)
        scores = tl.dot(q, tl.trans(k))
        scores = tl.where(valid[None, :], scores, float("-inf"))
        # Every tile holds at least one valid position, so new_max is finite.
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        p = tl.exp(scores - new_max[:, None])
        rescale = tl.exp(running_max - new_max)
        running_sum = running_sum * rescale + tl.sum(p, axis=1)
        acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v)
        running_max = new_max

    keep = rows < GROUP
    index = split * NUM_HEADS + heads
    tl.store(partial_max_ptr + index, running_max, mask=keep)
    tl.store(partial_sum_ptr + index, running_sum, mask=keep)
    tl.store(partial_out_ptr + index[:, None] * HEAD_DIM + dims[None, :], acc,
             mask=keep[:, None])


@triton.jit
def _write_new_kv_kernel(
    q_ptr, v_ptr, key_cache_ptr, value_cache_ptr, block_table_ptr, seq_lens_ptr,
    cache_offsets_ptr, block_table_offsets_ptr, stride_block, stride_offset, stride_head,
    eps, NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, CACHE_BLOCK: tl.constexpr,
    Q_STRIDE: tl.constexpr, V_STRIDE: tl.constexpr,
):
    """Caches one token's normalized key and its value for one KV head, exactly as
    _decode_attention_kernel's owner split does (multi-token layout)."""
    kv_head = tl.program_id(0)
    token = tl.program_id(1)
    dims = tl.arange(0, HEAD_DIM)
    k_new = tl.load(q_ptr + token * Q_STRIDE + (NUM_HEADS + kv_head) * HEAD_DIM + dims)
    v_new = tl.load(v_ptr + token * V_STRIDE + kv_head * HEAD_DIM + dims).to(tl.bfloat16)
    k_new = (k_new * tl.rsqrt(tl.sum(k_new * k_new, axis=0) + eps)).to(tl.bfloat16)
    last = tl.load(seq_lens_ptr + token) - 1
    table = block_table_ptr + tl.load(block_table_offsets_ptr + token)
    slot = tl.load(table + last // CACHE_BLOCK).to(tl.int64) * CACHE_BLOCK + last % CACHE_BLOCK
    offset = tl.load(cache_offsets_ptr + token)
    base = ((slot // CACHE_BLOCK) * stride_block + (slot % CACHE_BLOCK) * stride_offset
            + kv_head * stride_head + offset)
    tl.store(key_cache_ptr + base + dims, k_new)
    tl.store(value_cache_ptr + base + dims, v_new)


@triton.jit
def _combine_gate_kernel(
    partial_out_ptr, partial_max_ptr, partial_sum_ptr, seq_lens_ptr, gate_ptr,
    out_nibble_major_ptr, WINDOW: tl.constexpr, CACHE_BLOCK: tl.constexpr,
    NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, SPLITS: tl.constexpr,
    BLOCKS_PER_TILE: tl.constexpr, DIM_CHUNK: tl.constexpr,
):
    head = tl.program_id(0)
    dims = tl.program_id(1) * DIM_CHUNK + tl.arange(0, DIM_CHUNK)
    Q: tl.constexpr = NUM_HEADS * HEAD_DIM
    token = tl.program_id(2)
    partial_out_ptr += token * SPLITS * NUM_HEADS * HEAD_DIM
    partial_max_ptr += token * SPLITS * NUM_HEADS
    partial_sum_ptr += token * SPLITS * NUM_HEADS
    seq_lens_ptr += token
    gate_ptr += token * Q
    out_nibble_major_ptr += token * Q
    _, _, _, tiles, tiles_per_split = _split_layout(seq_lens_ptr, WINDOW, CACHE_BLOCK, SPLITS,
                                                    BLOCKS_PER_TILE)
    splits = tl.arange(0, SPLITS)
    active = splits < tl.cdiv(tiles, tiles_per_split)
    maxima = tl.load(partial_max_ptr + splits * NUM_HEADS + head, mask=active,
                     other=float("-inf"))
    sums = tl.load(partial_sum_ptr + splits * NUM_HEADS + head, mask=active, other=0.0)
    overall = tl.max(maxima, axis=0)
    weights = tl.where(active, tl.exp(maxima - overall), 0.0)
    partial = tl.load(partial_out_ptr + (splits[:, None] * NUM_HEADS + head) * HEAD_DIM
                      + dims[None, :], mask=active[:, None], other=0.0)
    total = tl.sum(weights * sums, axis=0)
    # A padding request attends to nothing; its output is zero.
    out = tl.sum(weights[:, None] * partial, axis=0) / tl.where(total > 0, total, 1.0)
    element = head * HEAD_DIM + dims
    out = out * tl.load(gate_ptr + element)
    tl.store(out_nibble_major_ptr + (element % 8) * (Q // 8) + element // 8, out)


class DecodeAttention:
    """Gated decode attention with scratch buffers for a fixed split count.

    One query token per call (__call__, from_raw), or up to max_tokens tokens
    at once (multi_from_raw), each with its own position, cache and block table.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, device: torch.device,
                 splits: int = 32, blocks_per_tile: int = 1, num_warps: int = 4,
                 num_stages: int = 2, dim_chunk: int = 64, max_tokens: int = 1):
        if num_heads % num_kv_heads:
            raise ValueError("query heads must be a multiple of KV heads")
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.group = num_heads // num_kv_heads
        self.splits, self.blocks_per_tile, self.num_warps = splits, blocks_per_tile, num_warps
        self.num_stages, self.dim_chunk = num_stages, dim_chunk
        self.max_tokens = max_tokens
        self.no_offset = torch.zeros(1, dtype=torch.int64, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        self.partial_out = torch.empty(max_tokens, splits, num_heads, head_dim, **f32)
        self.partial_max = torch.empty(max_tokens, splits, num_heads, **f32)
        self.partial_sum = torch.empty(max_tokens, splits, num_heads, **f32)

    def __call__(self, q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                 block_table: torch.Tensor, seq_lens: torch.Tensor, window: int,
                 softmax_scale: float, gate: torch.Tensor, out_nibble_major: torch.Tensor) -> None:
        """q: BF16 [heads * head_dim], already in the cache's k and v; block_table:
        [max_blocks] of this sequence; seq_lens: [1] tokens including this one;
        window: 0 for full attention; gate: FP32 [heads * head_dim];
        out_nibble_major: FP32 [8, heads * head_dim / 8]."""
        self._launch(q, key_cache, value_cache, block_table, seq_lens, window, softmax_scale,
                     gate, out_nibble_major, raw=None, tokens=1, cache_offsets=self.no_offset)

    def from_raw(self, qk: torch.Tensor, v: torch.Tensor, sqk: torch.Tensor, sqk_factor: float,
                 eps: float, slot_mapping: torch.Tensor, key_cache: torch.Tensor,
                 value_cache: torch.Tensor, block_table: torch.Tensor, seq_lens: torch.Tensor,
                 window: int, softmax_scale: float, gate: torch.Tensor,
                 out_nibble_major: torch.Tensor) -> None:
        """Like __call__, from the rotated, unnormalized q and k heads (FP32
        [(heads + kv_heads) * head_dim]) and v (FP32 [kv_heads * head_dim]); also
        writes this token's k and v into the cache at slot_mapping[0]."""
        self._launch(qk, key_cache, value_cache, block_table, seq_lens, window, softmax_scale,
                     gate, out_nibble_major, raw=(v, sqk, sqk_factor, eps, slot_mapping),
                     tokens=1, cache_offsets=self.no_offset)

    def multi_from_raw(self, qk: torch.Tensor, v: torch.Tensor, sqk: torch.Tensor,
                       sqk_factor: float, eps: float, cache_offsets: torch.Tensor,
                       key_cache: torch.Tensor, value_cache: torch.Tensor,
                       block_tables: torch.Tensor, block_table_offsets: torch.Tensor,
                       seq_lens: torch.Tensor, window: int, softmax_scale: float,
                       gate: torch.Tensor, out_nibble_major: torch.Tensor,
                       shared_cache: bool = False) -> None:
        """from_raw for T tokens of one sequence: qk [T, (heads + kv_heads) * head_dim],
        v [T, kv_heads * head_dim], gate [T, heads * head_dim], out_nibble_major
        [T, 8, heads * head_dim / 8], seq_lens [T]. Token t is at position
        seq_lens[t] - 1 and uses the cache at key_cache / value_cache shifted by
        cache_offsets[t] elements, so the caches of one layer's loops must share
        their strides, and the block table at block_tables shifted by
        block_table_offsets[t] elements; it writes its key and value at the slot
        that block table gives for its position. Offsets are int64.

        shared_cache: tokens may read each other's slots (several loops sharing one
        cache, where an older token's slot holds its latest loop). Every token's
        key and value are then cached by a separate kernel before attention, so
        each token reads the others' new entries rather than racing their writes."""
        tokens = qk.shape[0]
        if (tokens > self.max_tokens or cache_offsets.dtype != torch.int64
                or block_table_offsets.dtype != torch.int64):
            raise ValueError("too many tokens for the scratch buffers, or non-int64 offsets")
        if shared_cache:
            _write_new_kv_kernel[(self.num_kv_heads, tokens)](
                qk, v, key_cache, value_cache, block_tables, seq_lens, cache_offsets,
                block_table_offsets, key_cache.stride(0), key_cache.stride(1),
                key_cache.stride(2), eps, NUM_HEADS=self.num_heads, HEAD_DIM=self.head_dim,
                CACHE_BLOCK=key_cache.shape[1],
                Q_STRIDE=(self.num_heads + self.num_kv_heads) * self.head_dim,
                V_STRIDE=self.num_kv_heads * self.head_dim, num_warps=4)
        self._launch(qk, key_cache, value_cache, block_tables, seq_lens, window, softmax_scale,
                     gate, out_nibble_major, raw=(v, sqk, sqk_factor, eps, seq_lens),
                     tokens=tokens, cache_offsets=cache_offsets,
                     block_table_offsets=block_table_offsets, write_kv=not shared_cache)

    def _launch(self, q, key_cache, value_cache, block_table, seq_lens, window, softmax_scale,
                gate, out_nibble_major, raw, tokens, cache_offsets,
                block_table_offsets=None, write_kv=True) -> None:
        if key_cache.stride() != value_cache.stride() or key_cache.stride(-1) != 1:
            raise ValueError("key and value caches need equal strides and unit last stride")
        v, sqk, sqk_factor, eps, slot_mapping = raw or (q, q, 0.0, 0.0, seq_lens)
        q_stride = ((self.num_heads + self.num_kv_heads) if raw else self.num_heads) * self.head_dim
        _decode_attention_kernel[(self.num_kv_heads, self.splits, tokens)](
            q, key_cache, value_cache, block_table, seq_lens, self.partial_out,
            self.partial_max, self.partial_sum, softmax_scale,
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), v, sqk, sqk_factor,
            eps, slot_mapping, cache_offsets,
            self.no_offset if block_table_offsets is None else block_table_offsets,
            WINDOW=window, CACHE_BLOCK=key_cache.shape[1], HEAD_DIM=self.head_dim,
            GROUP=self.group, GROUP_PAD=max(16, triton.next_power_of_2(self.group)),
            NUM_HEADS=self.num_heads, SPLITS=self.splits, BLOCKS_PER_TILE=self.blocks_per_tile,
            RAW_QKV=raw is not None, Q_STRIDE=q_stride, V_STRIDE=self.num_kv_heads * self.head_dim,
            MULTI_TOKEN=cache_offsets is not self.no_offset, WRITE_KV=write_kv,
            num_warps=self.num_warps,
            num_stages=self.num_stages)
        _combine_gate_kernel[(self.num_heads, self.head_dim // self.dim_chunk, tokens)](
            self.partial_out, self.partial_max, self.partial_sum, seq_lens, gate,
            out_nibble_major, WINDOW=window, CACHE_BLOCK=key_cache.shape[1],
            NUM_HEADS=self.num_heads, HEAD_DIM=self.head_dim, SPLITS=self.splits,
            BLOCKS_PER_TILE=self.blocks_per_tile, DIM_CHUNK=self.dim_chunk, num_warps=4)
