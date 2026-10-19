"""Wavefront decoding inside vLLM: Dory's own decode step for a lone request.

When a single request decodes one token, the Dory model (with the config's
wavefront_decode on) runs this engine instead of its usual forward. Each vLLM
step stays an ordinary decode step: vLLM samples the next token from the
logits this step returns, with its own sampler, penalties and logprobs. The
engine keeps the request's next tokens in flight at different loop depths,
as in kernels/dory_wavefront.py, so that most steps only advance a pipeline.
In-flight row t holds the token at position q + t, where q is the position of
vLLM's token this step.

- Steady step (vLLM's token is the draft in flight for its position): the
  input block for the newest draft, one recurrent-block call that advances
  every in-flight token by one loop (vLLM's token through the last loop), and
  the output block and LM head for two rows: vLLM's token at full depth (this
  step's logits) and the newest token after loop 1 (the next draft).
- Catch-up step (anything else: the first decode step, a draft that vLLM did
  not sample, another request, any forward the engine did not run): vLLM's
  token goes through the input block and every loop while new drafts enter
  behind it one loop apart. This refills the pipeline and ends like a steady
  step.

Drafts reproduce vLLM's sampling: the argmax of the loop-1 logits after the
request's temperature and top-k/top-p, plus the Gumbel noise vLLM's sampler
draws for that position (vLLM's gumbel_sample, keyed by the request's seed,
the position and the token id). When the shallow and full-depth
distributions are close, both draws pick the same token. vLLM's sampled token
is always the one kept, so outputs follow the full-depth distribution
whatever the drafts are. (This holds for vLLM's Gumbel sampler, the default
with VLLM_USE_FLASHINFER_SAMPLER=0; another sampler only lowers how often
drafts match.)

KV: in-flight tokens write every cache at their own positions, up to loops - 1
positions past vLLM's token. The model's config asks vLLM's scheduler for those
slots (DoryConfig.kv_lookahead_tokens), which it reserves only while at most
one request runs, and the engine runs only when the request's allocated blocks
(vLLM's BlockTables.num_blocks) cover every in-flight position in every KV
cache group; past that, a block-table entry can name another request's block.
Every cache is vLLM's, read through each layer's own block table, because vLLM
can place one layer's loop caches in KV cache groups with different block
tables.

Each step reads two numbers back to the host (whether vLLM's token is the
draft, and its position) to choose between the steady and catch-up CUDA
graphs. vLLM's scheduling for the step is done before the forward starts, so
asynchronous scheduling keeps its overlap.
"""

import os

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample

from dory_common import EPS, to_residual
from dory_multi_token import MultiTokenSublayers
from dory_wavefront import Hidden

# Under vLLM's logger namespace, so its handlers write these lines to the server log.
logger = init_logger(f"vllm.{__name__}")

# Steps between progress lines in the server log.
LOG_EVERY = 2000
# "1" makes every step a catch-up step, which computes vLLM's token from
# scratch with the same kernels: the reference that steady steps must match
# token for token under greedy decoding.
FORCE_CATCH_UP = os.environ.get("DORY_WAVEFRONT_FORCE_CATCH_UP") == "1"


class _AttentionLayer:
    """One attention layer's vLLM caches: the key and value views of the loop-0
    cache, and per loop the element offset of that loop's cache from it and
    the index of its block table among the engine's copies."""

    def __init__(self, weights, rope: torch.Tensor, window: int, scale: float,
                 key: torch.Tensor, value: torch.Tensor, cache_offsets: list[int],
                 tables: list[int]):
        self.weights, self.rope, self.window, self.scale = weights, rope, window, scale
        self.key, self.value = key, value
        self.cache_offsets, self.tables = cache_offsets, tables
        # Per-token offsets for each pattern of loops, filled by the engine.
        self.token_cache_offsets: dict[tuple[int, ...], torch.Tensor] = {}
        self.token_table_offsets: dict[tuple[int, ...], torch.Tensor] = {}


class WavefrontEngine:
    def __init__(self, decoder, backbone, config, max_model_len: int, device: torch.device):
        """decoder: the model's FusedDoryDecoder (weights, single-token sublayers,
        LM head); backbone: vLLM's DoryBackbone (embeddings, attention layers)."""
        if decoder.lm_head is None:
            raise ValueError("wavefront decoding needs the fused 8-bit LM head")
        if config.n_recurrent_loops < 2:
            raise ValueError("wavefront decoding needs at least two recurrent loops")
        self.decoder, self.backbone, self.config = decoder, backbone, config
        self.max_model_len, self.device = max_model_len, device
        self.loops = L = config.n_recurrent_loops
        H = config.hidden_size
        first, recurrent = config.n_input_layers, config.n_recurrent_layers
        self.input_layers = list(range(first))
        self.recurrent_layers = list(range(first, first + recurrent))
        self.output_layers = list(range(first + recurrent, config.num_hidden_layers))
        self.multi = MultiTokenSublayers(
            H, config.intermediate_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, config.initializer_range,
            config.ngpt_alpha_init / config.initializer_range,
            config.attention_output_gate_logit_scale_init, device, max_tokens=L)

        # Per-row state of the in-flight tokens (row t at position q + t): the
        # loop input from the input block, the state after the loops so far, the
        # loop entry n(state + input), and the token itself.
        self.loop_input, self.state, self.entry = (Hidden(L, H, device) for _ in range(3))
        self.ping = [Hidden(L, H, device) for _ in range(2)]
        self.single = [Hidden(1, H, device) for _ in range(2)]
        self.readout = Hidden(2, H, device)
        self.shift_buffer = Hidden(L, H, device)
        i64 = dict(dtype=torch.int64, device=device)
        i32 = dict(dtype=torch.int32, device=device)
        self.tokens = torch.zeros(L, **i64)
        self.shift_tokens = torch.zeros(L, **i64)
        # The next token to enter the pipeline: vLLM's token at the start of a
        # catch-up step, otherwise the newest draft.
        self.pending = torch.zeros(1, **i64)
        # Graph inputs staged before each replay.
        self.in_token = torch.zeros(1, **i64)
        self.in_position = torch.zeros(1, **i64)
        self.req_index = torch.zeros(1, **i64)
        self.status = torch.zeros(2, **i64)
        self.row_offsets = torch.arange(L, **i64)
        self.row_positions = torch.zeros(L, **i64)
        self.row_seq_lens = torch.zeros(L, **i32)
        self.enter_position = torch.zeros(1, **i64)
        self.enter_seq_len = torch.zeros(1, **i32)
        self.readout_positions = torch.zeros(2, **i64)
        self.readout_seq_lens = torch.zeros(2, **i32)
        vocab = decoder.lm_head.vocab
        self.logits = torch.zeros(2, vocab, dtype=torch.float32, device=device)
        self.draft_logits = torch.zeros(1, vocab, dtype=torch.float32, device=device)
        self.commit_hidden = torch.zeros(1, H, dtype=torch.bfloat16, device=device)

        self.layers: dict[int, _AttentionLayer] | None = None  # bound at the first step
        self.disabled = False
        self.states = None  # the sampler's per-request sampling state
        self.allocated_blocks = None  # vLLM's BlockTables: blocks per group and request
        self.use_fp64_gumbel = False
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.pool = None

        self.step_request: tuple[str, int] | None = None
        self.valid = False
        self.request_id: str | None = None
        self.next_position = -1
        self.logits_ready = False
        self.steps = self.catch_ups = self.verified = self.matched = 0
        self.unallocated = 0  # single-request steps without the lookahead blocks

    # ---- hooks called by the model and vLLM's model runner ----

    def bind_sampler(self, sampler) -> None:
        self.states = sampler.sampling_states
        self.use_fp64_gumbel = sampler.use_fp64_gumbel

    def bind_block_tables(self, block_tables) -> None:
        self.allocated_blocks = block_tables

    def begin_step(self, req_ids: list[str], idx_mapping_np) -> None:
        """Called before every real forward with the batch's requests."""
        if len(req_ids) == 1:
            self.step_request = (req_ids[0], int(idx_mapping_np[0]))
        else:
            self.step_request = None
            self.invalidate()

    def invalidate(self) -> None:
        """A forward the engine did not run: the pipeline no longer matches the
        caches."""
        self.valid = False

    def take_logits(self) -> torch.Tensor | None:
        """This step's full-depth logits [1, V], once, after a step the engine ran."""
        if not self.logits_ready:
            return None
        self.logits_ready = False
        return self.logits[:1]

    def step(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor | None:
        """One decode step for vLLM's single token; returns the final hidden
        state [1, H], or None when the engine does not apply (the caller then
        runs the usual forward)."""
        request, self.step_request = self.step_request, None
        if (request is None or self.disabled or self.states is None
                or self.allocated_blocks is None):
            self.invalidate()
            return None
        if self.layers is None and not self._bind_caches():
            return None
        request_id, req_index = request
        context = get_forward_context().attn_metadata
        self.in_token.copy_(input_ids[:1])
        self.in_position.copy_(positions[:1])
        for g, name in enumerate(self.table_layers):
            table = context[name].block_table[0]
            self.block_tables[g, :table.shape[0]].copy_(table)
        self.req_index.fill_(req_index)
        self.status[0:1].copy_(self.in_token == self.tokens[:1])
        self.status[1:2].copy_(self.in_position)
        match, position = self.status.tolist()
        tables = self.allocated_blocks
        allocated = min(int(tables.num_blocks.np[g, req_index]) * size
                        for g, size in enumerate(tables.block_sizes))
        if position + self.loops > min(allocated, self.max_model_len):
            # Some in-flight position has no block of this request's own.
            self.unallocated += 1
            self.invalidate()
            return None
        continuing = (self.valid and request_id == self.request_id
                      and position == self.next_position)
        steady = continuing and bool(match) and not FORCE_CATCH_UP
        states = self.states
        sampling = (bool(states.top_k.np[req_index] != states.vocab_size),
                    bool(states.top_p.np[req_index] != 1.0))
        if steady:
            self._run(("steady",) + sampling, lambda: self._steady(*sampling))
        else:
            self._run(("catch_up",) + sampling, lambda: self._catch_up(*sampling))
        self.valid, self.request_id, self.next_position = True, request_id, position + 1
        self.logits_ready = True
        self.steps += 1
        self.catch_ups += not steady
        if continuing:
            self.verified += 1
            self.matched += steady
        if self.steps % LOG_EVERY == 0:
            logger.info("Dory wavefront: %d steps, %d catch-up steps, %.1f%% of %d drafts "
                        "sampled by vLLM, %d steps left to the fused path for want of "
                        "lookahead blocks", self.steps, self.catch_ups,
                        100 * self.matched / max(self.verified, 1), self.verified,
                        self.unallocated)
        return self.commit_hidden

    # ---- setup ----

    def _bind_caches(self) -> bool:
        """Finds every attention layer's vLLM caches and block tables (allocated
        after the model loads); False, and the engine stays off, when a layer
        does not use the BF16 Triton-backend cache the kernels read."""
        config, L = self.config, self.loops
        tables: dict[tuple[int, int], int] = {}
        self.table_layers: list[str] = []
        layers, width, block_size = {}, None, None
        for index, layer_type in enumerate(config.layer_types):
            if layer_type == "mlp":
                continue
            mixer = self.backbone.layers[index].mixer
            loops = L if index in self.recurrent_layers else 1
            caches, cache_tables = [], []
            for loop in range(loops):
                attention = mixer.attn[loop if mixer.per_loop_kv else 0]
                metadata, layer, kv_cache, _ = get_attention_context(attention.layer_name)
                if (metadata is None or type(layer.impl).__name__ != "TritonAttentionImpl"
                        or kv_cache.dtype != torch.bfloat16):
                    logger.warning("Dory wavefront decoding is off: layer %s does not use "
                                   "a BF16 Triton attention cache", attention.layer_name)
                    self.disabled = True
                    return False
                table = metadata.block_table
                key = (table.data_ptr(), table.shape[1])
                if key not in tables:
                    tables[key] = len(tables)
                    self.table_layers.append(attention.layer_name)
                cache_tables.append(tables[key])
                caches.append(kv_cache)
                width = table.shape[1] if width is None else width
                if table.shape[1] != width:
                    raise ValueError("block tables of different widths")
            # [blocks, kv_heads, block_size, 2 * head_dim] seen as key and value
            # [blocks, block_size, kv_heads, head_dim], as the fused decoder does.
            key_cache, value_cache = caches[0].transpose(1, 2).split(config.head_dim, dim=-1)
            block_size = key_cache.shape[1] if block_size is None else block_size
            offsets = []
            for cache in caches:
                delta = cache.data_ptr() - caches[0].data_ptr()
                if (cache.stride() != caches[0].stride() or cache.shape != caches[0].shape
                        or delta % cache.element_size()):
                    raise ValueError("a layer's loop caches differ in layout")
                offsets.append(delta // cache.element_size())
            impl = layer.impl
            window = config.sliding_window if layer_type == "sliding_attention" else 0
            if window and impl.sliding_window[0] + 1 != window:
                raise ValueError("vLLM's sliding window differs from the config's")
            layers[index] = _AttentionLayer(
                self.decoder.attention_weights[index], mixer.rotary_emb.cos_sin_cache, window,
                impl.scale, key_cache, value_cache, offsets, cache_tables)
        self.block_size, self.table_width = block_size, width
        self.block_tables = torch.zeros(len(tables), width, dtype=torch.int32,
                                        device=self.device)
        i64 = dict(dtype=torch.int64, device=self.device)
        # Per-token offsets for every loop pattern a call uses: in-flight row t of
        # a k-row recurrent call is at loop k - 1 - t; readouts use loop 0.
        patterns = [tuple(k - 1 - t for t in range(k)) for k in range(1, L + 1)]
        patterns += [(0,), (0, 0)]
        for layer in layers.values():
            for pattern in patterns:
                if max(pattern) >= len(layer.cache_offsets):
                    continue
                layer.token_cache_offsets[pattern] = torch.tensor(
                    [layer.cache_offsets[loop] for loop in pattern], **i64)
                layer.token_table_offsets[pattern] = torch.tensor(
                    [layer.tables[loop] * width for loop in pattern], **i64)
        self.layers = layers
        self.pool = torch.cuda.graph_pool_handle()
        logger.info("Dory wavefront decoding: %d loops, %d block tables, block size %d",
                    L, len(tables), block_size)
        return True

    # ---- the CUDA graph bodies ----

    def _run(self, key: tuple, fn) -> None:
        """Runs fn as a CUDA graph; the first call for a key runs fn eagerly (the
        real step, which also compiles its kernels) and then captures it."""
        graph = self.graphs.get(key)
        if graph is not None:
            graph.replay()
            return
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
            fn()
        self.graphs[key] = graph

    def _steady(self, top_k: bool, top_p: bool) -> None:
        L = self.loops
        self._set_positions()
        self._enter(L - 1)
        self._recurrent(L)
        self._readout(L, commit=True, top_k=top_k, top_p=top_p)
        self._shift()

    def _catch_up(self, top_k: bool, top_p: bool) -> None:
        L = self.loops
        self._set_positions()
        self.pending.copy_(self.in_token)
        for k in range(1, L + 1):
            self._enter(k - 1)
            self._recurrent(k)
            self._readout(k, commit=k == L, top_k=top_k, top_p=top_p)
        self._shift()

    def _set_positions(self) -> None:
        torch.add(self.row_offsets, self.in_position, out=self.row_positions)
        self.row_seq_lens.copy_(self.row_positions + 1)

    def _enter(self, row: int) -> None:
        """Embedding and input block for self.pending at position q + row; it
        becomes in-flight row `row`, before its first loop."""
        position = self.enter_position
        torch.add(self.in_position, row, out=position)
        self.enter_seq_len.copy_(position + 1)
        physical = self.block_tables[:, torch.div(position, self.block_size,
                                                   rounding_mode="floor")]
        slots = physical.view(-1).to(torch.int64) * self.block_size + position % self.block_size
        decoder, fused = self.decoder, self.decoder.attention
        x = self.single[0]
        to_residual(self.backbone.embeddings(self.pending)[0], None, x.bf16[0], x.nm[0],
                    normalize=False)
        for step, index in enumerate(self.input_layers):
            out = self.single[(step + 1) % 2]
            if index in decoder.mlp_weights:
                decoder.mlp(decoder.mlp_weights[index], x.bf16[0], x.nm[0], out.bf16[0],
                            out.nm[0])
            else:
                layer = self.layers[index]
                g = layer.tables[0]
                fused.project_qkvg(layer.weights, x.nm[0], position, layer.rope)
                decoder.decode_attention.from_raw(
                    fused.qk, fused.v, layer.weights.sqk, fused.sqk_factor, EPS, slots[g:g + 1],
                    layer.key, layer.value, self.block_tables[g], self.enter_seq_len,
                    layer.window, layer.scale, fused.gate, decoder.gated_attention)
                fused.post_attention(layer.weights, x.bf16[0], decoder.gated_attention,
                                     out.bf16[0], out.nm[0], gated_nibble_major=True)
            x = out
        to_residual(x.bf16[0], None, self.loop_input.bf16[row], self.loop_input.nm[row],
                    normalize=False)
        self.state.copy_row(row, self.loop_input, row)
        self.tokens[row:row + 1].copy_(self.pending)

    def _multi_block(self, layers: list[int], x: Hidden, positions: torch.Tensor,
                     seq_lens: torch.Tensor, pattern: tuple[int, ...]) -> Hidden:
        tokens = len(pattern)
        for step, index in enumerate(layers):
            out = self.ping[step % 2].rows(0, tokens)
            if index in self.decoder.mlp_weights:
                self.multi.mlp(self.decoder.mlp_weights[index], x.bf16, x.nm, out.bf16, out.nm)
            else:
                layer = self.layers[index]
                self.multi.attention(
                    layer.weights, x.bf16, x.nm, positions, layer.rope,
                    layer.token_cache_offsets[pattern], layer.key, layer.value,
                    self.block_tables, layer.token_table_offsets[pattern], seq_lens,
                    layer.window, layer.scale, out.bf16, out.nm)
            x = out
        return x

    def _recurrent(self, k: int) -> None:
        """One recurrent-block call for rows 0 .. k-1; row t runs loop k - 1 - t."""
        for t in range(k):
            to_residual(self.state.bf16[t], self.loop_input.bf16[t], self.entry.bf16[t],
                        self.entry.nm[t], normalize=True)
        h = self._multi_block(self.recurrent_layers, self.entry.rows(0, k),
                              self.row_positions[:k], self.row_seq_lens[:k],
                              tuple(k - 1 - t for t in range(k)))
        for t in range(k):
            to_residual(h.bf16[t], None, self.state.bf16[t], self.state.nm[t], normalize=True)

    def _readout(self, k: int, commit: bool, top_k: bool, top_p: bool) -> None:
        """Output block and LM head for row 0 at full depth (when commit) and for
        row k - 1 after loop 1, whose logits draft the next token."""
        rows = ([0] if commit else []) + [k - 1]
        n = len(rows)
        for r, row in enumerate(rows):
            self.readout.copy_row(r, self.state, row)
            self.readout_positions[r:r + 1].copy_(self.row_positions[row:row + 1])
        self.readout_seq_lens[:n].copy_(self.readout_positions[:n] + 1)
        h = self._multi_block(self.output_layers, self.readout.rows(0, n),
                              self.readout_positions[:n], self.readout_seq_lens[:n], (0,) * n)
        self.decoder.lm_head.rows(h.bf16, self.logits[:n])
        if commit:
            self.commit_hidden.copy_(h.bf16[:1])
        self._draft(self.logits[n - 1:n], self.readout_positions[n - 1:n], top_k, top_p)

    def _draft(self, logits: torch.Tensor, position: torch.Tensor, top_k: bool,
               top_p: bool) -> None:
        """The draft for position + 1, drawn as vLLM's sampler would draw from these
        logits: temperature, top-k/top-p, then Gumbel noise for (seed, position)."""
        states, draft = self.states, self.draft_logits
        draft.copy_(logits)
        apply_temperature(draft, self.req_index, states.temperature.gpu)
        if top_k or top_p:
            draft = apply_top_k_top_p(
                draft, states.top_k.gpu[self.req_index] if top_k else None,
                states.top_p.gpu[self.req_index] if top_p else None)
        token = gumbel_sample(draft, self.req_index, states.temperature.gpu, states.seeds.gpu,
                              position, apply_temperature=False, is_drafting=False,
                              use_fp64=self.use_fp64_gumbel)
        self.pending.copy_(token.view(1))

    def _shift(self) -> None:
        """Drops row 0 (vLLM's token, now complete): rows 1 .. L-1 move up one."""
        L, tmp = self.loops, self.shift_buffer
        for buffer in (self.state, self.loop_input):
            tmp.bf16[:L - 1].copy_(buffer.bf16[1:])
            tmp.nm[:L - 1].copy_(buffer.nm[1:])
            buffer.bf16[:L - 1].copy_(tmp.bf16[:L - 1])
            buffer.nm[:L - 1].copy_(tmp.nm[:L - 1])
        self.shift_tokens[:L - 1].copy_(self.tokens[1:])
        self.tokens[:L - 1].copy_(self.shift_tokens[:L - 1])
