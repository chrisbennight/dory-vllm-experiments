"""Stand-alone Dory decoder for evaluating wavefront self-speculative decoding.

Holds the whole model on the GPU (the fused sublayers' weight layouts, the
embeddings, the INT8 LM head) and its own KV caches, outside vLLM. One cache
per attention layer of the input and output blocks, and one per loop for each
recurrent attention layer. A recurrent layer's loop caches are consecutive,
so a token picks its loop's cache with an offset from the loop-0 cache.

Three ways to decode one sequence, greedily or by sampling:

- sequential, single-token kernels ("A"): the fused decode path, one token
  through every block, as in vLLM with DORY_FUSED_DECODE; the speed baseline;
- sequential, multi-token kernels ("B"): the same order, with the recurrent and
  output blocks on MultiTokenSublayers for one token; the exactness reference;
- wavefront: tokens at different loop depths in flight together. Each step
  runs the input block for the newest token, one recurrent-block call for
  every token in flight (each in its own loop), and the output block and head
  for the token that just finished loop 1 (its prediction drafts the next
  token) and the one that finished the last loop (its prediction verifies the
  next token's draft). On a mismatch the full-depth token is committed and
  every younger token in flight is discarded.

A commit only reads cache entries of committed or older in-flight positions,
computed by the same per-token kernels as in B, so the wavefront's committed
tokens equal B's. The draft token can read the output-block entry that the
committing token writes in the same launch, before or after the write; that
changes only how good a draft is, never what is committed.

With sampling (temperature > 0, optional top-p), drafts are sampled from the
shallow distribution and verified with speculative_accept against the
full-depth distribution, so committed tokens follow the same distribution as
sequential sampling (not the same tokens: the random draws differ).
"""

import json
from dataclasses import dataclass, field

import torch
from safetensors import safe_open

from dory_attention import FusedAttentionSublayer, load_attention_weights
from dory_common import EPS, LaunchConfig, to_residual
from dory_decode_attention import DecodeAttention
from dory_lm_head import QuantizedLmHead
from dory_mlp import FusedMlpSublayer, load_mlp_weights
from dory_multi_token import MultiTokenSublayers
from dory_sampling import filtered_probs, sample, speculative_accept

CACHE_BLOCK = 16
# The single-token launch configurations of kernels/dory_fused_model.py.
QKVG = LaunchConfig(block_n=4, block_w=64, num_warps=4)
ATTENTION_OUT = LaunchConfig(block_n=4, block_w=128, num_warps=4)
ATTENTION_OUT_NIBBLE_MAJOR = LaunchConfig(block_n=4, block_w=64, num_warps=4)
GATE_UP = LaunchConfig(block_n=4, block_w=64, num_warps=4)
DOWN_FOLDED = LaunchConfig(block_n=4, block_w=64, num_warps=4)


class Hidden:
    """Hidden states as BF16 [rows, H] plus nibble-major FP32 [rows, 8, H/8]."""

    def __init__(self, rows: int, hidden_size: int, device: torch.device):
        self.bf16 = torch.empty(rows, hidden_size, dtype=torch.bfloat16, device=device)
        self.nm = torch.empty(rows, 8, hidden_size // 8, dtype=torch.float32, device=device)

    def rows(self, start: int, stop: int) -> "Hidden":
        view = Hidden.__new__(Hidden)
        view.bf16, view.nm = self.bf16[start:stop], self.nm[start:stop]
        return view

    def copy_row(self, row: int, source: "Hidden", source_row: int) -> None:
        self.bf16[row].copy_(source.bf16[source_row])
        self.nm[row].copy_(source.nm[source_row])


@dataclass
class InFlight:
    position: int
    token: int
    depth: int = 0  # loops completed


@dataclass
class WavefrontStats:
    steps: int = 0
    commits: int = 0
    accepted: int = 0
    rejected: int = 0
    tokens_per_step: list = field(default_factory=list)


class DoryPrototype:
    def __init__(self, checkpoint, loops: int, max_len: int, device: torch.device,
                 max_tokens: int = 4):
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.rotary_embedding import get_rope

        config = json.loads((checkpoint / "config.json").read_text())
        self.config, self.loops, self.device = config, loops, device
        H, F = config["hidden_size"], config["intermediate_size"]
        NH, NKV, HD = (config["num_attention_heads"], config["num_key_value_heads"],
                       config["head_dim"])
        self.H, self.HD = H, HD
        std = config["initializer_range"]
        alpha_scale = config["ngpt_alpha_init"] / std
        gate_init = config["attention_output_gate_logit_scale_init"]
        n_in, n_rec = config["n_input_layers"], config["n_recurrent_layers"]
        types = config["layer_types"]
        self.input_layers = list(range(n_in))
        self.recurrent_layers = list(range(n_in, n_in + n_rec))
        self.output_layers = list(range(n_in + n_rec, config["num_hidden_layers"]))
        self.window = {i: config["sliding_window"] if t == "sliding_attention" else 0
                       for i, t in enumerate(types) if t != "mlp"}
        self.scale = HD ** -0.5

        with set_current_vllm_config(VllmConfig()):
            ropes = {kind: get_rope(HD, max_position=config["max_position_embeddings"],
                                    is_neox_style=True,
                                    rope_parameters=config["rope_parameters"][kind],
                                    dtype=torch.bfloat16).to(device).cos_sin_cache
                     for kind in ("sliding_attention", "full_attention")}
        self.rope = {i: ropes[t] for i, t in enumerate(types) if t != "mlp"}

        self.attention, self.mlp = {}, {}
        with safe_open(str(checkpoint / "model.safetensors"), framework="pt",
                       device=str(device)) as f:
            for i, t in enumerate(types):
                if t == "mlp":
                    self.mlp[i] = load_mlp_weights(f, i)
                else:
                    self.attention[i] = load_attention_weights(f, i, NH, NKV, HD)
            self.embeddings = f.get_tensor("backbone.embeddings.weight")
            lm_head = f.get_tensor("lm_head.weight")
            self.head = QuantizedLmHead(lm_head, f.get_tensor("lm_head.logit_scale"), std)
            del lm_head
        self.vocab = self.head.vocab

        # KV caches: one per attention layer, and one per loop for recurrent layers.
        self.cache_id = {}
        for i in self.input_layers + self.output_layers:
            if i in self.attention:
                self.cache_id[(i, 0)] = len(self.cache_id)
        for i in self.recurrent_layers:
            if i in self.attention:
                for loop in range(loops):
                    self.cache_id[(i, loop)] = len(self.cache_id)
        blocks = max_len // CACHE_BLOCK + 1
        self.max_len = blocks * CACHE_BLOCK - CACHE_BLOCK
        self.kv = torch.zeros(len(self.cache_id), blocks, CACHE_BLOCK, NKV, 2 * HD,
                              dtype=torch.bfloat16, device=device)
        self.cache_numel = self.kv[0].numel()
        self.block_table = torch.arange(blocks, dtype=torch.int32, device=device)

        self.fused_attention = FusedAttentionSublayer(
            H, NH, NKV, HD, std, alpha_scale, gate_init, device, QKVG, ATTENTION_OUT,
            o_nibble_major=ATTENTION_OUT_NIBBLE_MAJOR)
        self.decode_attention = DecodeAttention(NH, NKV, HD, device)
        self.fused_mlp = FusedMlpSublayer(H, F, alpha_scale, device, GATE_UP, DOWN_FOLDED)
        self.multi = MultiTokenSublayers(H, F, NH, NKV, HD, std, alpha_scale, gate_init, device,
                                         max_tokens=max_tokens)
        self.gated = torch.empty(8, NH * HD // 8, dtype=torch.float32, device=device)
        self.max_tokens = max_tokens
        T = max_tokens

        # Per-block ping-pong buffers and the wavefront's per-token state.
        self.ping = [Hidden(T, H, device) for _ in range(2)]
        self.state, self.loop_input, self.entry = (Hidden(T, H, device) for _ in range(3))
        self.p_out = Hidden(1, H, device)
        self.readout = Hidden(T, H, device)
        self.logits = torch.empty(T, self.vocab, dtype=torch.float32, device=device)
        self.next_tokens = torch.empty(T, dtype=torch.int64, device=device)
        # Sampling readout: the shallow distribution each in-flight token drafted
        # its successor from, in slot position % (T + 1) (the positions in flight
        # are consecutive, so slots never collide), whether the verified draft was
        # accepted, and the readout's inputs: the committing token's slot, the
        # drafting token's slot, the successor's draft token, and whether there
        # is a successor (int64 [4]).
        self.draft_slots = T + 1
        self.draft_probs = torch.zeros(T + 1, self.vocab, dtype=torch.float32, device=device)
        self.accepted = torch.zeros(1, dtype=torch.bool, device=device)
        self.readout_args = torch.zeros(4, dtype=torch.int64, device=device)
        i64 = dict(dtype=torch.int64, device=device)
        # Inputs of the graphs: token and position of a single token, and the
        # per-row positions, slots, sequence lengths and cache offsets of the
        # recurrent and output blocks.
        self.token = torch.zeros(1, **i64)
        self.position = torch.zeros(1, **i64)
        self.seq_len = torch.zeros(1, dtype=torch.int32, device=device)
        self.r_position = torch.zeros(T, **i64)
        self.r_seq_len = torch.zeros(T, dtype=torch.int32, device=device)
        self.r_offset = torch.zeros(T, **i64)
        self.c_position = torch.zeros(T, **i64)
        self.c_seq_len = torch.zeros(T, dtype=torch.int32, device=device)
        self.c_offset = torch.zeros(T, **i64)  # the output block's single cache
        self.loop_offsets = [torch.full((1,), loop * self.cache_numel, **i64)
                             for loop in range(loops)]
        self.same_table = torch.zeros(T, **i64)  # block-table offsets: all rows share one
        self.sequential_tokens = torch.zeros(max_len, **i64)
        self.sequential_count = torch.zeros(1, **i64)
        self.graphs = {}

    # ---- sublayers and blocks ----

    def _kv(self, cache: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv[cache].split(self.HD, dim=-1)

    def _single_block(self, layers, loop: int, x: Hidden) -> Hidden:
        """One token through `layers` on the single-token kernels; x has one row."""
        for step, i in enumerate(layers):
            out = self.ping[step % 2].rows(0, 1)
            if i in self.mlp:
                self.fused_mlp(self.mlp[i], x.bf16[0], x.nm[0], out.bf16[0], out.nm[0])
            else:
                w, fused = self.attention[i], self.fused_attention
                key, value = self._kv(self.cache_id[(i, loop)])
                fused.project_qkvg(w, x.nm[0], self.position, self.rope[i])
                self.decode_attention.from_raw(
                    fused.qk, fused.v, w.sqk, fused.sqk_factor, EPS, self.position, key, value,
                    self.block_table, self.seq_len, self.window[i], self.scale, fused.gate,
                    self.gated)
                fused.post_attention(w, x.bf16[0], self.gated, out.bf16[0], out.nm[0],
                                     gated_nibble_major=True)
            x = out
        return x

    def _multi_block(self, layers, x: Hidden, positions, seq_lens, offsets) -> Hidden:
        """T = rows of x tokens through `layers` on the multi-token kernels. Token t
        uses the cache at offset offsets[t] from each layer's loop-0 cache; every
        token uses the one identity block table, so a position is its own slot."""
        tokens = x.bf16.shape[0]
        for step, i in enumerate(layers):
            out = self.ping[step % 2].rows(0, tokens)
            if i in self.mlp:
                self.multi.mlp(self.mlp[i], x.bf16, x.nm, out.bf16, out.nm)
            else:
                key, value = self._kv(self.cache_id[(i, 0)])
                self.multi.attention(self.attention[i], x.bf16, x.nm, positions, self.rope[i],
                                     offsets, key, value, self.block_table,
                                     self.same_table[:tokens], seq_lens, self.window[i],
                                     self.scale, out.bf16, out.nm)
            x = out
        return x

    def _input_block(self) -> Hidden:
        """Embedding and input block for self.token at self.position; returns the
        loop input in self.p_out."""
        self.seq_len.copy_(self.position + 1)
        x = self.ping[1].rows(0, 1)
        embedded = self.embeddings.index_select(0, self.token)
        to_residual(embedded[0], None, x.bf16[0], x.nm[0], normalize=False)
        x = self._single_block(self.input_layers, 0, x)
        to_residual(x.bf16[0], None, self.p_out.bf16[0], self.p_out.nm[0], normalize=False)
        return self.p_out

    # ---- sequential decoding ----

    def _sequential_step(self, multi: bool, temperature: float, top_p: float) -> None:
        e = self._input_block()
        state, entry = self.state.rows(0, 1), self.entry.rows(0, 1)
        state.copy_row(0, e, 0)
        for loop in range(self.loops):
            to_residual(state.bf16[0], e.bf16[0], entry.bf16[0], entry.nm[0], normalize=True)
            if multi:
                h = self._multi_block(self.recurrent_layers, entry, self.position, self.seq_len,
                                      self.loop_offsets[loop])
            else:
                h = self._single_block(self.recurrent_layers, loop, entry)
            to_residual(h.bf16[0], None, state.bf16[0], state.nm[0], normalize=True)
        if multi:
            h = self._multi_block(self.output_layers, state, self.position, self.seq_len,
                                  self.c_offset[:1])
        else:
            h = self._single_block(self.output_layers, 0, state)
        self.head.rows(h.bf16, self.logits[:1])
        if temperature > 0:
            self.token.copy_(sample(filtered_probs(self.logits[:1], temperature, top_p)))
        else:
            torch.argmax(self.logits[:1], dim=-1, out=self.token)
        self.sequential_tokens.index_copy_(0, self.sequential_count, self.token)
        self.sequential_count.add_(1)
        self.position.add_(1)

    def _run(self, key, fn) -> None:
        """Runs fn as a CUDA graph. The first call for a key runs fn eagerly (that
        run is the real step, and compiles its kernels) and then captures it;
        capturing records without executing, so the step runs exactly once."""
        graph = self.graphs.get(key)
        if graph is not None:
            graph.replay()
            return
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        self.graphs[key] = graph

    def prefill(self, prompt: list[int], multi: bool, temperature: float = 0.0,
                top_p: float = 1.0) -> int:
        """Runs the prompt through sequential steps; returns the first generated token."""
        self.sequential_count.zero_()
        key = ("sequential", multi, temperature, top_p)
        for position, token in enumerate(prompt):
            self.token.fill_(token)
            self.position.fill_(position)
            self._run(key, lambda: self._sequential_step(multi, temperature, top_p))
        return int(self.token.item())

    def generate_sequential(self, prompt: list[int], count: int, multi: bool,
                            temperature: float = 0.0, top_p: float = 1.0
                            ) -> tuple[list[int], float]:
        """Decoding of `count` tokens after the prompt (greedy at temperature 0);
        returns the tokens (the first comes from the prompt's last step) and the
        decode seconds."""
        first = self.prefill(prompt, multi, temperature, top_p)
        graph = self.graphs[("sequential", multi, temperature, top_p)]
        self.sequential_count.zero_()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(count - 1):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        tokens = [first] + self.sequential_tokens[:count - 1].tolist()
        return tokens, start.elapsed_time(end) / 1000

    # ---- wavefront decoding ----

    def _recurrent_step(self, tokens: int) -> None:
        state, e, entry = (b.rows(0, tokens) for b in (self.state, self.loop_input, self.entry))
        for t in range(tokens):
            to_residual(state.bf16[t], e.bf16[t], entry.bf16[t], entry.nm[t], normalize=True)
        h = self._multi_block(self.recurrent_layers, entry, self.r_position[:tokens],
                              self.r_seq_len[:tokens], self.r_offset[:tokens])
        for t in range(tokens):
            to_residual(h.bf16[t], None, state.bf16[t], state.nm[t], normalize=True)

    def _readout_step(self, roles: tuple[str, ...], temperature: float, top_p: float) -> None:
        """Output block and head for the readout rows ("commit" first, then "draft"),
        and their tokens in self.next_tokens: argmax when greedy; with sampling,
        speculative acceptance of the successor's draft for the commit row and a
        new draft (whose distribution is kept) for the draft row."""
        tokens = len(roles)
        h = self._multi_block(self.output_layers, self.readout.rows(0, tokens),
                              self.c_position[:tokens], self.c_seq_len[:tokens],
                              self.c_offset[:tokens])
        self.head.rows(h.bf16, self.logits[:tokens])
        if temperature == 0:
            torch.argmax(self.logits[:tokens], dim=-1, out=self.next_tokens[:tokens])
            return
        probs = filtered_probs(self.logits[:tokens], temperature, top_p)
        commit_slot, draft_slot = self.readout_args[0:1], self.readout_args[1:2]
        successor, has_successor = self.readout_args[2:3], self.readout_args[3:4].bool()
        for r, role in enumerate(roles):
            if role == "commit":
                token, accepted = speculative_accept(
                    probs[r:r + 1], self.draft_probs.index_select(0, commit_slot), successor,
                    valid=has_successor)
                self.next_tokens[r:r + 1].copy_(token)
                self.accepted.copy_(accepted)
            else:
                self.draft_probs.index_copy_(0, draft_slot, probs[r:r + 1])
                self.next_tokens[r:r + 1].copy_(sample(probs[r:r + 1]))

    def generate_wavefront(self, prompt: list[int], count: int, temperature: float = 0.0,
                           top_p: float = 1.0, seed: int = 0
                           ) -> tuple[list[int], float, WavefrontStats]:
        """Wavefront decoding of `count` tokens after the prompt (drafting at loop 1,
        greedy at temperature 0); returns the committed tokens, the decode seconds
        and statistics."""
        loops = self.loops
        if loops > self.max_tokens:
            raise ValueError("the wavefront needs one row per loop")
        sampling = temperature > 0
        if sampling:
            torch.manual_seed(seed)  # the graphs draw from the default generator
        committed = [self.prefill(prompt, True, temperature, top_p)]
        flight: list[InFlight] = []
        pending = (committed[0], len(prompt))
        stats = WavefrontStats()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        while len(committed) < count:
            stats.steps += 1
            # Input block for the newest token; it starts at loop depth 0.
            if pending is not None:
                token, position = pending
                if position >= self.max_len:
                    raise ValueError("sequence longer than the caches")
                self.token.fill_(token)
                self.position.fill_(position)
                self._run("input", self._input_block)
                row = len(flight)
                self.loop_input.copy_row(row, self.p_out, 0)
                self.state.copy_row(row, self.p_out, 0)
                flight.append(InFlight(position, token))
                pending = None
            # One recurrent-block call for every token in flight.
            tokens = len(flight)
            positions = [f.position for f in flight]
            self.r_position[:tokens].copy_(torch.tensor(positions), non_blocking=True)
            self.r_seq_len[:tokens].copy_(torch.tensor([p + 1 for p in positions],
                                                       dtype=torch.int32), non_blocking=True)
            self.r_offset[:tokens].copy_(torch.tensor([f.depth * self.cache_numel
                                                       for f in flight]), non_blocking=True)
            self._run(("recurrent", tokens), lambda: self._recurrent_step(tokens))
            for f in flight:
                f.depth += 1
            # Readout: the token that finished the last loop commits, the one
            # that finished loop 1 drafts.
            readout = []
            if flight[0].depth == loops:
                readout.append(("commit", 0))
            if flight[-1].depth == 1 and not (readout and tokens == 1):
                readout.append(("draft", tokens - 1))
            for r, (_, row) in enumerate(readout):
                self.readout.copy_row(r, self.state, row)
            rows = len(readout)
            if rows == 0:
                raise AssertionError("every step drafts or commits")
            readout_positions = [flight[row].position for _, row in readout]
            self.c_position[:rows].copy_(torch.tensor(readout_positions), non_blocking=True)
            self.c_seq_len[:rows].copy_(torch.tensor([p + 1 for p in readout_positions],
                                                     dtype=torch.int32), non_blocking=True)
            roles = tuple(kind for kind, _ in readout)
            if sampling:
                args = [0, 0, 0, 0]
                if roles[0] == "commit":
                    args[0] = flight[0].position % self.draft_slots
                    if len(flight) > 1:
                        args[2], args[3] = flight[1].token, 1
                if roles[-1] == "draft":
                    args[1] = flight[-1].position % self.draft_slots
                self.readout_args.copy_(torch.tensor(args), non_blocking=True)
            self._run(("readout", roles, temperature, top_p),
                      lambda: self._readout_step(roles, temperature, top_p))
            values = torch.cat([self.next_tokens[:rows], self.accepted.long()]).tolist()
            predicted = values[:rows]
            if sampling:
                accepted = roles[0] == "commit" and bool(values[-1])
            else:
                accepted = len(flight) > 1 and roles[0] == "commit" and (
                    flight[1].token == predicted[0])
            draft = None
            for (kind, row), token in zip(readout, predicted):
                if kind == "draft":
                    draft = (token, flight[row].position + 1)
            if readout and readout[0][0] == "commit":
                token = predicted[0]
                stats.commits += 1
                committed.append(token)
                if accepted:
                    stats.accepted += 1
                else:
                    if len(flight) > 1:
                        stats.rejected += 1
                    flight = flight[:1]
                    draft = (token, flight[0].position + 1)
                flight.pop(0)
                # Keep the remaining tokens in rows 0 .. n-1, oldest first.
                for buffer in (self.state, self.loop_input):
                    if flight:
                        buffer.bf16[:len(flight)].copy_(buffer.bf16[1:len(flight) + 1].clone())
                        buffer.nm[:len(flight)].copy_(buffer.nm[1:len(flight) + 1].clone())
            pending = draft
            stats.tokens_per_step.append(tokens)
        end.record()
        torch.cuda.synchronize()
        return committed[:count], start.elapsed_time(end) / 1000, stats
