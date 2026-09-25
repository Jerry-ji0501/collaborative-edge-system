"""Collaborative inference engine for LLaMA-family models.

Every rank of the ``pp x sp x tp`` mesh constructs a :class:`CollabEngine`
(SPMD) and calls the same methods; inputs are taken from global rank 0 and
results are returned on every rank.

Execution model
---------------
* **Prefill** – the prompt is split over the SP group (ring / Ulysses
  attention); inside each SP rank the layers are tensor-parallel; stage
  outputs travel to the next pipeline stage through a :class:`P2PChannel`.
* **Decode** – the new token is replicated on the SP ranks, which attend to
  their part of the distributed KV cache and merge exactly; TP and PP work as
  in prefill.
* **Pipelining** – prompts are split into micro-batches.  The last stage
  samples, and tokens loop back to the first stage, which immediately feeds
  them into the next step while other micro-batches occupy later stages, so
  all stages stay busy (with ``num_microbatches >= pp_size``).  Finished
  micro-batches are retired with an in-band ``STOP`` message.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from ..config import ParallelConfig
from ..distributed.comm import CommStats
from ..distributed.mesh import ParallelContext
from ..models.cache import ForwardState, KVCache
from ..models.config import ModelConfig
from ..models.llama import LlamaStage
from ..models.weights import WeightSource, open_weights
from ..parallel.partition import SequenceLayout, offsets_of, split_sizes
from ..parallel.pipeline_parallel import P2PChannel, partition_layers
from ..parallel.sequence_parallel import gather_sequence
from .sampling import Sampler, SamplingParams

Prompts = Sequence[Sequence[int]]

# smallest automatically chosen prefill chunk (smaller chunks make GEMMs inefficient)
MIN_AUTO_CHUNK = 128


@dataclass
class GenerationResult:
    """Output of :meth:`CollabEngine.generate` (identical on every rank).

    Attributes:
        tokens: generated token ids per prompt (prompt not included; stops
            after the first ``eos_token_id``).
        scores: per prompt, the ``[num_generated, vocab]`` logits of every
            step.  Only on last-stage ranks and only if requested.
        stats: timing information measured on rank 0.
    """

    tokens: List[List[int]]
    scores: Optional[List[torch.Tensor]] = None
    stats: Dict[str, float] = field(default_factory=dict)


class _MicroBatch:
    """Per micro-batch generation state (every stage keeps its own copy).

    Steps ``0 .. num_chunks - 1`` feed the (left-padded) prompt chunk by chunk;
    the last chunk yields the first generated token and every later step feeds
    one generated token.
    """

    def __init__(
        self,
        prompt_ids: List[int],
        prompts: Prompts,
        device: torch.device,
        reserve_tokens: int = 0,
        chunk_size: int = 0,
    ) -> None:
        self.prompt_ids = prompt_ids
        self.seq_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)  # sampling noise keys
        B, S = len(prompts), max(len(p) for p in prompts)
        self.ids = torch.zeros(B, S, dtype=torch.long)
        self.positions = torch.full((B, S), -1, dtype=torch.long)
        for b, p in enumerate(prompts):  # left padding
            self.ids[b, S - len(p) :] = torch.tensor(list(p), dtype=torch.long)
            self.positions[b, S - len(p) :] = torch.arange(len(p))
        self.ids, self.positions = self.ids.to(device), self.positions.to(device)
        if chunk_size and S > chunk_size:
            self.chunks = [(a, min(a + chunk_size, S)) for a in range(0, S, chunk_size)]
        else:
            self.chunks = [(0, S)]
        self.num_chunks = len(self.chunks)
        self.next_pos = torch.tensor([len(p) for p in prompts], dtype=torch.long, device=device)
        # the first chunk allocates room for the whole prompt and the generated tokens
        self.cache = KVCache(reserve_tokens + S - self.chunks[0][1])
        self.step = 0
        self.generated: List[torch.Tensor] = []  # first stage
        self.done_first = torch.zeros(B, dtype=torch.bool, device=device)
        self.done_last = torch.zeros(B, dtype=torch.bool, device=device)
        self.pending: Optional[torch.Tensor] = None  # tokens when pp == 1
        self.scores: List[torch.Tensor] = []
        self.first_token_time: Optional[float] = None

    @property
    def in_prefill(self) -> bool:
        return self.step < self.num_chunks

    @property
    def emits_token(self) -> bool:
        """Whether the last stage samples at this step (not for non-final chunks)."""
        return self.step >= self.num_chunks - 1

    @property
    def gen_index(self) -> int:
        """Index of the token sampled at this step (sampling noise key)."""
        return self.step - (self.num_chunks - 1)

    def step_positions(self) -> torch.Tensor:
        if self.in_prefill:
            a, b = self.chunks[self.step]
            return self.positions[:, a:b]
        return self.next_pos.unsqueeze(1)

    def step_ids(self) -> torch.Tensor:
        if self.in_prefill:
            a, b = self.chunks[self.step]
            return self.ids[:, a:b]
        return self.generated[-1].unsqueeze(1)

    def advance(self) -> None:
        if not self.in_prefill:
            self.next_pos += 1
        self.step += 1


class CollabEngine:
    """Tensor/pipeline/sequence-parallel inference over a device mesh.

    Args:
        model_config: architecture of the model.
        parallel_config: strategy; world size must equal ``pp * sp * tp``.
        weights: full checkpoint (state dict, file, HF directory or
            :class:`WeightSource`); every rank loads only its shard.
        device: compute device of this rank.
        dtype: parameter/activation dtype.
        ctx: an existing :class:`ParallelContext` (created if omitted).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: Optional[ParallelConfig] = None,
        weights: Union[WeightSource, Mapping[str, torch.Tensor], str, None] = None,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        ctx: Optional[ParallelContext] = None,
    ) -> None:
        if weights is None:
            raise ValueError("weights are required (use models.random_state_dict for a random model)")
        self.model_config = model_config
        self.pc = (parallel_config or ParallelConfig()).validate(model_config.num_layers)
        self.ctx = ctx if ctx is not None else ParallelContext.from_config(self.pc, device=device)
        if (self.ctx.pp_size, self.ctx.sp_size, self.ctx.tp_size) != (self.pc.pp_size, self.pc.sp_size, self.pc.tp_size):
            raise ValueError("ParallelContext does not match ParallelConfig")
        self.device = self.ctx.device
        self.dtype = dtype
        counts = list(self.pc.pp_layers) if self.pc.pp_layers is not None else partition_layers(
            model_config.num_layers, self.pc.pp_size
        )
        self.layer_counts = [int(c) for c in counts]
        start = sum(self.layer_counts[: self.ctx.pp_rank])
        self.model = LlamaStage(
            model_config,
            self.ctx,
            range(start, start + self.layer_counts[self.ctx.pp_rank]),
            has_embedding=self.ctx.is_first_stage,
            has_head=self.ctx.is_last_stage,
            tp_weights=self.pc.stage_tp_weights(self.ctx.pp_rank),
            sp_mode=self.pc.sp_mode,
            sp_weights=self.pc.sp_weights,
            kv_block=self.pc.attn_kv_block,
            dtype=dtype,
            device=self.device,
        )
        self.model.load_weights(open_weights(weights))
        self.model.eval()
        self.channel = P2PChannel(self.ctx.pp)
        self.vocab_slice = None  # (global offset, first local row, rows) of the LM head this rank scores
        if self.ctx.is_last_stage:
            head = self.model.lm_head
            sizes = split_sizes(head.out_local, list(self.pc.sp_weights) if self.pc.sp_weights else self.ctx.sp_size)
            row = offsets_of(sizes)[self.ctx.sp_rank]
            self.vocab_slice = (head.out_start + row, row, sizes[self.ctx.sp_rank])
        self.last_kv_cache_bytes = 0  # KV cache held by this rank in the last generate()
        self.last_kv_cache_allocated_bytes = 0  # including reserved but unused slots

    # ------------------------------------------------------------ properties
    @property
    def comm_stats(self) -> CommStats:
        return self.ctx.stats

    def memory_report(self) -> Dict[str, Any]:
        return {
            "rank": self.ctx.rank,
            "coords": (self.ctx.pp_rank, self.ctx.sp_rank, self.ctx.tp_rank),
            "layers": self.model.layer_ids,
            "parameters": self.model.num_parameters(),
            "parameter_bytes": self.model.parameter_bytes(),
        }

    # ------------------------------------------------------------- internals
    def _make_state(
        self, positions: torch.Tensor, cache: Optional[KVCache]
    ) -> Tuple[ForwardState, Optional[SequenceLayout]]:
        """Decide how the tokens of this pass are laid out on the mesh.

        Every rank evaluates this with identical inputs and reaches the same
        decision, so no coordination messages are needed.
        """
        ctx, pc = self.ctx, self.pc
        S = positions.shape[1]
        sp = ctx.sp_size
        cache_empty = cache is None or cache.is_empty()
        sharded = sp > 1 and S >= sp and (pc.sp_mode == "ulysses" or cache_empty)
        layout, sp_sizes, owner = None, None, None
        local_pos = positions
        if sharded:
            layout = SequenceLayout(S, sp, pc.sp_weights, pc.sp_layout)
            local_pos = layout.shard(positions, ctx.sp_rank)
            sp_sizes = layout.sizes
        if cache is not None:
            if sp > 1 and pc.sp_mode == "ring":
                if cache.sp_lengths is None:
                    cache.sp_lengths = [0] * sp
                if sharded:
                    cache.sp_lengths = [n + s for n, s in zip(cache.sp_lengths, layout.sizes)]
                else:  # balance the distributed cache: least-loaded rank stores new tokens
                    owner = min(range(sp), key=lambda r: (cache.sp_lengths[r], r))
                    cache.sp_lengths[owner] += S
            cache.num_tokens += S
        s_local = local_pos.shape[1]
        # replicated tokens (decoding): SP ranks share the dense compute instead of repeating it
        sp_split = pc.sp_decode_split and sp > 1 and not sharded
        tp_sp_sizes = None
        if pc.megatron_sp and ctx.tp_size > 1 and s_local >= ctx.tp_size and not sp_split:
            tp_sp_sizes = split_sizes(s_local, ctx.tp_size)
        state = ForwardState(local_pos, sharded, sp_sizes, tp_sp_sizes, owner, cache, sp_split=sp_split)
        return state, layout

    def _embed(self, ids: torch.Tensor, state: ForwardState, layout: Optional[SequenceLayout]) -> torch.Tensor:
        if layout is not None:
            ids = layout.shard(ids, self.ctx.sp_rank)
        return self.model.embed(ids, state)

    def _last_hidden(
        self, x: torch.Tensor, state: ForwardState, layout: Optional[SequenceLayout]
    ) -> torch.Tensor:
        """Final hidden state of the last token, ``[B, H]``, on every rank of the stage."""
        h = self.model.final_hidden(x, state)
        if layout is None:
            h_last = h[:, -1]
        else:  # the last token lives on one SP rank: broadcast it to the others
            owner, index = layout.owner_of(-1)
            if self.ctx.sp_rank == owner:
                h_last = h[:, index].contiguous()
            else:
                h_last = h.new_empty((h.shape[0], h.shape[-1]))
            h_last = self.ctx.sp.broadcast(h_last, src=owner)
        return h_last

    def _select_tokens(self, h_last: torch.Tensor, mb: _MicroBatch, sampler: Sampler, eos: Optional[int]) -> torch.Tensor:
        """Score this rank's vocabulary slice and agree on the next token.

        The LM head is split over every rank of the last stage, and only a few
        candidates per rank are exchanged (see :mod:`.sampling`), instead of
        all-gathering full-vocabulary logits and broadcasting the choice.
        """
        offset, row, count = self.vocab_slice
        head = self.model.lm_head
        bias = None if head.bias is None else head.bias[row : row + count]
        logits = F.linear(h_last, head.weight[row : row + count], bias)
        tokens = sampler.select(logits, offset, mb.seq_ids, mb.gen_index, comm=self.ctx.stage)
        if eos is not None:
            tokens = torch.where(mb.done_last, torch.full_like(tokens, eos), tokens)
            mb.done_last |= tokens == eos
        return tokens

    def _chunk_size(self, prompt_len: int) -> int:
        """Tokens per prefill chunk for a prompt of ``prompt_len`` tokens (0 = no chunking)."""
        pc = self.pc
        if pc.prefill_chunk is not None:
            return pc.prefill_chunk
        if pc.pp_size == 1 or (pc.sp_size > 1 and pc.sp_mode == "ring"):
            return 0  # ring prefill needs the whole prompt at once
        # ~2 chunks per stage keep every stage busy; big enough for efficient GEMMs
        return max(MIN_AUTO_CHUNK, -(-prompt_len // (2 * pc.pp_size)))

    def _broadcast_inputs(self, obj: Any) -> Any:
        return self.ctx.world.broadcast_object(obj if self.ctx.rank == 0 else None, src=0)

    # --------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        num_microbatches: Optional[int] = None,
        broadcast: bool = False,
    ) -> Optional[torch.Tensor]:
        """Logits ``[B, S, vocab]`` for every position (no KV cache).

        ``input_ids``/``attention_mask`` are read on rank 0.  Logits are
        returned on the last pipeline stage, or on every rank with
        ``broadcast=True``; other ranks get ``None``.
        """
        input_ids, attention_mask = self._broadcast_inputs((input_ids, attention_mask))
        if input_ids is None:
            raise ValueError("rank 0 must provide input_ids")
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            positions = torch.arange(input_ids.shape[1], device=self.device).expand_as(input_ids)
        else:
            mask = attention_mask.to(self.device).bool()
            positions = torch.where(mask, mask.long().cumsum(-1) - 1, torch.full_like(input_ids, -1))
        B = input_ids.shape[0]
        m = max(1, min(num_microbatches or self.pc.num_microbatches or self.ctx.pp_size, B))
        ctx = self.ctx
        outputs = []
        for ids, pos in zip(input_ids.split(split_sizes(B, m)), positions.split(split_sizes(B, m))):
            state, layout = self._make_state(pos, None)
            if ctx.is_first_stage:
                x = self._embed(ids, state, layout)
            else:
                x = self.channel.recv(ctx.pp_rank - 1).tensor
            x = self.model.forward_layers(x, state)
            if ctx.is_last_stage:
                logits = self.model.logits(self.model.final_hidden(x, state))
                if layout is not None:
                    logits = gather_sequence(logits, layout, ctx.sp)
                outputs.append(logits)
            else:
                self.channel.send(x, ctx.pp_rank + 1)
        self.channel.flush()
        result = torch.cat(outputs, dim=0) if outputs else None
        if broadcast:
            shape = (B, input_ids.shape[1], self.model_config.vocab_size)
            if result is None:
                result = torch.empty(shape, dtype=self.dtype, device=self.device)
            result = ctx.world.broadcast(result, src=ctx.rank_of(ctx.pp_size - 1, 0, 0))
        return result

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self,
        prompts: Optional[Prompts] = None,
        max_new_tokens: int = 32,
        *,
        sampling: Optional[SamplingParams] = None,
        eos_token_id: Optional[int] = None,
        num_microbatches: Optional[int] = None,
        return_scores: bool = False,
    ) -> GenerationResult:
        """Autoregressively generate up to ``max_new_tokens`` per prompt.

        ``prompts`` (lists of token ids, possibly of different lengths) are
        read on rank 0 and the result is returned on every rank.
        """
        sampling = (sampling or SamplingParams()).resolved() if self.ctx.rank == 0 else None
        prompts, sampling = self._broadcast_inputs((prompts, sampling))
        if not prompts:
            raise ValueError("rank 0 must provide at least one prompt")
        vocab = self.model_config.vocab_size
        for p in prompts:
            if len(p) == 0:
                raise ValueError("prompts must not be empty")
            if min(p) < 0 or max(p) >= vocab:
                raise ValueError(f"token ids must be in [0, {vocab})")
        n = len(prompts)
        m = max(1, min(num_microbatches or self.pc.num_microbatches or self.ctx.pp_size, n))
        groups, start = [], 0
        for size in split_sizes(n, m):
            ids = list(range(start, start + size))
            group = [prompts[i] for i in ids]
            # decoding appends at most max_new_tokens - 1 tokens to any rank's cache
            groups.append(
                _MicroBatch(
                    ids,
                    group,
                    self.device,
                    max(0, max_new_tokens - 1),
                    self._chunk_size(max(len(p) for p in group)),
                )
            )
            start += size
        sampler = Sampler(sampling)
        t0 = time.perf_counter()
        if max_new_tokens > 0:
            self._generation_loop(groups, max_new_tokens, sampler, eos_token_id, return_scores)
        self.channel.flush()
        elapsed = time.perf_counter() - t0
        self.last_kv_cache_bytes = sum(mb.cache.nbytes() for mb in groups)
        self.last_kv_cache_allocated_bytes = sum(mb.cache.allocated_bytes() for mb in groups)

        payload = None
        if self.ctx.rank == 0:
            tokens: List[List[int]] = [[] for _ in range(n)]
            for mb in groups:
                if mb.generated:
                    steps = torch.stack(mb.generated, dim=1).tolist()
                    for b, pid in enumerate(mb.prompt_ids):
                        seq = steps[b]
                        if eos_token_id is not None and eos_token_id in seq:
                            seq = seq[: seq.index(eos_token_id) + 1]
                        tokens[pid] = seq
            generated = sum(len(t) for t in tokens)
            first_times = [mb.first_token_time for mb in groups if mb.first_token_time is not None]
            ttft = (min(first_times) - t0) if first_times else 0.0
            decode_time = max(elapsed - ttft, 1e-9)
            stats = {
                "total_s": elapsed,
                "ttft_s": ttft,
                "generated_tokens": float(generated),
                "decode_tokens_per_s": max(generated - n, 0) / decode_time,
                "num_microbatches": float(m),
            }
            payload = (tokens, stats)
        tokens, stats = self.ctx.world.broadcast_object(payload, src=0)
        scores = None
        if return_scores and self.ctx.is_last_stage:
            scores = [None] * n
            for mb in groups:
                if mb.scores:
                    stacked = torch.stack(mb.scores, dim=1)  # [B, T, V]
                    for b, pid in enumerate(mb.prompt_ids):
                        scores[pid] = stacked[b, : len(tokens[pid])]
        return GenerationResult(tokens=tokens, scores=scores, stats=stats)

    def _generation_loop(
        self,
        groups: List[_MicroBatch],
        max_new_tokens: int,
        sampler: Sampler,
        eos: Optional[int],
        return_scores: bool,
    ) -> None:
        ctx = self.ctx
        first, last, pp = ctx.is_first_stage, ctx.is_last_stage, ctx.pp_size
        prev, nxt = ctx.pp_rank - 1, ctx.pp_rank + 1
        active = list(groups)
        while active:
            still_active = []
            for mb in active:
                # ---- receive the input of this step (or retire the micro-batch)
                if first:
                    if mb.step >= mb.num_chunks:  # decoding: wait for the previous token
                        tokens = self.channel.recv(pp - 1).tensor if pp > 1 else mb.pending
                        if mb.first_token_time is None:
                            mb.first_token_time = time.perf_counter()
                        mb.generated.append(tokens)
                        if eos is not None:
                            mb.done_first |= tokens == eos
                        if len(mb.generated) >= max_new_tokens or bool(mb.done_first.all()):
                            if pp > 1:
                                self.channel.send_stop(nxt)
                            continue
                    x_in = None
                else:
                    msg = self.channel.recv(prev)
                    if msg.is_stop:
                        if not last:
                            self.channel.send_stop(nxt)
                        continue
                    x_in = msg.tensor
                # ---- run this stage
                state, layout = self._make_state(mb.step_positions(), mb.cache)
                if first:
                    x = self._embed(mb.step_ids(), state, layout)
                else:
                    x = x_in
                x = self.model.forward_layers(x, state)
                # ---- hand over to the next stage (tokens loop back to stage 0)
                if last and not mb.emits_token:
                    pass  # an intermediate prompt chunk: only the KV cache was needed
                elif last:
                    h_last = self._last_hidden(x, state, layout)
                    if return_scores:  # full-vocabulary logits only when asked for
                        mb.scores.append(self.model.logits(h_last))
                    tokens = self._select_tokens(h_last, mb, sampler, eos)
                    if pp > 1:
                        self.channel.send(tokens, 0)
                    else:
                        mb.pending = tokens
                else:
                    self.channel.send(x, nxt)
                mb.advance()
                still_active.append(mb)
            active = still_active


__all__ = ["CollabEngine", "GenerationResult"]
