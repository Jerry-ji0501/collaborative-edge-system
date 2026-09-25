"""Token sampling (greedy, temperature, top-k, top-p) over vocabulary-parallel logits.

The last pipeline stage splits the LM head over *all* its ranks, so each rank
only holds the logits of one vocabulary slice.  Instead of gathering full
vocabulary logits (``vocab_size`` floats per token per rank), every rank
proposes a few candidates from its slice and one small all-gather lets all
ranks agree on the token:

* greedy and pure temperature sampling gather one candidate per rank,
* top-k gathers ``k`` candidates per rank,
* top-p gathers a bounded candidate set plus each slice's log-normaliser and
  only falls back to the full logits when those candidates cannot contain the
  nucleus, so it stays exact.

Random sampling uses the Gumbel-max trick with *counter-based* noise: the
noise of token ``v`` for sequence ``s`` at generation step ``t`` is a hash of
``(seed, s, t, v)``.  Samples are exact draws from the (filtered) softmax, yet
identical for every parallel layout and micro-batching.  All paths except
top-p decide by pure comparisons of gathered values, so heterogeneous devices
agree bit for bit without an extra broadcast; top-p is decided by the first
rank and broadcast.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import torch

from ..distributed.comm import Communicator

_M64 = (1 << 64) - 1


def _signed(x: int) -> int:
    x &= _M64
    return x - (1 << 64) if x >= (1 << 63) else x


_C1, _C2, _C3 = _signed(0x9E3779B97F4A7C15), _signed(0xBF58476D1CE4E5B9), _signed(0x94D049BB133111EB)
_INT64_MAX = (1 << 63) - 1
TOP_P_CANDIDATES = 256  # per-rank candidates gathered for top-p without top-k


def _lsr(z: torch.Tensor, shift: int) -> torch.Tensor:
    """Logical right shift of int64 values."""
    return (z >> shift) & ((1 << (64 - shift)) - 1)


def splitmix64(x: torch.Tensor) -> torch.Tensor:
    """SplitMix64 hash of int64 tensors (wrapping arithmetic)."""
    x = x + _C1
    z = (x ^ _lsr(x, 30)) * _C2
    z = (z ^ _lsr(z, 27)) * _C3
    return z ^ _lsr(z, 31)


def uniform_noise(seed: int, seq_ids: torch.Tensor, step: int, token_ids: torch.Tensor) -> torch.Tensor:
    """Deterministic uniforms in ``(0, 1)`` keyed by ``(seed, sequence, step, token)``.

    ``seq_ids``: ``[B]``; ``token_ids``: ``[B, n]`` or ``[n]``.  Returns float64 ``[B, n]``.
    """
    seed_t = torch.tensor(_signed(seed), dtype=torch.int64, device=seq_ids.device)
    base = splitmix64(splitmix64(splitmix64(seed_t) + seq_ids.to(torch.int64)) + int(step))
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    keys = splitmix64(base.unsqueeze(1) + token_ids.to(torch.int64))
    return (_lsr(keys, 11).to(torch.float64) + 0.5) * (2.0**-53)


def gumbel_noise(seed: int, seq_ids: torch.Tensor, step: int, token_ids: torch.Tensor) -> torch.Tensor:
    return -torch.log(-torch.log(uniform_noise(seed, seq_ids, step, token_ids)))


@dataclass
class SamplingParams:
    """``temperature <= 0`` means greedy decoding."""

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0

    def resolved(self) -> "SamplingParams":
        """Copy with a concrete seed (drawn at random if unset)."""
        return self if self.seed is not None else replace(self, seed=random.getrandbits(62))


# --------------------------------------------------------------------------
# deterministic selection helpers (pure comparisons)
# --------------------------------------------------------------------------
def _argmax_min_id(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Per row: id of the maximum value; ties go to the smallest id."""
    top = values.max(dim=1, keepdim=True).values
    return torch.where(values == top, ids, torch.full_like(ids, _INT64_MAX)).min(dim=1).values


def _order(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Column order sorting rows by descending value, ties by ascending id."""
    by_id = torch.argsort(ids, dim=1, stable=True)
    by_value = torch.argsort(-values.gather(1, by_id), dim=1, stable=True)
    return by_id.gather(1, by_value)


class Sampler:
    """Chooses the next token from (possibly vocabulary-parallel) logits."""

    def __init__(self, params: Optional[SamplingParams] = None) -> None:
        self.params = (params or SamplingParams()).resolved()
        self.seed = int(self.params.seed)
        self._calls = 0

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Single-device convenience: ``logits [B, V]`` -> token ids ``[B]``."""
        seq_ids = torch.arange(logits.shape[0], device=logits.device)
        tokens = self.select(logits, 0, seq_ids, self._calls)
        self._calls += 1
        return tokens

    # ------------------------------------------------------------------ core
    def select(
        self,
        logits: torch.Tensor,
        offset: int,
        seq_ids: torch.Tensor,
        step: int,
        comm: Optional[Communicator] = None,
    ) -> torch.Tensor:
        """Pick one token per row; identical on every rank of ``comm``.

        Args:
            logits: ``[B, n]`` logits of vocabulary ids ``[offset, offset + n)``.
            seq_ids: ``[B]`` global ids of the sequences (noise keys).
            step: generation step (noise key).
            comm: group whose ranks together hold the whole vocabulary.
        """
        p = self.params
        B, n = logits.shape
        device = logits.device
        ids = torch.arange(offset, offset + n, device=device).expand(B, n)
        if p.greedy:
            values, index = logits.max(dim=1)
            return self._comparison_pick(values[:, None].double(), (index + offset)[:, None], comm)
        scaled = logits.double() / p.temperature
        if p.top_p >= 1.0 and p.top_k == 0:  # Gumbel-max over the whole vocabulary
            perturbed = scaled + gumbel_noise(self.seed, seq_ids, step, ids)
            values, index = perturbed.max(dim=1)
            return self._comparison_pick(values[:, None], (index + offset)[:, None], comm)
        k = p.top_k if p.top_k > 0 else TOP_P_CANDIDATES
        cand_values, cand_index = torch.topk(scaled, min(k, n), dim=1)
        cand_ids = cand_index + offset
        cand_pert = cand_values + gumbel_noise(self.seed, seq_ids, step, cand_ids)
        if min(k, n) < k:  # pad so every rank contributes the same shape
            pad = k - n
            cand_values = torch.cat([cand_values, cand_values.new_full((B, pad), float("-inf"))], 1)
            cand_pert = torch.cat([cand_pert, cand_pert.new_full((B, pad), float("-inf"))], 1)
            cand_ids = torch.cat([cand_ids, cand_ids.new_full((B, pad), -1)], 1)
        if p.top_p >= 1.0:  # top-k: comparisons only
            values, pert, gids = self._gather_candidates([cand_values, cand_pert], cand_ids, comm)
            order = _order(values, gids)[:, : p.top_k]
            return _argmax_min_id(pert.gather(1, order), gids.gather(1, order))
        return self._top_p(scaled, offset, seq_ids, step, cand_values, cand_pert, cand_ids, comm)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _gather_candidates(
        float_cols: List[torch.Tensor], ids: torch.Tensor, comm: Optional[Communicator]
    ) -> Tuple[torch.Tensor, ...]:
        """All-gather ``[B, m]`` candidate columns (ids travel exactly as float64)."""
        packed = torch.stack([c.double() for c in float_cols] + [ids.double()], dim=-1)  # [B, m, c]
        if comm is not None and comm.size > 1:
            packed = torch.cat(comm.all_gather_list(packed.contiguous(), dim=0), dim=1)
        cols = [packed[..., i] for i in range(len(float_cols))]
        return (*cols, packed[..., -1].to(torch.int64))

    def _comparison_pick(self, values: torch.Tensor, ids: torch.Tensor, comm: Optional[Communicator]) -> torch.Tensor:
        gathered_values, gathered_ids = self._gather_candidates([values], ids, comm)
        return _argmax_min_id(gathered_values, gathered_ids)

    def _top_p(self, scaled, offset, seq_ids, step, cand_values, cand_pert, cand_ids, comm):
        """Nucleus sampling; decided on the first rank and broadcast."""
        B, k = cand_values.shape
        n_local = scaled.shape[1]
        local_lse = torch.logsumexp(scaled, dim=1, keepdim=True).expand(B, k)
        complete = torch.full((B, k), 1.0 if n_local <= k else 0.0, dtype=torch.float64, device=scaled.device)
        gathered = self._gather_candidates([cand_values, cand_pert, local_lse, complete], cand_ids, comm)
        distributed = comm is not None and comm.size > 1
        tokens, need_full = None, False
        if not distributed or comm.rank == 0:
            tokens, need_full = self._nucleus(*gathered, world=comm.size if distributed else 1)
        if not distributed:
            return self._top_p_full(scaled, offset, seq_ids, step, comm) if need_full else tokens
        flag = torch.tensor([1 if need_full else 0], dtype=torch.int64, device=scaled.device)
        if int(comm.broadcast(flag, src=0).item()):
            return self._top_p_full(scaled, offset, seq_ids, step, comm)
        if tokens is None:
            tokens = torch.empty(B, dtype=torch.int64, device=scaled.device)
        return comm.broadcast(tokens, src=0)

    def _nucleus(self, values, pert, lse, complete, gids, world: int):
        """``(tokens, need_full)`` from gathered candidates (runs on one rank)."""
        p = self.params
        B, total = values.shape
        per_rank = total // world
        order = _order(values, gids)
        sv, sp, sg = values.gather(1, order), pert.gather(1, order), gids.gather(1, order)
        if p.top_k > 0:
            # the union of per-rank top-k holds the global top-k: always exact
            sv, sp, sg = sv[:, : p.top_k], sp[:, : p.top_k], sg[:, : p.top_k]
            norm = torch.logsumexp(sv, dim=1, keepdim=True)
        else:
            norm = torch.logsumexp(lse.view(B, world, per_rank)[:, :, 0], dim=1, keepdim=True)
        probs = torch.exp(sv - norm)
        keep = (torch.cumsum(probs, dim=1) - probs) <= p.top_p
        if p.top_k == 0:
            # tokens a truncated slice did not send are no larger than its smallest candidate;
            # the result is exact if the nucleus ends strictly above all of those bounds
            blocks = values.view(B, world, per_rank)
            smallest = blocks.masked_fill(torch.isinf(blocks), float("inf")).min(dim=2).values
            truncated = complete.view(B, world, per_rank)[:, :, 0] == 0
            bound = torch.where(truncated, smallest, torch.full_like(smallest, float("-inf"))).max(dim=1).values
            cutoff = torch.where(keep, sv, torch.full_like(sv, float("inf"))).min(dim=1).values
            if not bool((cutoff > bound).all()):
                return None, True
        tokens = _argmax_min_id(torch.where(keep, sp, torch.full_like(sp, float("-inf"))), sg)
        return tokens, False

    def _top_p_full(self, scaled, offset, seq_ids, step, comm):
        """Exact nucleus sampling on the gathered full vocabulary (rare fallback)."""
        p = self.params
        B, n = scaled.shape
        ids = torch.arange(offset, offset + n, device=scaled.device).expand(B, n).contiguous()
        distributed = comm is not None and comm.size > 1
        if distributed:
            sizes = [int(t.item()) for t in comm.all_gather_list(torch.tensor([n], device=scaled.device), dim=0)]
            scaled = torch.cat(comm.all_gather_list(scaled.contiguous(), dim=1, sizes=sizes), dim=1)
            ids = torch.cat(comm.all_gather_list(ids, dim=1, sizes=sizes), dim=1)
            if comm.rank != 0:
                return comm.broadcast(torch.empty(B, dtype=torch.int64, device=scaled.device), src=0)
        order = _order(scaled, ids)
        values, gids = scaled.gather(1, order), ids.gather(1, order)
        probs = torch.softmax(values, dim=1)
        keep = (torch.cumsum(probs, dim=1) - probs) <= p.top_p
        pert = values + gumbel_noise(self.seed, seq_ids, step, gids)
        tokens = _argmax_min_id(torch.where(keep, pert, torch.full_like(pert, float("-inf"))), gids)
        return comm.broadcast(tokens, src=0) if distributed else tokens


__all__ = ["Sampler", "SamplingParams", "gumbel_noise", "splitmix64", "uniform_noise"]
