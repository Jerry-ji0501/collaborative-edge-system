"""Partitioning helpers shared by all parallel strategies.

Everything here is deterministic pure computation: every rank evaluates the
same functions with the same arguments and therefore agrees on who owns which
slice without exchanging any messages.  All splits support *weights* so that
heterogeneous edge devices receive work proportional to their capability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch

WeightsLike = Union[int, Sequence[float]]


def split_sizes(
    total: int,
    weights: WeightsLike,
    granularity: int = 1,
    min_units: int = 1,
) -> List[int]:
    """Split ``total`` into parts proportional to ``weights``.

    Args:
        total: quantity to split (e.g. number of heads, tokens or features).
        weights: number of equal parts, or one positive weight per part.
        granularity: every part is a multiple of this (``total`` must be too).
        min_units: minimum number of granules per part.

    The largest-remainder method is used, so an even split is exact whenever
    possible and ties go to lower ranks.
    """
    if isinstance(weights, int):
        if weights < 1:
            raise ValueError("number of parts must be >= 1")
        weights = [1.0] * weights
    w = [float(x) for x in weights]
    n = len(w)
    if n == 0:
        raise ValueError("weights must not be empty")
    if any(x <= 0 or not math.isfinite(x) for x in w):
        raise ValueError(f"weights must be positive and finite, got {w}")
    if granularity < 1:
        raise ValueError("granularity must be >= 1")
    if total < 0 or total % granularity != 0:
        raise ValueError(f"total={total} is not a non-negative multiple of granularity={granularity}")
    units = total // granularity
    if units < n * min_units:
        raise ValueError(
            f"cannot split {total} (granularity {granularity}) into {n} parts "
            f"with at least {min_units} unit(s) each"
        )
    free = units - n * min_units
    wsum = sum(w)
    ideal = [free * x / wsum for x in w]
    base = [int(math.floor(v + 1e-9)) for v in ideal]
    leftover = free - sum(base)
    if leftover < 0:  # guard against floating point overshoot
        base = [int(math.floor(v)) for v in ideal]
        leftover = free - sum(base)
    order = sorted(range(n), key=lambda i: (-(ideal[i] - base[i]), i))
    for i in order[:leftover]:
        base[i] += 1
    return [(b + min_units) * granularity for b in base]


def offsets_of(sizes: Sequence[int]) -> List[int]:
    """Exclusive prefix sums: start offset of every part."""
    out, acc = [], 0
    for s in sizes:
        out.append(acc)
        acc += int(s)
    return out


# --------------------------------------------------------------------------
# attention heads
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HeadShard:
    """The attention heads owned by one rank.

    ``kv_map[i]`` is the (shard-local) KV head used by the ``i``-th local query
    head.  With grouped-query attention several query heads share a KV head;
    when there are fewer KV heads than ranks a KV head is replicated on every
    rank that needs it.
    """

    q_start: int
    q_count: int
    kv_start: int
    kv_count: int
    kv_map: Tuple[int, ...]

    @property
    def q_range(self) -> Tuple[int, int]:
        return self.q_start, self.q_start + self.q_count

    @property
    def kv_range(self) -> Tuple[int, int]:
        return self.kv_start, self.kv_start + self.kv_count


def default_kv_map(num_q_heads: int, num_kv_heads: int) -> List[int]:
    if num_kv_heads < 1 or num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads={num_q_heads} must be a multiple of num_kv_heads={num_kv_heads}"
        )
    group = num_q_heads // num_kv_heads
    return [h // group for h in range(num_q_heads)]


def partition_heads(
    num_q_heads: int,
    num_kv_heads: int,
    weights: WeightsLike,
    kv_map: Optional[Sequence[int]] = None,
) -> List[HeadShard]:
    """Partition query heads (and the KV heads they need) over ranks.

    If KV heads are uniformly grouped and there are at least as many KV heads
    as ranks, whole KV groups are assigned so no KV head is duplicated.
    Otherwise the query heads are split directly and the needed KV heads are
    replicated (e.g. multi-query attention with ``tp_size > 1``).
    """
    if kv_map is None:
        kv_map = default_kv_map(num_q_heads, num_kv_heads)
    kv_map = [int(x) for x in kv_map]
    if len(kv_map) != num_q_heads:
        raise ValueError("kv_map must have one entry per query head")
    if any(b < a for a, b in zip(kv_map, kv_map[1:])):
        raise ValueError("kv_map must be non-decreasing")
    if kv_map and (kv_map[0] < 0 or kv_map[-1] >= num_kv_heads):
        raise ValueError("kv_map entries out of range")
    n = weights if isinstance(weights, int) else len(weights)

    uniform = (
        num_kv_heads > 0
        and num_q_heads % num_kv_heads == 0
        and kv_map == default_kv_map(num_q_heads, num_kv_heads)
    )
    if uniform and num_kv_heads >= n:
        group = num_q_heads // num_kv_heads
        q_sizes = [s * group for s in split_sizes(num_kv_heads, weights)]
    else:
        if num_q_heads < n:
            raise ValueError(f"cannot split {num_q_heads} query heads over {n} ranks")
        q_sizes = split_sizes(num_q_heads, weights)

    shards = []
    for q_start, q_count in zip(offsets_of(q_sizes), q_sizes):
        kv_lo = kv_map[q_start]
        kv_hi = kv_map[q_start + q_count - 1] + 1
        local = tuple(kv_map[h] - kv_lo for h in range(q_start, q_start + q_count))
        shards.append(HeadShard(q_start, q_count, kv_lo, kv_hi - kv_lo, local))
    return shards


# --------------------------------------------------------------------------
# sequence layouts
# --------------------------------------------------------------------------
class SequenceLayout:
    """Assignment of the tokens of a sequence to sequence-parallel ranks.

    * ``contiguous``: rank ``r`` owns one consecutive chunk.
    * ``zigzag``: the sequence is cut into ``2n`` chunks and rank ``r`` owns
      chunks ``r`` and ``2n - 1 - r``.  With causal attention this gives every
      rank the same amount of work in ring attention.

    Attention in this framework masks by explicit token positions, so any
    layout is valid; the layout only decides load balance.
    """

    def __init__(
        self,
        seq_len: int,
        num_ranks: int,
        weights: Optional[Sequence[float]] = None,
        kind: str = "contiguous",
    ) -> None:
        if kind not in ("contiguous", "zigzag"):
            raise ValueError(f"unknown sequence layout {kind!r}")
        self.seq_len = int(seq_len)
        self.num_ranks = int(num_ranks)
        self.kind = kind
        self.sizes = split_sizes(self.seq_len, list(weights) if weights else self.num_ranks)
        segments: List[List[Tuple[int, int]]] = []
        if kind == "contiguous":
            for start, size in zip(offsets_of(self.sizes), self.sizes):
                segments.append([(start, size)])
        else:
            first = [s // 2 for s in self.sizes]
            second = [s - a for s, a in zip(self.sizes, first)]
            # global chunk order: first_0 .. first_{n-1}, second_{n-1} .. second_0
            chunk_sizes = first + second[::-1]
            starts = offsets_of(chunk_sizes)
            for r in range(self.num_ranks):
                j = 2 * self.num_ranks - 1 - r
                segments.append([(starts[r], first[r]), (starts[j], second[r])])
        self.segments = [[(s, l) for s, l in segs if l > 0] for segs in segments]

    def local_len(self, rank: int) -> int:
        return self.sizes[rank]

    def indices(self, rank: int) -> torch.Tensor:
        """Global token indices owned by ``rank`` (in local order)."""
        parts = [torch.arange(s, s + l) for s, l in self.segments[rank]]
        return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)

    def owner_of(self, index: int) -> Tuple[int, int]:
        """``(rank, local_index)`` of global token ``index``."""
        if index < 0:
            index += self.seq_len
        if not 0 <= index < self.seq_len:
            raise IndexError(f"token index {index} out of range for length {self.seq_len}")
        for rank, segs in enumerate(self.segments):
            local = 0
            for start, length in segs:
                if start <= index < start + length:
                    return rank, local + index - start
                local += length
        raise AssertionError("unreachable: layout does not cover the sequence")

    def shard(self, x: torch.Tensor, rank: int, dim: int = 1) -> torch.Tensor:
        """Select the tokens of ``rank`` from a full-sequence tensor."""
        pieces = [x.narrow(dim, s, l) for s, l in self.segments[rank]]
        if len(pieces) == 1:
            return pieces[0]
        return torch.cat(pieces, dim=dim)

    def unshard(self, parts: Sequence[torch.Tensor], dim: int = 1) -> torch.Tensor:
        """Inverse of :meth:`shard`: reassemble per-rank tensors in global order."""
        pieces = []
        for rank, part in enumerate(parts):
            offset = 0
            for start, length in self.segments[rank]:
                pieces.append((start, part.narrow(dim, offset, length)))
                offset += length
        pieces.sort(key=lambda item: item[0])
        return torch.cat([p for _, p in pieces], dim=dim)

    def __repr__(self) -> str:
        return f"SequenceLayout(seq_len={self.seq_len}, kind={self.kind}, sizes={self.sizes})"


__all__ = [
    "HeadShard",
    "SequenceLayout",
    "default_kv_map",
    "offsets_of",
    "partition_heads",
    "split_sizes",
]
