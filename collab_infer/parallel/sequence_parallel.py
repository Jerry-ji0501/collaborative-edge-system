"""Sequence parallelism: splitting the tokens of a sequence across devices.

Token-wise operators (embeddings, norms, MLPs, projections) run unchanged on
a rank's local tokens; only attention needs to see other ranks' tokens.  Two
classic algorithms are provided, plus the decoding counterpart:

* :func:`ring_attention` – every rank keeps its queries while the key/value
  blocks travel around a ring of ``n`` ranks; partial results are merged with
  their log-sum-exp.  Communication (``n - 1`` block transfers) overlaps with
  computation and the KV cache stays **sharded along the sequence**, which is
  what lets memory-constrained edge devices hold long contexts together.
* :func:`ulysses_attention` – two all-to-alls switch between sequence
  sharding and head sharding (DeepSpeed-Ulysses), so each rank runs ordinary
  attention over the full sequence for a subset of heads.  Communication
  volume is independent of the number of ranks; the KV cache ends up
  **sharded along heads**.
* :func:`distributed_kv_attention` – decoding with a replicated query against
  a KV cache that is split across ranks: each rank attends to its own
  entries and the partial results are merged exactly.

All functions accept uneven per-rank sizes (heterogeneous devices) and any
token layout, since masking uses explicit positions (see
:mod:`collab_infer.parallel.attention`).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from ..distributed.comm import Communicator
from .attention import (
    KVMap,
    attention,
    attention_with_lse,
    fully_masked,
    merge_attention,
    merge_attention_list,
)
from .partition import HeadShard, SequenceLayout, offsets_of

TAG_RING_KV = 21
TAG_RING_POS = 22


# --------------------------------------------------------------------------
# sharding helpers
# --------------------------------------------------------------------------
def shard_sequence(x: torch.Tensor, layout: SequenceLayout, rank: int, dim: int = 1) -> torch.Tensor:
    """This rank's tokens of a full-sequence tensor."""
    return layout.shard(x, rank, dim)


def gather_sequence(
    x_local: torch.Tensor, layout: SequenceLayout, comm: Communicator, dim: int = 1
) -> torch.Tensor:
    """All-gather sequence shards and restore global token order."""
    if comm.size == 1:
        return x_local
    parts = comm.all_gather_list(x_local, dim=dim, sizes=layout.sizes)
    return layout.unshard(parts, dim=dim)


# --------------------------------------------------------------------------
# ring attention
# --------------------------------------------------------------------------
def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    comm: Communicator,
    kv_sizes: Sequence[int],
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    kv_map: KVMap = None,
    kv_block: Optional[int] = None,
) -> torch.Tensor:
    """Attention of local queries over the keys/values of all ranks.

    Args:
        q, k, v: local shards ``[B, s_i, H(kv), D]``.
        q_pos, k_pos: positions of the local tokens (``-1`` = padding).
        kv_sizes: number of tokens held by each rank (needed to size receive
            buffers for uneven splits).

    At step ``t`` rank ``i`` holds the block that originated at rank
    ``i - t``, sends it to ``i + 1`` and receives the next one from ``i - 1``
    while computing.  Blocks invisible under the causal mask are skipped.
    """
    n, r = comm.size, comm.rank
    if n == 1:
        out, _ = attention_with_lse(q, k, v, q_pos, k_pos, causal=causal, scale=scale, kv_map=kv_map, kv_block=kv_block)
        return out.to(q.dtype)
    if len(kv_sizes) != n or kv_sizes[r] != k.shape[1]:
        raise ValueError(f"kv_sizes {list(kv_sizes)} inconsistent with local block of {k.shape[1]} tokens")
    B, _, Hkv, D = k.shape
    # K and V travel as one message, optionally in a narrower dtype; the local
    # block is used at full precision and received blocks are forwarded as-is
    wire = comm.to_wire(torch.stack([k, v])).contiguous()
    k_cur, v_cur, pos = k, v, k_pos.contiguous()
    out = lse = None
    nxt, prv = (r + 1) % n, (r - 1) % n
    for step in range(n):
        pending = []
        if step < n - 1:
            src = (r - step - 1) % n  # origin of the block arriving next
            pending = [
                comm.isend(wire, nxt, tag=TAG_RING_KV),
                comm.isend(pos, nxt, tag=TAG_RING_POS),
                comm.irecv((2, B, kv_sizes[src], Hkv, D), wire.dtype, prv, tag=TAG_RING_KV),
                comm.irecv((B, kv_sizes[src]), pos.dtype, prv, tag=TAG_RING_POS, emulate=False),
            ]
        if not fully_masked(q_pos, pos, causal):
            o, l = attention_with_lse(
                q, k_cur, v_cur, q_pos, pos, causal=causal, scale=scale, kv_map=kv_map, kv_block=kv_block
            )
            out, lse = (o, l) if out is None else merge_attention(out, lse, o, l)
        if pending:
            pending[0].wait()
            pending[1].wait()
            wire = pending[2].wait()
            pos = pending[3].wait()
            k_cur, v_cur = wire[0].to(k.dtype), wire[1].to(k.dtype)
    if out is None:  # every key invisible (e.g. all padding)
        return torch.zeros_like(q)
    return out.to(q.dtype)


# --------------------------------------------------------------------------
# Ulysses (all-to-all) attention
# --------------------------------------------------------------------------
def _check_ulysses(comm: Communicator, seq_sizes: Sequence[int], head_shards: Sequence[HeadShard], s_local: int):
    if len(seq_sizes) != comm.size or seq_sizes[comm.rank] != s_local:
        raise ValueError(f"seq_sizes {list(seq_sizes)} inconsistent with local shard of {s_local} tokens")
    if len(head_shards) != comm.size:
        raise ValueError("need one head shard per rank")


def ulysses_scatter(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    comm: Communicator,
    seq_sizes: Sequence[int],
    head_shards: Sequence[HeadShard],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First Ulysses all-to-all: sequence-sharded -> head-sharded.

    Takes local sequence shards ``[B, s_i, H(kv), D]`` with all heads and
    returns this rank's heads over the whole sequence (tokens concatenated in
    rank order) together with their positions ``[B, S]``.  Q, K and V travel
    in a single all-to-all.
    """
    n, r = comm.size, comm.rank
    B, s_local, _, D = q.shape
    _check_ulysses(comm, seq_sizes, head_shards, s_local)
    me = head_shards[r]
    sends = [
        torch.cat(
            [
                q[:, :, sh.q_start : sh.q_start + sh.q_count],
                k[:, :, sh.kv_start : sh.kv_start + sh.kv_count],
                v[:, :, sh.kv_start : sh.kv_start + sh.kv_count],
            ],
            dim=2,
        ).contiguous()
        for sh in head_shards
    ]
    width = me.q_count + 2 * me.kv_count
    received = comm.all_to_all(sends, [(B, seq_sizes[j], width, D) for j in range(n)], compress=True)
    full = torch.cat(received, dim=1) if n > 1 else received[0]
    q_f, k_f, v_f = full.split([me.q_count, me.kv_count, me.kv_count], dim=2)
    pos_f = comm.all_gather(positions, dim=1, sizes=seq_sizes)
    return q_f, k_f, v_f, pos_f


def ulysses_gather(
    out_heads: torch.Tensor,
    comm: Communicator,
    seq_sizes: Sequence[int],
    head_shards: Sequence[HeadShard],
) -> torch.Tensor:
    """Second Ulysses all-to-all: head-sharded ``[B, S, h_i, D]`` -> ``[B, s_i, H, D]``."""
    n, r = comm.size, comm.rank
    B, _, _, D = out_heads.shape
    starts = offsets_of(seq_sizes)
    sends = [out_heads[:, st : st + sz].contiguous() for st, sz in zip(starts, seq_sizes)]
    back = comm.all_to_all(sends, [(B, seq_sizes[r], head_shards[j].q_count, D) for j in range(n)], compress=True)
    return torch.cat(back, dim=2) if n > 1 else back[0]


def ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    comm: Communicator,
    seq_sizes: Sequence[int],
    head_shards: Sequence[HeadShard],
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    past: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    kv_block: Optional[int] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """All-to-all sequence parallel attention.

    Args:
        q, k, v: local sequence shards ``[B, s_i, H(kv), D]`` with *all* heads.
        positions: positions of the local tokens ``[B, s_i]``.
        seq_sizes: tokens per rank.
        head_shards: partition of the heads over the ranks
            (see :func:`~collab_infer.parallel.partition.partition_heads`).
        past: optional cached ``(k, v, pos)`` for this rank's heads over earlier
            tokens (``[B, T, kv_count, D]``), e.g. a previous prefill chunk.
            (With a :class:`~collab_infer.models.cache.LayerKVCache`, prefer
            :func:`ulysses_scatter` + append + attention + :func:`ulysses_gather`,
            which avoids copying the cached history.)

    Returns:
        ``(out, (k_heads, v_heads, pos_full))`` where ``out`` is
        ``[B, s_i, Hq, D]`` and the second element holds this rank's heads over
        the full new sequence, ready to be appended to a head-sharded KV cache.
    """
    me = head_shards[comm.rank]
    q_f, k_f, v_f, pos_f = ulysses_scatter(q, k, v, positions, comm, seq_sizes, head_shards)
    k_att, v_att, p_att = k_f, v_f, pos_f
    if past is not None and past[0].shape[1] > 0:
        k_att = torch.cat([past[0], k_f], dim=1)
        v_att = torch.cat([past[1], v_f], dim=1)
        p_att = torch.cat([past[2], pos_f], dim=1)
    o = attention(q_f, k_att, v_att, pos_f, p_att, causal=causal, scale=scale, kv_map=me.kv_map, kv_block=kv_block)
    out = ulysses_gather(o, comm, seq_sizes, head_shards)
    return out, (k_f.contiguous(), v_f.contiguous(), pos_f)


def gather_heads(out_local: torch.Tensor, comm: Communicator, head_shards: Sequence[HeadShard]) -> torch.Tensor:
    """All-gather per-rank head slices ``[B, S, h_i, D]`` into ``[B, S, H, D]``."""
    return comm.all_gather(out_local, dim=2, sizes=[sh.q_count for sh in head_shards])


# --------------------------------------------------------------------------
# decoding against a distributed KV cache
# --------------------------------------------------------------------------
def distributed_kv_attention(
    q: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos_local: torch.Tensor,
    comm: Communicator,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    kv_map: KVMap = None,
    kv_block: Optional[int] = None,
) -> torch.Tensor:
    """Attention of a query replicated on all ranks over a sharded KV cache.

    Each rank computes attention over the cache entries it stores and the
    results are merged with their log-sum-exp.  Communication is a single
    all-gather of ``[B, Sq, H, D + 1]`` per call, independent of the context
    length, while cache memory and attention compute are split across ranks.
    """
    o, l = attention_with_lse(
        q, k_local, v_local, q_pos, k_pos_local, causal=causal, scale=scale, kv_map=kv_map, kv_block=kv_block
    )
    if comm.size > 1:
        D = o.shape[-1]
        packed = torch.cat([o, l.transpose(1, 2).unsqueeze(-1)], dim=-1)
        parts = comm.all_gather_list(packed, dim=0)
        outs: List[torch.Tensor] = [p[..., :D] for p in parts]
        lses: List[torch.Tensor] = [p[..., D].transpose(1, 2) for p in parts]
        o, _ = merge_attention_list(outs, lses)
    return o.to(q.dtype)


__all__ = [
    "distributed_kv_attention",
    "gather_heads",
    "gather_sequence",
    "ring_attention",
    "shard_sequence",
    "ulysses_attention",
    "ulysses_gather",
    "ulysses_scatter",
]
