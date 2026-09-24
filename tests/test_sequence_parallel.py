import pytest
import torch

from collab_infer.distributed import ParallelContext
from collab_infer.parallel.attention import attention
from collab_infer.parallel.partition import SequenceLayout, partition_heads
from collab_infer.parallel.sequence_parallel import (
    distributed_kv_attention,
    gather_sequence,
    ring_attention,
    ulysses_attention,
)
from helpers import run_dist

B, S, HQ, HKV, D = 2, 13, 6, 3, 4


def _inputs():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(B, S, HQ, D, generator=g, dtype=torch.float64)
    k = torch.randn(B, S, HKV, D, generator=g, dtype=torch.float64)
    v = torch.randn(B, S, HKV, D, generator=g, dtype=torch.float64)
    pos = torch.arange(S).repeat(B, 1)
    pos[1, :3] = -1  # left padding in the second sequence
    pos[1, 3:] -= 3
    return q, k, v, pos


def _sp_worker(rank, world, kind, weights, causal):
    ctx = ParallelContext(1, world, 1)
    sp = ctx.sp
    q, k, v, pos = _inputs()
    ref = attention(q, k, v, pos, pos, causal=causal)
    valid = pos >= 0
    layout = SequenceLayout(S, world, weights, kind)
    ql, kl, vl, pl = (layout.shard(t, rank) for t in (q, k, v, pos))
    errs = {}
    out = ring_attention(ql, kl, vl, pl, pl, sp, layout.sizes, causal=causal)
    errs["ring"] = (gather_sequence(out, layout, sp) - ref)[valid].abs().max().item()
    shards = partition_heads(HQ, HKV, weights or world)
    out, (kh, vh, ph) = ulysses_attention(ql, kl, vl, pl, sp, layout.sizes, shards, causal=causal)
    errs["ulysses"] = (gather_sequence(out, layout, sp) - ref)[valid].abs().max().item()
    me = shards[rank]
    errs["ulysses_kv_heads"] = (kh - k[:, :, me.kv_start : me.kv_start + me.kv_count][:, layout_order(layout)]).abs().max().item()
    # decoding: the last two tokens (replicated) against the sequence-sharded cache
    out = distributed_kv_attention(q[:, -2:], kl, vl, pos[:, -2:], pl, sp, causal=causal)
    errs["distributed_kv"] = (out - ref[:, -2:]).abs().max().item()
    return errs


def layout_order(layout):
    return torch.cat([layout.indices(r) for r in range(layout.num_ranks)])


@pytest.mark.parametrize(
    "world,kind,weights,causal",
    [
        (2, "contiguous", None, True),
        (2, "zigzag", None, True),
        (3, "zigzag", [3, 2, 1], True),
        (3, "contiguous", [1, 2, 3], False),
    ],
)
def test_sequence_parallel_attention_matches_full(world, kind, weights, causal):
    for errs in run_dist(_sp_worker, world, kind, weights, causal):
        for name, err in errs.items():
            assert err < 1e-12, (name, err)


def _ulysses_chunked(rank, world):
    """Ulysses prefill in two chunks, the second attending to the cached first."""
    ctx = ParallelContext(1, world, 1)
    q, k, v, pos = _inputs()
    ref = attention(q, k, v, pos, pos)
    shards = partition_heads(HQ, HKV, world)
    split = 7
    outs, past = [], None
    for lo, hi in ((0, split), (split, S)):
        layout = SequenceLayout(hi - lo, world)
        sl = [layout.shard(t[:, lo:hi], rank) for t in (q, k, v, pos)]
        out, (kh, vh, ph) = ulysses_attention(*sl, ctx.sp, layout.sizes, shards, past=past)
        past = (kh, vh, ph) if past is None else tuple(torch.cat([a, b], 1) for a, b in zip(past, (kh, vh, ph)))
        outs.append(gather_sequence(out, layout, ctx.sp))
    full = torch.cat(outs, dim=1)
    return (full - ref)[pos >= 0].abs().max().item()


def test_ulysses_prefill_on_top_of_cache():
    for err in run_dist(_ulysses_chunked, 3):
        assert err < 1e-12
