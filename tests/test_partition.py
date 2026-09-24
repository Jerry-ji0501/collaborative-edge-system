import pytest
import torch

from collab_infer.parallel.partition import (
    SequenceLayout,
    default_kv_map,
    offsets_of,
    partition_heads,
    split_sizes,
)


def test_split_sizes_even_and_uneven():
    assert split_sizes(12, 3) == [4, 4, 4]
    assert split_sizes(10, 3) == [4, 3, 3]
    assert split_sizes(12, [2, 1, 1]) == [6, 3, 3]
    assert split_sizes(64, [0.7, 0.3], granularity=8) == [40, 24]
    assert sum(split_sizes(1000, [5, 2, 0.1, 0.1])) == 1000


def test_split_sizes_guarantees_minimum_and_validates():
    assert split_sizes(4, [100, 1, 1, 1]) == [1, 1, 1, 1]
    assert min(split_sizes(10, [1000, 1])) >= 1
    assert split_sizes(3, 5, min_units=0) == [1, 1, 1, 0, 0]
    with pytest.raises(ValueError):
        split_sizes(2, 3)
    with pytest.raises(ValueError):
        split_sizes(10, [1, -1])
    with pytest.raises(ValueError):
        split_sizes(10, 2, granularity=3)


def test_offsets():
    assert offsets_of([3, 1, 2]) == [0, 3, 4]


@pytest.mark.parametrize(
    "hq,hkv,weights",
    [(8, 8, 2), (8, 4, 3), (8, 2, 4), (8, 1, 2), (6, 2, [3, 1]), (32, 8, [5, 2, 1]), (12, 4, 5)],
)
def test_partition_heads_is_consistent(hq, hkv, weights):
    shards = partition_heads(hq, hkv, weights)
    n = weights if isinstance(weights, int) else len(weights)
    assert len(shards) == n
    assert sum(s.q_count for s in shards) == hq
    kv_of = default_kv_map(hq, hkv)
    covered = []
    for s in shards:
        assert s.q_count >= 1 and s.kv_count >= 1
        for i, local_kv in enumerate(s.kv_map):
            # every local query head must map to the KV head it uses globally
            assert s.kv_start + local_kv == kv_of[s.q_start + i]
        covered.extend(range(s.q_start, s.q_start + s.q_count))
    assert covered == list(range(hq))


def test_partition_heads_avoids_kv_replication_when_possible():
    shards = partition_heads(8, 4, 2)
    assert [(s.kv_start, s.kv_count) for s in shards] == [(0, 2), (2, 2)]
    mqa = partition_heads(8, 1, 2)  # a single KV head must be replicated
    assert all(s.kv_count == 1 and s.kv_start == 0 for s in mqa)


@pytest.mark.parametrize("kind", ["contiguous", "zigzag"])
@pytest.mark.parametrize("seq_len,weights", [(11, [2, 1, 1]), (8, None), (5, [1, 1, 1, 1, 1]), (16, [1, 3])])
def test_sequence_layout_roundtrip(kind, seq_len, weights):
    n = len(weights) if weights else 2
    layout = SequenceLayout(seq_len, n, weights, kind)
    x = torch.arange(seq_len).view(1, seq_len, 1).repeat(2, 1, 3)
    parts = [layout.shard(x, r) for r in range(n)]
    assert [p.shape[1] for p in parts] == layout.sizes
    assert torch.equal(layout.unshard(parts), x)
    all_idx = torch.cat([layout.indices(r) for r in range(n)])
    assert sorted(all_idx.tolist()) == list(range(seq_len))
    for index in range(seq_len):
        rank, local = layout.owner_of(index)
        assert layout.indices(rank)[local].item() == index


def test_zigzag_balances_causal_work():
    layout = SequenceLayout(16, 4, kind="zigzag")
    # causal work of a token ~ its index; zigzag gives equal sums per rank
    work = [int(layout.indices(r).sum()) + layout.sizes[r] for r in range(4)]
    assert len(set(work)) == 1
