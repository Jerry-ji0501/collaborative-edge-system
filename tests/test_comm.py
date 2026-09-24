import time

import pytest
import torch

from collab_infer.config import NetworkConfig
from collab_infer.distributed import ParallelContext
from helpers import run_dist


def _collectives(rank, world, fallback):
    ctx = ParallelContext(1, 1, world, force_comm_fallback=fallback)
    c = ctx.tp
    sizes = [r + 1 for r in range(world)]
    out = {}
    out["all_reduce"] = c.all_reduce(torch.full((3,), float(rank + 1))).tolist()
    out["all_reduce_max"] = c.all_reduce(torch.tensor([float(rank)]), op="max").item()
    out["all_gather"] = c.all_gather(torch.full((2, rank + 1), float(rank)), dim=1, sizes=sizes).tolist()
    full = torch.arange(2 * sum(sizes), dtype=torch.float64).view(2, -1) * (rank + 1)
    out["reduce_scatter"] = c.reduce_scatter(full, dim=1, sizes=sizes).tolist()
    ins = [torch.full((j + 1, rank + 1), float(rank * 10 + j)) for j in range(world)]
    out["all_to_all"] = [t.tolist() for t in c.all_to_all(ins, [(rank + 1, j + 1) for j in range(world)])]
    out["broadcast"] = c.broadcast(torch.arange(4.0) if rank == 1 else torch.zeros(4), src=1).tolist()
    out["object"] = c.broadcast_object({"a": [1, 2]} if rank == 0 else None)
    w_send = c.isend(torch.tensor([rank]), (rank + 1) % world)
    out["ring"] = c.irecv((1,), torch.int64, (rank - 1) % world).wait().item()
    w_send.wait()
    out["calls"] = ctx.stats.total_calls
    return out


@pytest.mark.parametrize("fallback", [False, True], ids=["native", "fallback"])
def test_collectives_with_uneven_sizes(fallback):
    world = 3
    results = run_dist(_collectives, world, fallback)
    total = torch.arange(2 * 6, dtype=torch.float64).view(2, -1) * 6  # sum over ranks of (r+1)
    for rank, out in enumerate(results):
        assert out["all_reduce"] == [6.0] * 3
        assert out["all_reduce_max"] == 2.0
        assert out["all_gather"] == [[0.0, 1.0, 1.0, 2.0, 2.0, 2.0]] * 2
        start = sum(range(1, rank + 1))
        assert out["reduce_scatter"] == total[:, start : start + rank + 1].tolist()
        for j, t in enumerate(out["all_to_all"]):
            assert torch.equal(torch.tensor(t), torch.full((rank + 1, j + 1), float(j * 10 + rank)))
        assert out["broadcast"] == [0.0, 1.0, 2.0, 3.0]
        assert out["object"] == {"a": [1, 2]}
        assert out["ring"] == (rank - 1) % world
        assert out["calls"] > 0


def _emulated(rank, world):
    ctx = ParallelContext(1, 1, world, network=NetworkConfig(bandwidth_mbps=8, latency_ms=20))
    ctx.tp.barrier()
    t0 = time.perf_counter()
    ctx.tp.all_reduce(torch.zeros(12_500))  # 50 kB
    elapsed = time.perf_counter() - t0
    return elapsed, ctx.stats.as_dict()


def test_network_emulation_delays_and_records():
    results = run_dist(_emulated, 2)
    # ring all-reduce of 50 kB over 2 ranks: 2 steps * 20 ms + 50 kB / 1 MB/s = 0.09 s
    for elapsed, stats in results:
        assert elapsed >= 0.085
        assert stats["tp.all_reduce"]["calls"] == 1
        assert stats["tp.all_reduce"]["bytes"] == 50_000
