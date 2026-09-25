import pytest
import torch
import torch.nn as nn

from collab_infer.distributed import ParallelContext
from collab_infer.parallel.pipeline_parallel import P2PChannel, PipelineRunner, split_sequential
from helpers import run_dist


def _model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)
    ).double()


def _runner(rank, world, counts, microbatches, return_to):
    ctx = ParallelContext(world, 1, 1)
    full = _model()
    x = torch.randn(10, 8, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    ref = full(x)
    stage = split_sequential(list(full), ctx.pp, layer_counts=counts)
    out = PipelineRunner(stage, ctx.pp).forward(x if rank == 0 else None, microbatches, return_to=return_to)
    return None if out is None else (out - ref).abs().max().item()


@pytest.mark.parametrize(
    "world,counts,microbatches,return_to",
    [(2, None, 1, "last"), (3, [2, 4, 1], 4, "last"), (4, [1, 2, 2, 2], 3, "first"), (2, [6, 1], 20, "first")],
)
def test_pipeline_runner_matches_sequential(world, counts, microbatches, return_to):
    results = run_dist(_runner, world, counts, microbatches, return_to)
    owner = world - 1 if return_to == "last" else 0
    for rank, err in enumerate(results):
        if rank == owner:
            assert err is not None and err < 1e-12
        else:
            assert err is None


def _channel(rank, world):
    ctx = ParallelContext(world, 1, 1)
    ch = P2PChannel(ctx.pp)
    payload = [
        torch.arange(6).view(2, 3),
        torch.ones(2, 2, dtype=torch.bfloat16),
        torch.tensor([True, False, True]),
        torch.zeros(0, 5),
        torch.randn(3, 1, 2, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(0)),
    ]
    if rank == 0:
        for _ in range(3):  # several messages in flight before the receiver reads
            ch.send(payload, 1)
        ch.send_stop(1)
        ch.flush()
        return None
    msgs = [ch.recv(0) for _ in range(4)]
    ok = all(
        all(torch.equal(a, b) and a.dtype == b.dtype for a, b in zip(m.tensors, payload)) for m in msgs[:3]
    )
    return ok and msgs[3].is_stop and not msgs[0].is_stop


def test_channel_roundtrip_and_stop():
    assert run_dist(_channel, 2)[1] is True


def _encoded_channel(rank, world, comm_dtype):
    ctx = ParallelContext(world, 1, 1, comm_dtype=comm_dtype)
    ch = P2PChannel(ctx.pp)
    g = torch.Generator().manual_seed(0)
    xs = [torch.randn(2, 5, 64, generator=g) * 3, torch.randn(3, 7, generator=g, dtype=torch.float64), torch.arange(6), torch.zeros(0, 4)]
    if rank == 0:
        ch.send(xs, 1)
        ch.send(xs[0], 1, compress=False)
        ch.flush()
        return None
    got, raw = ch.recv(0).tensors, ch.recv(0).tensor
    errors = [float((a.double() - b.double()).abs().max() / max(b.double().abs().max().item(), 1e-12)) if b.numel() else 0.0 for a, b in zip(got, xs)]
    same_meta = all(a.dtype == b.dtype and a.shape == b.shape for a, b in zip(got, xs))
    return errors, same_meta, bool(torch.equal(got[2], xs[2])), bool(torch.equal(raw, xs[0]))


@pytest.mark.parametrize("comm_dtype,tol", [("float16", 1e-3), ("bfloat16", 5e-3), ("int8", 1e-2)])
def test_channel_compression_roundtrip(comm_dtype, tol):
    errors, same_meta, ints_exact, raw_exact = run_dist(_encoded_channel, 2, comm_dtype)[1]
    assert same_meta and ints_exact and raw_exact
    assert max(errors) < tol
