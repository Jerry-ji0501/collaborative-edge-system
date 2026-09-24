import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from collab_infer.distributed import ParallelContext
from collab_infer.parallel.tensor_parallel import (
    ColumnParallelLinear,
    ColwiseParallel,
    ParallelLMHead,
    RowParallelLinear,
    RowwiseParallel,
    VocabParallel,
    VocabParallelEmbedding,
    parallelize_module,
)
from helpers import run_dist


def _g(seed):
    return torch.Generator().manual_seed(seed)


def _layers(rank, world, weights):
    ctx = ParallelContext(1, 1, world)
    tp = ctx.tp
    x = torch.randn(2, 5, 12, generator=_g(0), dtype=torch.float64)
    w1 = torch.randn(20, 12, generator=_g(1), dtype=torch.float64)
    b1 = torch.randn(20, generator=_g(2), dtype=torch.float64)
    w2 = torch.randn(12, 20, generator=_g(3), dtype=torch.float64)
    b2 = torch.randn(12, generator=_g(4), dtype=torch.float64)
    errs = {}
    col = ColumnParallelLinear(12, 20, tp, weights=weights, bias=True, gather_output=True, dtype=torch.float64)
    col.load_full(w1, b1)
    errs["column_gather"] = (col(x) - F.linear(x, w1, b1)).abs().max().item()
    # column (sharded output) feeding row parallel = one all-reduce
    col2 = ColumnParallelLinear(12, 20, tp, weights=weights, bias=True, dtype=torch.float64)
    col2.load_full(w1, b1)
    row = RowParallelLinear(20, 12, tp, sizes=col2.out_sizes, bias=True, dtype=torch.float64)
    row.load_full(w2, b2)
    ref = F.linear(torch.relu(F.linear(x, w1, b1)), w2, b2)
    errs["column_row"] = (row(torch.relu(col2(x))) - ref).abs().max().item()
    # reduce-scatter along the sequence (Megatron SP)
    seq_sizes = [5 - (world - 1)] + [1] * (world - 1)
    start = sum(seq_sizes[:rank])
    rs = row(torch.relu(col2(x)), reduce="reduce_scatter", scatter_dim=1, scatter_sizes=seq_sizes)
    errs["row_reduce_scatter"] = (rs - ref[:, start : start + seq_sizes[rank]]).abs().max().item()
    # vocabulary parallel embedding + LM head
    table = torch.randn(50, 12, generator=_g(5), dtype=torch.float64)
    ids = torch.randint(0, 50, (2, 7), generator=_g(6))
    emb = VocabParallelEmbedding(50, 12, tp, weights=weights, dtype=torch.float64)
    emb.load_full(table)
    errs["embedding"] = (emb(ids) - F.embedding(ids, table)).abs().max().item()
    head = ParallelLMHead(12, 50, tp, weights=weights, dtype=torch.float64)
    head.load_full(table)
    errs["lm_head"] = (head(x) - x @ table.T).abs().max().item()
    return errs


@pytest.mark.parametrize("world,weights", [(2, None), (3, [3, 2, 1]), (4, [1, 1, 2, 4])])
def test_tensor_parallel_layers_match_dense(world, weights):
    for errs in run_dist(_layers, world, weights):
        for name, err in errs.items():
            assert err < 1e-12, (name, err)


class TinyBlock(nn.Module):
    """A user-defined block whose head count is derived from the weights."""

    def __init__(self, dim=16, heads=4, vocab=40):
        super().__init__()
        self.head_dim = dim // heads
        self.embed = nn.Embedding(vocab, dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.fc1 = nn.Linear(dim, 3 * dim)
        self.fc2 = nn.Linear(3 * dim, dim)

    def forward(self, ids):
        x = self.embed(ids)
        B, S, _ = x.shape
        q = self.q(x).view(B, S, -1, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, S, -1, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, S, -1, self.head_dim).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, S, -1)
        x = x + self.o(a)
        return x + self.fc2(F.gelu(self.fc1(x)))


def _generic(rank, world, weights):
    torch.manual_seed(0)
    model = TinyBlock().double()
    ids = torch.randint(0, 40, (2, 6), generator=_g(1))
    ref = model(ids)
    ctx = ParallelContext(1, 1, world)
    plan = {
        "embed": VocabParallel(),
        "q": ColwiseParallel(granularity=4),
        "k": ColwiseParallel(granularity=4),
        "v": ColwiseParallel(granularity=4),
        "o": RowwiseParallel(granularity=4),
        "fc1": ColwiseParallel(),
        "fc2": RowwiseParallel(),
    }
    parallelize_module(model, ctx.tp, plan, weights=weights)
    local_params = sum(p.numel() for p in model.parameters())
    return (model(ids) - ref).abs().max().item(), local_params


@pytest.mark.parametrize("world,weights", [(2, None), (2, [3, 1])])
def test_parallelize_arbitrary_module(world, weights):
    results = run_dist(_generic, world, weights)
    dense = sum(p.numel() for p in TinyBlock().parameters())
    for err, local in results:
        assert err < 1e-12
        assert local < dense  # every rank only stores its shard
