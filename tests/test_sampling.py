"""Vocabulary-parallel sampling: layout independence and statistical exactness."""

import pytest
import torch

from collab_infer.distributed import ParallelContext
from collab_infer.engine.sampling import Sampler, SamplingParams, splitmix64, uniform_noise
from helpers import run_dist

MODES = {
    "greedy": SamplingParams(),
    "temperature": SamplingParams(temperature=0.8, seed=11),
    "top_k": SamplingParams(temperature=1.2, top_k=5, seed=12),
    "top_p": SamplingParams(temperature=0.7, top_p=0.8, seed=13),
    "top_k_top_p": SamplingParams(temperature=1.0, top_k=8, top_p=0.6, seed=14),
}


def _logits(kind):
    g = torch.Generator().manual_seed(0)
    if kind == "flat":  # nucleus far larger than the per-rank candidates -> exact fallback
        return torch.randn(6, 1000, generator=g, dtype=torch.float64) * 0.01
    return torch.randn(6, 1000, generator=g, dtype=torch.float64) * 3


def _split_worker(rank, world, kind):
    ctx = ParallelContext(1, 1, world)
    logits = _logits(kind)
    sizes = [500, 300, 200][:world] if world == 3 else [1000 // world] * world
    offset = sum(sizes[:rank])
    local = logits[:, offset : offset + sizes[rank]]
    seq_ids = torch.arange(10, 16)
    out = {}
    for name, params in MODES.items():
        out[name] = [Sampler(params).select(local, offset, seq_ids, step, comm=ctx.tp).tolist() for step in range(3)]
    return out


@pytest.mark.parametrize("kind", ["peaked", "flat"])
def test_vocab_parallel_selection_matches_single_device(kind):
    logits = _logits(kind)
    seq_ids = torch.arange(10, 16)
    single = {
        name: [Sampler(params).select(logits, 0, seq_ids, step).tolist() for step in range(3)]
        for name, params in MODES.items()
    }
    for result in run_dist(_split_worker, 3, kind):
        assert result == single


def test_greedy_is_argmax_with_first_index_ties():
    logits = torch.tensor([[1.0, 3.0, 3.0, 0.0], [2.0, 2.0, 2.0, 2.0]])
    assert Sampler(SamplingParams())(logits).tolist() == [1, 0]


@pytest.mark.parametrize(
    "params,expected",
    [
        (SamplingParams(temperature=1.0, seed=1), torch.softmax(torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0]), 0)),
        (SamplingParams(temperature=2.0, seed=2), torch.softmax(torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0]) / 2, 0)),
        (SamplingParams(temperature=1.0, top_k=2, seed=3), torch.softmax(torch.tensor([2.0, 1.0]), 0)),
        # softmax -> [.558, .205, .124, .075, .028]: the nucleus of 0.7 is the first two tokens
        (SamplingParams(temperature=1.0, top_p=0.7, seed=4), torch.softmax(torch.tensor([2.0, 1.0]), 0)),
    ],
    ids=["t1", "t2", "top_k", "top_p"],
)
def test_sampling_frequencies_match_distribution(params, expected):
    n = 40_000
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]], dtype=torch.float64).expand(n, 5)
    tokens = Sampler(params).select(logits, 0, torch.arange(n), step=0)
    freq = torch.bincount(tokens, minlength=5).double() / n
    expected = torch.cat([expected.double(), torch.zeros(5 - expected.numel(), dtype=torch.float64)])
    assert torch.allclose(freq, expected, atol=0.01), (freq, expected)


def test_noise_is_keyed_by_sequence_step_and_token():
    ids = torch.arange(50)
    a = uniform_noise(7, torch.tensor([0, 1]), 3, ids)
    assert torch.equal(a, uniform_noise(7, torch.tensor([0, 1]), 3, ids))
    assert not torch.equal(a[0], a[1])  # sequences differ
    assert not torch.equal(a, uniform_noise(7, torch.tensor([0, 1]), 4, ids))  # steps differ
    # slicing the vocabulary does not change a token's noise
    assert torch.equal(a[:, 20:30], uniform_noise(7, torch.tensor([0, 1]), 3, ids[20:30]))
    assert ((a > 0) & (a < 1)).all()


def test_splitmix64_matches_reference():
    m64 = (1 << 64) - 1

    def ref(x):
        x = (x + 0x9E3779B97F4A7C15) & m64
        z = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & m64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m64
        z ^= z >> 31
        return z - (1 << 64) if z >= (1 << 63) else z

    xs = [0, 1, 12345, (1 << 62) + 7, -5, 987654321987654321]
    assert splitmix64(torch.tensor(xs)).tolist() == [ref(x & m64) for x in xs]
