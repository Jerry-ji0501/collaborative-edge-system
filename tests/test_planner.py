import itertools
import random

import pytest

from collab_infer import ModelConfig, ParallelConfig
from collab_infer.planner import DeviceProfile, Workload, balance_layers, estimate, plan, search

TINYLLAMA = ModelConfig(
    vocab_size=32000, hidden_size=2048, intermediate_size=5632, num_layers=22, num_heads=32, num_kv_heads=4
)
CLUSTER = [
    DeviceProfile("jetson-orin", tflops=5.0, memory_gb=8, mem_bandwidth_gbps=100),
    DeviceProfile("laptop", tflops=2.0, memory_gb=16, mem_bandwidth_gbps=50),
    DeviceProfile("rpi5-a", tflops=0.1, memory_gb=4, mem_bandwidth_gbps=10),
    DeviceProfile("rpi5-b", tflops=0.1, memory_gb=4, mem_bandwidth_gbps=10),
]


def _bottleneck(counts, speeds, costs, extra):
    bounds = [0]
    for c in counts:
        bounds.append(bounds[-1] + c)
    return max((extra[k] + sum(costs[bounds[k] : bounds[k + 1]])) / speeds[k] for k in range(len(counts)))


def test_balance_layers_is_optimal():
    rng = random.Random(0)
    for _ in range(100):
        K = rng.randint(1, 4)
        L = rng.randint(K, 9)
        speeds = [rng.uniform(0.5, 3) for _ in range(K)]
        costs = [rng.uniform(0.5, 2) for _ in range(L)]
        extra = [rng.uniform(0, 2) for _ in range(K)]
        best = min(
            _bottleneck([b - a for a, b in zip((0,) + cuts, cuts + (L,))], speeds, costs, extra)
            for cuts in itertools.combinations(range(1, L), K - 1)
        )
        counts = balance_layers(speeds, costs, stage_extra_costs=extra)
        assert sum(counts) == L and min(counts) >= 1
        assert _bottleneck(counts, speeds, costs, extra) == pytest.approx(best)


def test_balance_layers_respects_memory():
    counts = balance_layers([10.0, 1.0], [1.0] * 10, layer_mems=[1.0] * 10, stage_mem_caps=[4.0, 100.0])
    assert counts[0] <= 4
    with pytest.raises(ValueError):
        balance_layers([1.0, 1.0], [1.0] * 10, layer_mems=[1.0] * 10, stage_mem_caps=[4.0, 4.0])


def test_plan_follows_device_capability():
    cfg = plan(TINYLLAMA, CLUSTER, pp_size=4, workload=Workload(prompt_len=256))
    assert cfg.pp_layers[0] > cfg.pp_layers[1] > cfg.pp_layers[2]
    assert sum(cfg.pp_layers) == TINYLLAMA.num_layers
    tp_cfg = plan(TINYLLAMA, CLUSTER, tp_size=4)
    assert tp_cfg.tp_weights == [5.0, 2.0, 0.1, 0.1]
    even = plan(TINYLLAMA, [DeviceProfile()] * 4, tp_size=2, sp_size=2)
    assert even.tp_weights is None and even.sp_weights is None
    for c in (cfg, tp_cfg, even):
        c.validate(TINYLLAMA.num_layers)


def test_estimate_and_search():
    uneven = estimate(TINYLLAMA, plan(TINYLLAMA, CLUSTER, tp_size=4), CLUSTER)
    even = estimate(TINYLLAMA, ParallelConfig(tp_size=4), CLUSTER)
    assert uneven.total_s < even.total_s  # capability-aware split beats the even split
    results = search(TINYLLAMA, CLUSTER, Workload(prompt_len=1024, new_tokens=64))
    assert results and results[0].estimate.feasible
    totals = [r.estimate.total_s for r in results if r.estimate.feasible]
    assert totals == sorted(totals)
    for r in results:
        assert r.config.world_size == len(CLUSTER)


def test_estimate_flags_out_of_memory():
    tiny_mem = [DeviceProfile(memory_gb=0.1)] * 2
    assert not estimate(TINYLLAMA, ParallelConfig(tp_size=2), tiny_mem).feasible
