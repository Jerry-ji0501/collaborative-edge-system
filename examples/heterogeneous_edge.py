"""Capability-aware partitioning on a heterogeneous (simulated) edge cluster.

Two processes on this machine play a fast device (3 CPU threads) and a slow
device (1 CPU thread).  The script

1. profiles both "devices" with a short matmul benchmark,
2. asks the planner for capability-proportional TP weights / PP layer splits,
3. runs tensor and pipeline parallel inference with the even and the planned
   partitions and compares their latency.

With an even split the slow device is the bottleneck; the planned split gives
it proportionally less work.  Results are identical in both cases.

    python examples/heterogeneous_edge.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from collab_infer import CollabEngine, ModelConfig, ParallelConfig, launch_local, random_state_dict  # noqa: E402
from collab_infer.planner import DeviceProfile, Workload, plan  # noqa: E402

THREADS = [3, 1]  # rank 0 = fast device, rank 1 = slow device
MODEL = ModelConfig(
    vocab_size=4096, hidden_size=1024, intermediate_size=2816, num_layers=8, num_heads=16, num_kv_heads=4
)
PROMPT_LEN, NEW_TOKENS = 512, 4


def profile(rank, world):
    """Sustained matmul throughput of this 'device' in TFLOP/s."""
    a, b = torch.randn(512, 1024), torch.randn(1024, 2816)
    for _ in range(3):
        a @ b
    start = time.perf_counter()
    for _ in range(10):
        a @ b
    return 2 * 512 * 1024 * 2816 * 10 / (time.perf_counter() - start) / 1e12


def run(rank, world, parallel):
    engine = CollabEngine(MODEL, ParallelConfig(**parallel), random_state_dict(MODEL, seed=0))
    prompt = torch.randint(0, MODEL.vocab_size, (1, PROMPT_LEN), generator=torch.Generator().manual_seed(0)).tolist()
    engine.generate(prompt, max_new_tokens=1)  # warm-up
    result = engine.generate(prompt, max_new_tokens=NEW_TOKENS)
    return result.stats, result.tokens


def main() -> None:
    speeds = launch_local(profile, 2, threads_per_rank=THREADS, return_results=True)
    devices = [DeviceProfile(f"device{r} ({t} threads)", tflops=s) for r, (t, s) in enumerate(zip(THREADS, speeds))]
    for d in devices:
        print(f"{d.name}: {d.tflops * 1e3:.0f} GFLOP/s")
    workload = Workload(prompt_len=PROMPT_LEN, new_tokens=NEW_TOKENS, dtype_bytes=4)

    cases = []
    for kind in ("tp", "pp"):
        sizes = dict(tp_size=2) if kind == "tp" else dict(pp_size=2)
        planned = plan(MODEL, devices, workload=workload, **sizes)
        cases.append((f"{kind.upper()}2 even", sizes))
        cases.append((f"{kind.upper()}2 planned", {k: v for k, v in planned.to_dict().items() if v is not None and k != "network"}))

    print(f"\n{'case':<16}{'TTFT s':>9}{'total s':>9}  partition")
    reference = None
    for label, parallel in cases:
        stats, tokens = launch_local(run, 2, args=(parallel,), threads_per_rank=THREADS, return_results=True)[0]
        reference = reference or tokens
        detail = parallel.get("tp_weights") or parallel.get("pp_layers") or "even"
        if isinstance(detail, list) and detail and isinstance(detail[0], float):
            detail = [round(x, 3) for x in detail]
        flag = "" if tokens == reference else "  (tokens differ!)"
        print(f"{label:<16}{stats['ttft_s']:>9.3f}{stats['total_s']:>9.3f}  {detail}{flag}")


if __name__ == "__main__":
    main()
