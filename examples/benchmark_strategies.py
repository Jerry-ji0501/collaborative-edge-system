"""Compare parallel strategies on a simulated edge cluster.

Every strategy runs on the same number of local processes; links can be
throttled with the network emulator to mimic LAN / Wi-Fi.  The table shows
time-to-first-token (prefill), decoding speed, and per-device memory and
traffic, which illustrates the trade-offs:

* TP moves activations twice per layer (sensitive to latency/bandwidth),
* PP only sends activations between stages (cheap links suffice),
* SP splits long prompts and the KV cache (memory per device drops).

    python examples/benchmark_strategies.py --world 4 --prompt-len 256
    python examples/benchmark_strategies.py --bandwidth-mbps 100 --latency-ms 5
    python examples/benchmark_strategies.py --bandwidth-mbps 100 --latency-ms 5 --comm-dtype float16
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from collab_infer import CollabEngine, ModelConfig, NetworkConfig, ParallelConfig, launch_local, random_state_dict  # noqa: E402


def strategies(world: int):
    """(label, ParallelConfig kwargs) for a given number of devices."""
    out = [
        (f"TP{world}", dict(tp_size=world)),
        (f"PP{world}", dict(pp_size=world)),
        (f"SP{world}-ring", dict(sp_size=world, sp_layout="zigzag")),
        (f"SP{world}-ulysses", dict(sp_size=world, sp_mode="ulysses")),
    ]
    if world == 4:
        out += [
            ("PP2xTP2", dict(pp_size=2, tp_size=2)),
            ("SP2xTP2", dict(sp_size=2, tp_size=2)),
            ("PP2xSP2", dict(pp_size=2, sp_size=2)),
        ]
    return out


def worker(rank, world, model_dict, parallel, network, workload):
    model = ModelConfig(**model_dict)
    config = ParallelConfig(**parallel, network=NetworkConfig(**network) if network else None)
    engine = CollabEngine(model, config, random_state_dict(model, seed=0))
    g = torch.Generator().manual_seed(0)
    prompts = torch.randint(0, model.vocab_size, (workload["batch"], workload["prompt_len"]), generator=g).tolist()
    engine.generate(prompts, max_new_tokens=2)  # warm-up
    engine.comm_stats.reset()
    result = engine.generate(prompts, max_new_tokens=workload["new_tokens"])
    return dict(
        stats=result.stats,
        sent_mb=engine.comm_stats.total_bytes / 1e6,
        kv_mb=engine.last_kv_cache_bytes / 1e6,
        param_mb=engine.model.parameter_bytes() / 1e6,
        tokens=result.tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--bandwidth-mbps", type=float)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--only", help="comma separated strategy labels to run")
    parser.add_argument("--comm-dtype", choices=["float16", "bfloat16", "int8"], help="compress activation traffic")
    parser.add_argument("--prefill-chunk", type=int, help="pipelined prefill chunk size (0 = off, default auto)")
    args = parser.parse_args()
    extra = {k: v for k, v in (("comm_dtype", args.comm_dtype), ("prefill_chunk", args.prefill_chunk)) if v is not None}

    model = dict(vocab_size=2048, hidden_size=256, intermediate_size=688, num_layers=8, num_heads=8, num_kv_heads=4)
    network = None
    if args.bandwidth_mbps or args.latency_ms:
        network = dict(bandwidth_mbps=args.bandwidth_mbps, latency_ms=args.latency_ms)
    workload = dict(batch=args.batch, prompt_len=args.prompt_len, new_tokens=args.new_tokens)
    net = f"{args.bandwidth_mbps or 'unlimited'} Mbps / {args.latency_ms} ms" if network else "no emulation"
    print(f"model {model}\nworkload {workload}, network: {net}, options: {extra or 'defaults'}\n")
    print(f"{'strategy':<14}{'TTFT s':>9}{'decode tok/s':>14}{'total s':>9}{'max sent MB':>13}{'max KV MB':>11}{'max param MB':>14}")
    baseline = None
    selected = set(args.only.split(",")) if args.only else None
    for label, parallel in strategies(args.world):
        if selected and label not in selected:
            continue
        rows = launch_local(worker, args.world, args=(model, {**parallel, **extra}, network, workload), return_results=True)
        s = rows[0]["stats"]
        baseline = baseline or rows[0]["tokens"]
        same = "" if rows[0]["tokens"] == baseline else "  (tokens differ!)"
        print(
            f"{label:<14}{s['ttft_s']:>9.3f}{s['decode_tokens_per_s']:>14.1f}{s['total_s']:>9.3f}"
            f"{max(r['sent_mb'] for r in rows):>13.2f}{max(r['kv_mb'] for r in rows):>11.2f}"
            f"{max(r['param_mb'] for r in rows):>14.2f}{same}"
        )


if __name__ == "__main__":
    main()
