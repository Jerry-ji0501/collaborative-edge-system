"""Repeatable prefill / decode benchmark for collab_infer.

Runs a fixed model (default: hidden 1024, 8 layers, GQA 16/4 heads, 32k vocab,
random weights) under one or more parallel strategies on local processes and
reports, per strategy, the median over ``--repeats`` timed runs of

* prefill latency (time to first token: embedding, all layers over the prompt,
  LM head on the last position and sampling),
* decode latency per step and decode throughput (tokens/s over the batch).

Everything that affects timing is pinned so runs are comparable: seeds, prompt
tokens, greedy sampling, and the number of CPU threads per simulated device
(``--threads-per-device``, default 1, so adding devices adds compute the way a
real edge cluster does).  Results can be saved as JSON together with the
environment (torch version, CPU, commit) and compared against a saved
baseline to spot regressions:

    python benchmarks/bench_inference.py
    python benchmarks/bench_inference.py --strategies single,TP2,PP2,SP2 --prompt-len 512
    python benchmarks/bench_inference.py --output results.json
    python benchmarks/bench_inference.py --compare benchmarks/baselines/cpu-4core.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from collab_infer import CollabEngine, ModelConfig, NetworkConfig, ParallelConfig, launch_local, random_state_dict  # noqa: E402

DEFAULT_MODEL = dict(
    vocab_size=32000,
    hidden_size=1024,
    intermediate_size=2816,
    num_layers=8,
    num_heads=16,
    num_kv_heads=4,
    max_position_embeddings=8192,
)

# label -> ParallelConfig kwargs
STRATEGIES: Dict[str, Dict[str, Any]] = {
    "single": dict(),
    "TP2": dict(tp_size=2),
    "PP2": dict(pp_size=2),
    "SP2": dict(sp_size=2, sp_layout="zigzag"),
    "SP2-ulysses": dict(sp_size=2, sp_mode="ulysses"),
    "TP4": dict(tp_size=4),
    "PP4": dict(pp_size=4),
    "SP4": dict(sp_size=4, sp_layout="zigzag"),
    "PP2xTP2": dict(pp_size=2, tp_size=2),
    "PP2xSP2": dict(pp_size=2, sp_size=2, sp_layout="zigzag"),
    "SP2xTP2": dict(sp_size=2, tp_size=2, sp_layout="zigzag"),
}

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def world_size(parallel: Dict[str, Any]) -> int:
    return parallel.get("tp_size", 1) * parallel.get("pp_size", 1) * parallel.get("sp_size", 1)


def worker(rank: int, world: int, model_dict, parallel, network, workload) -> Optional[Dict[str, Any]]:
    model = ModelConfig(**model_dict)
    dtype = _DTYPES[workload["dtype"]]
    config = ParallelConfig(**parallel, network=NetworkConfig(**network) if network else None)
    engine = CollabEngine(model, config, random_state_dict(model, seed=0, dtype=dtype), dtype=dtype)
    g = torch.Generator().manual_seed(1234)
    prompts = torch.randint(0, model.vocab_size, (workload["batch"], workload["prompt_len"]), generator=g).tolist()
    new_tokens = workload["new_tokens"]
    for _ in range(workload["warmup"]):
        engine.generate(prompts, max_new_tokens=new_tokens)
    runs = []
    for _ in range(workload["repeats"]):
        engine.comm_stats.reset()
        result = engine.generate(prompts, max_new_tokens=new_tokens)
        s = result.stats
        steps = max(new_tokens - 1, 1)
        runs.append(
            dict(
                prefill_s=s["ttft_s"],
                decode_ms_per_step=(s["total_s"] - s["ttft_s"]) / steps * 1e3,
                decode_tok_per_s=s["decode_tokens_per_s"],
                total_s=s["total_s"],
                sent_mb=engine.comm_stats.total_bytes / 1e6,
                tokens=result.tokens,
            )
        )
    return dict(
        runs=runs,
        param_mb=engine.model.parameter_bytes() / 1e6,
        kv_mb=engine.last_kv_cache_bytes / 1e6,
    )


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Median / min / max over repeats (timings from rank 0, memory and traffic max over ranks)."""
    runs = rows[0]["runs"]
    out: Dict[str, Any] = {}
    for key in ("prefill_s", "decode_ms_per_step", "decode_tok_per_s", "total_s"):
        vals = [r[key] for r in runs]
        out[key] = dict(median=statistics.median(vals), min=min(vals), max=max(vals))
    out["max_sent_mb_per_run"] = max(r["runs"][-1]["sent_mb"] for r in rows)
    out["max_param_mb"] = max(r["param_mb"] for r in rows)
    out["max_kv_mb"] = max(r["kv_mb"] for r in rows)
    out["tokens"] = runs[-1]["tokens"]
    return out


def environment(threads_per_device: int) -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return dict(
        torch=torch.__version__,
        python=platform.python_version(),
        cpu=cpu,
        cpu_count=os.cpu_count(),
        threads_per_device=threads_per_device,
        commit=commit,
    )


def fmt_median(stat: Dict[str, float], scale: float = 1.0, digits: int = 1) -> str:
    return f"{stat['median'] * scale:.{digits}f}"


def print_table(results: Dict[str, Dict[str, Any]], baseline: Optional[Dict[str, Any]]) -> None:
    head = f"{'strategy':<13}{'devices':>8}{'prefill ms':>12}{'decode ms/step':>16}{'decode tok/s':>14}{'sent MB':>10}{'param MB':>10}{'KV MB':>8}"
    if baseline:
        head += f"{'Δprefill':>10}{'Δdecode':>10}"
    print(head)
    base_rows = (baseline or {}).get("results", {})
    for label, r in results.items():
        line = (
            f"{label:<13}{r['devices']:>8}{fmt_median(r['prefill_s'], 1e3):>12}"
            f"{fmt_median(r['decode_ms_per_step'], 1.0, 2):>16}{fmt_median(r['decode_tok_per_s']):>14}"
            f"{r['max_sent_mb_per_run']:>10.2f}{r['max_param_mb']:>10.1f}{r['max_kv_mb']:>8.2f}"
        )
        b = base_rows.get(label)
        if baseline:
            if b:
                dp = r["prefill_s"]["median"] / b["prefill_s"]["median"] - 1
                dd = r["decode_ms_per_step"]["median"] / b["decode_ms_per_step"]["median"] - 1
                line += f"{dp:>+10.1%}{dd:>+10.1%}"
            else:
                line += f"{'-':>10}{'-':>10}"
        print(line)
    note = "; Δ = change vs baseline median, negative is faster" if baseline else ""
    print(f"\n(median over repeats; prefill = time to first token{note})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--strategies", default="single,TP2,PP2,SP2", help=f"comma separated, from: {','.join(STRATEGIES)}")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--dtype", default="float32", choices=sorted(_DTYPES))
    p.add_argument("--threads-per-device", type=int, default=1, help="CPU threads per simulated device")
    p.add_argument("--bandwidth-mbps", type=float, help="emulated link bandwidth")
    p.add_argument("--latency-ms", type=float, default=0.0, help="emulated link latency")
    p.add_argument("--model-json", help="JSON dict of ModelConfig overrides for the default model")
    p.add_argument("--output", help="write results (and environment) to this JSON file")
    p.add_argument("--compare", help="baseline JSON (from --output) to compare against")
    args = p.parse_args()

    model = dict(DEFAULT_MODEL, **(json.loads(args.model_json) if args.model_json else {}))
    labels = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in labels if s not in STRATEGIES]
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; choose from {list(STRATEGIES)}")
    network = None
    if args.bandwidth_mbps or args.latency_ms:
        network = dict(bandwidth_mbps=args.bandwidth_mbps, latency_ms=args.latency_ms)
    workload = dict(
        batch=args.batch,
        prompt_len=args.prompt_len,
        new_tokens=args.new_tokens,
        repeats=args.repeats,
        warmup=args.warmup,
        dtype=args.dtype,
    )
    env = environment(args.threads_per_device)
    print(f"model    {model}")
    print(f"workload {workload}, network: {network or 'no emulation'}")
    print(f"env      {env}\n")

    torch.manual_seed(0)
    results: Dict[str, Dict[str, Any]] = {}
    reference: Optional[Tuple[str, Any]] = None
    for label in labels:
        parallel = STRATEGIES[label]
        world = world_size(parallel)
        rows = launch_local(
            worker,
            world,
            args=(model, parallel, network, workload),
            threads_per_rank=args.threads_per_device,
            return_results=True,
        )
        summary = summarize(rows)
        summary["devices"] = world
        summary["parallel"] = parallel
        if reference is None:
            reference = (label, summary["tokens"])
        elif summary["tokens"] != reference[1]:
            print(f"warning: {label} generated different tokens than {reference[0]}")
        results[label] = summary
        print(f"  {label}: done", flush=True)
    print()

    baseline = None
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            baseline = json.load(f)
        if baseline.get("model") != model or baseline.get("workload") != workload:
            print("warning: baseline was recorded with a different model or workload; deltas are not comparable\n")
    print_table(results, baseline)

    if args.output:
        for r in results.values():
            r.pop("tokens", None)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(dict(model=model, workload=workload, network=network, environment=env, results=results), f, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
