"""Quickstart: run one model on a simulated multi-device cluster.

Spawns ``pp * sp * tp`` processes on this machine (gloo backend), runs
generation with the chosen strategy and checks the result against a
single-device run of the same model.

    python examples/quickstart.py                      # pp=2, sp=1, tp=2
    python examples/quickstart.py --pp 2 --sp 2 --tp 2 --sp-mode ulysses
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from collab_infer import (  # noqa: E402
    CollabEngine,
    ModelConfig,
    ParallelConfig,
    launch_local,
    random_state_dict,
)

MODEL = ModelConfig.tiny(num_layers=4)
PROMPTS = [[1, 5, 9, 14, 3, 7], [2, 4, 8], [11, 12, 13, 14, 15, 16, 17, 18]]


def run(rank: int, world_size: int, parallel: dict):
    # Every rank builds the same (random) full checkpoint here for simplicity;
    # with a real model pass a checkpoint path and each rank reads only its shard.
    weights = random_state_dict(MODEL, seed=0)
    engine = CollabEngine(MODEL, ParallelConfig(**parallel), weights)
    print(f"[rank {rank}] {engine.ctx.describe()} layers={engine.model.layer_ids}", flush=True)
    result = engine.generate(PROMPTS if rank == 0 else None, max_new_tokens=8)
    return result.tokens, result.stats, engine.comm_stats.total_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--sp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--sp-mode", default="ring", choices=["ring", "ulysses"])
    args = parser.parse_args()
    parallel = dict(pp_size=args.pp, sp_size=args.sp, tp_size=args.tp, sp_mode=args.sp_mode)
    world = args.pp * args.sp * args.tp

    results = launch_local(run, world, args=(parallel,), return_results=True)
    tokens, stats, _ = results[0]
    reference = CollabEngine(MODEL, ParallelConfig(), random_state_dict(MODEL, seed=0)).generate(PROMPTS, 8)

    print(f"\n{ParallelConfig(**parallel).describe()} on {world} processes")
    for prompt, out in zip(PROMPTS, tokens):
        print(f"  {prompt} -> {out}")
    print(f"  time {stats['total_s']:.3f}s, bytes sent per rank: {[r[2] for r in results]}")
    print("  matches single-device run:", tokens == reference.tokens)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
