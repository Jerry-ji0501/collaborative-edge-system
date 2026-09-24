"""Shared helpers for the multi-process tests."""

from collab_infer.distributed import launch_local


def run_dist(fn, world_size, *args, **kwargs):
    """Run ``fn(rank, world_size, *args)`` on ``world_size`` gloo processes."""
    return launch_local(fn, world_size, args=args, return_results=True, threads_per_rank=1, timeout_s=300, **kwargs)
