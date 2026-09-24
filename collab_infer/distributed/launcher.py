"""Process-group initialisation and local multi-process launching.

Two ways to run a collaborative deployment:

* **Real devices** – start one process per device (``torchrun`` or plain
  ``python`` with ``RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT``) and
  call :func:`init_distributed` at the top of the script.
* **Simulation on one host** – :func:`launch_local` spawns ``world_size``
  processes, initialises a gloo process group among them and runs a function
  in each; handy for development, tests and strategy studies (optionally
  combined with :class:`~collab_infer.config.NetworkConfig` emulation).
"""

from __future__ import annotations

import os
import pickle
import socket
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def resolve_device(device: Optional[str] = None, backend: Optional[str] = None, local_rank: int = 0) -> torch.device:
    """Pick the compute device of this process.

    ``None`` means CPU unless the backend is NCCL; ``"auto"`` uses CUDA when
    available.  Mixed clusters (some ranks with GPUs, some without) should use
    the gloo backend with ``device="auto"``.
    """
    if device is None or device == "auto":
        use_cuda = torch.cuda.is_available() and (backend == "nccl" or device == "auto")
        if use_cuda:
            dev = torch.device("cuda", local_rank % torch.cuda.device_count())
            torch.cuda.set_device(dev)
            return dev
        if backend == "nccl":
            raise RuntimeError("the nccl backend requires CUDA devices")
        return torch.device("cpu")
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    return dev


def init_distributed(
    backend: Optional[str] = None,
    *,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    master_addr: Optional[str] = None,
    master_port: Optional[int] = None,
    init_method: Optional[str] = None,
    device: Optional[str] = None,
    timeout_s: float = 1800.0,
) -> torch.device:
    """Initialise the default process group from arguments or environment.

    Reads the ``torchrun`` variables ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``,
    ``MASTER_ADDR`` and ``MASTER_PORT`` when arguments are omitted.  The
    backend defaults to ``$COLLAB_BACKEND`` or ``gloo`` (portable across CPU,
    GPU and ARM edge devices).  Returns the compute device of this process.

    On multi-homed edge devices set ``GLOO_SOCKET_IFNAME`` (e.g. ``wlan0``) so
    gloo binds to the interface that reaches the other devices.
    """
    rank = int(os.environ.get("RANK", 0)) if rank is None else rank
    world_size = int(os.environ.get("WORLD_SIZE", 1)) if world_size is None else world_size
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = backend or os.environ.get("COLLAB_BACKEND", "gloo")
    if world_size > 1 and not dist.is_initialized():
        if init_method is None:
            addr = master_addr or os.environ.get("MASTER_ADDR", "127.0.0.1")
            port = master_port or int(os.environ.get("MASTER_PORT", 29500))
            init_method = f"tcp://{addr}:{port}"
        dist.init_process_group(
            backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_s),
        )
    return resolve_device(device, backend, local_rank)


def _local_entry(
    rank: int,
    fn: Callable[..., Any],
    world_size: int,
    port: int,
    backend: str,
    args: Sequence[Any],
    threads: Sequence[int],
    queue: Any,
    timeout_s: float,
    env: Optional[Dict[str, str]],
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    if env:
        os.environ.update(env)
    torch.set_num_threads(max(1, threads[rank]))
    dist.init_process_group(
        backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_s),
    )
    try:
        result = fn(rank, world_size, *args)
        if queue is not None:
            queue.put((rank, pickle.dumps(result)))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def launch_local(
    fn: Callable[..., Any],
    world_size: int,
    args: Sequence[Any] = (),
    *,
    backend: str = "gloo",
    threads_per_rank: Optional[Union[int, Sequence[int]]] = None,
    return_results: bool = False,
    timeout_s: float = 600.0,
    env: Optional[Dict[str, str]] = None,
) -> Optional[List[Any]]:
    """Run ``fn(rank, world_size, *args)`` in ``world_size`` local processes.

    ``fn`` must be importable (defined at module top level).  Exceptions in any
    process are re-raised in the caller.  With ``return_results=True`` the
    per-rank return values (which must be picklable) are returned as a list.
    ``threads_per_rank`` may be a list, which gives the simulated devices
    different compute capabilities (a heterogeneous cluster on one host).
    """
    port = find_free_port()
    if threads_per_rank is None:
        threads_per_rank = max(1, (os.cpu_count() or 1) // world_size)
    if isinstance(threads_per_rank, int):
        threads = [threads_per_rank] * world_size
    else:
        threads = [int(t) for t in threads_per_rank]
        if len(threads) != world_size:
            raise ValueError("threads_per_rank needs one entry per rank")
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue() if return_results else None
    # read by libtorch at import time, so it must be inherited by the children
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    context = mp.start_processes(
        _local_entry,
        args=(fn, world_size, port, backend, tuple(args), threads, queue, timeout_s, env),
        nprocs=world_size,
        join=False,
        start_method="spawn",
    )
    results: Dict[int, Any] = {}

    def drain() -> None:
        while queue is not None and not queue.empty():
            r, payload = queue.get()
            results[r] = pickle.loads(payload)

    while not context.join(timeout=0.05):
        drain()
    drain()
    if not return_results:
        return None
    missing = [r for r in range(world_size) if r not in results]
    if missing:
        raise RuntimeError(f"ranks {missing} did not return a result")
    return [results[r] for r in range(world_size)]


__all__ = ["find_free_port", "init_distributed", "launch_local", "resolve_device"]
