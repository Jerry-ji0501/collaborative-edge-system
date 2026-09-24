"""Group-bound communication primitives.

:class:`Communicator` wraps one ``torch.distributed`` process group and
exposes exactly the collectives the parallel strategies need, with the
properties an edge cluster requires:

* **uneven sizes everywhere** – heterogeneous devices receive uneven shards,
  so ``all_gather``/``reduce_scatter``/``all_to_all`` accept per-rank sizes;
* **backend independence** – works with ``gloo`` (CPU / mixed clusters),
  ``nccl`` and ``mpi``; when a backend lacks an operation a mathematically
  equivalent fallback built from supported primitives is used;
* **mixed devices** – compute may run on CUDA while communication runs over
  gloo on CPU (e.g. a Jetson GPU talking to a Raspberry Pi);
* **accounting and emulation** – every call is recorded in :class:`CommStats`
  and can be slowed down by a :class:`NetworkEmulator` to mimic Wi-Fi/LAN
  links when experimenting on a single host.

Ranks passed to the methods (``src``/``dst``) are *group-local* indices.
"""

from __future__ import annotations

import os
import time
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist

from ..config import NetworkConfig

_REDUCE_OPS = {
    "sum": dist.ReduceOp.SUM,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
    "prod": dist.ReduceOp.PRODUCT,
}

# P2P tags used internally (user code may use any other tag).
TAG_ALL_TO_ALL = 91


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _even_split(total: int, parts: int) -> List[int]:
    return [total // parts + (1 if i < total % parts else 0) for i in range(parts)]


def _is_unsupported(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        key in msg
        for key in ("not supported", "does not support", "unsupported", "not implemented", "invalid tensor size")
    )


# --------------------------------------------------------------------------
# statistics and network emulation
# --------------------------------------------------------------------------
class CommStats:
    """Per-process record of communication calls, bytes and time."""

    def __init__(self) -> None:
        self.calls: Dict[str, int] = defaultdict(int)
        self.bytes: Dict[str, int] = defaultdict(int)
        self.seconds: Dict[str, float] = defaultdict(float)

    def record(self, op: str, nbytes: int, seconds: float) -> None:
        self.calls[op] += 1
        self.bytes[op] += int(nbytes)
        self.seconds[op] += float(seconds)

    @property
    def total_calls(self) -> int:
        return sum(self.calls.values())

    @property
    def total_bytes(self) -> int:
        """Bytes sent by this process (receives are not double counted)."""
        return sum(b for op, b in self.bytes.items() if not op.endswith(".recv"))

    @property
    def total_seconds(self) -> float:
        return sum(self.seconds.values())

    def reset(self) -> None:
        self.calls.clear()
        self.bytes.clear()
        self.seconds.clear()

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            op: {"calls": self.calls[op], "bytes": self.bytes[op], "seconds": self.seconds[op]}
            for op in sorted(self.calls)
        }

    def format(self) -> str:
        lines = [f"{'op':<24}{'calls':>8}{'MB':>12}{'seconds':>10}"]
        for op, row in self.as_dict().items():
            lines.append(
                f"{op:<24}{int(row['calls']):>8}{row['bytes'] / 1e6:>12.3f}{row['seconds']:>10.3f}"
            )
        lines.append(
            f"{'total':<24}{self.total_calls:>8}{self.total_bytes / 1e6:>12.3f}{self.total_seconds:>10.3f}"
        )
        return "\n".join(lines)


class NetworkEmulator:
    """Adds latency/bandwidth delays so one host can mimic an edge network.

    The model is deliberately simple: an operation that moves ``nbytes`` in
    ``steps`` sequential messages costs ``steps * latency + nbytes / bandwidth``.
    """

    def __init__(self, bandwidth_mbps: Optional[float] = None, latency_ms: float = 0.0) -> None:
        self.bandwidth_mbps = bandwidth_mbps
        self.latency_ms = latency_ms
        self._bytes_per_s = None if bandwidth_mbps is None else bandwidth_mbps * 1e6 / 8.0
        self._latency_s = latency_ms / 1e3

    @classmethod
    def from_config(cls, cfg: Optional[NetworkConfig]) -> Optional["NetworkEmulator"]:
        if cfg is None or (cfg.bandwidth_mbps is None and not cfg.latency_ms):
            return None
        return cls(cfg.bandwidth_mbps, cfg.latency_ms)

    def cost(self, nbytes: int, steps: int = 1) -> float:
        t = max(steps, 0) * self._latency_s
        if self._bytes_per_s:
            t += nbytes / self._bytes_per_s
        return t

    @staticmethod
    def sleep_until(deadline: float, clock: Callable[[], float] = time.perf_counter) -> None:
        delay = deadline - clock()
        if delay > 0:
            time.sleep(delay)

    def __repr__(self) -> str:
        return f"NetworkEmulator(bandwidth_mbps={self.bandwidth_mbps}, latency_ms={self.latency_ms})"


class CommWork:
    """Handle of an asynchronous operation; :meth:`wait` returns its result."""

    def __init__(
        self,
        work: Any = None,
        finalize: Optional[Callable[[], Any]] = None,
        deadline: Optional[float] = None,
        keepalive: Any = None,
    ) -> None:
        self._work = work
        self._finalize = finalize
        self._deadline = deadline
        self._keepalive = keepalive  # buffers that must outlive the transfer
        self._done = False
        self._result: Any = None

    def is_completed(self) -> bool:
        return self._done or self._work is None or self._work.is_completed()

    def wait(self) -> Any:
        if not self._done:
            if self._work is not None:
                self._work.wait()
            if self._finalize is not None:
                self._result = self._finalize()
            if self._deadline is not None:
                NetworkEmulator.sleep_until(self._deadline)
            self._done = True
            self._keepalive = None
        return self._result


# --------------------------------------------------------------------------
# communicator
# --------------------------------------------------------------------------
class Communicator:
    """Collective and point-to-point operations over one process group.

    Args:
        ranks: global ranks of the group members, in group order.
        group: the ``torch.distributed`` group handle (``None`` if size 1).
        name: prefix used in :class:`CommStats` (e.g. ``"tp"``).
        device: compute device of this process; results are returned there.
        stats: shared statistics object.
        emulator: optional network emulator.
        force_fallback: always use the portable fallback implementations of
            ``all_to_all``/``reduce_scatter`` (useful when the ranks of an
            edge cluster run different PyTorch builds).  Also enabled by the
            environment variable ``COLLAB_INFER_COMM_FALLBACK=1``.
    """

    def __init__(
        self,
        ranks: Sequence[int],
        group: Any = None,
        *,
        name: str = "world",
        device: Optional[torch.device] = None,
        stats: Optional[CommStats] = None,
        emulator: Optional[NetworkEmulator] = None,
        force_fallback: bool = False,
    ) -> None:
        self.ranks = [int(r) for r in ranks]
        self.size = len(self.ranks)
        self.global_rank = dist.get_rank() if dist_ready() else 0
        if self.global_rank not in self.ranks:
            raise ValueError(f"rank {self.global_rank} is not a member of group {self.ranks}")
        self.rank = self.ranks.index(self.global_rank)
        if self.size > 1 and group is None:
            raise ValueError("a process group is required for groups with more than one rank")
        self.group = group
        self.name = name
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.backend = dist.get_backend(group) if self.size > 1 else None
        # NCCL communicates device tensors; every other backend uses host memory.
        self.comm_device = self.device if self.backend == "nccl" else torch.device("cpu")
        self.stats = stats if stats is not None else CommStats()
        self.emulator = emulator
        force_fallback = force_fallback or os.environ.get("COLLAB_INFER_COMM_FALLBACK", "0") == "1"
        self._native = {"all_to_all": not force_fallback, "reduce_scatter": not force_fallback}

    # ------------------------------------------------------------- utilities
    def __repr__(self) -> str:
        return (
            f"Communicator(name={self.name!r}, rank={self.rank}/{self.size}, "
            f"ranks={self.ranks}, backend={self.backend})"
        )

    @property
    def is_trivial(self) -> bool:
        return self.size == 1

    def _to_comm(self, t: torch.Tensor) -> torch.Tensor:
        if t.device != self.comm_device:
            t = t.to(self.comm_device)
        return t.contiguous()

    @staticmethod
    def _to(t: torch.Tensor, device: torch.device) -> torch.Tensor:
        return t if t.device == device else t.to(device)

    def _timed(self, op: str, nbytes: int, steps: int, fn: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        result = fn()
        if self.emulator is not None:
            NetworkEmulator.sleep_until(start + self.emulator.cost(nbytes, steps))
        self.stats.record(f"{self.name}.{op}", nbytes, time.perf_counter() - start)
        return result

    def _native_or_fallback(self, key: str, native: Callable[[], Any], fallback: Callable[[], Any]) -> Any:
        if self._native[key]:
            try:
                return native()
            except (RuntimeError, NotImplementedError, ValueError) as exc:
                if not _is_unsupported(exc):
                    raise
                self._native[key] = False
                warnings.warn(
                    f"backend {self.backend!r} does not support {key}; using the portable fallback ({exc})"
                )
        return fallback()

    # ----------------------------------------------------------- collectives
    def all_reduce(self, t: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """Reduce ``t`` over the group.  ``t`` may be modified in place."""
        if self.size == 1:
            return t
        buf = self._to_comm(t)
        n = self.size

        def run() -> torch.Tensor:
            dist.all_reduce(buf, op=_REDUCE_OPS[op], group=self.group)
            return buf

        out = self._timed("all_reduce", 2 * (n - 1) * _nbytes(buf) // n, 2 * (n - 1), run)
        return self._to(out, t.device)

    def all_gather_list(
        self, t: torch.Tensor, dim: int = 0, sizes: Optional[Sequence[int]] = None
    ) -> List[torch.Tensor]:
        """Gather ``t`` from every rank; ``sizes[i]`` is rank ``i``'s extent along ``dim``."""
        if self.size == 1:
            return [t]
        dim = dim % t.dim()
        sizes = [int(s) for s in sizes] if sizes is not None else [t.shape[dim]] * self.size
        if len(sizes) != self.size or sizes[self.rank] != t.shape[dim]:
            raise ValueError(f"all_gather sizes {sizes} inconsistent with local shape {tuple(t.shape)}")
        longest = max(sizes)
        buf = t
        if t.shape[dim] < longest:  # gloo needs equal sizes: pad and trim
            pad_shape = list(t.shape)
            pad_shape[dim] = longest - t.shape[dim]
            buf = torch.cat([t, t.new_zeros(pad_shape)], dim=dim)
        buf = self._to_comm(buf)
        outs = [torch.empty_like(buf) for _ in range(self.size)]

        def run() -> None:
            dist.all_gather(outs, buf, group=self.group)

        self._timed("all_gather", _nbytes(buf) * (self.size - 1), self.size - 1, run)
        return [self._to(o.narrow(dim, 0, s), t.device) for o, s in zip(outs, sizes)]

    def all_gather(
        self, t: torch.Tensor, dim: int = 0, sizes: Optional[Sequence[int]] = None
    ) -> torch.Tensor:
        """Concatenate ``t`` from all ranks along ``dim`` (uneven sizes allowed)."""
        if self.size == 1:
            return t
        return torch.cat(self.all_gather_list(t, dim, sizes), dim=dim)

    def reduce_scatter(
        self, t: torch.Tensor, dim: int = 0, sizes: Optional[Sequence[int]] = None
    ) -> torch.Tensor:
        """Sum ``t`` over ranks and return this rank's chunk along ``dim``."""
        if self.size == 1:
            return t
        dim = dim % t.dim()
        total = t.shape[dim]
        sizes = [int(s) for s in sizes] if sizes is not None else _even_split(total, self.size)
        if len(sizes) != self.size or sum(sizes) != total:
            raise ValueError(f"reduce_scatter sizes {sizes} do not match extent {total}")
        start = sum(sizes[: self.rank])
        n = self.size

        def native() -> torch.Tensor:
            chunks = [self._to_comm(c) for c in t.split(sizes, dim=dim)]
            longest = max(sizes)
            if self.backend != "gloo" and min(sizes) != longest:
                # not every backend accepts uneven chunks: pad, reduce, trim
                padded = []
                for c in chunks:
                    if c.shape[dim] < longest:
                        pad_shape = list(c.shape)
                        pad_shape[dim] = longest - c.shape[dim]
                        c = torch.cat([c, c.new_zeros(pad_shape)], dim=dim)
                    padded.append(c)
                out = torch.empty_like(padded[self.rank])
                dist.reduce_scatter(out, padded, group=self.group)
                return out.narrow(dim, 0, sizes[self.rank]).contiguous()
            out = torch.empty_like(chunks[self.rank])
            dist.reduce_scatter(out, chunks, group=self.group)
            return out

        def fallback() -> torch.Tensor:
            full = self._to_comm(t.clone())
            dist.all_reduce(full, group=self.group)
            return full.narrow(dim, start, sizes[self.rank]).contiguous()

        out = self._timed(
            "reduce_scatter",
            (n - 1) * _nbytes(t) // n,
            n - 1,
            lambda: self._native_or_fallback("reduce_scatter", native, fallback),
        )
        return self._to(out, t.device)

    def all_to_all(
        self, inputs: Sequence[torch.Tensor], output_shapes: Sequence[Sequence[int]]
    ) -> List[torch.Tensor]:
        """Send ``inputs[j]`` to rank ``j``; receive tensors of ``output_shapes``.

        Shapes may differ per peer (uneven head/sequence splits).  All tensors
        must share one dtype.
        """
        if len(inputs) != self.size or len(output_shapes) != self.size:
            raise ValueError("all_to_all needs one input and one output shape per rank")
        if self.size == 1:
            return [inputs[0]]
        device = inputs[self.rank].device
        dtype = inputs[self.rank].dtype
        send = [self._to_comm(x) for x in inputs]
        out_numels = [int(torch.Size(s).numel()) for s in output_shapes]
        sent_bytes = sum(_nbytes(x) for j, x in enumerate(send) if j != self.rank)

        def native() -> List[torch.Tensor]:
            flat_in = torch.cat([x.reshape(-1) for x in send])
            flat_out = torch.empty(sum(out_numels), dtype=dtype, device=self.comm_device)
            dist.all_to_all_single(
                flat_out,
                flat_in,
                output_split_sizes=out_numels,
                input_split_sizes=[x.numel() for x in send],
                group=self.group,
            )
            return [p.view(tuple(s)) for p, s in zip(flat_out.split(out_numels), output_shapes)]

        def fallback() -> List[torch.Tensor]:
            outs = [torch.empty(tuple(s), dtype=dtype, device=self.comm_device) for s in output_shapes]
            works = []
            for j in range(self.size):
                if j == self.rank:
                    outs[j].copy_(send[j].view(tuple(output_shapes[j])))
                    continue
                works.append(dist.isend(send[j], self.ranks[j], group=self.group, tag=TAG_ALL_TO_ALL))
                works.append(dist.irecv(outs[j], self.ranks[j], group=self.group, tag=TAG_ALL_TO_ALL))
            for w in works:
                w.wait()
            return outs

        outs = self._timed(
            "all_to_all",
            sent_bytes,
            1,
            lambda: self._native_or_fallback("all_to_all", native, fallback),
        )
        return [self._to(o, device) for o in outs]

    def broadcast(self, t: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast from group-local rank ``src``; other ranks pass a buffer."""
        if self.size == 1:
            return t
        buf = self._to_comm(t)
        steps = max(1, (self.size - 1).bit_length())

        def run() -> None:
            dist.broadcast(buf, src=self.ranks[src], group=self.group)

        self._timed("broadcast", _nbytes(buf), steps, run)
        return self._to(buf, t.device)

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast a picklable Python object (used for small control data)."""
        if self.size == 1:
            return obj
        box = [obj]
        kwargs = {"device": self.comm_device} if self.backend == "nccl" else {}
        self._timed(
            "broadcast_object",
            0,
            1,
            lambda: dist.broadcast_object_list(box, src=self.ranks[src], group=self.group, **kwargs),
        )
        return box[0]

    def barrier(self) -> None:
        if self.size > 1:
            dist.barrier(group=self.group)

    # --------------------------------------------------------- point-to-point
    def isend(self, t: torch.Tensor, dst: int, tag: int = 0) -> CommWork:
        """Asynchronously send ``t`` to group-local rank ``dst``."""
        buf = self._to_comm(t)
        start = time.perf_counter()
        work = dist.isend(buf, self.ranks[dst], group=self.group, tag=tag)
        self.stats.record(f"{self.name}.send", _nbytes(buf), time.perf_counter() - start)
        return CommWork(work, keepalive=buf)

    def irecv(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        *,
        emulate: bool = True,
    ) -> CommWork:
        """Post an asynchronous receive; ``wait()`` returns the tensor.

        With emulation the data is considered available ``cost`` seconds after
        the receive was posted (both peers are assumed to be in lock-step, as
        in ring attention).
        """
        buf = torch.empty(tuple(shape), dtype=dtype, device=self.comm_device)
        post = time.perf_counter()
        work = dist.irecv(buf, self.ranks[src], group=self.group, tag=tag)
        deadline = None
        if emulate and self.emulator is not None:
            deadline = post + self.emulator.cost(_nbytes(buf), 1)
        name = f"{self.name}.recv"

        def finalize() -> torch.Tensor:
            self.stats.record(name, _nbytes(buf), time.perf_counter() - post)
            return self._to(buf, self.device)

        return CommWork(work, finalize=finalize, deadline=deadline)

    def send(self, t: torch.Tensor, dst: int, tag: int = 0) -> None:
        self.isend(t, dst, tag).wait()

    def recv(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        *,
        emulate: bool = True,
    ) -> torch.Tensor:
        """Blocking receive of a tensor with known shape and dtype."""
        buf = torch.empty(tuple(shape), dtype=dtype, device=self.comm_device)
        start = time.perf_counter()
        dist.recv(buf, self.ranks[src], group=self.group, tag=tag)
        if emulate and self.emulator is not None:
            time.sleep(self.emulator.cost(_nbytes(buf), 1))
        self.stats.record(f"{self.name}.recv", _nbytes(buf), time.perf_counter() - start)
        return self._to(buf, self.device)


__all__ = ["CommStats", "CommWork", "Communicator", "NetworkEmulator", "dist_ready"]
