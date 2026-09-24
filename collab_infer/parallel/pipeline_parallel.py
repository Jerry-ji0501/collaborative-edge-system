"""Pipeline parallelism: consecutive layers on consecutive devices.

Building blocks:

* :class:`P2PChannel` – self-describing point-to-point transport between
  stages.  Every message carries a small fixed-size header (message kind,
  dtypes and shapes), so stages need no prior knowledge of activation shapes
  and control messages such as ``STOP`` travel in-band.  Sends are
  asynchronous, which lets a stage start the next micro-batch while the
  previous activations are still on the wire.
* :func:`partition_layers` / :func:`split_sequential` – assign layers to
  stages, optionally proportional to device capability.
* :class:`PipelineRunner` – micro-batched forward execution of *any* stage
  module (e.g. the blocks of a CNN or MLP split with
  :func:`split_sequential`).  With ``M`` micro-batches all stages work
  concurrently after a fill phase of ``pp - 1`` steps (GPipe schedule).

The LLM engine (:mod:`collab_infer.engine`) uses the same channel with an
autoregressive schedule in which generated tokens loop back from the last
stage to the first.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from ..distributed.comm import CommWork, Communicator, NetworkEmulator
from .partition import split_sizes

TensorOrList = Union[torch.Tensor, Sequence[torch.Tensor]]

_DTYPES = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.complex64,
    torch.complex128,
]
_DTYPE_CODE = {dt: i for i, dt in enumerate(_DTYPES)}

HEADER_SLOTS = 64
MAX_DIMS = 8
_FIXED = 4  # kind, number of tensors, payload bytes, arrival time (us)
MAX_TENSORS = (HEADER_SLOTS - _FIXED) // (2 + MAX_DIMS)
_ALIGN = 16

MSG_DATA = 1
MSG_STOP = 2


@dataclass
class Message:
    kind: int
    tensors: List[torch.Tensor] = field(default_factory=list)

    @property
    def is_stop(self) -> bool:
        return self.kind == MSG_STOP

    @property
    def tensor(self) -> torch.Tensor:
        return self.tensors[0]


class P2PChannel:
    """Asynchronous, self-describing tensor transport over a communicator.

    Every message costs two transfers (a 512-byte header and one packed
    payload) regardless of the number of tensors it carries.  With a
    :class:`NetworkEmulator` the sender stamps each message with its modelled
    arrival time (serialising messages on the link), and the receiver waits
    until then, so pipeline overlap is modelled faithfully.
    """

    TAG_HEADER = 31
    TAG_PAYLOAD = 32

    def __init__(self, comm: Communicator) -> None:
        self.comm = comm
        self._pending: Deque[List[CommWork]] = deque()
        self._link_free_at: Dict[int, float] = defaultdict(float)

    # ---------------------------------------------------------------- sending
    def send(self, tensors: TensorOrList, dst: int, kind: int = MSG_DATA) -> None:
        tensors = [tensors] if isinstance(tensors, torch.Tensor) else list(tensors)
        if len(tensors) > MAX_TENSORS:
            raise ValueError(f"a message carries at most {MAX_TENSORS} tensors")
        header = torch.zeros(HEADER_SLOTS, dtype=torch.int64)
        header[0], header[1] = kind, len(tensors)
        segments: List[torch.Tensor] = []
        nbytes = 0
        for i, t in enumerate(tensors):
            if t.dim() > MAX_DIMS:
                raise ValueError(f"tensors with more than {MAX_DIMS} dims are not supported")
            if t.dtype not in _DTYPE_CODE:
                raise TypeError(f"unsupported dtype {t.dtype}")
            base = _FIXED + i * (2 + MAX_DIMS)
            header[base] = _DTYPE_CODE[t.dtype]
            header[base + 1] = t.dim()
            for d, extent in enumerate(t.shape):
                header[base + 2 + d] = extent
            raw = t.detach().contiguous().reshape(-1).view(torch.uint8)
            if raw.device != self.comm.comm_device:
                raw = raw.to(self.comm.comm_device)
            segments.append(raw)
            pad = (-raw.numel()) % _ALIGN
            if pad:
                segments.append(torch.zeros(pad, dtype=torch.uint8, device=raw.device))
            nbytes += raw.numel() + pad
        header[2] = nbytes
        emulator = self.comm.emulator
        if emulator is not None:
            now = time.time()
            start = max(now, self._link_free_at[dst])
            arrival = start + emulator.cost(nbytes + HEADER_SLOTS * 8, 1)
            self._link_free_at[dst] = arrival
            header[3] = int(arrival * 1e6)
        works = [self.comm.isend(header, dst, tag=self.TAG_HEADER)]
        if nbytes:
            payload = torch.cat(segments) if len(segments) > 1 else segments[0]
            works.append(self.comm.isend(payload, dst, tag=self.TAG_PAYLOAD))
        self._pending.append(works)
        self._reap()

    def send_stop(self, dst: int) -> None:
        self.send([], dst, kind=MSG_STOP)

    def _reap(self) -> None:
        while self._pending and all(w.is_completed() for w in self._pending[0]):
            for w in self._pending.popleft():
                w.wait()

    def flush(self) -> None:
        """Wait until every queued send has completed."""
        while self._pending:
            for w in self._pending.popleft():
                w.wait()

    # -------------------------------------------------------------- receiving
    def recv(self, src: int) -> Message:
        header = self.comm.recv((HEADER_SLOTS,), torch.int64, src, tag=self.TAG_HEADER, emulate=False)
        h = header.tolist()
        kind, count, nbytes, arrival_us = h[0], h[1], h[2], h[3]
        payload = None
        if nbytes:
            payload = self.comm.recv((nbytes,), torch.uint8, src, tag=self.TAG_PAYLOAD, emulate=False)
        tensors = []
        offset = 0
        for i in range(count):
            base = _FIXED + i * (2 + MAX_DIMS)
            dtype = _DTYPES[h[base]]
            shape = tuple(h[base + 2 : base + 2 + h[base + 1]])
            numel = 1
            for extent in shape:
                numel *= extent
            size = numel * torch.empty(0, dtype=dtype).element_size()
            raw = payload[offset : offset + size] if size else torch.zeros(0, dtype=torch.uint8)
            tensors.append(raw.view(dtype).reshape(shape))
            offset += size + (-size) % _ALIGN
        if arrival_us:
            NetworkEmulator.sleep_until(arrival_us / 1e6, clock=time.time)
        return Message(kind, tensors)


# --------------------------------------------------------------------------
# layer partitioning
# --------------------------------------------------------------------------
def partition_layers(num_layers: int, num_stages: int, weights: Optional[Sequence[float]] = None) -> List[int]:
    """Number of layers per stage, proportional to ``weights`` if given."""
    return split_sizes(num_layers, list(weights) if weights is not None else num_stages)


def split_sequential(
    layers: Sequence[nn.Module],
    comm: Communicator,
    layer_counts: Optional[Sequence[int]] = None,
    weights: Optional[Sequence[float]] = None,
) -> nn.Sequential:
    """The layers of ``comm.rank``'s stage as an ``nn.Sequential``."""
    layers = list(layers)
    counts = list(layer_counts) if layer_counts is not None else partition_layers(len(layers), comm.size, weights)
    if len(counts) != comm.size or sum(counts) != len(layers):
        raise ValueError(f"layer counts {counts} do not split {len(layers)} layers into {comm.size} stages")
    start = sum(counts[: comm.rank])
    return nn.Sequential(*layers[start : start + counts[comm.rank]])


# --------------------------------------------------------------------------
# generic micro-batched pipeline execution
# --------------------------------------------------------------------------
def _as_list(x: Any) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [x]
    return list(x)


def _concat(outputs: List[List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
    merged = [torch.cat(parts, dim=0) for parts in zip(*outputs)]
    return merged[0] if len(merged) == 1 else merged


class PipelineRunner:
    """Run a stage module in a micro-batched pipeline.

    Every rank of the pipeline communicator calls :meth:`forward`; only the
    first stage passes ``inputs`` (a tensor or list of tensors whose first
    dimension is the batch).  Stage outputs may be a tensor or a tuple/list of
    tensors and are forwarded as the next stage's positional arguments.
    """

    def __init__(self, stage_module: nn.Module, comm: Communicator) -> None:
        self.stage = stage_module
        self.comm = comm
        self.channel = P2PChannel(comm)

    @property
    def is_first(self) -> bool:
        return self.comm.rank == 0

    @property
    def is_last(self) -> bool:
        return self.comm.rank == self.comm.size - 1

    def _run_stage(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        return _as_list(self.stage(*tensors))

    @torch.no_grad()
    def forward(
        self,
        inputs: Optional[TensorOrList] = None,
        num_microbatches: int = 1,
        return_to: str = "last",
    ) -> Optional[Union[torch.Tensor, List[torch.Tensor]]]:
        """Pipeline ``inputs`` through all stages.

        Returns the concatenated outputs on the last stage (``return_to="last"``)
        or ships them back to the first stage (``return_to="first"``);
        ``None`` is returned on the other ranks.
        """
        if return_to not in ("last", "first"):
            raise ValueError("return_to must be 'last' or 'first'")
        n, stage = self.comm.size, self.comm.rank
        if self.is_first:
            if inputs is None:
                raise ValueError("the first stage must provide inputs")
            tensors = _as_list(inputs)
            batch = tensors[0].shape[0]
            m = max(1, min(num_microbatches, batch))
            chunks = list(zip(*[t.split(split_sizes(batch, m), dim=0) for t in tensors]))
        outputs: List[List[torch.Tensor]] = []
        if n == 1:
            outputs = [self._run_stage(list(c)) for c in chunks]
            return _concat(outputs)
        if self.is_first:
            for chunk in chunks:
                self.channel.send(self._run_stage(list(chunk)), stage + 1)
            self.channel.send_stop(stage + 1)
            if return_to == "first":
                outputs = [self.channel.recv(n - 1).tensors for _ in chunks]
        else:
            while True:
                msg = self.channel.recv(stage - 1)
                if msg.is_stop:
                    if not self.is_last:
                        self.channel.send_stop(stage + 1)
                    break
                out = self._run_stage(msg.tensors)
                if self.is_last:
                    outputs.append(out)
                    if return_to == "first":
                        self.channel.send(out, 0)
                else:
                    self.channel.send(out, stage + 1)
        self.channel.flush()
        if outputs and (self.is_last if return_to == "last" else self.is_first):
            return _concat(outputs)
        return None


__all__ = [
    "MSG_DATA",
    "MSG_STOP",
    "Message",
    "P2PChannel",
    "PipelineRunner",
    "partition_layers",
    "split_sequential",
]
