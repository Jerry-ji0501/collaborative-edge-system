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
_PER_TENSOR = 3 + MAX_DIMS  # dtype, ndim, encoding, dims
MAX_TENSORS = (HEADER_SLOTS - _FIXED) // _PER_TENSOR
_ALIGN = 16

# wire encodings of floating point tensors (ParallelConfig.comm_dtype)
ENC_RAW, ENC_FP16, ENC_BF16, ENC_INT8 = 0, 1, 2, 3
_ENCODING = {"float16": ENC_FP16, "bfloat16": ENC_BF16, "int8": ENC_INT8}

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
    @staticmethod
    def _raw(t: torch.Tensor) -> torch.Tensor:
        return t.detach().contiguous().reshape(-1).view(torch.uint8)

    def _encode(self, t: torch.Tensor, encoding: int) -> List[torch.Tensor]:
        """Byte segments of ``t`` in the given wire encoding."""
        if encoding == ENC_FP16:  # saturate instead of overflowing to inf
            return [self._raw(t.clamp(-65504.0, 65504.0).to(torch.float16))]
        if encoding == ENC_BF16:
            return [self._raw(t.to(torch.bfloat16))]
        if encoding == ENC_INT8:  # symmetric per-row (last dimension) quantisation
            rows = t.detach().reshape(-1, t.shape[-1] if t.dim() else 1).to(torch.float32)
            scale = rows.abs().amax(dim=1, keepdim=True) / 127.0
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            q = torch.round(rows / scale).clamp_(-127, 127).to(torch.int8)
            return [self._raw(q), self._raw(scale)]
        return [self._raw(t)]

    def send(self, tensors: TensorOrList, dst: int, kind: int = MSG_DATA, compress: bool = True) -> None:
        """Queue ``tensors`` for ``dst``.

        With ``compress`` (the default) floating point tensors travel in the
        communicator's ``comm_dtype`` encoding when one is configured.
        """
        tensors = [tensors] if isinstance(tensors, torch.Tensor) else list(tensors)
        if len(tensors) > MAX_TENSORS:
            raise ValueError(f"a message carries at most {MAX_TENSORS} tensors")
        header = torch.zeros(HEADER_SLOTS, dtype=torch.int64)
        header[0], header[1] = kind, len(tensors)
        wire_encoding = _ENCODING.get(self.comm.comm_dtype, ENC_RAW) if compress else ENC_RAW
        segments: List[torch.Tensor] = []
        nbytes = 0
        for i, t in enumerate(tensors):
            if t.dim() > MAX_DIMS:
                raise ValueError(f"tensors with more than {MAX_DIMS} dims are not supported")
            if t.dtype not in _DTYPE_CODE:
                raise TypeError(f"unsupported dtype {t.dtype}")
            encoding = wire_encoding if t.is_floating_point() and t.element_size() > 1 and t.numel() else ENC_RAW
            if encoding in (ENC_FP16, ENC_BF16) and t.element_size() <= 2:
                encoding = ENC_RAW  # already narrow
            base = _FIXED + i * _PER_TENSOR
            header[base] = _DTYPE_CODE[t.dtype]
            header[base + 1] = t.dim()
            header[base + 2] = encoding
            for d, extent in enumerate(t.shape):
                header[base + 3 + d] = extent
            for raw in self._encode(t, encoding):
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

        def take(nbytes: int, dtype: torch.dtype) -> torch.Tensor:
            nonlocal offset
            raw = payload[offset : offset + nbytes] if nbytes else torch.zeros(0, dtype=torch.uint8)
            offset += nbytes + (-nbytes) % _ALIGN
            return raw.view(dtype)

        for i in range(count):
            base = _FIXED + i * _PER_TENSOR
            dtype = _DTYPES[h[base]]
            encoding = h[base + 2]
            shape = tuple(h[base + 3 : base + 3 + h[base + 1]])
            numel = 1
            for extent in shape:
                numel *= extent
            if encoding == ENC_INT8:  # only used for non-empty tensors
                cols = shape[-1] if shape else 1
                rows = numel // cols
                q = take(numel, torch.int8)
                scale = take(rows * 4, torch.float32)
                t = (q.reshape(rows, cols).to(torch.float32) * scale.reshape(rows, 1)).to(dtype).reshape(shape)
            elif encoding in (ENC_FP16, ENC_BF16):
                wire = torch.float16 if encoding == ENC_FP16 else torch.bfloat16
                t = take(numel * 2, wire).to(dtype).reshape(shape)
            else:
                t = take(numel * torch.empty(0, dtype=dtype).element_size(), dtype).reshape(shape)
            tensors.append(t)
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
