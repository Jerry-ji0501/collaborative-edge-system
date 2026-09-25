"""Configuration objects describing *how* a model is spread over devices.

A deployment is a 3-D device mesh ``pp_size x sp_size x tp_size``:

* **pipeline parallelism (PP)** splits the layers into consecutive stages,
* **sequence parallelism (SP)** splits the tokens of a sequence inside a stage
  (ring attention or DeepSpeed-Ulysses style all-to-all attention),
* **tensor parallelism (TP)** splits every weight matrix inside a stage
  (Megatron-LM style column/row parallel layers).

Global rank ``r`` is mapped to mesh coordinates as
``r = pp_rank * (sp_size * tp_size) + sp_rank * tp_size + tp_rank`` so that
TP peers (which communicate the most) have adjacent ranks.

Edge clusters are usually heterogeneous, therefore every dimension accepts
optional *weights* that make the split proportional to device capability
(e.g. ``tp_weights=[2, 1]`` gives the first TP rank two thirds of the heads).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Union

SP_MODES = ("ring", "ulysses")
SP_LAYOUTS = ("contiguous", "zigzag")
COMM_DTYPES = ("float16", "bfloat16", "int8")

Weights = Sequence[float]


@dataclass
class NetworkConfig:
    """Emulated link characteristics, used to study strategies on one host.

    When set, every communication call takes at least
    ``latency_ms * steps + bytes / bandwidth`` of wall-clock time.  Leave it
    unset on a real multi-device cluster, where the physical network already
    imposes these costs.
    """

    bandwidth_mbps: Optional[float] = None
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.bandwidth_mbps is not None and self.bandwidth_mbps <= 0:
            raise ValueError("bandwidth_mbps must be positive")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")


def _is_nested(weights: Any) -> bool:
    return (
        weights is not None
        and len(weights) > 0
        and isinstance(weights[0], (list, tuple))
    )


@dataclass
class ParallelConfig:
    """Parallelization strategy of a collaborative deployment.

    Attributes:
        tp_size: tensor-parallel degree (devices sharing every layer).
        pp_size: pipeline-parallel degree (number of stages).
        sp_size: sequence-parallel degree (devices sharing the tokens).
        sp_mode: ``"ring"`` (ring attention, KV blocks circulate; the KV cache
            stays sharded along the sequence) or ``"ulysses"`` (all-to-all
            switches between sequence- and head-sharding; the KV cache is
            sharded along heads).
        sp_layout: ``"contiguous"`` or ``"zigzag"`` token assignment.  Zigzag
            balances causal attention work across ring-attention ranks.
        megatron_sp: additionally apply Megatron-LM sequence parallelism inside
            each TP group (norms/residuals run on 1/tp of the tokens and the
            all-reduces become reduce-scatter + all-gather).
        sp_decode_split: while decoding, let the SP ranks of a stage share
            the MLP and attention output projection like extra TP ranks (they
            already hold those weights) instead of all recomputing the whole
            layer.  Lossless, but it adds one or two collectives per layer
            (fused with the TP reductions when ``tp_size > 1``), so it only
            pays off when decode compute dominates communication latency
            (large models, fast links).  Off by default.
        tp_weights: relative capability of TP ranks; either one list shared by
            all stages or one list per pipeline stage.
        sp_weights: relative capability of SP ranks (uneven token split).
        pp_layers: number of decoder layers per pipeline stage.
        num_microbatches: micro-batches used to keep the pipeline busy.
            Defaults to ``pp_size``.
        prefill_chunk: split prompts into chunks of this many tokens that flow
            through the pipeline stages concurrently (lower time-to-first-token
            when a single request occupies the pipeline).  ``None`` picks a
            size automatically when ``pp_size > 1``; ``0`` disables chunking.
        attn_kv_block: if set, attention is computed in key blocks of this
            size (bounded memory for long contexts).
        comm_dtype: lossy compression of activations on the network for
            bandwidth-limited links: ``"float16"``/``"bfloat16"`` (half the
            bytes of float32) or ``"int8"`` (row-quantised pipeline
            activations, bfloat16 collectives).  ``None`` sends activations
            unmodified.
        network: optional emulated network characteristics.
    """

    tp_size: int = 1
    pp_size: int = 1
    sp_size: int = 1
    sp_mode: str = "ring"
    sp_layout: str = "contiguous"
    megatron_sp: bool = False
    sp_decode_split: bool = False
    tp_weights: Optional[Union[Weights, Sequence[Weights]]] = None
    sp_weights: Optional[Weights] = None
    pp_layers: Optional[Sequence[int]] = None
    num_microbatches: Optional[int] = None
    prefill_chunk: Optional[int] = None
    attn_kv_block: Optional[int] = None
    comm_dtype: Optional[str] = None
    network: Optional[NetworkConfig] = None

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size * self.sp_size

    # ------------------------------------------------------------------ checks
    def validate(self, num_layers: Optional[int] = None) -> "ParallelConfig":
        for name in ("tp_size", "pp_size", "sp_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.sp_mode not in SP_MODES:
            raise ValueError(f"sp_mode must be one of {SP_MODES}, got {self.sp_mode!r}")
        if self.sp_layout not in SP_LAYOUTS:
            raise ValueError(f"sp_layout must be one of {SP_LAYOUTS}, got {self.sp_layout!r}")
        if self.tp_weights is not None:
            per_stage = self.tp_weights if _is_nested(self.tp_weights) else [self.tp_weights]
            if _is_nested(self.tp_weights) and len(per_stage) != self.pp_size:
                raise ValueError("per-stage tp_weights must have pp_size entries")
            for w in per_stage:
                _check_weights(w, self.tp_size, "tp_weights")
        if self.sp_weights is not None:
            _check_weights(self.sp_weights, self.sp_size, "sp_weights")
        if self.pp_layers is not None:
            if len(self.pp_layers) != self.pp_size:
                raise ValueError(
                    f"pp_layers has {len(self.pp_layers)} entries but pp_size={self.pp_size}"
                )
            if any(int(n) < 0 for n in self.pp_layers):
                raise ValueError("pp_layers entries must be non-negative")
            if num_layers is not None and sum(self.pp_layers) != num_layers:
                raise ValueError(
                    f"pp_layers sums to {sum(self.pp_layers)} but the model has {num_layers} layers"
                )
        elif num_layers is not None and num_layers < self.pp_size:
            raise ValueError(f"cannot split {num_layers} layers into {self.pp_size} stages")
        if self.num_microbatches is not None and self.num_microbatches < 1:
            raise ValueError("num_microbatches must be >= 1")
        if self.prefill_chunk is not None and self.prefill_chunk < 0:
            raise ValueError("prefill_chunk must be >= 0")
        if self.attn_kv_block is not None and self.attn_kv_block < 1:
            raise ValueError("attn_kv_block must be >= 1")
        if self.comm_dtype is not None and self.comm_dtype not in COMM_DTYPES:
            raise ValueError(f"comm_dtype must be one of {COMM_DTYPES}, got {self.comm_dtype!r}")
        return self

    # ----------------------------------------------------------------- helpers
    def stage_tp_weights(self, stage: int) -> Optional[List[float]]:
        """TP weights used by pipeline stage ``stage`` (``None`` = even split)."""
        if self.tp_weights is None:
            return None
        if _is_nested(self.tp_weights):
            return [float(x) for x in self.tp_weights[stage]]
        return [float(x) for x in self.tp_weights]

    def describe(self) -> str:
        parts = [f"pp={self.pp_size}", f"sp={self.sp_size}", f"tp={self.tp_size}"]
        if self.sp_size > 1:
            parts.append(f"sp_mode={self.sp_mode}")
            if self.sp_layout != "contiguous":
                parts.append(f"layout={self.sp_layout}")
        if self.megatron_sp and self.tp_size > 1:
            parts.append("megatron_sp")
        if self.comm_dtype:
            parts.append(f"comm_dtype={self.comm_dtype}")
        if self.prefill_chunk is not None:
            parts.append(f"prefill_chunk={self.prefill_chunk}")
        if self.sp_decode_split and self.sp_size > 1:
            parts.append("sp_decode_split")
        if self.tp_weights is not None:
            parts.append(f"tp_weights={self.tp_weights}")
        if self.sp_weights is not None:
            parts.append(f"sp_weights={list(self.sp_weights)}")
        if self.pp_layers is not None:
            parts.append(f"pp_layers={list(self.pp_layers)}")
        return "ParallelConfig(" + ", ".join(parts) + ")"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParallelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown ParallelConfig fields: {sorted(unknown)}")
        data = dict(data)
        if isinstance(data.get("network"), dict):
            data["network"] = NetworkConfig(**data["network"])
        return cls(**data)


def _check_weights(weights: Weights, expected: int, name: str) -> None:
    if len(weights) != expected:
        raise ValueError(f"{name} has {len(weights)} entries, expected {expected}")
    if any(float(w) <= 0 for w in weights):
        raise ValueError(f"{name} entries must be positive, got {list(weights)}")


__all__ = ["COMM_DTYPES", "NetworkConfig", "ParallelConfig", "SP_LAYOUTS", "SP_MODES"]

