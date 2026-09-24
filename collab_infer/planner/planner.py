"""Heterogeneity-aware parallelisation planning for edge clusters.

Given a model, a list of device profiles and a workload, the planner

1. derives capability-proportional **TP/SP split weights** and a
   **pipeline layer partition** (dynamic programming that minimises the
   slowest stage subject to per-device memory limits), see :func:`plan`;
2. **estimates** prefill latency, per-token decode latency, communication
   volume and per-device memory with an analytic cost model, see
   :func:`estimate`;
3. **searches** all ``pp x sp x tp`` factorisations (and SP algorithms) and
   ranks them, see :func:`search`.

The cost model is intentionally simple (roofline compute/memory time, alpha-
beta communication cost, ring attention overlapped with compute) — it is
meant to rank strategies and pick uneven splits, not to predict absolute
latency precisely.  Profile devices with realistic *sustained* numbers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..config import ParallelConfig
from ..models.config import ModelConfig
from ..parallel.partition import split_sizes


@dataclass
class DeviceProfile:
    """Capabilities of one edge device.

    Attributes:
        tflops: sustained matmul throughput in TFLOP/s for the model dtype.
        memory_gb: memory available for weights and KV cache.
        mem_bandwidth_gbps: memory bandwidth in GB/s (bounds decoding).
        bandwidth_mbps: network bandwidth to the other devices (Mbit/s).
        latency_ms: one-way network latency.
    """

    name: str = "device"
    tflops: float = 1.0
    memory_gb: float = 8.0
    mem_bandwidth_gbps: float = 50.0
    bandwidth_mbps: float = 1000.0
    latency_ms: float = 1.0


@dataclass
class Workload:
    batch_size: int = 1
    prompt_len: int = 512
    new_tokens: int = 64
    dtype_bytes: int = 2


@dataclass
class Estimate:
    prefill_s: float
    decode_step_s: float
    total_s: float
    comm_bytes: float
    device_memory_gb: List[float]
    feasible: bool
    breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def decode_tokens_per_s(self) -> float:
        return 1.0 / self.decode_step_s if self.decode_step_s > 0 else float("inf")


@dataclass
class PlanResult:
    config: ParallelConfig
    estimate: Estimate

    def __repr__(self) -> str:
        e = self.estimate
        return (
            f"{self.config.describe()}: total={e.total_s:.3f}s prefill={e.prefill_s:.3f}s "
            f"decode={e.decode_step_s * 1e3:.1f}ms/token feasible={e.feasible}"
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _normalize(weights: Sequence[float]) -> List[float]:
    total = float(sum(weights))
    return [float(w) / total for w in weights]


def _rank_of(p: int, s: int, t: int, sp: int, tp: int) -> int:
    return (p * sp + s) * tp + t


def _allreduce_time(nbytes: float, n: int, bw_bytes: float, lat_s: float) -> float:
    if n <= 1:
        return 0.0
    return 2 * (n - 1) / n * nbytes / bw_bytes + 2 * (n - 1) * lat_s


def _link(devs: Sequence[DeviceProfile]):
    """Slowest link of a group: (bytes/s, seconds)."""
    return min(d.bandwidth_mbps for d in devs) * 1e6 / 8, max(d.latency_ms for d in devs) / 1e3


def layer_flops(cfg: ModelConfig, tokens: float, context: float) -> float:
    """FLOPs of one decoder layer for ``tokens`` new tokens (causal attention)."""
    linear = 2.0 * tokens * (cfg.layer_param_count() - 2 * cfg.hidden_size)
    avg_ctx = max(context - (tokens - 1) / 2.0, 1.0)
    return linear + 4.0 * tokens * avg_ctx * cfg.q_size


# --------------------------------------------------------------------------
# pipeline layer partitioning
# --------------------------------------------------------------------------
def balance_layers(
    stage_speeds: Sequence[float],
    layer_costs: Sequence[float],
    *,
    stage_extra_costs: Optional[Sequence[float]] = None,
    layer_mems: Optional[Sequence[float]] = None,
    stage_mem_caps: Optional[Sequence[float]] = None,
    stage_extra_mems: Optional[Sequence[float]] = None,
    min_layers: int = 1,
) -> List[int]:
    """Contiguous layer partition minimising the slowest stage.

    Stage ``k`` holding layers ``[a, b)`` takes
    ``(extra_cost[k] + sum(layer_costs[a:b])) / stage_speeds[k]`` and needs
    ``extra_mem[k] + sum(layer_mems[a:b]) <= stage_mem_caps[k]``.
    Solved exactly by dynamic programming in ``O(K * L^2)``.
    """
    K, L = len(stage_speeds), len(layer_costs)
    extra_c = list(stage_extra_costs) if stage_extra_costs is not None else [0.0] * K
    extra_m = list(stage_extra_mems) if stage_extra_mems is not None else [0.0] * K
    mems = list(layer_mems) if layer_mems is not None else [0.0] * L
    caps = list(stage_mem_caps) if stage_mem_caps is not None else [float("inf")] * K
    if L < K * min_layers:
        raise ValueError(f"cannot place {L} layers on {K} stages with >= {min_layers} each")
    pc = [0.0]
    pm = [0.0]
    for c, m in zip(layer_costs, mems):
        pc.append(pc[-1] + c)
        pm.append(pm[-1] + m)
    inf = float("inf")
    best = [[inf] * (L + 1) for _ in range(K + 1)]
    choice = [[-1] * (L + 1) for _ in range(K + 1)]
    best[0][0] = 0.0
    for k in range(1, K + 1):
        for i in range(k * min_layers, L + 1):
            for j in range((k - 1) * min_layers, i - min_layers + 1):
                if best[k - 1][j] == inf:
                    continue
                if extra_m[k - 1] + pm[i] - pm[j] > caps[k - 1]:
                    continue
                t = (extra_c[k - 1] + pc[i] - pc[j]) / stage_speeds[k - 1]
                value = max(best[k - 1][j], t)
                if value < best[k][i] - 1e-15:
                    best[k][i], choice[k][i] = value, j
    if best[K][L] == inf:
        raise ValueError("no layer partition satisfies the memory limits")
    counts, i = [], L
    for k in range(K, 0, -1):
        j = choice[k][i]
        counts.append(i - j)
        i = j
    return counts[::-1]


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------
def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _all_equal(values: Sequence[float]) -> bool:
    return max(values) - min(values) <= 1e-12 * max(abs(v) for v in values)


def plan(
    model: ModelConfig,
    devices: Sequence[DeviceProfile],
    pp_size: int = 1,
    sp_size: int = 1,
    tp_size: int = 1,
    workload: Optional[Workload] = None,
    *,
    sp_mode: str = "ring",
    uneven: bool = True,
    **config_kwargs,
) -> ParallelConfig:
    """Derive split weights and a layer partition for a fixed mesh shape.

    Device ``r`` is placed at global rank ``r``, i.e. mesh coordinates
    ``(r // (sp*tp), (r // tp) % sp, r % tp)``.  Order ``devices`` so that TP
    groups (consecutive ranks) contain well-connected devices.

    * a TP rank's share of heads/units follows its speed (averaged over the
      SP ranks, which must use the same head partition);
    * an SP rank's share of tokens follows the aggregate speed of its TP
      group (averaged over stages, which must use the same token layout);
    * layers are assigned to stages by :func:`balance_layers`, using the
      aggregate speed of each stage and the tightest device memory.
    """
    workload = workload or Workload()
    if len(devices) != pp_size * sp_size * tp_size:
        raise ValueError(f"{len(devices)} devices for a {pp_size}x{sp_size}x{tp_size} mesh")

    def dev(p: int, s: int, t: int) -> DeviceProfile:
        return devices[_rank_of(p, s, t, sp_size, tp_size)]

    stage_tp = [[1.0] * tp_size for _ in range(pp_size)]
    sp_w = [1.0] * sp_size
    if uneven:
        stage_tp = [[_mean(dev(p, s, t).tflops for s in range(sp_size)) for t in range(tp_size)] for p in range(pp_size)]
        sp_w = [_mean(sum(dev(p, s, t).tflops for t in range(tp_size)) for p in range(pp_size)) for s in range(sp_size)]

    tp_weights = None
    if tp_size > 1 and not all(_all_equal(w) for w in stage_tp):
        tp_weights = stage_tp[0] if all(w == stage_tp[0] for w in stage_tp) else stage_tp
    sp_weights = sp_w if sp_size > 1 and not _all_equal(sp_w) else None

    pp_layers = None
    if pp_size > 1:
        B, S, bpe = workload.batch_size, workload.prompt_len, workload.dtype_bytes
        tpn = [_normalize(w) for w in stage_tp]
        spn = _normalize(sp_w)
        # full (un-split) KV cache of one layer, and the share an SP rank keeps
        kv_layer = 2 * model.kv_size * bpe * B * (S + workload.new_tokens)
        kv_share = max(spn) if (sp_size > 1 and sp_mode == "ring") else 1.0 / sp_size
        layer_mem = model.layer_param_count() * bpe + kv_layer * kv_share
        embed = model.vocab_size * model.hidden_size * bpe
        # weights are split over TP ranks, so a stage fits if every device can
        # hold its TP fraction of it: capacity in "whole stage" bytes
        caps = [
            min(dev(p, s, t).memory_gb * 1e9 / tpn[p][t] for s in range(sp_size) for t in range(tp_size))
            for p in range(pp_size)
        ]
        extra_mem = [0.0] * pp_size
        extra_mem[0] += embed
        if not model.tie_word_embeddings:
            extra_mem[-1] += embed
        extra_cost = [0.0] * pp_size
        extra_cost[-1] += 2.0 * B * model.vocab_size * model.hidden_size  # LM head
        speeds = [sum(dev(p, s, t).tflops for s in range(sp_size) for t in range(tp_size)) / sp_size for p in range(pp_size)]
        pp_layers = balance_layers(
            speeds,
            [layer_flops(model, B * S, S)] * model.num_layers,
            stage_extra_costs=extra_cost,
            layer_mems=[layer_mem] * model.num_layers,
            stage_mem_caps=caps,
            stage_extra_mems=extra_mem,
        )
    return ParallelConfig(
        tp_size=tp_size,
        pp_size=pp_size,
        sp_size=sp_size,
        sp_mode=sp_mode,
        tp_weights=tp_weights,
        sp_weights=sp_weights,
        pp_layers=pp_layers,
        **config_kwargs,
    )


# --------------------------------------------------------------------------
# cost model
# --------------------------------------------------------------------------
def estimate(
    model: ModelConfig,
    config: ParallelConfig,
    devices: Sequence[DeviceProfile],
    workload: Optional[Workload] = None,
) -> Estimate:
    """Analytic latency / memory estimate of running ``workload``."""
    w = workload or Workload()
    pp, sp, tp = config.pp_size, config.sp_size, config.tp_size
    if len(devices) != pp * sp * tp:
        raise ValueError("number of devices does not match the mesh")
    B, S, T, bpe = w.batch_size, w.prompt_len, w.new_tokens, w.dtype_bytes
    H, V = model.hidden_size, model.vocab_size
    counts = list(config.pp_layers) if config.pp_layers is not None else split_sizes(model.num_layers, pp)
    spw = _normalize(config.sp_weights or [1] * sp)
    layer_params = model.layer_param_count()
    linear_params = layer_params - 2 * H
    kv_per_token_layer = 2 * model.kv_size * bpe
    sharded = sp > 1 and S >= sp

    def dev(p, s, t):
        return devices[_rank_of(p, s, t, sp, tp)]

    prefill_stages, decode_stages = [], []
    comm_bytes = 0.0
    mem = [0.0] * len(devices)
    for p in range(pp):
        tpw = _normalize(config.stage_tp_weights(p) or [1] * tp)
        n_layers = counts[p]
        first, last = p == 0, p == pp - 1
        pre_dev, dec_dev = [], []
        for s in range(sp):
            tp_group = [dev(p, s, t) for t in range(tp)]
            tp_bw, tp_lat = _link(tp_group)
            sp_group = [dev(p, x, 0) for x in range(sp)]
            sp_bw, sp_lat = _link(sp_group)
            tokens = B * S * (spw[s] if sharded else 1.0)
            for t in range(tp):
                d = dev(p, s, t)
                flops_s = d.tflops * 1e12
                # ---------------- prefill
                lin = 2.0 * tokens * linear_params * tpw[t] * n_layers
                attn = 4.0 * tokens * (S / 2.0) * model.q_size * tpw[t] * n_layers
                if last:
                    lin += 2.0 * B * V * H * tpw[t]  # LM head on the last token
                t_lin, t_attn = lin / flops_s, attn / flops_s
                act = tokens * H * bpe
                t_tp = n_layers * 2 * _allreduce_time(act, tp, tp_bw, tp_lat)
                if sharded and config.sp_mode == "ring":
                    block = B * S * max(spw) * kv_per_token_layer * tpw[t]
                    ring = n_layers * (sp - 1) * (block / sp_bw + sp_lat)
                    t_sp = max(0.0, ring - t_attn)  # overlapped with attention compute
                    comm_bytes += n_layers * (sp - 1) * block
                elif sharded:
                    a2a = B * S * spw[s] * (2 * model.q_size + 2 * model.kv_size) * bpe * tpw[t]
                    t_sp = n_layers * (2 * (sp - 1) / sp * a2a / sp_bw + 2 * sp_lat)
                    comm_bytes += n_layers * (sp - 1) / sp * a2a
                else:
                    t_sp = 0.0
                comm_bytes += n_layers * 2 * 2 * (tp - 1) / tp * act if tp > 1 else 0.0
                pre_dev.append(t_lin + t_attn + t_tp + t_sp)
                # ---------------- decode (one step for the batch, mid-generation context)
                ctx_len = S + T / 2.0
                cache_share = spw[s] if (sp > 1 and config.sp_mode == "ring") else 1.0 / sp
                dflops = n_layers * (2.0 * B * linear_params + 4.0 * B * ctx_len * model.q_size * cache_share) * tpw[t]
                dbytes = n_layers * (linear_params * bpe + B * ctx_len * kv_per_token_layer * cache_share) * tpw[t]
                if last:
                    dflops += 2.0 * B * V * H * tpw[t]
                    dbytes += V * H * bpe * tpw[t]
                t_dec = max(dflops / flops_s, dbytes / (d.mem_bandwidth_gbps * 1e9))
                t_dec += n_layers * 2 * _allreduce_time(B * H * bpe, tp, tp_bw, tp_lat)
                if sp > 1:
                    t_dec += n_layers * ((sp - 1) * (B * model.q_size * bpe * tpw[t]) / sp_bw + (sp - 1) * sp_lat)
                dec_dev.append(t_dec)
                # ---------------- memory
                weights = n_layers * layer_params * bpe * tpw[t]
                if first:
                    weights += V * H * bpe * tpw[t]
                if last and not model.tie_word_embeddings:
                    weights += V * H * bpe * tpw[t]
                kv = n_layers * B * (S + T) * kv_per_token_layer * tpw[t] * cache_share
                mem[_rank_of(p, s, t, sp, tp)] = (weights + kv) / 1e9
        prefill_stages.append(max(pre_dev))
        decode_stages.append(max(dec_dev))

    link_bw, link_lat = _link(devices)
    boundary = (pp - 1) * (B * S * max(spw if sharded else [1.0]) * H * bpe / link_bw + link_lat)
    prefill = sum(prefill_stages) + boundary
    token_loop = (pp if pp > 1 else 0) * (B * H * bpe / link_bw + link_lat)
    m = min(config.num_microbatches or pp, B)
    if m > 1:  # micro-batches overlap in the pipeline
        per_mb = [t / m for t in decode_stages]
        decode_step = max(sum(per_mb) + token_loop, m * max(per_mb))
    else:
        decode_step = sum(decode_stages) + token_loop
    comm_bytes += (pp - 1) * B * S * H * bpe + T * pp * B * H * bpe
    feasible = all(used <= d.memory_gb for used, d in zip(mem, devices))
    return Estimate(
        prefill_s=prefill,
        decode_step_s=decode_step,
        total_s=prefill + T * decode_step,
        comm_bytes=comm_bytes,
        device_memory_gb=mem,
        feasible=feasible,
        breakdown={"pp_boundary_s": boundary, "token_loop_s": token_loop},
    )


def _factorizations(n: int):
    for pp in range(1, n + 1):
        if n % pp:
            continue
        for sp in range(1, n // pp + 1):
            if (n // pp) % sp:
                continue
            yield pp, sp, n // (pp * sp)


def search(
    model: ModelConfig,
    devices: Sequence[DeviceProfile],
    workload: Optional[Workload] = None,
    *,
    sp_modes: Sequence[str] = ("ring", "ulysses"),
    uneven: bool = True,
    top_k: Optional[int] = None,
) -> List[PlanResult]:
    """Enumerate mesh shapes for ``devices`` and rank them by estimated latency.

    Infeasible (out-of-memory) plans are listed after feasible ones.
    """
    workload = workload or Workload()
    results: List[PlanResult] = []
    for (pp, sp, tp), mode in itertools.product(_factorizations(len(devices)), sp_modes):
        if sp == 1 and mode != sp_modes[0]:
            continue  # SP algorithm irrelevant without SP
        if tp > model.num_heads or pp > model.num_layers or (sp > 1 and sp > workload.prompt_len):
            continue
        if mode == "ulysses" and sp > max(1, model.num_heads // tp):
            continue
        try:
            config = plan(model, devices, pp, sp, tp, workload, sp_mode=mode, uneven=uneven)
            config.validate(model.num_layers)
        except ValueError:
            continue
        results.append(PlanResult(config, estimate(model, config, devices, workload)))
    results.sort(key=lambda r: (not r.estimate.feasible, r.estimate.total_s))
    return results[:top_k] if top_k else results


__all__ = [
    "DeviceProfile",
    "Estimate",
    "PlanResult",
    "Workload",
    "balance_layers",
    "estimate",
    "layer_flops",
    "plan",
    "search",
]
