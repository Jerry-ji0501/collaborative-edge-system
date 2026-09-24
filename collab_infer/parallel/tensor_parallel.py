"""Tensor parallelism (Megatron-LM style), inference only.

A linear layer ``y = x W^T`` is split either

* **column-wise** – every rank owns a slice of the output features; the
  output stays sharded (e.g. attention heads, MLP hidden units) or is
  all-gathered, or
* **row-wise** – every rank owns a slice of the input features and produces a
  partial sum that is all-reduced (or reduce-scattered along the sequence for
  Megatron sequence parallelism).

A column-parallel layer followed by a row-parallel layer needs a single
reduction, which is how attention and MLP blocks are parallelised.  Shards may
be uneven so heterogeneous devices get work proportional to their speed.

Besides the building blocks used by the bundled transformer, this module
offers :func:`parallelize_module`, which shards the ``nn.Linear`` and
``nn.Embedding`` layers of *any* model according to a name-pattern plan.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributed.comm import Communicator
from .partition import split_sizes

REDUCE_MODES = ("all_reduce", "reduce_scatter", "none")


def _resolve_partition(
    total: int,
    comm: Communicator,
    sizes: Optional[Sequence[int]],
    weights: Optional[Sequence[float]],
    granularity: int,
    local_range: Optional[Tuple[int, int]],
) -> Tuple[Optional[List[int]], int, int]:
    """Return ``(all_sizes or None, local_start, local_size)``."""
    if local_range is not None:
        start, size = int(local_range[0]), int(local_range[1])
        if start < 0 or size < 0 or start + size > total:
            raise ValueError(f"local range {local_range} outside [0, {total})")
        return None, start, size
    if sizes is None:
        sizes = split_sizes(total, list(weights) if weights is not None else comm.size, granularity)
    sizes = [int(s) for s in sizes]
    if len(sizes) != comm.size or sum(sizes) != total:
        raise ValueError(f"partition {sizes} does not split {total} over {comm.size} ranks")
    return sizes, sum(sizes[: comm.rank]), sizes[comm.rank]


def _empty_param(*shape: int, dtype=None, device=None) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=dtype, device=device), requires_grad=False)


def _copy_into(param: torch.Tensor, value: torch.Tensor) -> None:
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(f"shape mismatch: parameter {tuple(param.shape)} vs value {tuple(value.shape)}")
    param.data.copy_(value.to(dtype=param.dtype, device=param.device))


def reduce_partial(
    y: torch.Tensor,
    comm: Communicator,
    reduce: str = "all_reduce",
    scatter_dim: int = 1,
    scatter_sizes: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Combine row-parallel partial sums."""
    if reduce == "all_reduce":
        return comm.all_reduce(y)
    if reduce == "reduce_scatter":
        return comm.reduce_scatter(y, dim=scatter_dim, sizes=scatter_sizes)
    if reduce == "none":
        return y
    raise ValueError(f"unknown reduce mode {reduce!r}; expected one of {REDUCE_MODES}")


class ColumnParallelLinear(nn.Module):
    """Linear layer whose output features are split across ranks.

    Args:
        sizes: explicit output partition (one entry per rank).
        weights: relative rank capabilities (ignored if ``sizes`` is given).
        granularity: partition unit, e.g. ``head_dim`` to never split a head.
        local_range: ``(start, size)`` of this rank's slice; allows overlapping
            slices such as KV heads replicated for multi-query attention.
        gather_output: all-gather the full output (e.g. for the LM head).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        comm: Communicator,
        *,
        sizes: Optional[Sequence[int]] = None,
        weights: Optional[Sequence[float]] = None,
        granularity: int = 1,
        local_range: Optional[Tuple[int, int]] = None,
        bias: bool = False,
        gather_output: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.comm = comm
        self.in_features, self.out_features = in_features, out_features
        self.out_sizes, self.out_start, self.out_local = _resolve_partition(
            out_features, comm, sizes, weights, granularity, local_range
        )
        if gather_output and self.out_sizes is None:
            raise ValueError("gather_output requires a partition, not a local_range")
        self.gather_output = gather_output
        self.weight = _empty_param(self.out_local, in_features, dtype=dtype, device=device)
        self.bias = _empty_param(self.out_local, dtype=dtype, device=device) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output and self.comm.size > 1:
            y = self.comm.all_gather(y, dim=-1, sizes=self.out_sizes)
        return y

    @torch.no_grad()
    def load_full(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        """Load this rank's shard from the full (unsharded) parameters."""
        _copy_into(self.weight, weight.narrow(0, self.out_start, self.out_local))
        if self.bias is not None:
            if bias is None:
                raise ValueError("layer has a bias but none was provided")
            _copy_into(self.bias, bias.narrow(0, self.out_start, self.out_local))

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, local=[{self.out_start}:"
            f"{self.out_start + self.out_local}], bias={self.bias is not None}, gather={self.gather_output}"
        )


class RowParallelLinear(nn.Module):
    """Linear layer whose input features are split across ranks.

    ``forward`` consumes the rank-local input slice and returns the reduced
    output.  ``reduce="reduce_scatter"`` leaves the output sharded along
    ``scatter_dim`` (Megatron sequence parallelism); ``reduce="none"`` returns
    the raw partial sum *without* bias so callers can fuse the reduction.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        comm: Communicator,
        *,
        sizes: Optional[Sequence[int]] = None,
        weights: Optional[Sequence[float]] = None,
        granularity: int = 1,
        local_range: Optional[Tuple[int, int]] = None,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.comm = comm
        self.in_features, self.out_features = in_features, out_features
        self.in_sizes, self.in_start, self.in_local = _resolve_partition(
            in_features, comm, sizes, weights, granularity, local_range
        )
        self.weight = _empty_param(out_features, self.in_local, dtype=dtype, device=device)
        self.bias = _empty_param(out_features, dtype=dtype, device=device) if bias else None

    def forward(
        self,
        x: torch.Tensor,
        reduce: str = "all_reduce",
        scatter_dim: int = 1,
        scatter_sizes: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        y = F.linear(x, self.weight)
        y = reduce_partial(y, self.comm, reduce, scatter_dim, scatter_sizes)
        if self.bias is not None and reduce != "none":
            y = y + self.bias
        return y

    @torch.no_grad()
    def load_full(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        _copy_into(self.weight, weight.narrow(1, self.in_start, self.in_local))
        if self.bias is not None:
            if bias is None:
                raise ValueError("layer has a bias but none was provided")
            _copy_into(self.bias, bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, local=[{self.in_start}:"
            f"{self.in_start + self.in_local}], bias={self.bias is not None}"
        )


class VocabParallelEmbedding(nn.Module):
    """Embedding table split along the vocabulary.

    Each rank looks up the ids that fall into its vocabulary range and writes
    zeros elsewhere; summing over ranks yields the full embedding.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        comm: Communicator,
        *,
        sizes: Optional[Sequence[int]] = None,
        weights: Optional[Sequence[float]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.comm = comm
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.vocab_sizes, self.vocab_start, self.vocab_local = _resolve_partition(
            num_embeddings, comm, sizes, weights, 1, None
        )
        self.weight = _empty_param(self.vocab_local, embedding_dim, dtype=dtype, device=device)

    def forward(
        self,
        ids: torch.Tensor,
        reduce: str = "all_reduce",
        scatter_dim: int = 1,
        scatter_sizes: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if self.comm.size == 1:
            return F.embedding(ids, self.weight)
        outside = (ids < self.vocab_start) | (ids >= self.vocab_start + self.vocab_local)
        local_ids = (ids - self.vocab_start).masked_fill(outside, 0)
        out = F.embedding(local_ids, self.weight).masked_fill(outside.unsqueeze(-1), 0.0)
        return reduce_partial(out, self.comm, reduce, scatter_dim, scatter_sizes)

    @torch.no_grad()
    def load_full(self, weight: torch.Tensor) -> None:
        _copy_into(self.weight, weight.narrow(0, self.vocab_start, self.vocab_local))

    def extra_repr(self) -> str:
        return (
            f"vocab={self.num_embeddings}, dim={self.embedding_dim}, "
            f"local=[{self.vocab_start}:{self.vocab_start + self.vocab_local}]"
        )


class ParallelLMHead(ColumnParallelLinear):
    """Output projection split over the vocabulary; returns full logits."""

    def __init__(self, hidden_size: int, vocab_size: int, comm: Communicator, **kwargs) -> None:
        kwargs.setdefault("gather_output", True)
        super().__init__(hidden_size, vocab_size, comm, **kwargs)


# --------------------------------------------------------------------------
# generic, plan-based parallelisation of arbitrary models
# --------------------------------------------------------------------------
@dataclass
class ParallelStyle:
    """Base class of the entries of a :func:`parallelize_module` plan."""

    granularity: int = 1

    def build(self, module: nn.Module, comm: Communicator, weights: Optional[Sequence[float]]) -> nn.Module:
        raise NotImplementedError


@dataclass
class ColwiseParallel(ParallelStyle):
    """Shard an ``nn.Linear`` along its output features."""

    gather_output: bool = False

    def build(self, module, comm, weights):
        if not isinstance(module, nn.Linear):
            raise TypeError(f"ColwiseParallel expects nn.Linear, got {type(module).__name__}")
        layer = ColumnParallelLinear(
            module.in_features,
            module.out_features,
            comm,
            weights=weights,
            granularity=self.granularity,
            bias=module.bias is not None,
            gather_output=self.gather_output,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        layer.load_full(module.weight.detach(), None if module.bias is None else module.bias.detach())
        return layer


@dataclass
class RowwiseParallel(ParallelStyle):
    """Shard an ``nn.Linear`` along its input features (output all-reduced)."""

    def build(self, module, comm, weights):
        if not isinstance(module, nn.Linear):
            raise TypeError(f"RowwiseParallel expects nn.Linear, got {type(module).__name__}")
        layer = RowParallelLinear(
            module.in_features,
            module.out_features,
            comm,
            weights=weights,
            granularity=self.granularity,
            bias=module.bias is not None,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        layer.load_full(module.weight.detach(), None if module.bias is None else module.bias.detach())
        return layer


@dataclass
class VocabParallel(ParallelStyle):
    """Shard an ``nn.Embedding`` along the vocabulary."""

    def build(self, module, comm, weights):
        if not isinstance(module, nn.Embedding):
            raise TypeError(f"VocabParallel expects nn.Embedding, got {type(module).__name__}")
        layer = VocabParallelEmbedding(
            module.num_embeddings,
            module.embedding_dim,
            comm,
            weights=weights,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        layer.load_full(module.weight.detach())
        return layer


def parallelize_module(
    module: nn.Module,
    comm: Communicator,
    plan: Dict[str, ParallelStyle],
    weights: Optional[Sequence[float]] = None,
) -> nn.Module:
    """Replace sub-modules matching ``plan`` patterns by tensor-parallel ones.

    ``plan`` maps ``fnmatch`` patterns over qualified module names (e.g.
    ``"layers.*.attn.q_proj"``) to a :class:`ParallelStyle`.  The dense
    weights of the matched modules are sharded in place.  The model's own
    forward logic must be consistent with the sharding: e.g. an attention
    block must derive its number of local heads from the projection sizes,
    and consecutive column/row layers must use the same ``granularity`` so
    their partitions line up.

    Returns ``module`` (modified in place).
    """
    matched = set()
    for name, sub in list(module.named_modules()):
        if not name:
            continue
        for pattern, style in plan.items():
            if fnmatch.fnmatchcase(name, pattern):
                parent_name, _, child = name.rpartition(".")
                parent = module.get_submodule(parent_name) if parent_name else module
                setattr(parent, child, style.build(sub, comm, weights))
                matched.add(pattern)
                break
    unmatched = set(plan) - matched
    if unmatched:
        raise ValueError(f"plan patterns matched no module: {sorted(unmatched)}")
    return module


__all__ = [
    "ColumnParallelLinear",
    "ColwiseParallel",
    "ParallelLMHead",
    "ParallelStyle",
    "RowParallelLinear",
    "RowwiseParallel",
    "VocabParallel",
    "VocabParallelEmbedding",
    "parallelize_module",
    "reduce_partial",
]
