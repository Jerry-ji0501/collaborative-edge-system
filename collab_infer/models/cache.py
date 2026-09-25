"""Per-rank runtime state: KV caches and the forward-pass descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


class LayerKVCache:
    """Growable key/value cache of one attention layer on one rank.

    Keys/values are stored head-major, ``[B, H_local, capacity, D]``, so the
    attention kernels read each head's history contiguously; :meth:`view`
    returns token-major ``[B, T, H_local, D]`` views of the same memory (no
    copies).  Positions of the cached tokens are kept as ``[B, T]``.  Which
    tokens/heads a rank caches depends on the parallel strategy (TP: its
    heads; ring SP: its tokens; Ulysses SP: its heads over all tokens), but
    attention only relies on the stored positions.
    """

    def __init__(self, reserve: int = 0) -> None:
        self.k: Optional[torch.Tensor] = None  # [B, H, capacity, D]
        self.v: Optional[torch.Tensor] = None
        self.pos: Optional[torch.Tensor] = None  # [B, capacity]
        self.length = 0
        # extra slots allocated on top of what is needed (e.g. the tokens that
        # decoding will append), so memory-constrained devices avoid regrowth
        self.reserve = max(0, int(reserve))

    def __len__(self) -> int:
        return self.length

    @property
    def capacity(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def _grow(self, k: torch.Tensor, v: torch.Tensor, pos: torch.Tensor, needed: int) -> None:
        capacity = max(needed + self.reserve, int(self.capacity * 1.5))
        B, _, H, D = k.shape
        new_k = k.new_empty((B, H, capacity, D))
        new_v = v.new_empty((B, v.shape[2], capacity, v.shape[3]))
        new_pos = pos.new_full((B, capacity), -1)
        if self.length:
            new_k[:, :, : self.length] = self.k[:, :, : self.length]
            new_v[:, :, : self.length] = self.v[:, :, : self.length]
            new_pos[:, : self.length] = self.pos[:, : self.length]
        self.k, self.v, self.pos = new_k, new_v, new_pos

    def append(self, k: torch.Tensor, v: torch.Tensor, pos: torch.Tensor) -> None:
        """Append token-major ``k``/``v`` (``[B, n, H, D]``) and their positions."""
        n = k.shape[1]
        if n == 0:
            return
        if self.k is None or self.length + n > self.capacity:
            self._grow(k, v, pos, self.length + n)
        end = self.length + n
        self.k[:, :, self.length : end] = k.transpose(1, 2)
        self.v[:, :, self.length : end] = v.transpose(1, 2)
        self.pos[:, self.length : end] = pos
        self.length = end

    def view(
        self, like_k: Optional[torch.Tensor] = None, like_pos: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cached ``(k, v, pos)`` as token-major views; empty tensors shaped like ``like_*`` if unused."""
        if self.k is None:
            if like_k is None or like_pos is None:
                raise RuntimeError("empty cache: provide templates to build empty views")
            empty = like_k.new_empty((like_k.shape[0], 0) + tuple(like_k.shape[2:]))
            return empty, empty, like_pos.new_empty((like_pos.shape[0], 0))
        n = self.length
        return self.k[:, :, :n].transpose(1, 2), self.v[:, :, :n].transpose(1, 2), self.pos[:, :n]

    def nbytes(self) -> int:
        if self.k is None:
            return 0
        per_token = (self.k[:, :, :1].numel() + self.v[:, :, :1].numel()) * self.k.element_size()
        return per_token * self.length


class KVCache:
    """KV caches of all local layers for one (micro-)batch on one rank.

    ``sp_lengths`` tracks how many tokens every sequence-parallel rank caches
    (ring mode); all SP ranks update it identically, so the owner of newly
    generated tokens is agreed on without communication.
    """

    def __init__(self, reserve_tokens: int = 0) -> None:
        self.layers: Dict[int, LayerKVCache] = {}
        self.sp_lengths: Optional[List[int]] = None
        self.num_tokens = 0
        self.reserve_tokens = reserve_tokens

    def layer(self, idx: int) -> LayerKVCache:
        if idx not in self.layers:
            self.layers[idx] = LayerKVCache(self.reserve_tokens)
        return self.layers[idx]

    def allocated_bytes(self) -> int:
        """Bytes of the cache buffers including unused reserved slots."""
        total = 0
        for layer in self.layers.values():
            if layer.k is not None:
                total += (layer.k.numel() + layer.v.numel()) * layer.k.element_size()
        return total

    def is_empty(self) -> bool:
        return self.num_tokens == 0

    def nbytes(self) -> int:
        return sum(layer.nbytes() for layer in self.layers.values())


@dataclass
class ForwardState:
    """Describes how the tokens of the current forward pass are laid out.

    Attributes:
        positions: positions of this rank's tokens, ``[B, s]`` (``-1`` = pad).
            With Megatron sequence parallelism these are the positions of the
            whole SP-local sequence (TP ranks all-gather before attention).
        seq_sharded: tokens are split across the SP group (prefill); otherwise
            every SP rank holds the same tokens (decode or short inputs).
        sp_sizes: tokens per SP rank when ``seq_sharded``.
        tp_sp_sizes: Megatron-SP split of the local tokens over TP ranks, or
            ``None`` when activations are replicated inside the TP group.
        decode_owner: ring mode, replicated tokens: SP rank that caches them.
        cache: KV cache to read and extend, or ``None`` for cache-free passes.
        rope: rotary ``(cos, sin)`` for ``positions``, computed once per pass
            and shared by all layers.
        sp_split: tokens are replicated on the SP ranks and those ranks split
            the dense compute of each layer (see ``ParallelConfig.sp_decode_split``).
    """

    positions: torch.Tensor
    seq_sharded: bool = False
    sp_sizes: Optional[List[int]] = None
    tp_sp_sizes: Optional[List[int]] = None
    decode_owner: Optional[int] = None
    cache: Optional[KVCache] = None
    rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    sp_split: bool = False

    @property
    def tp_reduce(self) -> str:
        return "reduce_scatter" if self.tp_sp_sizes else "all_reduce"


__all__ = ["ForwardState", "KVCache", "LayerKVCache"]
