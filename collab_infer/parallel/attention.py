"""Attention kernels that also return log-sum-exp (LSE) statistics.

Partial attention results over disjoint key sets can be merged exactly with
their LSE, which is the foundation of ring attention and of decoding with a
KV cache that is distributed over several devices.

Masking is defined by explicit token *positions* rather than tensor indices::

    key j is visible to query i  <=>  k_pos[j] >= 0  and  (not causal or k_pos[j] <= q_pos[i])

Position ``-1`` marks padding.  Because of this, attention does not care how
tokens are distributed over ranks (contiguous, zigzag, cached out of order):
correctness only depends on every token carrying its true position.

Tensor layout: ``q`` is ``[B, Sq, Hq, D]``, ``k``/``v`` are ``[B, Sk, Hkv, D]``,
positions are ``[B, S]`` int64.  Grouped-query attention is expressed with a
``kv_map`` giving the KV head of every query head.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch

KVMap = Optional[Union[torch.Tensor, Sequence[int]]]


def accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _as_index(kv_map: KVMap, device: torch.device) -> Optional[torch.Tensor]:
    if kv_map is None:
        return None
    if isinstance(kv_map, torch.Tensor):
        return kv_map.to(device=device, dtype=torch.long)
    return torch.tensor(list(kv_map), dtype=torch.long, device=device)


def expand_kv(x: torch.Tensor, num_q_heads: int, kv_map: KVMap = None) -> torch.Tensor:
    """Repeat KV heads so that head ``h`` of the result serves query head ``h``."""
    index = _as_index(kv_map, x.device)
    if index is None:
        num_kv = x.shape[2]
        if num_kv == num_q_heads:
            return x
        if num_q_heads % num_kv != 0:
            raise ValueError(f"{num_q_heads} query heads cannot share {num_kv} KV heads evenly")
        return x.repeat_interleave(num_q_heads // num_kv, dim=2)
    if index.numel() != num_q_heads:
        raise ValueError("kv_map must have one entry per query head")
    if x.shape[2] == num_q_heads and torch.equal(index, torch.arange(num_q_heads, device=x.device)):
        return x
    return x.index_select(2, index)


def visibility_mask(q_pos: torch.Tensor, k_pos: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Boolean ``[B, 1, Sq, Sk]`` mask of visible (query, key) pairs."""
    mask = (k_pos >= 0)[:, None, None, :]
    if causal:
        mask = mask & (k_pos[:, None, None, :] <= q_pos[:, None, :, None])
    return mask


def fully_masked(q_pos: torch.Tensor, k_pos: torch.Tensor, causal: bool = True) -> bool:
    """Cheap check whether no query can see any key (block can be skipped)."""
    if k_pos.numel() == 0 or q_pos.numel() == 0:
        return True
    valid_k = k_pos[k_pos >= 0]
    if valid_k.numel() == 0:
        return True
    if not causal:
        return False
    return bool(q_pos.max() < valid_k.min())


def _attention_block(qh, kh, vh, q_pos, k_pos, causal, scale):
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale  # [B, H, Sq, Sk]
    scores = scores.masked_fill(~visibility_mask(q_pos, k_pos, causal), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [B, H, Sq]; -inf for fully masked rows
    safe = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    probs = torch.exp(scores - safe.unsqueeze(-1))
    out = torch.matmul(probs, vh)  # [B, H, Sq, D]
    return out.transpose(1, 2), lse


def attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    kv_map: KVMap = None,
    kv_block: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masked softmax attention returning ``(out, lse)`` in accumulation precision.

    ``out`` is ``[B, Sq, Hq, D]`` and ``lse`` is ``[B, Hq, Sq]``.  Rows without
    any visible key produce ``out = 0`` and ``lse = -inf`` (never NaN), which
    makes them neutral elements for :func:`merge_attention`.
    ``kv_block`` bounds memory by processing keys in blocks.
    """
    B, Sq, Hq, D = q.shape
    acc = accumulation_dtype(q.dtype)
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    Sk = k.shape[1]
    if Sk == 0 or Sq == 0:
        return (
            torch.zeros(B, Sq, Hq, D, dtype=acc, device=q.device),
            torch.full((B, Hq, Sq), float("-inf"), dtype=acc, device=q.device),
        )
    qh = q.transpose(1, 2).to(acc)
    kh = expand_kv(k, Hq, kv_map).transpose(1, 2).to(acc)
    vh = expand_kv(v, Hq, kv_map).transpose(1, 2).to(acc)
    if kv_block is None or kv_block >= Sk:
        return _attention_block(qh, kh, vh, q_pos, k_pos, causal, scale)
    out = lse = None
    for start in range(0, Sk, kv_block):
        end = min(start + kv_block, Sk)
        kp = k_pos[:, start:end]
        if fully_masked(q_pos, kp, causal):
            continue
        o, l = _attention_block(qh, kh[:, :, start:end], vh[:, :, start:end], q_pos, kp, causal, scale)
        out, lse = (o, l) if out is None else merge_attention(out, lse, o, l)
    if out is None:
        return (
            torch.zeros(B, Sq, Hq, D, dtype=acc, device=q.device),
            torch.full((B, Hq, Sq), float("-inf"), dtype=acc, device=q.device),
        )
    return out, lse


def merge_attention(
    out_a: torch.Tensor, lse_a: torch.Tensor, out_b: torch.Tensor, lse_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exactly combine attention over two disjoint key sets."""
    lse = torch.logaddexp(lse_a, lse_b)
    safe = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    w_a = torch.exp(lse_a - safe).transpose(1, 2).unsqueeze(-1)  # [B, Sq, H, 1]
    w_b = torch.exp(lse_b - safe).transpose(1, 2).unsqueeze(-1)
    return out_a * w_a + out_b * w_b, lse


def merge_attention_list(
    outs: Sequence[torch.Tensor], lses: Sequence[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine any number of partial attention results."""
    if len(outs) == 1:
        return outs[0], lses[0]
    stacked = torch.stack(list(lses))  # [n, B, H, Sq]
    lse = torch.logsumexp(stacked, dim=0)
    safe = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    weights = torch.exp(stacked - safe)  # [n, B, H, Sq]
    out = sum(o * w.transpose(1, 2).unsqueeze(-1) for o, w in zip(outs, weights))
    return out, lse


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Plain (single-device) attention; returns ``q.dtype``."""
    out, _ = attention_with_lse(q, k, v, q_pos, k_pos, **kwargs)
    return out.to(q.dtype)


__all__ = [
    "accumulation_dtype",
    "attention",
    "attention_with_lse",
    "expand_kv",
    "fully_masked",
    "merge_attention",
    "merge_attention_list",
    "visibility_mask",
]
