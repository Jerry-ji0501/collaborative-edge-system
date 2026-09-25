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

Performance: the work is dispatched to PyTorch's fused kernels
(``scaled_dot_product_attention`` and, when the LSE is needed, the CPU flash
kernel).  No mask is materialised when the positions allow it (aligned causal
prefill uses ``is_causal``; decoding where every cached key is visible uses no
mask), grouped-query attention never copies K/V, and when a mask is needed the
queries are processed in chunks so its memory stays bounded.  Inputs are only
viewed as ``[B, H, S, D]``; a head-major KV cache therefore costs no copies.
"""

from __future__ import annotations

import math
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

KVMap = Optional[Union[torch.Tensor, Sequence[int]]]

# max elements of a materialised mask per kernel call (bounded memory)
MASK_CHUNK_ELEMENTS = 1 << 22

_FLASH_CPU = getattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu", None)


def _sdpa_supports_gqa() -> bool:
    try:
        x = torch.zeros(1, 2, 1, 2)
        F.scaled_dot_product_attention(x, x[:, :1], x[:, :1], enable_gqa=True)
        return True
    except (TypeError, RuntimeError):
        return False


_SDPA_GQA = _sdpa_supports_gqa()


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


def gqa_group(num_q_heads: int, num_kv_heads: int, kv_map: KVMap = None) -> Optional[int]:
    """Group size ``g`` if query head ``h`` uses KV head ``h // g``, else ``None``."""
    if num_kv_heads < 1 or num_q_heads % num_kv_heads:
        return None
    group = num_q_heads // num_kv_heads
    if kv_map is None:
        return group
    mapping = kv_map.tolist() if isinstance(kv_map, torch.Tensor) else list(kv_map)
    return group if mapping == [h // group for h in range(num_q_heads)] else None


def visibility_mask(q_pos: torch.Tensor, k_pos: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Boolean ``[B, 1, Sq, Sk]`` mask of visible (query, key) pairs."""
    mask = (k_pos >= 0)[:, None, None, :]
    if causal:
        mask = mask & (k_pos[:, None, None, :] <= q_pos[:, None, :, None])
    return mask.expand(q_pos.shape[0], 1, q_pos.shape[1], k_pos.shape[1])


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


def _mask_mode(q_pos: torch.Tensor, k_pos: torch.Tensor, causal: bool) -> str:
    """``"none"`` (every query sees every key), ``"causal"`` (aligned causal) or ``"mask"``."""
    if not bool((k_pos >= 0).all()):
        return "mask"
    if not causal:
        return "none"
    if bool((q_pos >= 0).all()) and bool((k_pos.amax(dim=1) <= q_pos.amin(dim=1)).all()):
        return "none"  # e.g. decoding: all cached keys precede the query
    if q_pos.shape == k_pos.shape and torch.equal(q_pos, k_pos):
        if q_pos.shape[1] == 1 or bool((q_pos[:, 1:] - q_pos[:, :-1] == 1).all()):
            return "causal"  # e.g. prefill without padding
    return "mask"


def _plan(q_pos: torch.Tensor, k_pos: torch.Tensor, causal: bool) -> Iterator[Tuple[int, int, str, Optional[torch.Tensor]]]:
    """Yield ``(q_start, q_end, mode, bool_mask)`` chunks for one attention call."""
    B, Sq = q_pos.shape
    Sk = k_pos.shape[1]
    mode = _mask_mode(q_pos, k_pos, causal)
    if mode != "mask":
        yield 0, Sq, mode, None
        return
    step = max(1, MASK_CHUNK_ELEMENTS // max(1, B * Sk))
    for start in range(0, Sq, step):
        end = min(Sq, start + step)
        yield start, end, mode, visibility_mask(q_pos[:, start:end], k_pos, causal)


def _prepare_kv(k: torch.Tensor, v: torch.Tensor, num_q_heads: int, kv_map: KVMap, native_gqa: bool):
    group = gqa_group(num_q_heads, k.shape[2], kv_map)
    if group is None or (group > 1 and not native_gqa):
        k, v = expand_kv(k, num_q_heads, kv_map), expand_kv(v, num_q_heads, kv_map)
        group = 1
    return k, v, group


def _empty_result(q: torch.Tensor, acc: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    B, Sq, Hq, D = q.shape
    return (
        torch.zeros(B, Sq, Hq, D, dtype=acc, device=q.device),
        torch.full((B, Hq, Sq), float("-inf"), dtype=acc, device=q.device),
    )


# --------------------------------------------------------------------------
# attention with LSE (needed to merge partial results)
# --------------------------------------------------------------------------
def _flash_lse(q, k, v, q_pos, k_pos, causal, scale, kv_map, acc):
    k, v, _ = _prepare_kv(k, v, q.shape[2], kv_map, native_gqa=True)
    qh, kh, vh = (t.to(acc).transpose(1, 2) for t in (q, k, v))
    outs: List[torch.Tensor] = []
    lses: List[torch.Tensor] = []
    for start, end, mode, mask in _plan(q_pos, k_pos, causal):
        bias = None
        if mask is not None:
            bias = torch.zeros(mask.shape, dtype=acc, device=q.device).masked_fill(~mask, float("-inf"))
        o, l = _FLASH_CPU(qh[:, :, start:end], kh, vh, 0.0, mode == "causal", attn_mask=bias, scale=scale)
        if mask is not None:  # the kernel reports lse = 0 for rows without visible keys
            empty = ~mask.any(dim=-1)  # [B, 1, sq]
            l = l.masked_fill(empty, float("-inf"))
            o = o.masked_fill(empty.unsqueeze(-1), 0.0)
        outs.append(o)
        lses.append(l)
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
    lse = lses[0] if len(lses) == 1 else torch.cat(lses, dim=2)
    return out.transpose(1, 2), lse


def _math_lse(q, k, v, q_pos, k_pos, causal, scale, kv_map, acc):
    """Portable reference path; grouped-query attention without copying K/V."""
    B, Sq, Hq, D = q.shape
    k, v, group = _prepare_kv(k, v, Hq, kv_map, native_gqa=True)
    Hkv = k.shape[2]
    qg = q.to(acc).permute(0, 2, 1, 3).reshape(B, Hkv, group * Sq, D)  # head h = kv * group + i
    kh, vh = k.to(acc).transpose(1, 2), v.to(acc).transpose(1, 2)
    scores = torch.matmul(qg, kh.transpose(-1, -2)) * scale  # [B, Hkv, group * Sq, Sk]
    mask = visibility_mask(q_pos, k_pos, causal).repeat(1, 1, group, 1)
    scores = scores.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    safe = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    out = torch.matmul(torch.exp(scores - safe.unsqueeze(-1)), vh)  # [B, Hkv, group * Sq, D]
    out = out.reshape(B, Hq, Sq, D).transpose(1, 2)
    return out, lse.reshape(B, Hq, Sq)


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
    ``kv_block`` additionally processes the keys in blocks.
    """
    acc = accumulation_dtype(q.dtype)
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    Sk = k.shape[1]
    if Sk == 0 or q.shape[1] == 0:
        return _empty_result(q, acc)
    if kv_block is not None and kv_block < Sk:
        out = lse = None
        for start in range(0, Sk, kv_block):
            end = min(start + kv_block, Sk)
            kp = k_pos[:, start:end]
            if fully_masked(q_pos, kp, causal):
                continue
            o, l = attention_with_lse(
                q, k[:, start:end], v[:, start:end], q_pos, kp, causal=causal, scale=scale, kv_map=kv_map
            )
            out, lse = (o, l) if out is None else merge_attention(out, lse, o, l)
        return _empty_result(q, acc) if out is None else (out, lse)
    if _FLASH_CPU is not None and q.device.type == "cpu":
        return _flash_lse(q, k, v, q_pos, k_pos, causal, scale, kv_map, acc)
    return _math_lse(q, k, v, q_pos, k_pos, causal, scale, kv_map, acc)


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


# --------------------------------------------------------------------------
# plain attention
# --------------------------------------------------------------------------
def attention(
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
) -> torch.Tensor:
    """Single-device attention; returns ``[B, Sq, Hq, D]`` in ``q.dtype``."""
    Sk = k.shape[1]
    if kv_block is not None and kv_block < Sk:
        out, _ = attention_with_lse(q, k, v, q_pos, k_pos, causal=causal, scale=scale, kv_map=kv_map, kv_block=kv_block)
        return out.to(q.dtype)
    if Sk == 0 or q.shape[1] == 0:
        return torch.zeros_like(q)
    k, v, group = _prepare_kv(k, v, q.shape[2], kv_map, native_gqa=_SDPA_GQA)
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    extra = {"enable_gqa": True} if group > 1 else {}
    outs = []
    for start, end, mode, mask in _plan(q_pos, k_pos, causal):
        o = F.scaled_dot_product_attention(
            qh[:, :, start:end], kh, vh, attn_mask=mask, is_causal=mode == "causal", scale=scale, **extra
        )
        if mask is not None:  # rows without visible keys are 0 (some releases return NaN)
            o = o.masked_fill(~mask.any(dim=-1, keepdim=True), 0.0)
        outs.append(o)
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
    return out.transpose(1, 2)


__all__ = [
    "MASK_CHUNK_ELEMENTS",
    "accumulation_dtype",
    "attention",
    "attention_with_lse",
    "expand_kv",
    "fully_masked",
    "gqa_group",
    "merge_attention",
    "merge_attention_list",
    "visibility_mask",
]
