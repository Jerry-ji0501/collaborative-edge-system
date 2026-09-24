"""Independent single-device reference implementation used by the tests.

Deliberately written without any framework code: plain tensor math, PyTorch's
``scaled_dot_product_attention``, one unpadded sequence at a time and no KV
cache (the whole sequence is recomputed at every generation step).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def _rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    # x: [S, H, D]
    S, _, D = x.shape
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    ang = torch.arange(S, dtype=torch.float64)[:, None] * inv[None, :]  # [S, D/2]
    ang = torch.cat([ang, ang], dim=-1)[:, None, :].to(x.dtype)
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * ang.cos() + rotated * ang.sin()


def dense_logits(sd: Dict[str, torch.Tensor], cfg, ids: List[int]) -> torch.Tensor:
    """Logits ``[S, V]`` of one sequence."""
    d, hq, hkv = cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    x = sd["model.embed_tokens.weight"][torch.tensor(ids)]
    S = x.shape[0]
    for layer in range(cfg.num_layers):
        p = f"model.layers.{layer}."

        def lin(h, name, bias):
            y = h @ sd[p + name + ".weight"].T
            return y + sd[p + name + ".bias"] if bias else y

        h = _rms(x, sd[p + "input_layernorm.weight"], cfg.rms_norm_eps)
        q = lin(h, "self_attn.q_proj", cfg.qkv_bias).view(S, hq, d)
        k = lin(h, "self_attn.k_proj", cfg.qkv_bias).view(S, hkv, d)
        v = lin(h, "self_attn.v_proj", cfg.qkv_bias).view(S, hkv, d)
        q, k = _rope(q, cfg.rope_theta), _rope(k, cfg.rope_theta)
        k = k.repeat_interleave(hq // hkv, dim=1)
        v = v.repeat_interleave(hq // hkv, dim=1)
        att = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True, scale=1 / math.sqrt(d)
        )
        x = x + lin(att.transpose(0, 1).reshape(S, hq * d), "self_attn.o_proj", cfg.o_bias)
        h = _rms(x, sd[p + "post_attention_layernorm.weight"], cfg.rms_norm_eps)
        mlp = F.silu(lin(h, "mlp.gate_proj", cfg.mlp_bias)) * lin(h, "mlp.up_proj", cfg.mlp_bias)
        x = x + lin(mlp, "mlp.down_proj", cfg.mlp_bias)
    x = _rms(x, sd["model.norm.weight"], cfg.rms_norm_eps)
    head = sd["model.embed_tokens.weight"] if cfg.tie_word_embeddings else sd["lm_head.weight"]
    return x @ head.T


def dense_generate(
    sd: Dict[str, torch.Tensor], cfg, prompt: List[int], max_new_tokens: int
) -> Tuple[List[int], torch.Tensor]:
    """Greedy generation; returns tokens and the ``[T, V]`` logits of each step."""
    ids = list(prompt)
    steps = []
    for _ in range(max_new_tokens):
        logits = dense_logits(sd, cfg, ids)[-1]
        steps.append(logits)
        ids.append(int(logits.argmax()))
    return ids[len(prompt) :], torch.stack(steps)
