"""Architecture description of LLaMA-family decoder-only transformers."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Union

import torch

SUPPORTED_MODEL_TYPES = ("llama", "mistral", "qwen2")

# Architectures of popular edge-scale models (for planning and benchmarks
# without downloading a checkpoint).
PRESETS: Dict[str, Dict[str, Any]] = {
    "tinyllama-1.1b": dict(vocab_size=32000, hidden_size=2048, intermediate_size=5632, num_layers=22, num_heads=32, num_kv_heads=4, max_position_embeddings=2048),
    "llama2-7b": dict(vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_layers=32, num_heads=32, num_kv_heads=32, rms_norm_eps=1e-5),
    "llama3-8b": dict(vocab_size=128256, hidden_size=4096, intermediate_size=14336, num_layers=32, num_heads=32, num_kv_heads=8, rope_theta=500000.0, rms_norm_eps=1e-5, max_position_embeddings=8192),
    "qwen2-0.5b": dict(vocab_size=151936, hidden_size=896, intermediate_size=4864, num_layers=24, num_heads=14, num_kv_heads=2, rope_theta=1e6, tie_word_embeddings=True, qkv_bias=True, model_type="qwen2", max_position_embeddings=32768),
    "qwen2-1.5b": dict(vocab_size=151936, hidden_size=1536, intermediate_size=8960, num_layers=28, num_heads=12, num_kv_heads=2, rope_theta=1e6, tie_word_embeddings=True, qkv_bias=True, model_type="qwen2", max_position_embeddings=32768),
}


@dataclass
class ModelConfig:
    """Hyper-parameters of a LLaMA-style model (RMSNorm, RoPE, SwiGLU, GQA).

    Parameter names follow Hugging Face ``transformers`` so LLaMA, Mistral and
    Qwen2 checkpoints load directly (see :meth:`from_hf`).
    """

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = False
    qkv_bias: bool = False
    o_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"
    model_type: str = "llama"

    def __post_init__(self) -> None:
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            if self.hidden_size % self.num_heads:
                raise ValueError("hidden_size must be divisible by num_heads (or set head_dim)")
            self.head_dim = self.hidden_size // self.num_heads
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")

    # -------------------------------------------------------------- factories
    @classmethod
    def tiny(cls, **overrides: Any) -> "ModelConfig":
        """A small configuration for tests and demos."""
        base = dict(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=160,
            num_layers=4,
            num_heads=8,
            num_kv_heads=4,
            max_position_embeddings=512,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> "ModelConfig":
        """Architecture of a well-known model, e.g. ``"tinyllama-1.1b"``."""
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})

    @classmethod
    def from_hf(cls, config: Union[str, os.PathLike, Dict[str, Any], Any]) -> "ModelConfig":
        """Build from a HF ``config.json`` path/dir, dict or ``PretrainedConfig``."""
        if isinstance(config, (str, os.PathLike)):
            path = os.fspath(config)
            if os.path.isdir(path):
                path = os.path.join(path, "config.json")
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        elif hasattr(config, "to_dict"):
            config = config.to_dict()
        cfg = dict(config)
        model_type = cfg.get("model_type", "llama")
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise NotImplementedError(
                f"model_type {model_type!r} is not supported (supported: {SUPPORTED_MODEL_TYPES})"
            )
        num_heads = cfg["num_attention_heads"]
        rope_theta = cfg.get("rope_theta", 10000.0)
        rope_scaling = cfg.get("rope_scaling")
        rope_params = cfg.get("rope_parameters")  # newer transformers releases
        if isinstance(rope_params, dict):
            rope_theta = rope_params.get("rope_theta", rope_theta)
            if rope_params.get("rope_type", "default") != "default":
                rope_scaling = rope_params
        attention_bias = bool(cfg.get("attention_bias", False))
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=cfg.get("num_key_value_heads") or num_heads,
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // num_heads,
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            qkv_bias=True if model_type == "qwen2" else attention_bias,
            o_bias=False if model_type == "qwen2" else attention_bias,
            mlp_bias=bool(cfg.get("mlp_bias", False)),
            hidden_act=cfg.get("hidden_act", "silu"),
            model_type=model_type,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    # ---------------------------------------------------------------- derived
    @property
    def q_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    def layer_param_count(self) -> int:
        h, i = self.hidden_size, self.intermediate_size
        attn = h * self.q_size + 2 * h * self.kv_size + self.q_size * h
        if self.qkv_bias:
            attn += self.q_size + 2 * self.kv_size
        if self.o_bias:
            attn += h
        mlp = 3 * h * i + (2 * i + h if self.mlp_bias else 0)
        return attn + mlp + 2 * h

    def param_count(self) -> int:
        embed = self.vocab_size * self.hidden_size
        head = 0 if self.tie_word_embeddings else embed
        return embed + head + self.hidden_size + self.num_layers * self.layer_param_count()

    def rope_inv_freq(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Inverse rotary frequencies (matches HF, incl. linear/llama3 scaling)."""
        dim = self.head_dim
        exponent = torch.arange(0, dim, 2, dtype=torch.int64).to(dtype) / dim
        inv_freq = 1.0 / (self.rope_theta ** exponent)
        scaling = self.rope_scaling
        if not scaling:
            return inv_freq
        kind = scaling.get("rope_type", scaling.get("type", "default"))
        if kind == "default":
            return inv_freq
        if kind == "linear":
            return inv_freq / scaling["factor"]
        if kind == "llama3":
            factor = scaling["factor"]
            low, high = scaling["low_freq_factor"], scaling["high_freq_factor"]
            old_ctx = scaling["original_max_position_embeddings"]
            low_wavelen, high_wavelen = old_ctx / low, old_ctx / high
            wavelen = 2 * math.pi / inv_freq
            scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
            smooth = (old_ctx / wavelen - low) / (high - low)
            smoothed = (1 - smooth) * scaled / factor + smooth * scaled
            medium = ~(wavelen < high_wavelen) & ~(wavelen > low_wavelen)
            return torch.where(medium, smoothed, scaled)
        raise NotImplementedError(f"rope scaling type {kind!r} is not supported")


__all__ = ["ModelConfig", "PRESETS", "SUPPORTED_MODEL_TYPES"]
