"""Parallel LLaMA-family decoder (LLaMA, Mistral, Qwen2 architectures).

:class:`LlamaStage` is the part of the model owned by one rank:

* **pipeline**: only the decoder layers of this rank's stage are built; the
  first stage owns the embedding, the last stage the final norm and LM head;
* **tensor**: attention heads, MLP units and the vocabulary are split over
  the TP group (Q/K/V/gate/up column-parallel, O/down row-parallel), with
  optional Megatron sequence parallelism for norms and residuals;
* **sequence**: attention runs as ring or Ulysses attention over the SP group
  during prefill and against the distributed KV cache during decoding.

All three compose: a stage is a ``sp x tp`` sub-mesh of the device mesh.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributed.mesh import ParallelContext
from ..parallel.attention import accumulation_dtype, attention, gqa_group
from ..parallel.partition import HeadShard, offsets_of, partition_heads, split_sizes
from ..parallel.sequence_parallel import (
    distributed_kv_attention,
    gather_heads,
    ring_attention,
    ulysses_gather,
    ulysses_scatter,
)
from ..parallel.tensor_parallel import (
    FusedColumnParallelLinear,
    ParallelLMHead,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from .cache import ForwardState, LayerKVCache
from .config import ModelConfig
from .weights import WeightSource

_ACTIVATIONS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, dtype=None, device=None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.to(accumulation_dtype(x.dtype))
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding evaluated at explicit (global) positions."""

    def __init__(self, config: ModelConfig, device=None) -> None:
        super().__init__()
        self.register_buffer("inv_freq32", config.rope_inv_freq(torch.float32).to(device), persistent=False)
        self.register_buffer("inv_freq64", config.rope_inv_freq(torch.float64).to(device), persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype):
        inv_freq = self.inv_freq64 if dtype == torch.float64 else self.inv_freq32
        pos = positions.clamp(min=0).to(inv_freq.dtype)
        freqs = pos.unsqueeze(-1) * inv_freq  # [B, S, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x``: ``[B, S, H, D]``; ``cos``/``sin``: ``[B, S, D]``."""
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    return x * cos + _rotate_half(x) * sin


class ParallelAttention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        ctx: ParallelContext,
        tp_shards: Sequence[HeadShard],
        *,
        sp_mode: str = "ring",
        sp_weights: Optional[Sequence[float]] = None,
        kv_block: Optional[int] = None,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.tp, self.sp, self.stage = ctx.tp, ctx.sp, ctx.stage
        self.sp_mode = sp_mode
        self.kv_block = kv_block
        me = tp_shards[self.tp.rank]
        d, hidden = config.head_dim, config.hidden_size
        self.head_dim = d
        self.num_q, self.num_kv = me.q_count, me.kv_count
        kw = dict(dtype=dtype, device=device)
        kv_part = (config.kv_size, me.kv_start * d, me.kv_count * d)
        # Q, K and V in one GEMM (this rank's heads only)
        self.qkv_proj = FusedColumnParallelLinear(
            hidden, [(config.q_size, me.q_start * d, me.q_count * d), kv_part, kv_part], self.tp, bias=config.qkv_bias, **kw
        )
        self.o_proj = RowParallelLinear(
            config.q_size, hidden, self.tp, sizes=[s.q_count * d for s in tp_shards], bias=config.o_bias, **kw
        )
        # query->KV head map; None when heads are regularly grouped (fused GQA kernels)
        self.kv_map = self._kv_map_arg(me, device)
        self.ulysses_shards: Optional[List[HeadShard]] = None
        if self.sp.size > 1 and sp_mode == "ulysses":
            self.ulysses_shards = partition_heads(
                me.q_count, me.kv_count, list(sp_weights) if sp_weights else self.sp.size, kv_map=me.kv_map
            )
            self.ulysses_kv_map = self._kv_map_arg(self.ulysses_shards[self.sp.rank], device)
        # heads whose output projection this SP rank computes when decoding with sp_split
        self.sp_split_heads: Optional[tuple] = None
        if self.sp.size > 1:
            if self.ulysses_shards is not None:
                mine = self.ulysses_shards[self.sp.rank]
                self.sp_split_heads = (mine.q_start, mine.q_count)
            elif me.q_count >= self.sp.size:
                sizes = split_sizes(me.q_count, list(sp_weights) if sp_weights else self.sp.size)
                self.sp_split_heads = (offsets_of(sizes)[self.sp.rank], sizes[self.sp.rank])

    @staticmethod
    def _kv_map_arg(shard: HeadShard, device) -> Optional[torch.Tensor]:
        if gqa_group(shard.q_count, shard.kv_count, shard.kv_map) is not None:
            return None
        return torch.tensor(shard.kv_map, dtype=torch.long, device=device)

    def forward(
        self,
        x: torch.Tensor,
        state: ForwardState,
        cache: Optional[LayerKVCache],
        rotary: RotaryEmbedding,
    ) -> torch.Tensor:
        if state.sp_split and self.sp_split_heads is not None:
            return self._forward_sp_split(x, state, cache, rotary)
        B, S, _ = x.shape
        d = self.head_dim
        q, k, v = self.qkv_proj(x)
        q = q.view(B, S, self.num_q, d)
        k = k.view(B, S, self.num_kv, d)
        v = v.view(B, S, self.num_kv, d)
        cos, sin = self._rope(state, q.dtype, rotary)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        out = self._attend(q, k, v, state, cache)
        return self.o_proj(
            out.reshape(B, S, self.num_q * d),
            reduce=state.tp_reduce,
            scatter_dim=1,
            scatter_sizes=state.tp_sp_sizes,
        )

    @staticmethod
    def _rope(state: ForwardState, dtype: torch.dtype, rotary: RotaryEmbedding):
        if state.rope is None or state.rope[0].dtype != dtype:
            state.rope = rotary(state.positions, dtype)  # shared by all layers of this pass
        return state.rope

    def _qkv_rows(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Rows ``[start, end)`` of the fused Q/K/V projection (a view, no copy)."""
        bias = self.qkv_proj.bias
        return F.linear(x, self.qkv_proj.weight[start:end], None if bias is None else bias[start:end])

    def _forward_sp_split(self, x, state: ForwardState, cache: Optional[LayerKVCache], rotary) -> torch.Tensor:
        """Replicated tokens: SP ranks share the work of this layer (decode)."""
        B, S, _ = x.shape
        d, pos, sp = self.head_dim, state.positions, self.sp
        kw = dict(causal=True, kv_block=self.kv_block)
        cos, sin = self._rope(state, x.dtype, rotary)
        q_rows, kv_rows = self.num_q * d, self.num_kv * d
        head_start, head_count = self.sp_split_heads
        if self.ulysses_shards is not None:
            # this rank attends for its heads only, so it only projects those heads
            me = self.ulysses_shards[sp.rank]
            kv_lo, kv_hi = me.kv_start * d, (me.kv_start + me.kv_count) * d
            q = self._qkv_rows(x, me.q_start * d, (me.q_start + me.q_count) * d).view(B, S, me.q_count, d)
            k = self._qkv_rows(x, q_rows + kv_lo, q_rows + kv_hi).view(B, S, me.kv_count, d)
            v = self._qkv_rows(x, q_rows + kv_rows + kv_lo, q_rows + kv_rows + kv_hi).view(B, S, me.kv_count, d)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
            kpos = pos
            if cache is not None:
                cache.append(k, v, pos)
                k, v, kpos = cache.view()
            local = attention(q, k, v, pos, kpos, kv_map=self.ulysses_kv_map, **kw)
        else:
            # ring: every rank needs all query heads; only the cache owner needs K/V
            owner = cache is None or state.decode_owner == sp.rank
            q = apply_rotary(self._qkv_rows(x, 0, q_rows).view(B, S, self.num_q, d), cos, sin)
            if owner:
                k, v = self._qkv_rows(x, q_rows, q_rows + 2 * kv_rows).view(B, S, 2 * self.num_kv, d).split(self.num_kv, dim=2)
                k = apply_rotary(k, cos, sin)
            if cache is None:
                out = attention(q, k, v, pos, pos, kv_map=self.kv_map, **kw)
            else:
                if owner:
                    cache.append(k, v, pos)
                kc, vc, pc = cache.view(like_k=q.new_empty(B, 0, self.num_kv, d), like_pos=pos)
                out = distributed_kv_attention(q, kc, vc, pos, pc, sp, kv_map=self.kv_map, **kw)
            local = out[:, :, head_start : head_start + head_count]
        # each rank projects its heads; one all-reduce over the stage sums SP and TP partials
        weight = self.o_proj.weight[:, head_start * d : (head_start + head_count) * d]
        out = self.stage.all_reduce(F.linear(local.reshape(B, S, head_count * d), weight), compress=True)
        return out if self.o_proj.bias is None else out + self.o_proj.bias

    def _attend(self, q, k, v, state: ForwardState, cache: Optional[LayerKVCache]) -> torch.Tensor:
        pos, sp = state.positions, self.sp
        kw = dict(causal=True, kv_block=self.kv_block)
        if sp.size == 1:
            kpos = pos
            if cache is not None:
                cache.append(k, v, pos)
                k, v, kpos = cache.view()
            return attention(q, k, v, pos, kpos, kv_map=self.kv_map, **kw)

        if state.seq_sharded:  # prefill: tokens split over the SP group
            if self.sp_mode == "ring":
                if cache is not None and len(cache) > 0:
                    raise RuntimeError("ring attention prefill requires an empty cache")
                out = ring_attention(q, k, v, pos, pos, sp, state.sp_sizes, kv_map=self.kv_map, **kw)
                if cache is not None:
                    cache.append(k, v, pos)  # KV cache stays sharded along the sequence
                return out
            q_h, k_h, v_h, pos_all = ulysses_scatter(q, k, v, pos, sp, state.sp_sizes, self.ulysses_shards)
            k_att, v_att, p_att = k_h, v_h, pos_all
            if cache is not None:  # KV cache sharded along heads; attend to it in place
                cache.append(k_h, v_h, pos_all)
                k_att, v_att, p_att = cache.view()
            local = attention(q_h, k_att, v_att, pos_all, p_att, kv_map=self.ulysses_kv_map, **kw)
            return ulysses_gather(local, sp, state.sp_sizes, self.ulysses_shards)

        # tokens replicated on every SP rank (decoding or very short inputs)
        if cache is None:
            return attention(q, k, v, pos, pos, kv_map=self.kv_map, **kw)
        if self.sp_mode == "ring":
            if state.decode_owner == sp.rank:
                cache.append(k, v, pos)
            kc, vc, pc = cache.view(like_k=k, like_pos=pos)
            return distributed_kv_attention(q, kc, vc, pos, pc, sp, kv_map=self.kv_map, **kw)
        me = self.ulysses_shards[sp.rank]
        cache.append(
            k[:, :, me.kv_start : me.kv_start + me.kv_count],
            v[:, :, me.kv_start : me.kv_start + me.kv_count],
            pos,
        )
        kc, vc, pc = cache.view()
        local = attention(
            q[:, :, me.q_start : me.q_start + me.q_count], kc, vc, pos, pc, kv_map=self.ulysses_kv_map, **kw
        )
        return gather_heads(local, sp, self.ulysses_shards)


class ParallelMLP(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        ctx: ParallelContext,
        ffn_sizes: Sequence[int],
        dtype=None,
        device=None,
        sp_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        rank = ctx.tp.rank
        self.stage = ctx.stage
        self.local_units = ffn_sizes[rank]
        # units of the TP-local MLP this SP rank computes when decoding with sp_split
        self.sp_units: Optional[tuple] = None
        if ctx.sp.size > 1 and self.local_units >= ctx.sp.size:
            sizes = split_sizes(self.local_units, list(sp_weights) if sp_weights else ctx.sp.size)
            self.sp_units = (offsets_of(sizes)[ctx.sp.rank], sizes[ctx.sp.rank])
        part = (i, offsets_of(ffn_sizes)[rank], ffn_sizes[rank])
        # gate and up projections in one GEMM
        self.gate_up_proj = FusedColumnParallelLinear(
            h, [part, part], ctx.tp, bias=config.mlp_bias, dtype=dtype, device=device
        )
        self.down_proj = RowParallelLinear(
            i, h, ctx.tp, sizes=ffn_sizes, bias=config.mlp_bias, dtype=dtype, device=device
        )
        if config.hidden_act not in _ACTIVATIONS:
            raise NotImplementedError(f"activation {config.hidden_act!r} is not supported")
        self.act = _ACTIVATIONS[config.hidden_act]

    def forward(self, x: torch.Tensor, state: ForwardState) -> torch.Tensor:
        if state.sp_split and self.sp_units is not None:
            return self._forward_sp_split(x)
        gate, up = self.gate_up_proj(x)
        h = self.act(gate) * up
        return self.down_proj(h, reduce=state.tp_reduce, scatter_dim=1, scatter_sizes=state.tp_sp_sizes)

    def _forward_sp_split(self, x: torch.Tensor) -> torch.Tensor:
        """Replicated tokens: each SP rank computes a slice of the units (views, no copies)."""
        start, count = self.sp_units
        units = self.local_units
        weight, bias = self.gate_up_proj.weight, self.gate_up_proj.bias
        gate = F.linear(x, weight[start : start + count], None if bias is None else bias[start : start + count])
        up_lo = units + start
        up = F.linear(x, weight[up_lo : up_lo + count], None if bias is None else bias[up_lo : up_lo + count])
        partial = F.linear(self.act(gate) * up, self.down_proj.weight[:, start : start + count])
        out = self.stage.all_reduce(partial, compress=True)  # sums SP and TP partials at once
        return out if self.down_proj.bias is None else out + self.down_proj.bias


class DecoderLayer(nn.Module):
    def __init__(self, config, ctx, tp_shards, ffn_sizes, *, sp_mode, sp_weights, kv_block, dtype, device):
        super().__init__()
        self.tp = ctx.tp
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype, device)
        self.self_attn = ParallelAttention(
            config, ctx, tp_shards, sp_mode=sp_mode, sp_weights=sp_weights, kv_block=kv_block, dtype=dtype, device=device
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype, device)
        self.mlp = ParallelMLP(config, ctx, ffn_sizes, dtype, device, sp_weights=sp_weights)

    def _gather(self, h: torch.Tensor, state: ForwardState) -> torch.Tensor:
        if state.tp_sp_sizes:  # Megatron SP: norms ran on 1/tp of the tokens
            return self.tp.all_gather(h, dim=1, sizes=state.tp_sp_sizes, compress=True)
        return h

    def forward(self, x, state: ForwardState, cache: Optional[LayerKVCache], rotary: RotaryEmbedding):
        x = x + self.self_attn(self._gather(self.input_layernorm(x), state), state, cache, rotary)
        return x + self.mlp(self._gather(self.post_attention_layernorm(x), state), state)


class LlamaStage(nn.Module):
    """The share of a LLaMA-family model owned by one rank.

    Args:
        layer_ids: global indices of the decoder layers of this stage.
        has_embedding / has_head: whether this is the first / last stage.
        tp_weights: relative capability of the TP ranks of this stage.
        sp_mode / sp_weights: sequence-parallel algorithm and split weights.
        kv_block: optional key block size for memory-bounded attention.
    """

    def __init__(
        self,
        config: ModelConfig,
        ctx: ParallelContext,
        layer_ids: Iterable[int],
        *,
        has_embedding: bool,
        has_head: bool,
        tp_weights: Optional[Sequence[float]] = None,
        sp_mode: str = "ring",
        sp_weights: Optional[Sequence[float]] = None,
        kv_block: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config, self.ctx = config, ctx
        self.layer_ids = [int(i) for i in layer_ids]
        self.has_embedding, self.has_head = has_embedding, has_head
        tp = ctx.tp
        parts = list(tp_weights) if tp_weights is not None else tp.size
        self.vocab_sizes = split_sizes(config.vocab_size, parts)
        self.tp_shards = partition_heads(config.num_heads, config.num_kv_heads, parts)
        self.ffn_sizes = split_sizes(config.intermediate_size, parts)
        kw = dict(dtype=dtype, device=device)
        if has_embedding:
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, tp, sizes=self.vocab_sizes, **kw)
        self.layers = nn.ModuleDict(
            {
                str(i): DecoderLayer(
                    config, ctx, self.tp_shards, self.ffn_sizes, sp_mode=sp_mode, sp_weights=sp_weights, kv_block=kv_block, **kw
                )
                for i in self.layer_ids
            }
        )
        if has_head:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, **kw)
            self.lm_head = ParallelLMHead(config.hidden_size, config.vocab_size, tp, sizes=self.vocab_sizes, **kw)
        self.rotary = RotaryEmbedding(config, device)

    # ------------------------------------------------------------- execution
    def embed(self, input_ids: torch.Tensor, state: ForwardState) -> torch.Tensor:
        return self.embed_tokens(input_ids, reduce=state.tp_reduce, scatter_dim=1, scatter_sizes=state.tp_sp_sizes)

    def forward_layers(self, x: torch.Tensor, state: ForwardState) -> torch.Tensor:
        for idx, layer in self.layers.items():
            cache = state.cache.layer(int(idx)) if state.cache is not None else None
            x = layer(x, state, cache, self.rotary)
        return x

    def final_hidden(self, x: torch.Tensor, state: ForwardState) -> torch.Tensor:
        """Final norm; returns hidden states of all SP-local tokens."""
        h = self.norm(x)
        if state.tp_sp_sizes:
            h = self.ctx.tp.all_gather(h, dim=1, sizes=state.tp_sp_sizes, compress=True)
        return h

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    # --------------------------------------------------------------- weights
    @torch.no_grad()
    def load_weights(self, source: WeightSource) -> None:
        """Load this rank's shards from a full checkpoint (HF parameter names)."""
        cfg = self.config

        def bias(name: str, enabled: bool) -> Optional[torch.Tensor]:
            return source.get(name) if enabled else None

        if self.has_embedding:
            self.embed_tokens.load_full(source.get("model.embed_tokens.weight"))
        for idx, layer in self.layers.items():
            p = f"model.layers.{idx}."
            attn, mlp = layer.self_attn, layer.mlp
            qkv = ("q_proj", "k_proj", "v_proj")
            attn.qkv_proj.load_full(
                [source.get(p + f"self_attn.{n}.weight") for n in qkv],
                [bias(p + f"self_attn.{n}.bias", cfg.qkv_bias) for n in qkv],
            )
            attn.o_proj.load_full(source.get(p + "self_attn.o_proj.weight"), bias(p + "self_attn.o_proj.bias", cfg.o_bias))
            gate_up = ("gate_proj", "up_proj")
            mlp.gate_up_proj.load_full(
                [source.get(p + f"mlp.{n}.weight") for n in gate_up],
                [bias(p + f"mlp.{n}.bias", cfg.mlp_bias) for n in gate_up],
            )
            mlp.down_proj.load_full(source.get(p + "mlp.down_proj.weight"), bias(p + "mlp.down_proj.bias", cfg.mlp_bias))
            layer.input_layernorm.weight.copy_(source.get(p + "input_layernorm.weight"))
            layer.post_attention_layernorm.weight.copy_(source.get(p + "post_attention_layernorm.weight"))
        if self.has_head:
            self.norm.weight.copy_(source.get("model.norm.weight"))
            head = "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight"
            self.lm_head.load_full(source.get(head))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())


def random_state_dict(
    config: ModelConfig, seed: int = 0, dtype: torch.dtype = torch.float32, std: float = 0.02
) -> Dict[str, torch.Tensor]:
    """Deterministic random full checkpoint with HF names (tests and demos).

    Every process that calls this with the same arguments obtains identical
    tensors, so each rank can shard the same model without communication.
    """
    g = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=g, dtype=torch.float64) * std).to(dtype)

    def norm(dim: int) -> torch.Tensor:
        return (1.0 + 0.1 * torch.randn(dim, generator=g, dtype=torch.float64)).to(dtype)

    h, i, v = config.hidden_size, config.intermediate_size, config.vocab_size
    sd: Dict[str, torch.Tensor] = {"model.embed_tokens.weight": rand(v, h)}
    for layer in range(config.num_layers):
        p = f"model.layers.{layer}."
        sd[p + "self_attn.q_proj.weight"] = rand(config.q_size, h)
        sd[p + "self_attn.k_proj.weight"] = rand(config.kv_size, h)
        sd[p + "self_attn.v_proj.weight"] = rand(config.kv_size, h)
        sd[p + "self_attn.o_proj.weight"] = rand(h, config.q_size)
        if config.qkv_bias:
            sd[p + "self_attn.q_proj.bias"] = rand(config.q_size)
            sd[p + "self_attn.k_proj.bias"] = rand(config.kv_size)
            sd[p + "self_attn.v_proj.bias"] = rand(config.kv_size)
        if config.o_bias:
            sd[p + "self_attn.o_proj.bias"] = rand(h)
        sd[p + "mlp.gate_proj.weight"] = rand(i, h)
        sd[p + "mlp.up_proj.weight"] = rand(i, h)
        sd[p + "mlp.down_proj.weight"] = rand(h, i)
        if config.mlp_bias:
            sd[p + "mlp.gate_proj.bias"] = rand(i)
            sd[p + "mlp.up_proj.bias"] = rand(i)
            sd[p + "mlp.down_proj.bias"] = rand(h)
        sd[p + "input_layernorm.weight"] = norm(h)
        sd[p + "post_attention_layernorm.weight"] = norm(h)
    sd["model.norm.weight"] = norm(h)
    if not config.tie_word_embeddings:
        sd["lm_head.weight"] = rand(v, h)
    return sd


__all__ = [
    "DecoderLayer",
    "LlamaStage",
    "ParallelAttention",
    "ParallelMLP",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary",
    "random_state_dict",
]
