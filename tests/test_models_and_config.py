import json

import pytest
import torch

from collab_infer import ModelConfig, ParallelConfig
from collab_infer.engine import Sampler, SamplingParams
from collab_infer.models import (
    DictWeightSource,
    KVCache,
    LayerKVCache,
    TorchFileSource,
    open_weights,
    random_state_dict,
)


def test_parallel_config_validation():
    ParallelConfig(tp_size=2, pp_size=2, pp_layers=[1, 3]).validate(4)
    with pytest.raises(ValueError):
        ParallelConfig(pp_size=2, pp_layers=[1, 2]).validate(4)
    with pytest.raises(ValueError):
        ParallelConfig(sp_size=2, sp_mode="bogus").validate()
    with pytest.raises(ValueError):
        ParallelConfig(tp_size=2, tp_weights=[1, 2, 3]).validate()
    with pytest.raises(ValueError):
        ParallelConfig(tp_size=2, pp_size=2, tp_weights=[[1, 1]]).validate()
    cfg = ParallelConfig(tp_size=2, pp_size=2, tp_weights=[[1, 2], [3, 4]], network={"bandwidth_mbps": 100})
    cfg = ParallelConfig.from_dict({**cfg.to_dict(), "network": {"bandwidth_mbps": 100, "latency_ms": 1}})
    assert cfg.stage_tp_weights(1) == [3.0, 4.0] and cfg.network.bandwidth_mbps == 100
    assert "pp=2" in cfg.describe()


def test_model_config_from_hf_dict(tmp_path):
    hf = {
        "model_type": "qwen2",
        "vocab_size": 1000,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 3,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "rope_theta": 1e6,
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(hf))
    cfg = ModelConfig.from_hf(str(tmp_path))
    assert (cfg.num_layers, cfg.num_kv_heads, cfg.head_dim) == (3, 2, 16)
    assert cfg.qkv_bias and not cfg.o_bias and cfg.tie_word_embeddings
    with pytest.raises(NotImplementedError):
        ModelConfig.from_hf({**hf, "model_type": "gpt2"})
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg


def test_param_count_matches_state_dict():
    cfg = ModelConfig.tiny(qkv_bias=True, mlp_bias=True)
    sd = random_state_dict(cfg)
    assert cfg.param_count() == sum(t.numel() for t in sd.values())


def test_weight_sources(tmp_path):
    cfg = ModelConfig.tiny(num_layers=1)
    sd = random_state_dict(cfg)
    path = tmp_path / "model.pt"
    torch.save(sd, path)
    src = open_weights(str(path))
    assert isinstance(src, TorchFileSource)
    name = "model.layers.0.self_attn.q_proj.weight"
    assert torch.equal(src.get(name), sd[name])
    # checkpoints saved without the "model." prefix are resolved too
    bare = DictWeightSource({k.removeprefix("model."): v for k, v in sd.items()})
    assert torch.equal(bare.get(name), sd[name]) and bare.has(name)
    with pytest.raises(KeyError):
        src.get("missing.weight")


def test_kv_cache_growth():
    cache = LayerKVCache()
    k = torch.randn(2, 3, 4, 5)
    for step in range(20):
        cache.append(k, k * 2, torch.full((2, 3), step))
    kc, vc, pc = cache.view()
    assert kc.shape == (2, 60, 4, 5) and torch.equal(vc, kc * 2)
    assert pc[:, -1].tolist() == [19, 19]
    empty_k, _, empty_p = LayerKVCache().view(like_k=k, like_pos=torch.zeros(2, 3, dtype=torch.long))
    assert empty_k.shape == (2, 0, 4, 5) and empty_p.shape == (2, 0)
    assert KVCache().is_empty()


def test_sampler():
    logits = torch.tensor([[0.0, 5.0, 1.0], [3.0, 0.0, 0.0]])
    assert Sampler(SamplingParams()).__call__(logits).tolist() == [1, 0]
    a = Sampler(SamplingParams(temperature=1.0, seed=3))(logits.repeat(50, 1))
    b = Sampler(SamplingParams(temperature=1.0, seed=3))(logits.repeat(50, 1))
    assert torch.equal(a, b)
    top1 = Sampler(SamplingParams(temperature=1.0, top_k=1, seed=0))(logits.repeat(20, 1))
    assert top1.tolist() == [1, 0] * 20
    nucleus = Sampler(SamplingParams(temperature=1.0, top_p=0.5, seed=0))(logits.repeat(20, 1))
    assert nucleus.tolist() == [1, 0] * 20


def test_kv_cache_is_sized_exactly_for_generation():
    from collab_infer import CollabEngine

    cfg = ModelConfig.tiny(num_layers=2)
    engine = CollabEngine(cfg, ParallelConfig(), random_state_dict(cfg))
    engine.generate([[1, 2, 3, 4, 5], [6, 7]], max_new_tokens=7)
    assert engine.last_kv_cache_bytes > 0
    assert engine.last_kv_cache_allocated_bytes == engine.last_kv_cache_bytes
