"""Parity with Hugging Face ``transformers`` (skipped if it is not installed)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from collab_infer import CollabEngine, ModelConfig, ParallelConfig  # noqa: E402
from helpers import run_dist  # noqa: E402

SMALL = dict(vocab_size=100, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)


def _hf_model(kind):
    torch.manual_seed(0)
    if kind == "qwen2":
        model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**SMALL))
    elif kind == "llama3_rope":
        scaling = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 16}
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(rope_scaling=scaling, **SMALL))
    else:
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**SMALL))
    with torch.no_grad():  # non-trivial norms and biases so that loading them is tested
        for name, p in model.named_parameters():
            if "norm" in name:
                p.add_(torch.randn_like(p) * 0.1)
            elif name.endswith("bias"):
                p.normal_(0, 0.1)
    return model.eval()


def _parallel_worker(rank, world, path, ids, mask):
    eng = CollabEngine(ModelConfig.from_hf(path), ParallelConfig(tp_size=2, pp_size=2), path)
    return eng.forward(ids, mask, broadcast=True)


@pytest.mark.parametrize("kind", ["llama", "qwen2", "llama3_rope"])
def test_matches_transformers(kind, tmp_path):
    model = _hf_model(kind)
    model.save_pretrained(tmp_path)
    cfg = ModelConfig.from_hf(str(tmp_path))
    eng = CollabEngine(cfg, ParallelConfig(), str(tmp_path))
    ids = torch.randint(0, 100, (2, 11), generator=torch.Generator().manual_seed(1))
    mask = torch.ones_like(ids)
    mask[1, :4] = 0  # left padding
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=mask).logits
    torch.testing.assert_close(eng.forward(ids, mask)[mask.bool()], ref[mask.bool()], atol=1e-5, rtol=1e-5)
    # greedy generation, stopping at the model's EOS like transformers does
    eos = model.generation_config.eos_token_id
    ours = eng.generate([ids[0].tolist()], max_new_tokens=8, eos_token_id=eos).tokens[0]
    with torch.no_grad():
        theirs = model.generate(ids[:1], max_new_tokens=8, do_sample=False, pad_token_id=0, eos_token_id=eos)
    assert ours == theirs[0, ids.shape[1] :].tolist()
    # the same checkpoint on a 2x2 (pp x tp) mesh
    for logits in run_dist(_parallel_worker, 4, str(tmp_path), ids, mask):
        torch.testing.assert_close(logits[mask.bool()], ref[mask.bool()], atol=1e-5, rtol=1e-5)
