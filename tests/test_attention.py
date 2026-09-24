import math

import pytest
import torch
import torch.nn.functional as F

from collab_infer.parallel.attention import (
    attention,
    attention_with_lse,
    merge_attention,
    merge_attention_list,
)


def _rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


@pytest.mark.parametrize("causal", [True, False])
def test_matches_sdpa(causal):
    q, k, v = _rand(2, 7, 4, 8, seed=1), _rand(2, 7, 4, 8, seed=2), _rand(2, 7, 4, 8, seed=3)
    pos = torch.arange(7).repeat(2, 1)
    ours = attention(q, k, v, pos, pos, causal=causal)
    ref = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
    torch.testing.assert_close(ours, ref.transpose(1, 2))


def test_gqa_kv_map_and_blocks():
    q, k, v = _rand(1, 9, 6, 4, seed=4), _rand(1, 9, 2, 4, seed=5), _rand(1, 9, 2, 4, seed=6)
    pos = torch.arange(9).view(1, 9)
    ref = attention(q, k.repeat_interleave(3, 2), v.repeat_interleave(3, 2), pos, pos)
    torch.testing.assert_close(attention(q, k, v, pos, pos), ref)
    torch.testing.assert_close(attention(q, k, v, pos, pos, kv_map=[0, 0, 0, 1, 1, 1]), ref)
    torch.testing.assert_close(attention(q, k, v, pos, pos, kv_block=2), ref)


def test_merge_of_disjoint_key_sets_is_exact():
    q, k, v = _rand(2, 5, 3, 4, seed=7), _rand(2, 12, 3, 4, seed=8), _rand(2, 12, 3, 4, seed=9)
    qpos = torch.arange(7, 12).repeat(2, 1)
    kpos = torch.arange(12).repeat(2, 1)
    full, full_lse = attention_with_lse(q, k, v, qpos, kpos)
    idx = torch.randperm(12, generator=torch.Generator().manual_seed(0))
    parts = [idx[:3], idx[3:8], idx[8:]]
    outs, lses = zip(*[attention_with_lse(q, k[:, p], v[:, p], qpos, kpos[:, p]) for p in parts])
    out, lse = merge_attention_list(list(outs), list(lses))
    torch.testing.assert_close(out, full)
    torch.testing.assert_close(lse, full_lse)
    out2, _ = merge_attention(*merge_attention(outs[0], lses[0], outs[1], lses[1]), outs[2], lses[2])
    torch.testing.assert_close(out2, full)


def test_padding_and_fully_masked_rows_are_zero_not_nan():
    q, k, v = _rand(1, 4, 2, 4, seed=10), _rand(1, 4, 2, 4, seed=11), _rand(1, 4, 2, 4, seed=12)
    pos = torch.tensor([[-1, -1, 0, 1]])
    out, lse = attention_with_lse(q, k, v, pos, pos)
    assert torch.isfinite(out).all()
    assert torch.all(out[:, :2] == 0) and torch.isinf(lse[..., :2]).all()
    ref = attention(q[:, 2:], k[:, 2:], v[:, 2:], pos[:, 2:], pos[:, 2:])
    torch.testing.assert_close(out[:, 2:], ref)
    # merging with an empty partial is the identity
    empty_o, empty_l = attention_with_lse(q, k[:, :0], v[:, :0], pos, pos[:, :0])
    merged, _ = merge_attention(out, lse, empty_o, empty_l)
    torch.testing.assert_close(merged, out)


def test_scale_default():
    q, k, v = _rand(1, 3, 1, 16, seed=13), _rand(1, 3, 1, 16, seed=14), _rand(1, 3, 1, 16, seed=15)
    pos = torch.arange(3).view(1, 3)
    torch.testing.assert_close(attention(q, k, v, pos, pos), attention(q, k, v, pos, pos, scale=1 / math.sqrt(16)))
