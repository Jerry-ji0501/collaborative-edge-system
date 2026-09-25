"""End-to-end equivalence of every parallel configuration with a dense model.

Each case runs the engine on a gloo process group and compares, on every
rank, the prefill logits of a left-padded batch, the greedily generated
tokens and the per-step logits against the independent reference in
``reference.py`` (float64, so any mistake shows up far above round-off).
"""

import functools

import pytest
import torch

from collab_infer import CollabEngine, ModelConfig, ParallelConfig, SamplingParams, random_state_dict
from helpers import run_dist
from reference import dense_generate, dense_logits

BASE = dict(vocab_size=96, hidden_size=64, intermediate_size=96, num_layers=4, num_heads=8, num_kv_heads=4)
PROMPTS = [[5, 9, 2, 33, 7, 1, 60, 3, 11], [4, 8, 15, 16, 23], [42, 1, 2, 3, 4, 5, 6], [70, 71]]
NEW_TOKENS = 6

CASES = [
    # tensor parallelism
    ("tp2", dict(tp_size=2), {}),
    ("tp3_uneven_heads", dict(tp_size=3), {}),
    ("tp2_weighted", dict(tp_size=2, tp_weights=[3, 1]), {}),
    ("tp4_replicated_kv", dict(tp_size=4), dict(num_kv_heads=2)),
    ("tp2_megatron_sp", dict(tp_size=2, megatron_sp=True), {}),
    # pipeline parallelism
    ("pp2", dict(pp_size=2), {}),
    ("pp3_layers_microbatches", dict(pp_size=3, pp_layers=[1, 2, 1], num_microbatches=3), {}),
    ("pp2_embedding_only_stage", dict(pp_size=2, pp_layers=[0, 4]), {}),
    # sequence parallelism
    ("sp2_ring", dict(sp_size=2), {}),
    ("sp3_ring_zigzag_weighted", dict(sp_size=3, sp_layout="zigzag", sp_weights=[3, 2, 1]), {}),
    ("sp3_ring_short_prompts", dict(sp_size=3, num_microbatches=4), {}),
    ("sp2_ulysses", dict(sp_size=2, sp_mode="ulysses"), {}),
    ("sp2_ulysses_mqa", dict(sp_size=2, sp_mode="ulysses"), dict(num_kv_heads=1)),
    ("sp4_ulysses_weighted", dict(sp_size=4, sp_mode="ulysses", sp_weights=[2, 1, 1, 1]), {}),
    ("sp2_ring_kv_blocks", dict(sp_size=2, attn_kv_block=3), {}),
    # hybrids
    ("tp2_sp2_ring", dict(tp_size=2, sp_size=2), {}),
    ("tp2_sp2_ulysses_megatron", dict(tp_size=2, sp_size=2, sp_mode="ulysses", megatron_sp=True), {}),
    # TP-local head groups become irregular (3 ranks, 2 KV heads) before Ulysses splits them again
    ("tp3_sp2_ulysses_irregular_heads", dict(tp_size=3, sp_size=2, sp_mode="ulysses"), dict(num_kv_heads=2)),
    ("pp2_tp2_per_stage_weights", dict(pp_size=2, tp_size=2, tp_weights=[[2, 1], [1, 2]]), {}),
    ("pp2_sp2_zigzag", dict(pp_size=2, sp_size=2, sp_layout="zigzag", num_microbatches=4), {}),
    ("pp2_sp2_tp2_ring_megatron", dict(pp_size=2, sp_size=2, tp_size=2, sp_layout="zigzag", megatron_sp=True), {}),
    ("pp2_sp2_tp2_ulysses", dict(pp_size=2, sp_size=2, tp_size=2, sp_mode="ulysses"), {}),
    ("pp2_tp2_biases_tied", dict(pp_size=2, tp_size=2), dict(qkv_bias=True, o_bias=True, mlp_bias=True, tie_word_embeddings=True)),
    # SP ranks share the dense compute of each layer while decoding (opt-in)
    ("sp2_ring_decode_split", dict(sp_size=2, sp_decode_split=True), {}),
    ("sp2_ulysses_decode_split", dict(sp_size=2, sp_mode="ulysses", sp_decode_split=True), {}),
    ("sp3_ring_short_prompts_decode_split", dict(sp_size=3, num_microbatches=4, sp_decode_split=True), {}),
    ("sp2_ulysses_mqa_decode_split", dict(sp_size=2, sp_mode="ulysses", sp_decode_split=True), dict(num_kv_heads=1)),
    ("tp2_sp2_ring_decode_split", dict(tp_size=2, sp_size=2, sp_decode_split=True), {}),
    ("tp3_sp2_ulysses_decode_split", dict(tp_size=3, sp_size=2, sp_mode="ulysses", sp_decode_split=True), dict(num_kv_heads=2)),
    ("pp2_sp2_tp2_decode_split", dict(pp_size=2, sp_size=2, tp_size=2, sp_layout="zigzag", sp_decode_split=True), {}),
    ("sp4_heads_lt_ranks_decode_split", dict(sp_size=4, sp_decode_split=True), dict(num_heads=2, num_kv_heads=1, head_dim=32)),
    # chunked (pipelined) prefill: later chunks attend to the cache of earlier ones
    ("single_chunked_prefill", dict(prefill_chunk=4), {}),
    ("pp2_chunked_prefill", dict(pp_size=2, prefill_chunk=3), {}),
    ("pp3_chunked_microbatches", dict(pp_size=3, prefill_chunk=2, num_microbatches=2), {}),
    ("pp2_sp2_ulysses_chunked", dict(pp_size=2, sp_size=2, sp_mode="ulysses", prefill_chunk=4), {}),
    ("pp2_sp2_ring_chunked", dict(pp_size=2, sp_size=2, prefill_chunk=4), {}),
    ("pp2_tp2_megatron_chunked", dict(pp_size=2, tp_size=2, megatron_sp=True, prefill_chunk=3), {}),
]


def _model(overrides):
    return ModelConfig.tiny(**{**BASE, **overrides})


def _padded(prompts):
    S = max(len(p) for p in prompts)
    ids = torch.zeros(len(prompts), S, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for b, p in enumerate(prompts):
        ids[b, S - len(p) :] = torch.tensor(p)
        mask[b, S - len(p) :] = 1
    return ids, mask


@functools.lru_cache(maxsize=None)
def _reference(overrides_items):
    cfg = _model(dict(overrides_items))
    sd = random_state_dict(cfg, seed=0, dtype=torch.float64)
    logits = [dense_logits(sd, cfg, p) for p in PROMPTS]
    gens = [dense_generate(sd, cfg, p, NEW_TOKENS) for p in PROMPTS]
    return logits, [g[0] for g in gens], [g[1] for g in gens]


def _engine_worker(rank, world, pc_dict, overrides, ref_logits, ref_scores):
    cfg = _model(overrides)
    eng = CollabEngine(cfg, ParallelConfig.from_dict(pc_dict), random_state_dict(cfg, seed=0, dtype=torch.float64), dtype=torch.float64)
    ids, mask = _padded(PROMPTS)
    logits = eng.forward(ids if rank == 0 else None, mask if rank == 0 else None, broadcast=True)
    S = ids.shape[1]
    fwd_err = max((logits[b, S - len(p) :] - ref_logits[b]).abs().max().item() for b, p in enumerate(PROMPTS))
    res = eng.generate(PROMPTS if rank == 0 else None, max_new_tokens=NEW_TOKENS, return_scores=True)
    score_err = None
    if res.scores is not None:
        score_err = max((res.scores[b] - ref_scores[b]).abs().max().item() for b in range(len(PROMPTS)))
    return dict(fwd_err=fwd_err, tokens=res.tokens, score_err=score_err, last=eng.ctx.is_last_stage)


@pytest.mark.parametrize("name,pc,overrides", CASES, ids=[c[0] for c in CASES])
def test_engine_matches_dense_reference(name, pc, overrides):
    ref_logits, ref_tokens, ref_scores = _reference(tuple(sorted(overrides.items())))
    world = ParallelConfig.from_dict(pc).world_size
    results = run_dist(_engine_worker, world, pc, overrides, ref_logits, ref_scores)
    for rank, r in enumerate(results):
        assert r["fwd_err"] < 1e-10, (rank, r["fwd_err"])
        assert r["tokens"] == ref_tokens, rank
        if r["last"]:
            assert r["score_err"] is not None and r["score_err"] < 1e-10, (rank, r["score_err"])
        else:
            assert r["score_err"] is None


# --------------------------------------------------------------------------
def _eos_worker(rank, world, pc_dict, eos):
    cfg = _model({})
    eng = CollabEngine(cfg, ParallelConfig.from_dict(pc_dict), random_state_dict(cfg, seed=0, dtype=torch.float64), dtype=torch.float64)
    return eng.generate(PROMPTS, max_new_tokens=NEW_TOKENS, eos_token_id=eos).tokens


@pytest.mark.parametrize("pc", [dict(), dict(pp_size=2, num_microbatches=4), dict(pp_size=2, sp_size=2)])
def test_eos_stops_sequences(pc):
    _, ref_tokens, _ = _reference(())
    eos = ref_tokens[0][2]
    expected = [t[: t.index(eos) + 1] if eos in t else t for t in ref_tokens]
    world = ParallelConfig.from_dict(pc).world_size
    for tokens in run_dist(_eos_worker, world, pc, eos):
        assert tokens == expected


# --------------------------------------------------------------------------
def _sampling_worker(rank, world, pc_dict):
    cfg = _model({})
    eng = CollabEngine(cfg, ParallelConfig.from_dict(pc_dict), random_state_dict(cfg, seed=0, dtype=torch.float64), dtype=torch.float64)
    params = SamplingParams(temperature=0.9, top_k=30, top_p=0.95, seed=7)
    return eng.generate(PROMPTS, max_new_tokens=NEW_TOKENS, sampling=params).tokens


@pytest.mark.parametrize(
    "pc",
    [dict(pp_size=2, sp_size=2, tp_size=2), dict(pp_size=2, prefill_chunk=3), dict(sp_size=2, sp_mode="ulysses")],
    ids=["pp2_sp2_tp2", "pp2_chunked", "sp2_ulysses"],
)
def test_sampling_is_consistent_across_ranks_and_layouts(pc):
    single = run_dist(_sampling_worker, 1, {})[0]
    parallel = run_dist(_sampling_worker, ParallelConfig.from_dict(pc).world_size, pc)
    assert all(tokens == single for tokens in parallel)


# --------------------------------------------------------------------------
def _kv_worker(rank, world, pc_dict):
    cfg = _model({})
    eng = CollabEngine(cfg, ParallelConfig.from_dict(pc_dict), random_state_dict(cfg, seed=0, dtype=torch.float64), dtype=torch.float64)
    eng.generate(PROMPTS, max_new_tokens=NEW_TOKENS)
    return eng.last_kv_cache_bytes


@pytest.mark.parametrize(
    "pc",
    # one micro-batch everywhere so padding (which is cached too) is identical
    [dict(sp_size=2), dict(sp_size=2, sp_mode="ulysses"), dict(tp_size=2), dict(pp_size=2, num_microbatches=1)],
    ids=["ring", "ulysses", "tp", "pp"],
)
def test_kv_cache_is_distributed(pc):
    total = run_dist(_kv_worker, 1, {})[0]
    per_rank = run_dist(_kv_worker, 2, pc)
    assert sum(per_rank) == total  # nothing is replicated
    assert max(per_rank) <= 0.6 * total  # and the load is balanced


# --------------------------------------------------------------------------
def _compressed_worker(rank, world, pc_dict, ref_logits):
    cfg = _model({})
    eng = CollabEngine(cfg, ParallelConfig.from_dict(pc_dict), random_state_dict(cfg, seed=0, dtype=torch.float64), dtype=torch.float64)
    ids, mask = _padded(PROMPTS)
    logits = eng.forward(ids, mask, broadcast=True)
    S = ids.shape[1]
    err = max(
        ((logits[b, S - len(p) :] - ref_logits[b]).abs().max() / ref_logits[b].abs().max()).item()
        for b, p in enumerate(PROMPTS)
    )
    tokens = eng.generate(PROMPTS, max_new_tokens=NEW_TOKENS).tokens
    return err, eng.comm_stats.total_bytes, tokens


@pytest.mark.parametrize(
    "pc,tol",
    [
        (dict(pp_size=2, comm_dtype="int8"), 5e-2),
        (dict(tp_size=2, comm_dtype="float16"), 5e-3),
        (dict(sp_size=2, comm_dtype="bfloat16"), 5e-2),
        (dict(sp_size=2, sp_mode="ulysses", comm_dtype="float16"), 5e-3),
        (dict(pp_size=2, sp_size=2, tp_size=2, comm_dtype="bfloat16"), 5e-2),
    ],
    ids=["pp2_int8", "tp2_fp16", "sp2_ring_bf16", "sp2_ulysses_fp16", "pp2_sp2_tp2_bf16"],
)
def test_compressed_communication_is_close_and_smaller(pc, tol):
    ref_logits, ref_tokens, _ = _reference(())
    world = ParallelConfig.from_dict(pc).world_size
    compressed = run_dist(_compressed_worker, world, pc, ref_logits)
    plain = run_dist(_compressed_worker, world, {k: v for k, v in pc.items() if k != "comm_dtype"}, ref_logits)
    assert max(r[0] for r in compressed) < tol
    assert sum(r[1] for r in compressed) < 0.8 * sum(r[1] for r in plain)  # fewer bytes on the wire
    for r in compressed:
        assert len(r[2]) == len(PROMPTS) and all(len(t) == NEW_TOKENS for t in r[2])
