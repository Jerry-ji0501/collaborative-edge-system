"""Command line interface.

``collab-infer generate`` runs text generation with any parallel strategy:

* ``--nproc N`` simulates an N-device cluster on this machine (processes are
  spawned automatically, optionally with ``--bandwidth-mbps/--latency-ms``
  network emulation);
* under ``torchrun`` (or with ``RANK``/``WORLD_SIZE``/``MASTER_ADDR``/
  ``MASTER_PORT`` set) every device runs the same command and joins the
  cluster.

``collab-infer plan`` ranks parallel strategies for a list of edge devices.

Examples::

    collab-infer generate --tiny --nproc 4 --tp 2 --pp 2 --token-ids 1,2,3
    collab-infer generate --model ./TinyLlama-1.1B --nproc 4 --sp 2 --tp 2 --prompt "Hello"
    collab-infer plan --tiny --devices devices.json --prompt-len 2048
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch

from .config import NetworkConfig, ParallelConfig
from .models.config import PRESETS, ModelConfig

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "float64": torch.float64}


def _floats(text: Optional[str]) -> Optional[List[float]]:
    return None if not text else [float(x) for x in text.split(",")]


def _ints(text: Optional[str]) -> Optional[List[int]]:
    return None if not text else [int(x) for x in text.split(",")]


def _tp_weights(text: Optional[str]):
    """``2,1`` (all stages) or ``2,1;1,1`` (per stage)."""
    if not text:
        return None
    stages = [_floats(part) for part in text.split(";")]
    return stages[0] if len(stages) == 1 else stages


def _parallel_config(args: argparse.Namespace) -> ParallelConfig:
    network = None
    if args.bandwidth_mbps or args.latency_ms:
        network = NetworkConfig(bandwidth_mbps=args.bandwidth_mbps, latency_ms=args.latency_ms or 0.0)
    return ParallelConfig(
        tp_size=args.tp,
        pp_size=args.pp,
        sp_size=args.sp,
        sp_mode=args.sp_mode,
        sp_layout=args.sp_layout,
        megatron_sp=args.megatron_sp,
        tp_weights=_tp_weights(args.tp_weights),
        sp_weights=_floats(args.sp_weights),
        pp_layers=_ints(args.pp_layers),
        num_microbatches=args.microbatches,
        attn_kv_block=args.kv_block,
        network=network,
    )


def _model_config(args: argparse.Namespace) -> ModelConfig:
    if args.model:
        return ModelConfig.from_hf(args.model)
    if args.preset:
        return ModelConfig.preset(args.preset)
    return ModelConfig.tiny(num_layers=args.tiny_layers)


def _tokenizer(args: argparse.Namespace):
    if not args.model or not os.path.isdir(args.model):
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    return AutoTokenizer.from_pretrained(args.model)


def _run_generate(rank: int, world: int, args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Body executed by every rank."""
    from .distributed.launcher import resolve_device
    from .engine import CollabEngine, SamplingParams
    from .models.llama import random_state_dict

    args = argparse.Namespace(**args_dict)
    device = resolve_device(args.device, os.environ.get("COLLAB_BACKEND", args.backend), int(os.environ.get("LOCAL_RANK", rank)))
    model_cfg = _model_config(args)
    weights = args.model if args.model else random_state_dict(model_cfg, seed=args.seed or 0)
    engine = CollabEngine(model_cfg, _parallel_config(args), weights, device=device, dtype=_DTYPES[args.dtype])
    tokenizer = _tokenizer(args) if rank == 0 else None
    prompts = None
    if rank == 0:
        if args.token_ids:
            prompts = [_ints(p) for p in args.token_ids.split(";")]
        elif tokenizer is not None:
            prompts = [tokenizer(p)["input_ids"] for p in args.prompt]
        else:
            raise SystemExit("--prompt needs a tokenizer (HF model dir + transformers); use --token-ids instead")
    eos = args.eos_token_id
    if eos is None and tokenizer is not None:
        eos = tokenizer.eos_token_id
    sampling = SamplingParams(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed)
    for _ in range(args.warmup):
        engine.generate(prompts, max_new_tokens=2, sampling=sampling)
    engine.comm_stats.reset()
    result = engine.generate(
        prompts, max_new_tokens=args.max_new_tokens, sampling=sampling, eos_token_id=eos, num_microbatches=args.microbatches
    )
    info = {
        "rank": rank,
        "coords": (engine.ctx.pp_rank, engine.ctx.sp_rank, engine.ctx.tp_rank),
        "layers": engine.model.layer_ids,
        "param_mb": engine.model.parameter_bytes() / 1e6,
        "kv_cache_mb": engine.last_kv_cache_bytes / 1e6,
        "comm_mb": engine.comm_stats.total_bytes / 1e6,
        "comm_calls": engine.comm_stats.total_calls,
    }
    if rank == 0:
        texts = None
        if tokenizer is not None:
            texts = [tokenizer.decode(t, skip_special_tokens=True) for t in result.tokens]
        info.update(tokens=result.tokens, texts=texts, stats=result.stats, config=engine.pc.describe())
    return info


def _print_report(infos: List[Dict[str, Any]]) -> None:
    head = infos[0]
    print(f"\n{head['config']}")
    for i, tokens in enumerate(head["tokens"]):
        print(f"[prompt {i}] tokens: {tokens}")
        if head.get("texts"):
            print(f"[prompt {i}] text:   {head['texts'][i]!r}")
    s = head["stats"]
    print(
        f"\ntotal {s['total_s']:.3f}s | time-to-first-token {s['ttft_s']:.3f}s | "
        f"decode {s['decode_tokens_per_s']:.1f} tok/s | micro-batches {int(s['num_microbatches'])}"
    )
    if len(infos) > 1 or "param_mb" in head:
        print(f"\n{'rank':>4} {'(pp,sp,tp)':>11} {'layers':>12} {'params MB':>10} {'KV MB':>8} {'sent MB':>9} {'calls':>7}")
        for info in infos:
            layers = info["layers"]
            span = f"{layers[0]}-{layers[-1]}" if layers else "-"
            print(
                f"{info['rank']:>4} {str(info['coords']):>11} {span:>12} {info['param_mb']:>10.2f} "
                f"{info['kv_cache_mb']:>8.3f} {info['comm_mb']:>9.3f} {info['comm_calls']:>7}"
            )


def cmd_generate(args: argparse.Namespace) -> None:
    pc = _parallel_config(args)
    pc.validate()
    if not args.token_ids and not args.prompt:
        args.token_ids = "1,2,3,4,5,6,7,8"
    args_dict = vars(args).copy()
    args_dict.pop("func", None)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:  # launched by torchrun / on a real device
        from .distributed.launcher import init_distributed

        init_distributed(args.backend)
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        if world != pc.world_size:
            raise SystemExit(f"WORLD_SIZE={world} but tp*pp*sp={pc.world_size}")
        info = _run_generate(rank, world, args_dict)
        if rank == 0:
            _print_report([info])
        return
    world = args.nproc or pc.world_size
    if world != pc.world_size:
        raise SystemExit(f"--nproc {world} does not match tp*pp*sp = {pc.world_size}")
    if world == 1:
        _print_report([_run_generate(0, 1, args_dict)])
        return
    from .distributed.launcher import launch_local

    _print_report(launch_local(_run_generate, world, args=(args_dict,), backend=args.backend, return_results=True))


def cmd_plan(args: argparse.Namespace) -> None:
    from .planner import DeviceProfile, Workload, search

    with open(args.devices, "r", encoding="utf-8") as f:
        devices = [DeviceProfile(**d) for d in json.load(f)]
    workload = Workload(
        batch_size=args.batch_size, prompt_len=args.prompt_len, new_tokens=args.new_tokens, dtype_bytes=args.dtype_bytes
    )
    model_cfg = _model_config(args)
    print(f"{len(devices)} devices: " + ", ".join(d.name for d in devices))
    for i, result in enumerate(search(model_cfg, devices, workload, top_k=args.top_k)):
        e = result.estimate
        mem = ", ".join(f"{m:.2f}" for m in e.device_memory_gb)
        print(
            f"{i + 1:>2}. {result.config.describe()}\n"
            f"    prefill {e.prefill_s:.3f}s, decode {e.decode_step_s * 1e3:.1f} ms/token, "
            f"total {e.total_s:.2f}s, memory/device GB [{mem}]{'' if e.feasible else '  (OUT OF MEMORY)'}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collab-infer",
        description="Collaborative inference over edge devices with tensor, pipeline and sequence parallelism.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", help="Hugging Face model directory (LLaMA / Mistral / Qwen2)")
        p.add_argument("--preset", choices=sorted(PRESETS), help="architecture of a known model (random weights)")
        p.add_argument("--tiny", action="store_true", help="use a small random model (the default)")
        p.add_argument("--tiny-layers", type=int, default=4)

    g = sub.add_parser("generate", help="generate text with a parallel strategy")
    model_args(g)
    g.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    g.add_argument("--pp", type=int, default=1, help="pipeline-parallel size")
    g.add_argument("--sp", type=int, default=1, help="sequence-parallel size")
    g.add_argument("--sp-mode", default="ring", choices=["ring", "ulysses"])
    g.add_argument("--sp-layout", default="contiguous", choices=["contiguous", "zigzag"])
    g.add_argument("--megatron-sp", action="store_true", help="Megatron sequence parallelism inside TP groups")
    g.add_argument("--tp-weights", help="e.g. '2,1' or per stage '2,1;1,1'")
    g.add_argument("--sp-weights", help="e.g. '3,1'")
    g.add_argument("--pp-layers", help="layers per stage, e.g. '10,12'")
    g.add_argument("--microbatches", type=int)
    g.add_argument("--kv-block", type=int, help="attention key block size (bounded memory)")
    g.add_argument("--nproc", type=int, help="simulate this many devices locally")
    g.add_argument("--backend", default="gloo")
    g.add_argument("--device", default=None, help="cpu | cuda | auto")
    g.add_argument("--dtype", default="float32", choices=sorted(_DTYPES))
    g.add_argument("--prompt", action="append", help="prompt text (repeatable)")
    g.add_argument("--token-ids", help="prompts as token ids: '1,2,3;4,5'")
    g.add_argument("--max-new-tokens", type=int, default=16)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--eos-token-id", type=int)
    g.add_argument("--warmup", type=int, default=0, help="untimed warm-up generations")
    g.add_argument("--bandwidth-mbps", type=float, help="emulated link bandwidth")
    g.add_argument("--latency-ms", type=float, help="emulated link latency")
    g.set_defaults(func=cmd_generate)

    p = sub.add_parser("plan", help="rank parallel strategies for a set of devices")
    model_args(p)
    p.add_argument("--devices", required=True, help="JSON list of DeviceProfile fields")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--dtype-bytes", type=int, default=2)
    p.add_argument("--top-k", type=int, default=8)
    p.set_defaults(func=cmd_plan)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
