# collab-infer: a general collaborative inference framework for edge devices

[![CI](https://github.com/Jerry-ji0501/collaborative-edge-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Jerry-ji0501/collaborative-edge-system/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

English | [简体中文](README.zh-CN.md)

`collab_infer` lets several (possibly heterogeneous) edge devices run inference for one model together. It supports three kinds of parallelism and any combination of them:

| Parallelism | What is split | Communication | Main benefit |
| --- | --- | --- | --- |
| **Tensor parallelism, TP** (Megatron-LM style) | The weight matrices of every layer (attention heads / MLP units / vocabulary) | 2 all-reduces per layer (optionally Megatron-SP: reduce-scatter + all-gather) | Per-layer compute and weight memory are divided across devices; lowest latency (needs a fast link) |
| **Pipeline parallelism, PP** | Consecutive layers → consecutive devices (stages) | Point-to-point activation transfers at stage boundaries only | Least traffic, suited to weak networks such as Wi‑Fi; micro-batches keep every stage busy |
| **Sequence parallelism, SP** (Ring Attention / DeepSpeed‑Ulysses) | The tokens of the sequence | Ring: KV blocks passed around a ring (overlapped with compute); Ulysses: 2 all-to-alls | Faster long-context prefill; **the KV cache is spread over several devices**, lifting the single-device memory limit |

Together they form a `pp × sp × tp` 3D device mesh and can be combined freely (for example `pp=2, sp=2, tp=2` for 8 devices). Every dimension supports **uneven splits weighted by device capability** (heterogeneous edge clusters), and the framework ships an **automatic planner**, a **network simulator** and **communication statistics**.

> Correctness: in float64, the test suite compares 37 parallel configurations (all combinations, uneven splits, zigzag, Megatron-SP, GQA/MQA, padding, chunked pipelined prefill, SP decode split, EOS, sampling) logit by logit against an independently written single-device reference model (error ~1e-16), and checks parity with the Hugging Face `transformers` LLaMA / Qwen2 implementations (float32 error ~2e-7).

---

## Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Deploying on multiple real devices](#deploying-on-multiple-real-devices)
- [Loading Hugging Face models](#loading-hugging-face-models)
- [Heterogeneous clusters and automatic planning](#heterogeneous-clusters-and-automatic-planning)
- [Using the parallel primitives on your own models](#using-the-parallel-primitives-on-your-own-models)
- [Network simulation and strategy comparison](#network-simulation-and-strategy-comparison)
- [Inference optimizations](#inference-optimizations)
- [Configuration reference](#configuration-reference)
- [Key design decisions](#key-design-decisions)
- [Tests](#tests)
- [Limitations and future work](#limitations-and-future-work)
- [Repository layout](#repository-layout)
- [License](#license)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ planner/     device profiles → TP/SP weights, PP layer split (DP),   │
│              cost model, strategy search                             │
├──────────────────────────────────────────────────────────────────────┤
│ engine/      CollabEngine: prefill / decode / generation scheduling  │
│              (micro-batch pipelining, token loop-back, STOP control  │
│              messages), sampling                                     │
├──────────────────────────────────────────────────────────────────────┤
│ models/      parallelized LLaMA-family models (LLaMA / Mistral /     │
│              Qwen2), KV cache, sharded weight loading (state dict /  │
│              torch mmap / safetensors / HF)                          │
├──────────────────────────────────────────────────────────────────────┤
│ parallel/    model-agnostic parallel primitives                      │
│   tensor_parallel   Column/Row/VocabParallel layers,                 │
│                     parallelize_module                               │
│   sequence_parallel ring_attention, ulysses_attention,               │
│                     distributed KV decoding                          │
│   pipeline_parallel P2PChannel, PipelineRunner, split_sequential     │
│   attention         attention kernels with log-sum-exp and exact     │
│                     merging                                          │
│   partition         weighted splits, GQA-aware head partitioning,    │
│                     sequence layouts                                 │
├──────────────────────────────────────────────────────────────────────┤
│ distributed/ ParallelContext (3D mesh and process groups),           │
│              Communicator (uneven collectives, gloo fallbacks,       │
│              statistics, network simulation), launcher               │
└──────────────────────────────────────────────────────────────────────┘
```

Global ranks map to mesh coordinates as `rank = (pp_rank * sp + sp_rank) * tp + tp_rank`, so the members of a TP group (which communicates most often) have adjacent ranks. Each process builds and loads **only its own share of the model**: the layers of its stage and its TP shard of their weights.

---

## Installation

```bash
pip install -e .            # requires torch >= 2.1
pip install -e ".[hf,test]" # optional: load HF models / tokenizers (transformers, safetensors) and run the tests
```

CPU-only edge devices (Raspberry Pi, ARM boards, ...) use the `gloo` backend. GPU devices can use `nccl`; mixed GPU/CPU clusters use `gloo` (tensors are staged through host memory automatically).

---

## Quick start

### 1. Command line: simulate a multi-device cluster on one machine

```bash
# 8 processes simulate 8 devices: 2 pipeline stages × 2-way sequence parallel × 2-way tensor parallel
python -m collab_infer generate --nproc 8 --pp 2 --sp 2 --tp 2 \
    --sp-layout zigzag --megatron-sp --token-ids "1,2,3,4;5,6" --max-new-tokens 8

# Also simulate a 100 Mbps / 2 ms Wi‑Fi link
python -m collab_infer generate --nproc 4 --tp 2 --pp 2 --bandwidth-mbps 100 --latency-ms 2
```

The output contains the generated tokens, time to first token, decode speed, and for every rank its layer range, parameter size, KV cache size and traffic:

```
rank  (pp,sp,tp)       layers  params MB    KV MB   sent MB   calls
   0   (0, 0, 0)          0-1       0.19    0.002     0.030     152
   ...
   7   (1, 1, 1)          2-3       0.19    0.002     0.027     167
```

### 2. Python API

```python
import torch
from collab_infer import CollabEngine, ModelConfig, ParallelConfig, launch_local, random_state_dict

MODEL = ModelConfig.tiny()          # or ModelConfig.from_hf("/path/to/hf-model")

def run(rank, world_size):
    engine = CollabEngine(
        MODEL,
        ParallelConfig(pp_size=2, sp_size=1, tp_size=2),
        random_state_dict(MODEL),   # or a checkpoint path / HF directory: each rank reads only its own shard
    )
    result = engine.generate([[1, 5, 9, 14], [2, 4, 8]] if rank == 0 else None, max_new_tokens=16)
    return result.tokens            # identical on every rank

print(launch_local(run, world_size=4, return_results=True)[0])
```

`CollabEngine` is SPMD: every rank constructs the same engine and calls the same method. Inputs are read on rank 0 and broadcast; results are returned on every rank.

- `engine.generate(prompts, max_new_tokens, sampling=SamplingParams(...), eos_token_id=..., return_scores=...)`: batched generation; prompts may have different lengths (left-padded automatically).
- `engine.forward(input_ids, attention_mask, broadcast=True)`: returns the logits for every position.
- `engine.comm_stats`, `engine.memory_report()`, `engine.last_kv_cache_bytes`: communication and memory statistics.

More examples in [`examples/quickstart.py`](examples/quickstart.py).

---

## Deploying on multiple real devices

Every device runs **the same command** and joins the cluster through `torchrun` or environment variables (the device hosting rank 0 is the master):

```bash
# Device A (192.168.1.10, node_rank=0)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr=192.168.1.10 --master_port=29500 \
    -m collab_infer generate --model /models/TinyLlama-1.1B --pp 2 --prompt "Hello"

# Device B (node_rank=1)
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
    --master_addr=192.168.1.10 --master_port=29500 \
    -m collab_infer generate --model /models/TinyLlama-1.1B --pp 2 --prompt "Hello"
```

Without torchrun, set `RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT` and run `python -m collab_infer generate ...` directly. From your own scripts, call `collab_infer.init_distributed()`.

Tips:
- On devices with several network interfaces, set `GLOO_SOCKET_IFNAME=wlan0` (or `eth0`) so that gloo binds to an interface the other devices can reach.
- If the devices run different PyTorch versions, set `COLLAB_INFER_COMM_FALLBACK=1` to use the point-to-point implementations of all-to-all / reduce-scatter everywhere.

---

## Loading Hugging Face models

The `llama`, `mistral` and `qwen2` architectures are supported (GQA, QKV bias, tied embeddings, `linear` / `llama3` RoPE scaling):

```python
cfg = ModelConfig.from_hf("/models/Qwen2-0.5B")                 # reads config.json
engine = CollabEngine(cfg, ParallelConfig(tp_size=2), "/models/Qwen2-0.5B")  # reads safetensors / .bin (including sharded indexes)
```

Weights are read by their HF parameter names and each rank takes only the slices it needs. Weights in `torch.save` format are opened with mmap, so a memory-constrained device only touches the pages that hold its own shard.

---

## Heterogeneous clusters and automatic planning

Edge clusters often mix devices with very different compute and memory. Every split can be weighted:

```python
ParallelConfig(
    tp_size=2, tp_weights=[3, 1],        # the fast device gets 3/4 of the attention heads and MLP units
    sp_size=2, sp_weights=[2, 1],        # the fast device handles 2/3 of the tokens
    pp_size=2, pp_layers=[20, 12],       # number of layers per stage
)
# tp_weights can also be given per stage: [[3, 1], [1, 1]]
```

The planner derives these parameters from device profiles and ranks every `pp × sp × tp` combination with an analytical cost model:

```python
from collab_infer.planner import DeviceProfile, Workload, plan, search

devices = [
    DeviceProfile("jetson-orin", tflops=5.0, memory_gb=8,  mem_bandwidth_gbps=100),
    DeviceProfile("laptop",      tflops=2.0, memory_gb=16, mem_bandwidth_gbps=50),
    DeviceProfile("rpi5-a",      tflops=0.1, memory_gb=4,  mem_bandwidth_gbps=10),
    DeviceProfile("rpi5-b",      tflops=0.1, memory_gb=4,  mem_bandwidth_gbps=10),
]
model = ModelConfig.preset("tinyllama-1.1b")
print(plan(model, devices, pp_size=4).pp_layers)      # [14, 6, 1, 1]: dynamic programming minimizes the slowest stage under memory constraints
for r in search(model, devices, Workload(prompt_len=1024, new_tokens=64), top_k=3):
    print(r)
```

Command line: `python -m collab_infer plan --preset tinyllama-1.1b --devices devices.json` (`devices.json` is a list of `DeviceProfile` fields).

[`examples/heterogeneous_edge.py`](examples/heterogeneous_edge.py) simulates a fast and a slow device on one machine by giving them different CPU thread counts (measured 363 vs 132 GFLOP/s). It measures their compute first, hands the numbers to the planner, and compares an even split with the planned split:

```
case               TTFT s  total s  partition
TP2 even            0.920    1.165  even
TP2 planned         0.844    1.039  [0.363, 0.132]
PP2 even            1.050    1.220  even
PP2 planned         0.830    0.976  [6, 2]
```

(Both splits generate exactly the same tokens. Numbers come from a 4-core CPU container and are only indicative.)

---

## Using the parallel primitives on your own models

The primitives in `parallel/` are model-agnostic and work on your own models directly (see [`examples/custom_models.py`](examples/custom_models.py)):

```python
from collab_infer import ParallelContext
from collab_infer.parallel import (ColwiseParallel, RowwiseParallel, parallelize_module,
                                   PipelineRunner, split_sequential,
                                   SequenceLayout, ring_attention, gather_sequence)

# Tensor parallelism: shard the nn.Linear / nn.Embedding layers of any model by name pattern
ctx = ParallelContext(tp_size=2)
parallelize_module(block, ctx.tp, {
    "qkv.*": ColwiseParallel(granularity=head_dim),   # split on whole heads
    "proj":  RowwiseParallel(granularity=head_dim),
    "fc1":   ColwiseParallel(), "fc2": RowwiseParallel(),
})

# Pipeline parallelism: cut a sequential model such as a CNN into stages and pipeline micro-batches through them
ctx = ParallelContext(pp_size=3)
stage = split_sequential(list(cnn), ctx.pp, layer_counts=[3, 3, 5])
logits = PipelineRunner(stage, ctx.pp).forward(images, num_microbatches=4, return_to="first")

# Sequence parallelism: replace dense attention with ring_attention in your own attention module
out = ring_attention(q, k, v, q_pos, k_pos, ctx.sp, layout.sizes, causal=False)
```

---

## Network simulation and strategy comparison

`NetworkConfig(bandwidth_mbps, latency_ms)` adds `latency × steps + bytes / bandwidth` to every communication (pipeline messages queue on the link; ring attention transfers overlap with compute), so strategies can be evaluated under different network conditions on a single machine. [`examples/benchmark_strategies.py`](examples/benchmark_strategies.py) compares strategies on 4 simulated devices (8 layers, hidden size 256, batch 2, 256-token prompt, 16 generated tokens).

No network limit (local loopback):

```
strategy         TTFT s  decode tok/s  total s  max sent MB  max KV MB  max param MB
TP4               0.168          51.5    0.751        14.94       1.11          6.87
PP4               0.067         114.3    0.329         0.57       1.11          7.90
SP4-ring          0.110          55.0    0.655         3.93       1.11         27.41
SP4-ulysses       0.124          65.8    0.580         2.57       1.11         27.41
PP2xTP2           0.148          56.3    0.681         5.57       1.11          6.86
SP2xTP2           0.125          45.1    0.790         6.17       1.11         13.71
PP2xSP2           0.062          57.7    0.582         1.49       1.11         13.71
```

Simulated 100 Mbps / 5 ms Wi‑Fi:

```
strategy         TTFT s  decode tok/s  total s  max sent MB  max KV MB  max param MB
TP4               1.659          17.2    3.405        14.94       1.11          6.87
PP4               0.135          49.4    0.742         0.57       1.11          7.90
SP4-ring          0.444          28.4    1.500         3.93       1.11         27.41
SP4-ulysses       0.380          31.3    1.339         2.57       1.11         27.41
PP2xTP2           0.826          14.9    2.845         5.57       1.11          6.86
SP2xTP2           0.722          12.7    3.082         6.17       1.11         13.71
PP2xSP2           0.210          23.4    1.494         1.49       1.11         13.71
```

These show the typical edge trade-offs: TP runs two collectives per layer and is the most sensitive to latency and bandwidth; PP moves the least data and is the most robust on weak networks; SP spreads the KV cache evenly across devices (1/4 per device here), but weights are replicated within the SP group (hence the larger param MB), so it is usually combined with TP/PP.

---

## Inference optimizations

The table compares the version at the time [PR #1](https://github.com/Jerry-ji0501/collaborative-edge-system/pull/1) was merged ("before") with the current version ("after"), on the same 4-core x86 CPU, in the same session, with the same model and request: hidden size 1024, 8 layers, GQA with 16/4 heads, a 32k vocabulary (123M parameters, float32), one 1024-token prompt, 32 generated tokens, greedy decoding. Each simulated device gets a fixed number of CPU threads; "@100Mbps/2ms" means the edge link is simulated with the network simulator. Every configuration generates exactly the same tokens before and after.

| Scenario | Time to first token (before → after) | Decode ms/token (before → after) | MB sent per rank per generation (before → after) |
| --- | --- | --- | --- |
| single (4 threads) | 1376 → **701 ms** (2.0×) | 51.1 → **26.1** (2.0×) | 0.00 → 0.00 |
| TP2 | 1479 → **744 ms** (2.0×) | 92.8 → **42.7** (2.2×) | 73.51 → 72.42 |
| PP2 | 2525 → **1064 ms** (2.4×) | 75.7 → **52.1** (1.5×) | 4.27 → 4.27 |
| SP2 ring | 1182 → **743 ms** (1.6×) | 85.9 → **53.0** (1.6×) | 8.96 → 8.96 |
| SP2 ulysses | 1380 → **816 ms** (1.7×) | 77.7 → **59.6** (1.3×) | 21.27 → 21.27 |
| PP4 (1 thread each) | 4769 → **1102 ms** (4.3×) | 125.5 → **76.5** (1.6×) | 4.27 → 4.27 |
| TP2 @100Mbps/2ms | 7182 → **6597 ms** (1.1×) | 154.6 → **79.9** (1.9×) | 73.51 → 72.42 |
| PP2 @100Mbps/2ms | 2951 → **1111 ms** (2.7×) | 78.3 → **55.3** (1.4×) | 4.27 → 4.27 |
| SP2 ulysses @100Mbps/2ms | 3240 → **2529 ms** (1.3×) | 87.8 → **72.1** (1.2×) | 21.27 → 21.27 |
| TP2 @100Mbps/2ms +fp16 (new option) | 7182 → **3764 ms** (1.9×) | 154.6 → **78.9** (2.0×) | 73.51 → 36.21 |
| SP2 ulysses @100Mbps/2ms +fp16 (new option) | 3240 → **1695 ms** (1.9×) | 87.8 → **72.5** (1.2×) | 21.27 → 10.79 |

Notes: on the simulated link, TP2 prefill is bandwidth-bound (about 72 MB of all-reduce traffic per generation), so without compression the first token is only 1.1× faster, and 1.9× faster with `comm_dtype="float16"`. The PP4 time-to-first-token gain comes mainly from chunked pipelined prefill.

The optimizations (all lossless except communication compression):

| Optimization | How | Effect |
| --- | --- | --- |
| Fused attention kernels | PyTorch SDPA / the CPU flash kernel; no mask is built when positions allow it; GQA does not replicate K/V | Attention is 11–23× faster at 1024 tokens; its share of single-device prefill drops from about 60% to about 16% |
| Head-contiguous KV cache | The cache is stored as `[B, H, T, D]` and exposed as a token-major view | Decoding reads the cache without copies; long-context decode attention is about 30% faster |
| Fused projections | One GEMM each for Q/K/V and for gate/up; RoPE computed once per forward | Prefill projection GEMMs are 6–22% faster |
| Direct exchange for small messages | All-reduce / all-gather / broadcast of ≤64 KiB is done in one hop with a direct exchange, summed in rank order | Collective latency during decoding drops to about 1/3, bit-identical results; the cost is that all-reduce sends more bytes with more than 2 ranks (about 2× a ring at 4 ranks) |
| SP decode split (`sp_decode_split`, optional, off by default) | During decoding the SP ranks share the MLP and the attention output projection, and the TP and SP reductions are merged into one stage-wide all-reduce | Less duplicated compute but 1–2 extra collectives per layer: Ulysses decoding is about 10–30% faster on local loopback, but 1.35–2.2× slower on a simulated 100 Mbps/5 ms link, so it is off by default |
| Vocabulary-parallel sampling | The LM head is split over the vocabulary across the whole stage and only a few candidates are exchanged; Gumbel-max sampling with counter-based noise | No full-vocabulary logits gathered per token; sampling results do not depend on the parallel layout |
| Chunked pipelined prefill (`prefill_chunk`) | The prompt flows through the stages in chunks; later chunks use the KV cache of earlier ones | All stages work at once on a single request, lowering time to first token |
| Activation compression (`comm_dtype`, optional, lossy) | TP/SP collectives and pipeline activations travel as float16 / bfloat16, or pipeline activations are quantized to per-row int8 | Half the bytes (a quarter for int8 pipeline activations); time to first token drops noticeably on weak networks |

Recommendations:

- For **communication compression**, prefer `comm_dtype="float16"` (10-bit mantissa; out-of-range values saturate instead of overflowing). On random models with standard initialization (std 0.02), the relative logit error is ≤7e-4 for float16, ≤6e-3 for bfloat16 and ≤8e-3 for the int8 pipeline, and all 16 generated tokens match. On random models with larger initialization (std 0.05), where perturbations grow from layer to layer, bfloat16 / int8 change the output noticeably. Validate bfloat16 / int8 accuracy on your target model before using them.
- **SP decode split** is off by default: it trades extra collectives for less duplicated compute, and communication costs more on high-latency edge links. On low-latency links, with larger models or in Ulysses mode, enable it (`sp_decode_split=True` / `--sp-decode-split`) and measure.
- **Chunked pipelined prefill** is on by default when `pp_size > 1` (about 2 chunks per stage, at least 128 tokens each). It made PP4 time to first token about 1.9× faster on a single request in our measurements; when several devices are simulated on one machine the processes share memory bandwidth, so the gain is smaller than on a real multi-device deployment.
- **OpenMP wait policy**: when `launch_local` simulates several devices it sets `OMP_WAIT_POLICY=PASSIVE` by default, so idle compute threads do not starve the communication threads (TP2 decoding measured about 20% faster). It makes waking up small ops slower in a single process, though, so on real devices it is worth measuring both.
- **Weight precision**: the x86 CPU this repository was tested on has no native bf16 instructions, and PyTorch's int8 weight kernels were 4.6–58× slower than fp32 there, so int8 weight quantization is not provided. `dtype=torch.bfloat16` decodes about 1.4× faster with half the memory, but prefill is about 4× slower. The conclusions may differ on devices with native bf16/int8 instructions.

---

## Configuration reference

`ParallelConfig` fields:

| Field | Default | Description |
| --- | --- | --- |
| `tp_size` / `pp_size` / `sp_size` | 1 | Degree of each parallel dimension; their product equals the number of processes |
| `sp_mode` | `"ring"` | `"ring"` (KV blocks passed around a ring, KV cache split along the sequence) or `"ulysses"` (all-to-all, KV cache split across attention heads) |
| `sp_layout` | `"contiguous"` | `"zigzag"` balances the per-rank compute of causal attention |
| `megatron_sp` | `False` | Megatron sequence parallelism inside the TP group (norm/residual only process 1/tp of the tokens) |
| `tp_weights` | `None` | Relative capability of the TP ranks; a flat list (shared by all stages) or one list per stage |
| `sp_weights` | `None` | Relative capability of the SP ranks (uneven token split) |
| `pp_layers` | `None` | Number of layers per stage (even split by default) |
| `num_microbatches` | `pp_size` | Number of pipeline micro-batches |
| `prefill_chunk` | auto | Cuts the prompt into chunks that flow through the pipeline stages in parallel (lower time to first token for a single request); `None` picks a chunk size automatically when `pp_size > 1`, `0` disables it |
| `sp_decode_split` | `False` | During decoding the SP ranks share the MLP and attention output projection instead of each recomputing the whole layer (lossless, but 1–2 extra collectives per layer; enable only when decode compute clearly outweighs communication latency, e.g. larger models on fast links) |
| `comm_dtype` | `None` | Lossy activation compression: `"float16"` / `"bfloat16"` (half the bytes), `"int8"` (per-row int8 for pipeline activations, bfloat16 for collectives) |
| `attn_kv_block` | `None` | Computes attention in key blocks to bound peak memory on long sequences |
| `network` | `None` | `NetworkConfig(bandwidth_mbps, latency_ms)` network simulation |

---

## Key design decisions

- **Position-masked attention.** The attention mask is derived from the true position carried by each token (`k_pos <= q_pos`, `-1` for padding) rather than from tensor indices. Correctness therefore does not depend on how tokens are distributed across devices (contiguous, zigzag, uneven, or cached on any rank). Attention also returns the log-sum-exp, so partial results can be **merged exactly**; this is the basis of ring attention and distributed KV decoding.
- **Two shapes of SP: prefill and decode.** During prefill, tokens are split over the SP group according to a `SequenceLayout` (ring or Ulysses). During decoding the new token is replicated within the SP group; each rank attends only over the part of the KV cache it holds, and the results are merged with the LSE (one small all-gather per layer), so the KV cache always stays distributed. In ring mode, the KV of a new token is written to the rank whose cache is currently smallest; every rank reaches the same decision with the same deterministic rule, with no extra communication.
- **GQA-aware head partitioning.** Attention heads are assigned by whole KV groups where possible to avoid replicating KV; when there are fewer KV heads than devices (e.g. MQA), the needed KV heads are replicated automatically. After TP, Ulysses splits the local heads once more and handles irregular groupings the same way.
- **Self-describing pipeline channels.** Every `P2PChannel` message carries a fixed-size header (message type, dtype, shape), so stages do not need to know activation shapes in advance. A `STOP` control message travels with the data stream to retire micro-batches that have finished (EOS) early. During generation the last stage samples and sends the token back to the first stage, and several micro-batches interleave in the pipeline to keep every stage busy.
- **Vocabulary-parallel sampling.** The LM head is split over the vocabulary across all ranks of the last stage. Each rank proposes a few candidates, and after one small all-gather every rank arrives at the same token by pure comparison (greedy, temperature, top-k), without gathering full-vocabulary logits or broadcasting again. Top-p is decided by one rank and broadcast; when the candidates cannot cover the nucleus it falls back to the full vocabulary, so the result stays exact. Random sampling uses Gumbel-max with counter-based noise computed from `(seed, sequence, step, token)`, so sampling results do not depend on the parallel layout or micro-batch split, and all ranks agree exactly even if heterogeneous hardware produces slightly different logits.
- **Backend compatibility.** In some versions gloo does not support uneven all-gather / all-to-all (observed on torch 2.14 in this repository): all-gather pads and then trims, all-to-all uses uneven splits of `all_to_all_single`, and the code falls back to point-to-point implementations automatically when those are unsupported.
- **Direct exchange for small messages.** gloo's all-reduce needs several round trips for the small tensors of the decode phase. All-reduce / all-gather / broadcast of at most 64 KiB is instead done by every rank in a single direct exchange (measured latency about 1/3 of before); all-reduce sums in a fixed rank order, so the result is bit-identical on every rank. The cost is that all-reduce sends more bytes with more than 2 ranks (each rank sends the whole tensor directly to the other N−1 ranks, about 2× a ring algorithm at 4 ranks), but these messages are small and dominated by latency. The threshold can be set with the `COLLAB_INFER_SMALL_MSG_BYTES` environment variable or `ParallelContext(small_message_bytes=...)`; 0 disables it.
- **Attention kernels.** Attention is delegated to PyTorch's fused operators (SDPA; the CPU flash kernel when the LSE is needed). No mask is built when positions allow it (`is_causal` for prefill without padding; no mask during decoding when every cached key is visible), GQA does not replicate K/V, and queries are processed in chunks to bound memory when a mask is needed. The KV cache is stored head-contiguous.

---

## Tests

```bash
pip install pytest            # optional: pip install transformers safetensors to run the HF parity tests
python -m pytest tests -q
```

The tests launch multiple local processes over gloo and cover: communication primitives (native and fallback implementations, direct small-message exchange, uneven sizes, compression, network simulation), TP layers and `parallelize_module`, ring / Ulysses / distributed-KV attention, pipeline channels (including every compression encoding) and the runner, vocabulary-parallel sampling (token-by-token identical to a single device; sampling frequencies match the target distribution), the planner (DP checked against brute force), and the full engine matrix: for every configuration, every rank checks that prefill logits, greedily generated tokens and per-step logits match the independent reference implementation. EOS handling, sampling consistency across parallel layouts, even distribution of the KV cache, and the error and byte counts of compressed communication are tested too.

CI runs the whole suite, including the Hugging Face parity tests, on every push and pull request (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

---

## Limitations and future work

- Only tested on CPU + gloo. CUDA / NCCL and mixed CPU/GPU paths are supported by design but have not been verified in this repository's test environment.
- On CPU, attention uses PyTorch's fused operators; on GPU, the paths that need the LSE (ring attention, distributed KV decoding) currently use the math implementation and are not yet wired to CUDA fused kernels.
- The supported model families are currently LLaMA / Mistral / Qwen2; other architectures can be built directly from the primitives in `parallel/`.
- Continuous batching, speculative decoding, weight quantization and dynamic re-partitioning at runtime (device dropout / load changes) are not implemented yet. On the x86 CPU this repository was tested on (AVX-512, no native bf16), PyTorch's built-in int8 weight kernels were 4.6–58× slower than fp32, so int8 weight quantization is not provided; bf16 weights (`dtype=torch.bfloat16`) make decoding about 1.4× faster with half the memory, but prefill about 4× slower.
- The planner's cost model is an analytical estimate meant for comparing strategies and choosing uneven split ratios, not for predicting absolute latency precisely.

---

## Repository layout

```
collab_infer/
  config.py              ParallelConfig / NetworkConfig
  distributed/           comm.py (Communicator), mesh.py (ParallelContext), launcher.py
  parallel/              partition.py, attention.py, tensor_parallel.py, sequence_parallel.py, pipeline_parallel.py
  models/                config.py (ModelConfig, presets), llama.py (parallel model), cache.py, weights.py
  engine/                engine.py (CollabEngine), sampling.py
  planner/               planner.py (planning, cost model, strategy search)
  cli.py                 command line (generate / plan)
examples/                quickstart / custom_models / heterogeneous_edge / benchmark_strategies
tests/                   unit tests and multi-process equivalence tests
```

---

## License

[MIT](LICENSE)
