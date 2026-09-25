# collab-infer：面向边缘设备的通用协同推理框架

`collab_infer` 让多台（可能异构的）边缘设备协同完成一个模型的推理，支持三种并行方式及其任意组合：

| 并行方式 | 切分对象 | 通信 | 主要收益 |
| --- | --- | --- | --- |
| **张量并行 TP**（Megatron-LM 风格） | 每层的权重矩阵（注意力头 / MLP 单元 / 词表） | 每层 2 次 all-reduce（可选 Megatron-SP：reduce-scatter + all-gather） | 单层计算与权重内存按设备数切分，时延最低（需要高速链路） |
| **流水线并行 PP** | 连续的层 → 连续的设备（stage） | 仅在 stage 边界点对点传输激活 | 通信量最小，适合 Wi‑Fi 等弱网络；micro-batch 让所有 stage 同时工作 |
| **序列并行 SP**（Ring Attention / DeepSpeed‑Ulysses） | 序列中的 token | Ring：KV 块环形传递（与计算重叠）；Ulysses：2 次 all-to-all | 长上下文 prefill 加速，**KV Cache 分布在多台设备上**，突破单设备内存限制 |

三者构成一个 `pp × sp × tp` 的 3D 设备网格，可以任意组合（如 `pp=2, sp=2, tp=2` 共 8 台设备）。所有维度都支持**按设备能力的非均匀切分**（异构边缘集群），并提供**自动规划器**、**网络模拟器**和**通信统计**。

> 正确性：测试套件在 float64 下将 37 种并行配置（含所有组合、非均匀切分、zigzag、Megatron-SP、GQA/MQA、padding、分块流水 prefill、SP 解码拆分、EOS、采样）与一个独立实现的单设备参考模型逐 logit 比较（误差 ~1e-16），并与 Hugging Face `transformers` 的 LLaMA / Qwen2 实现对齐（float32 误差 ~2e-7）。

---

## 目录

- [架构](#架构)
- [安装](#安装)
- [快速开始](#快速开始)
- [在多台真实设备上部署](#在多台真实设备上部署)
- [加载 Hugging Face 模型](#加载-hugging-face-模型)
- [异构集群与自动规划](#异构集群与自动规划)
- [在自定义模型上使用并行原语](#在自定义模型上使用并行原语)
- [网络模拟与策略对比](#网络模拟与策略对比)
- [推理优化](#推理优化)
- [配置参考](#配置参考)
- [关键设计](#关键设计)
- [测试](#测试)
- [局限与后续工作](#局限与后续工作)

---

## 架构

```
┌──────────────────────────────────────────────────────────────────────┐
│ planner/     设备画像 → TP/SP 权重、PP 层划分（DP）、代价模型、策略搜索    │
├──────────────────────────────────────────────────────────────────────┤
│ engine/      CollabEngine：prefill / decode / 生成调度（micro-batch 流水、 │
│              token 回环、STOP 控制消息）、采样                            │
├──────────────────────────────────────────────────────────────────────┤
│ models/      并行化的 LLaMA 系列模型（LLaMA / Mistral / Qwen2）、KV Cache、 │
│              按分片加载权重（state dict / torch mmap / safetensors / HF） │
├──────────────────────────────────────────────────────────────────────┤
│ parallel/    模型无关的并行原语                                          │
│   tensor_parallel   Column/Row/VocabParallel 层、parallelize_module     │
│   sequence_parallel ring_attention、ulysses_attention、分布式 KV 解码     │
│   pipeline_parallel P2PChannel、PipelineRunner、split_sequential         │
│   attention         带 log-sum-exp 的注意力核与精确合并                   │
│   partition         加权切分、GQA 感知的注意力头划分、序列布局              │
├──────────────────────────────────────────────────────────────────────┤
│ distributed/ ParallelContext（3D 网格与通信组）、Communicator（非均匀集合 │
│              通信、gloo 回退实现、统计、网络模拟）、launcher                 │
└──────────────────────────────────────────────────────────────────────┘
```

全局 rank 与网格坐标的映射为 `rank = (pp_rank * sp + sp_rank) * tp + tp_rank`，TP 组（通信最频繁）的 rank 相邻。每个进程只构建并加载**属于自己的那一份模型**：自己 stage 的层、自己 TP 分片的权重。

---

## 安装

```bash
pip install -e .            # 依赖：torch >= 2.1
pip install -e ".[hf,test]" # 可选：加载 HF 模型 / 分词器（transformers、safetensors）及测试
```

纯 CPU 的边缘设备（树莓派、ARM 开发板等）使用 `gloo` 后端即可；GPU 设备可用 `nccl`，GPU 与 CPU 混合集群使用 `gloo`（通信时自动在主存中中转）。

---

## 快速开始

### 1. 命令行：在一台机器上模拟多设备集群

```bash
# 8 个进程模拟 8 台设备：2 个流水线 stage × 2 路序列并行 × 2 路张量并行
python -m collab_infer generate --nproc 8 --pp 2 --sp 2 --tp 2 \
    --sp-layout zigzag --megatron-sp --token-ids "1,2,3,4;5,6" --max-new-tokens 8

# 同时模拟 100 Mbps / 2 ms 的 Wi‑Fi 链路
python -m collab_infer generate --nproc 4 --tp 2 --pp 2 --bandwidth-mbps 100 --latency-ms 2
```

输出包含生成结果、首 token 时延、解码速度，以及每个 rank 的层范围、参数量、KV Cache 大小和通信量：

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

MODEL = ModelConfig.tiny()          # 或 ModelConfig.from_hf("/path/to/hf-model")

def run(rank, world_size):
    engine = CollabEngine(
        MODEL,
        ParallelConfig(pp_size=2, sp_size=1, tp_size=2),
        random_state_dict(MODEL),   # 或 checkpoint 路径 / HF 目录：每个 rank 只读取自己的分片
    )
    result = engine.generate([[1, 5, 9, 14], [2, 4, 8]] if rank == 0 else None, max_new_tokens=16)
    return result.tokens            # 所有 rank 上结果相同

print(launch_local(run, world_size=4, return_results=True)[0])
```

`CollabEngine` 采用 SPMD 方式：每个 rank 构造同一个引擎、调用同一个方法；输入从 rank 0 读取并广播，结果在所有 rank 上返回。

- `engine.generate(prompts, max_new_tokens, sampling=SamplingParams(...), eos_token_id=..., return_scores=...)`：批量生成，prompt 长度可不同（自动左填充）。
- `engine.forward(input_ids, attention_mask, broadcast=True)`：返回所有位置的 logits。
- `engine.comm_stats`、`engine.memory_report()`、`engine.last_kv_cache_bytes`：通信与内存统计。

更多示例见 [`examples/quickstart.py`](examples/quickstart.py)。

---

## 在多台真实设备上部署

每台设备运行**同一条命令**，通过 `torchrun` 或环境变量加入集群（rank 0 所在设备作为 master）：

```bash
# 设备 A（192.168.1.10，node_rank=0）
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr=192.168.1.10 --master_port=29500 \
    -m collab_infer generate --model /models/TinyLlama-1.1B --pp 2 --prompt "你好"

# 设备 B（node_rank=1）
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
    --master_addr=192.168.1.10 --master_port=29500 \
    -m collab_infer generate --model /models/TinyLlama-1.1B --pp 2 --prompt "你好"
```

不使用 torchrun 时，设置 `RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT` 后直接运行 `python -m collab_infer generate ...`。在自己的脚本中调用 `collab_infer.init_distributed()` 即可。

提示：
- 多网卡设备请设置 `GLOO_SOCKET_IFNAME=wlan0`（或 `eth0`），让 gloo 绑定到能互通的网卡。
- 设备间 PyTorch 版本不一致时，可设置 `COLLAB_INFER_COMM_FALLBACK=1`，统一使用基于点对点的 all-to-all / reduce-scatter 实现。

---

## 加载 Hugging Face 模型

支持 `llama`、`mistral`、`qwen2` 架构（GQA、QKV bias、tied embedding、`linear` / `llama3` RoPE scaling）：

```python
cfg = ModelConfig.from_hf("/models/Qwen2-0.5B")                 # 读取 config.json
engine = CollabEngine(cfg, ParallelConfig(tp_size=2), "/models/Qwen2-0.5B")  # 读取 safetensors / .bin（含分片 index）
```

权重按 HF 参数名读取，每个 rank 只取自己需要的切片；`torch.save` 格式的权重以 mmap 方式打开，内存受限的设备只会触及自己分片所在的页。

---

## 异构集群与自动规划

边缘集群往往由算力、内存差别很大的设备组成。所有切分都可以按权重进行：

```python
ParallelConfig(
    tp_size=2, tp_weights=[3, 1],        # 快设备分到 3/4 的注意力头和 MLP 单元
    sp_size=2, sp_weights=[2, 1],        # 快设备处理 2/3 的 token
    pp_size=2, pp_layers=[20, 12],       # 每个 stage 的层数
)
# tp_weights 也可以按 stage 分别指定：[[3, 1], [1, 1]]
```

规划器根据设备画像自动给出这些参数，并用解析代价模型对所有 `pp × sp × tp` 组合排序：

```python
from collab_infer.planner import DeviceProfile, Workload, plan, search

devices = [
    DeviceProfile("jetson-orin", tflops=5.0, memory_gb=8,  mem_bandwidth_gbps=100),
    DeviceProfile("laptop",      tflops=2.0, memory_gb=16, mem_bandwidth_gbps=50),
    DeviceProfile("rpi5-a",      tflops=0.1, memory_gb=4,  mem_bandwidth_gbps=10),
    DeviceProfile("rpi5-b",      tflops=0.1, memory_gb=4,  mem_bandwidth_gbps=10),
]
model = ModelConfig.preset("tinyllama-1.1b")
print(plan(model, devices, pp_size=4).pp_layers)      # [14, 6, 1, 1]：动态规划最小化最慢 stage，满足内存约束
for r in search(model, devices, Workload(prompt_len=1024, new_tokens=64), top_k=3):
    print(r)
```

命令行：`python -m collab_infer plan --preset tinyllama-1.1b --devices devices.json`（`devices.json` 为 `DeviceProfile` 字段列表）。

[`examples/heterogeneous_edge.py`](examples/heterogeneous_edge.py) 用不同的 CPU 线程数在一台机器上模拟快/慢两台设备（实测 363 vs 132 GFLOP/s），先测算力再交给规划器，比较均匀切分与规划切分：

```
case               TTFT s  total s  partition
TP2 even            0.920    1.165  even
TP2 planned         0.844    1.039  [0.363, 0.132]
PP2 even            1.050    1.220  even
PP2 planned         0.830    0.976  [6, 2]
```

（两种切分的生成结果完全相同；数据来自 4 核 CPU 容器，仅供参考。）

---

## 在自定义模型上使用并行原语

`parallel/` 中的原语与模型无关，可直接用于你自己的模型（见 [`examples/custom_models.py`](examples/custom_models.py)）：

```python
from collab_infer import ParallelContext
from collab_infer.parallel import (ColwiseParallel, RowwiseParallel, parallelize_module,
                                   PipelineRunner, split_sequential,
                                   SequenceLayout, ring_attention, gather_sequence)

# 张量并行：按名字模式切分任意模型的 nn.Linear / nn.Embedding
ctx = ParallelContext(tp_size=2)
parallelize_module(block, ctx.tp, {
    "qkv.*": ColwiseParallel(granularity=head_dim),   # 按整头切分
    "proj":  RowwiseParallel(granularity=head_dim),
    "fc1":   ColwiseParallel(), "fc2": RowwiseParallel(),
})

# 流水线并行：把 CNN 等顺序模型切成 stage，micro-batch 流水执行
ctx = ParallelContext(pp_size=3)
stage = split_sequential(list(cnn), ctx.pp, layer_counts=[3, 3, 5])
logits = PipelineRunner(stage, ctx.pp).forward(images, num_microbatches=4, return_to="first")

# 序列并行：在自定义注意力中用 ring_attention 替换稠密注意力
out = ring_attention(q, k, v, q_pos, k_pos, ctx.sp, layout.sizes, causal=False)
```

---

## 网络模拟与策略对比

`NetworkConfig(bandwidth_mbps, latency_ms)` 为每次通信加上 `延迟 × 步数 + 字节数 / 带宽` 的时间（流水线消息在链路上排队，ring attention 的传输与计算重叠），可在单机上评估不同网络条件下的策略。[`examples/benchmark_strategies.py`](examples/benchmark_strategies.py) 在 4 个模拟设备上对比各策略（8 层、hidden 256 的模型，batch 2，prompt 256，生成 16 个 token）：

无网络限制（本机回环）：

```
strategy         TTFT s  decode tok/s  total s  max sent MB  max KV MB  max param MB
TP4               0.142          33.8    1.031        14.35       1.11          6.87
PP4               0.124         116.1    0.382         0.57       1.11          7.90
SP4-ring          0.114          54.5    0.664         3.93       1.11         27.41
SP4-ulysses       0.140          60.7    0.634         2.57       1.11         27.41
PP2xTP2           0.120          41.2    0.849         5.57       1.11          6.86
SP2xTP2           0.127          37.1    0.935         6.30       1.11         13.71
PP2xSP2           0.090          62.5    0.570         1.49       1.11         13.71
```

模拟 100 Mbps / 5 ms 的 Wi‑Fi：

```
strategy         TTFT s  decode tok/s  total s  max sent MB  max KV MB  max param MB
TP4               1.700           3.6   10.150        14.35       1.11          6.87
PP4               0.219          52.3    0.792         0.57       1.11          7.90
SP4-ring          0.452          12.7    2.809         3.93       1.11         27.41
SP4-ulysses       0.512          13.4    2.757         2.57       1.11         27.41
PP2xTP2           0.665           8.5    4.187         5.57       1.11          6.86
SP2xTP2           0.740           7.7    4.642         6.30       1.11         13.71
PP2xSP2           0.200          24.6    1.421         1.49       1.11         13.71
```

可以看到边缘场景的典型取舍：TP 每层两次集合通信，对时延和带宽最敏感；PP 通信量最小，在弱网下最稳健；SP 把 KV Cache 均分到各设备（此处每设备 1/4），但权重在 SP 组内是复制的（param MB 更大），因此通常与 TP/PP 组合使用。

---

## 推理优化

下表是同一台 4 核 x86 CPU、同一会话中，对同一模型和请求在优化前（[PR #1](https://github.com/Jerry-ji0501/collaborative-edge-system/pull/1) 合并时的版本）与当前版本的对比：hidden 1024、8 层、GQA 16/4 头、3.2 万词表（1.23 亿参数，float32），单条 1024 token 的 prompt，生成 32 个 token，贪心解码。每个模拟设备分配固定的 CPU 线程数，"@100Mbps/2ms" 表示用网络模拟器模拟的边缘链路。所有配置优化前后生成的 token 完全相同。

| 场景 | 首 token 时延（优化前 → 优化后） | 解码 ms/token（优化前 → 优化后） | 每次生成每 rank 发送 MB（优化前 → 优化后） |
| --- | --- | --- | --- |
| single (4 threads) | 1376 → **701 ms**（2.0×） | 51.1 → **26.1**（2.0×） | 0.00 → 0.00 |
| TP2 | 1479 → **744 ms**（2.0×） | 92.8 → **42.7**（2.2×） | 73.51 → 72.42 |
| PP2 | 2525 → **1064 ms**（2.4×） | 75.7 → **52.1**（1.5×） | 4.27 → 4.27 |
| SP2 ring | 1182 → **743 ms**（1.6×） | 85.9 → **53.0**（1.6×） | 8.96 → 8.96 |
| SP2 ulysses | 1380 → **816 ms**（1.7×） | 77.7 → **59.6**（1.3×） | 21.27 → 21.27 |
| PP4 (1 thread each) | 4769 → **1102 ms**（4.3×） | 125.5 → **76.5**（1.6×） | 4.27 → 4.27 |
| TP2 @100Mbps/2ms | 7182 → **6597 ms**（1.1×） | 154.6 → **79.9**（1.9×） | 73.51 → 72.42 |
| PP2 @100Mbps/2ms | 2951 → **1111 ms**（2.7×） | 78.3 → **55.3**（1.4×） | 4.27 → 4.27 |
| SP2 ulysses @100Mbps/2ms | 3240 → **2529 ms**（1.3×） | 87.8 → **72.1**（1.2×） | 21.27 → 21.27 |
| TP2 @100Mbps/2ms +fp16（新增选项） | 7182 → **3764 ms**（1.9×） | 154.6 → **78.9**（2.0×） | 73.51 → 36.21 |
| SP2 ulysses @100Mbps/2ms +fp16（新增选项） | 3240 → **1695 ms**（1.9×） | 87.8 → **72.5**（1.2×） | 21.27 → 10.79 |

几点说明：TP2 在模拟链路上的 prefill 受带宽限制（每次生成约 72 MB 的 all-reduce 流量），不压缩时首 token 只快 1.1 倍，开启 `comm_dtype="float16"` 后快 1.9 倍；PP4 的首 token 时延主要得益于分块流水 prefill。

各项优化（除通信压缩外都是无损的）：

| 优化 | 做法 | 效果 |
| --- | --- | --- |
| 注意力融合内核 | 使用 PyTorch SDPA / CPU flash kernel；位置允许时不构造掩码，GQA 不复制 K/V | 1024 token 时注意力快 11–23 倍；单设备 prefill 中注意力占比从约 60% 降到约 16% |
| KV Cache 按 head 连续存储 | 缓存存为 `[B, H, T, D]`，对外仍是 token-major 视图 | 解码读取缓存无需拷贝，长上下文解码注意力约快 30% |
| 融合投影 | Q/K/V、gate/up 各合成一次 GEMM；RoPE 每次前向只算一次 | prefill 的投影 GEMM 快 6–22% |
| 小消息直接交换 | ≤64 KiB 的 all-reduce / all-gather / broadcast 一跳直接交换，按 rank 顺序求和 | 解码时的集合通信延迟约降到 1/3，结果逐位一致 |
| SP 解码拆分（`sp_decode_split`，可选，默认关闭） | 解码时 SP 各 rank 分担 MLP 和注意力输出投影，TP 与 SP 的归约合并为一次 stage 级 all-reduce | 减少重复计算，但每层多 1–2 次集合通信：本机回环下 Ulysses 解码约快 10–30%，模拟 100 Mbps/5 ms 链路下反而慢 1.35–2.2 倍，因此默认关闭 |
| 词表并行采样 | LM head 按词表切到整个 stage，只交换少量候选；计数器噪声的 Gumbel-max 采样 | 每个 token 不再收集完整词表 logits；采样结果与并行方式无关 |
| 分块流水 prefill（`prefill_chunk`） | prompt 分块依次流过各 stage，后面的块使用前面块的 KV Cache | 单请求时各 stage 同时工作，降低首 token 时延 |
| 激活通信压缩（`comm_dtype`，可选，有损） | TP/SP 集合通信与流水线激活以 float16 / bfloat16 传输，或流水线激活按行 int8 量化 | 通信字节减半（int8 流水线为 1/4），弱网下首 token 时延明显下降 |

使用建议：

- **通信压缩**优先用 `comm_dtype="float16"`（10 位尾数，超出范围时饱和而不是溢出）。在标准初始化（std 0.02）的随机模型上，float16 的 logits 相对误差 ≤7e-4，bfloat16 ≤6e-3，int8 流水线 ≤8e-3，生成的 16 个 token 全部一致；但在初始化更大（std 0.05）、扰动会逐层放大的随机模型上，bfloat16 / int8 会明显改变输出。请在目标模型上验证 bfloat16 / int8 的精度后再使用。
- **SP 解码拆分**默认关闭：它用额外的集合通信换取更少的重复计算。在高延迟的边缘链路上通信代价更高；在低延迟链路上、模型较大或 Ulysses 模式时可以开启（`sp_decode_split=True` / `--sp-decode-split`）并实测对比。
- **分块流水 prefill** 在 `pp_size > 1` 时默认开启（每个 stage 约 2 块，至少 128 token）。实测单请求时 PP4 首 token 时延快约 1.9 倍；在一台机器上模拟多设备时，各进程共享内存带宽，收益会小于真实的多设备部署。
- **OpenMP 线程等待策略**：`launch_local` 模拟多设备时默认设置 `OMP_WAIT_POLICY=PASSIVE`，避免空转的计算线程抢占通信线程（实测 TP2 解码约快 20%）；但它会让单进程的小算子唤醒变慢。在真实设备上可以两种都测一下。
- **权重精度**：本仓库测试的 x86 CPU 没有原生 bf16 指令，PyTorch 的 int8 权重算子比 fp32 慢 4.6–58 倍，因此没有提供 int8 权重量化；`dtype=torch.bfloat16` 解码约快 1.4 倍、内存减半，但 prefill 约慢 4 倍。在带原生 bf16/int8 指令的设备上结论可能不同。

---

## 配置参考

`ParallelConfig` 字段：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `tp_size` / `pp_size` / `sp_size` | 1 | 三个维度的并行度，乘积等于进程数 |
| `sp_mode` | `"ring"` | `"ring"`（KV 块环形传递，KV Cache 按序列切分）或 `"ulysses"`（all-to-all，KV Cache 按注意力头切分） |
| `sp_layout` | `"contiguous"` | `"zigzag"` 让因果注意力下各 rank 的计算量均衡 |
| `megatron_sp` | `False` | TP 组内再做 Megatron 序列并行（norm/残差只处理 1/tp 的 token） |
| `tp_weights` | `None` | TP 各 rank 的相对能力；一维列表（所有 stage 共用）或每个 stage 一个列表 |
| `sp_weights` | `None` | SP 各 rank 的相对能力（非均匀 token 划分） |
| `pp_layers` | `None` | 每个 stage 的层数（默认均分） |
| `num_microbatches` | `pp_size` | 流水线 micro-batch 数 |
| `prefill_chunk` | 自动 | 把 prompt 切成若干块在流水线各 stage 间并行流动（降低单请求首 token 时延）；`None` 在 `pp_size > 1` 时自动选择块大小，`0` 关闭 |
| `sp_decode_split` | `False` | 解码时 SP 各 rank 分担 MLP 和注意力输出投影，而不是各自重复计算整层（无损，但每层多 1–2 次集合通信；只在解码计算明显多于通信延迟时开启，例如模型较大、链路很快） |
| `comm_dtype` | `None` | 有损的激活通信压缩：`"float16"` / `"bfloat16"`（字节减半），`"int8"`（流水线激活按行 int8 量化，集合通信用 bfloat16） |
| `attn_kv_block` | `None` | 注意力按 key 分块计算，限制长序列的峰值内存 |
| `network` | `None` | `NetworkConfig(bandwidth_mbps, latency_ms)` 网络模拟 |

---

## 关键设计

- **按位置掩码的注意力。** 注意力掩码由每个 token 携带的真实位置决定（`k_pos <= q_pos`，`-1` 表示 padding），而不是依赖张量下标。因此 token 在设备间如何分布（连续、zigzag、非均匀、被缓存在任意 rank）都不影响正确性。注意力同时返回 log-sum-exp，多个部分结果可以**精确合并**，这是 ring attention 和分布式 KV 解码的基础。
- **SP 的 prefill / decode 两种形态。** Prefill 时 token 按 `SequenceLayout` 切分到 SP 组（ring 或 Ulysses）；decode 时新 token 在 SP 组内复制，各 rank 只对自己持有的那部分 KV Cache 计算注意力，再用 LSE 合并（每层一次很小的 all-gather），KV Cache 始终分布式存储。Ring 模式下新 token 的 KV 写入当前缓存最少的 rank，所有 rank 用同样的确定性规则得出一致结论，无需额外通信。
- **GQA 感知的头划分。** 优先按整个 KV 组分配注意力头，避免复制 KV；KV 头少于设备数（如 MQA）时自动复制所需的 KV 头。TP 之后 Ulysses 还会在本地头上再划分一次，同样处理不规则分组。
- **自描述的流水线通道。** `P2PChannel` 的每条消息带一个固定大小的头（消息类型、dtype、shape），stage 之间无需预先知道激活形状；`STOP` 控制消息随数据流传递，用于提前结束已生成完毕（EOS）的 micro-batch。生成时，最后一个 stage 采样后把 token 送回第一个 stage，多个 micro-batch 在流水线中交错，使所有 stage 保持忙碌。
- **词表并行采样。** LM head 在最后一个 stage 的所有 rank 上按词表切分，每个 rank 只提出少量候选，一次很小的 all-gather 后所有 rank 用纯比较得出相同的 token（贪心、温度采样、top-k），不需要收集完整词表的 logits，也不需要再广播；top-p 由一个 rank 决定后广播，候选不足以覆盖 nucleus 时自动退回完整词表，保证精确。随机采样使用 Gumbel-max 与按 `(seed, 序列, 步数, token)` 计算的计数器噪声，因此采样结果与并行方式和 micro-batch 划分无关，且即使异构硬件的 logits 有细微差别，各 rank 的结果也严格一致。
- **后端兼容。** gloo 在一些版本中不支持非均匀的 all-gather / all-to-all（本仓库在 torch 2.14 上实测）：all-gather 采用补齐后裁剪的方式，all-to-all 采用 `all_to_all_single` 的非均匀 split，不支持时自动回退到点对点实现。
- **小消息直接交换。** gloo 的 all-reduce 对解码阶段的小张量需要多步往返；不超过 64 KiB 的 all-reduce / all-gather / broadcast 改为所有 rank 一跳直接交换（实测延迟约为原来的 1/3），all-reduce 按固定 rank 顺序求和，结果在所有 rank 上逐位相同。
- **注意力内核。** 注意力交给 PyTorch 的融合算子（SDPA；需要 LSE 时使用 CPU flash kernel）。位置允许时不构造掩码（无 padding 的 prefill 用 `is_causal`，解码时所有缓存的 key 都可见则不需要掩码），GQA 不复制 K/V，需要掩码时按 query 分块以限制内存；KV Cache 按 head 连续存储。

---

## 测试

```bash
pip install pytest            # 可选：pip install transformers safetensors 以运行 HF 对齐测试
python -m pytest tests -q
```

测试用 gloo 在本机启动多进程，覆盖：通信原语（原生/回退实现、小消息直接交换、非均匀大小、压缩、网络模拟）、TP 层与 `parallelize_module`、ring / Ulysses / 分布式 KV 注意力、流水线通道（含各种压缩编码）与 runner、词表并行采样（与单设备逐 token 一致、采样频率符合目标分布）、规划器（DP 与暴力搜索对照）、以及引擎的完整矩阵——每种配置在每个 rank 上检查 prefill logits、贪心生成的 token 和逐步 logits 与独立参考实现一致，另外还测试 EOS、采样跨并行方式一致性、KV Cache 确实被均分，以及压缩通信的误差与字节数。

---

## 局限与后续工作

- 只在 CPU + gloo 上实际测试过；CUDA / NCCL 与 CPU/GPU 混合路径按设计支持，但未在本仓库的测试环境中验证。
- 注意力在 CPU 上使用 PyTorch 融合算子；GPU 上需要 LSE 的路径（ring attention、分布式 KV 解码）目前使用数学实现，尚未接入 CUDA 融合算子。
- 模型族目前为 LLaMA / Mistral / Qwen2；其他结构可以直接组合 `parallel/` 中的原语。
- 暂未实现连续批处理（continuous batching）、投机解码、权重量化，以及运行时的动态重新划分（设备掉线 / 负载变化）。在本仓库测试的 x86 CPU（AVX-512，无原生 bf16）上，PyTorch 自带的 int8 权重算子比 fp32 慢 4.6–58 倍，因此没有提供 int8 权重量化；bf16 权重（`dtype=torch.bfloat16`）可使解码快约 1.4 倍、内存减半，但 prefill 会慢约 4 倍。
- 规划器的代价模型是解析估计，用于比较策略、确定非均匀切分比例，不追求绝对时延的精确预测。

## 目录结构

```
collab_infer/
  config.py              ParallelConfig / NetworkConfig
  distributed/           comm.py（Communicator）、mesh.py（ParallelContext）、launcher.py
  parallel/              partition.py、attention.py、tensor_parallel.py、sequence_parallel.py、pipeline_parallel.py
  models/                config.py（ModelConfig、预设）、llama.py（并行模型）、cache.py、weights.py
  engine/                engine.py（CollabEngine）、sampling.py
  planner/               planner.py（规划、代价模型、策略搜索）
  cli.py                 命令行（generate / plan）
examples/                quickstart / custom_models / heterogeneous_edge / benchmark_strategies
tests/                   单元测试与多进程等价性测试
```
