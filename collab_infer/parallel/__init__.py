"""Parallel primitives: partitioning, tensor, sequence and pipeline parallelism.

These building blocks are model-agnostic and can be used to parallelise
custom models; :mod:`collab_infer.models` shows how they compose into a
fully parallel transformer.
"""

from .attention import attention, attention_with_lse, merge_attention, merge_attention_list
from .partition import HeadShard, SequenceLayout, partition_heads, split_sizes
from .pipeline_parallel import (
    MSG_DATA,
    MSG_STOP,
    Message,
    P2PChannel,
    PipelineRunner,
    partition_layers,
    split_sequential,
)
from .sequence_parallel import (
    distributed_kv_attention,
    gather_heads,
    gather_sequence,
    ring_attention,
    shard_sequence,
    ulysses_attention,
)
from .tensor_parallel import (
    ColumnParallelLinear,
    ColwiseParallel,
    ParallelLMHead,
    ParallelStyle,
    RowParallelLinear,
    RowwiseParallel,
    VocabParallel,
    VocabParallelEmbedding,
    parallelize_module,
)

__all__ = [
    "ColumnParallelLinear",
    "ColwiseParallel",
    "HeadShard",
    "MSG_DATA",
    "MSG_STOP",
    "Message",
    "P2PChannel",
    "ParallelLMHead",
    "ParallelStyle",
    "PipelineRunner",
    "RowParallelLinear",
    "RowwiseParallel",
    "SequenceLayout",
    "VocabParallel",
    "VocabParallelEmbedding",
    "attention",
    "attention_with_lse",
    "distributed_kv_attention",
    "gather_heads",
    "gather_sequence",
    "merge_attention",
    "merge_attention_list",
    "parallelize_module",
    "partition_heads",
    "partition_layers",
    "ring_attention",
    "shard_sequence",
    "split_sequential",
    "split_sizes",
    "ulysses_attention",
]
