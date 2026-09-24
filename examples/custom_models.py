"""Using the parallel primitives with *your own* models (no LLM engine).

1. Tensor parallelism for an arbitrary module via a name-pattern plan.
2. Pipeline parallelism for a CNN split into stages, with micro-batching.
3. Sequence parallelism (ring attention) inside a custom encoder layer.

    python examples/custom_models.py
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from collab_infer import ParallelContext, launch_local  # noqa: E402
from collab_infer.parallel import (  # noqa: E402
    ColwiseParallel,
    PipelineRunner,
    RowwiseParallel,
    SequenceLayout,
    gather_sequence,
    parallelize_module,
    ring_attention,
    split_sequential,
)


# --------------------------------------------------------------------------
# 1. tensor parallelism for an arbitrary module
# --------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    """A ViT/BERT-style block; the head count follows the projection size."""

    def __init__(self, dim=64, heads=8):
        super().__init__()
        self.head_dim = dim // heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.ModuleDict({n: nn.Linear(dim, dim) for n in ("q", "k", "v")})
        self.proj = nn.Linear(dim, dim)
        self.fc1, self.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

    def attend(self, h):
        B, S, _ = h.shape
        q, k, v = (self.qkv[n](h).view(B, S, -1, self.head_dim).transpose(1, 2) for n in ("q", "k", "v"))
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, S, -1)

    def forward(self, x):
        x = x + self.proj(self.attend(self.norm1(x)))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


def tensor_parallel_demo(rank, world):
    torch.manual_seed(0)
    block = EncoderBlock()
    x = torch.randn(2, 10, 64)
    reference = block(x)
    ctx = ParallelContext(tp_size=world)
    plan = {
        "qkv.*": ColwiseParallel(granularity=block.head_dim),  # whole heads per rank
        "proj": RowwiseParallel(granularity=block.head_dim),
        "fc1": ColwiseParallel(),
        "fc2": RowwiseParallel(),
    }
    parallelize_module(block, ctx.tp, plan)
    return (block(x) - reference).abs().max().item()


# --------------------------------------------------------------------------
# 2. pipeline parallelism for a CNN
# --------------------------------------------------------------------------
def make_cnn():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(64, 10),
    )


def pipeline_demo(rank, world):
    cnn = make_cnn().eval()
    images = torch.randn(16, 3, 32, 32)
    reference = cnn(images)
    ctx = ParallelContext(pp_size=world)
    stage = split_sequential(list(cnn), ctx.pp, layer_counts=[3, 3, 5])  # conv blocks on devices
    out = PipelineRunner(stage, ctx.pp).forward(images if rank == 0 else None, num_microbatches=4, return_to="first")
    return None if out is None else (out - reference).abs().max().item()


# --------------------------------------------------------------------------
# 3. sequence parallelism inside a custom attention layer
# --------------------------------------------------------------------------
def sequence_parallel_demo(rank, world):
    torch.manual_seed(0)
    block = EncoderBlock()
    x = torch.randn(1, 30, 64)
    reference = block(x)
    ctx = ParallelContext(sp_size=world)
    layout = SequenceLayout(30, world, weights=[2, 1, 1])  # uneven: rank 0 is faster
    x_local = layout.shard(x, rank)
    positions = layout.shard(torch.arange(30).unsqueeze(0), rank)

    def ring_attend(h):  # replaces the dense attention: keys/values circulate
        B, S, _ = h.shape
        q, k, v = (block.qkv[n](h).view(B, S, -1, block.head_dim) for n in ("q", "k", "v"))
        out = ring_attention(q, k, v, positions, positions, ctx.sp, layout.sizes, causal=False)
        return out.reshape(B, S, -1)

    block.attend = ring_attend
    y_local = block(x_local)  # norms / MLP run on the local tokens only
    return (gather_sequence(y_local, layout, ctx.sp) - reference).abs().max().item()


if __name__ == "__main__":
    torch.set_num_threads(1)
    print("tensor parallel (2 ranks) max error:", launch_local(tensor_parallel_demo, 2, return_results=True))
    print("pipeline parallel CNN (3 stages) max error on stage 0:", launch_local(pipeline_demo, 3, return_results=True)[0])
    print("ring-attention sequence parallel (3 ranks) max error:", launch_local(sequence_parallel_demo, 3, return_results=True))
