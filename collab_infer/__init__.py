"""collab_infer: a general collaborative inference framework for edge devices.

Split a model over several devices with any combination of

* tensor parallelism (Megatron-style, optional Megatron sequence parallelism),
* pipeline parallelism (micro-batched, token loop-back for generation),
* sequence parallelism (ring attention or DeepSpeed-Ulysses),

with uneven, capability-proportional partitioning for heterogeneous devices.
"""

from .config import NetworkConfig, ParallelConfig
from .distributed import (
    CommStats,
    Communicator,
    ParallelContext,
    init_distributed,
    launch_local,
)
from .engine import CollabEngine, GenerationResult, SamplingParams
from .models import ModelConfig, open_weights, random_state_dict

__version__ = "0.1.0"

__all__ = [
    "CollabEngine",
    "CommStats",
    "Communicator",
    "GenerationResult",
    "ModelConfig",
    "NetworkConfig",
    "ParallelConfig",
    "ParallelContext",
    "SamplingParams",
    "__version__",
    "init_distributed",
    "launch_local",
    "open_weights",
    "random_state_dict",
]
