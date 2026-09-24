"""Weight sources: where each rank reads the parameters of its shard from.

Every rank builds only its own part of the model and pulls the corresponding
slices out of a full checkpoint.  Torch checkpoints are memory-mapped, so a
memory-constrained edge device only touches the pages of its own shard.

Names follow Hugging Face ``transformers`` (``model.layers.0.self_attn.q_proj.weight``).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Union

import torch


class WeightSource:
    """Read-only mapping from parameter names to full (unsharded) tensors."""

    def keys(self) -> Iterable[str]:
        raise NotImplementedError

    def _get(self, name: str) -> torch.Tensor:
        raise NotImplementedError

    def _resolve(self, name: str) -> Optional[str]:
        names = self._key_set()
        if name in names:
            return name
        # checkpoints saved from the bare decoder lack the "model." prefix
        if name.startswith("model.") and name[len("model.") :] in names:
            return name[len("model.") :]
        if "model." + name in names:
            return "model." + name
        return None

    def _key_set(self) -> set:
        if not hasattr(self, "_keys_cache"):
            self._keys_cache = set(self.keys())
        return self._keys_cache

    def has(self, name: str) -> bool:
        return self._resolve(name) is not None

    def get(self, name: str) -> torch.Tensor:
        resolved = self._resolve(name)
        if resolved is None:
            raise KeyError(f"weight {name!r} not found in {self!r}")
        return self._get(resolved)


class DictWeightSource(WeightSource):
    """Weights held in memory, e.g. a ``state_dict()``."""

    def __init__(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.state_dict = dict(state_dict)

    def keys(self) -> Iterable[str]:
        return self.state_dict.keys()

    def _get(self, name: str) -> torch.Tensor:
        return self.state_dict[name]

    def __repr__(self) -> str:
        return f"DictWeightSource({len(self.state_dict)} tensors)"


class TorchFileSource(WeightSource):
    """One or more ``torch.save`` files, memory-mapped (lazy, low memory)."""

    def __init__(self, paths: Union[str, List[str]]) -> None:
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self._tensors: Dict[str, torch.Tensor] = {}
        for path in self.paths:
            try:
                data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            except (RuntimeError, TypeError):  # legacy (non-zip) format cannot be mmap'ed
                data = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(data, dict) and "state_dict" in data and isinstance(data["state_dict"], dict):
                data = data["state_dict"]
            self._tensors.update({k: v for k, v in data.items() if isinstance(v, torch.Tensor)})

    def keys(self) -> Iterable[str]:
        return self._tensors.keys()

    def _get(self, name: str) -> torch.Tensor:
        return self._tensors[name]

    def __repr__(self) -> str:
        return f"TorchFileSource({self.paths})"


class SafetensorsSource(WeightSource):
    """One or more ``.safetensors`` files (requires the ``safetensors`` package)."""

    def __init__(self, paths: Union[str, List[str]]) -> None:
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("install 'safetensors' to read .safetensors checkpoints") from exc
        self._safe_open = safe_open
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self._where: Dict[str, str] = {}
        for path in self.paths:
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    self._where[key] = path

    def keys(self) -> Iterable[str]:
        return self._where.keys()

    def _get(self, name: str) -> torch.Tensor:
        with self._safe_open(self._where[name], framework="pt") as f:
            return f.get_tensor(name)

    def __repr__(self) -> str:
        return f"SafetensorsSource({len(self.paths)} files)"


def hf_checkpoint(directory: str) -> WeightSource:
    """Weights of a Hugging Face model directory (sharded or not)."""
    for index_name, loader in (
        ("model.safetensors.index.json", SafetensorsSource),
        ("pytorch_model.bin.index.json", TorchFileSource),
    ):
        index_path = os.path.join(directory, index_name)
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                files = sorted(set(json.load(f)["weight_map"].values()))
            return loader([os.path.join(directory, name) for name in files])
    single = os.path.join(directory, "model.safetensors")
    if os.path.exists(single):
        return SafetensorsSource(single)
    single = os.path.join(directory, "pytorch_model.bin")
    if os.path.exists(single):
        return TorchFileSource(single)
    raise FileNotFoundError(f"no checkpoint found in {directory}")


def open_weights(weights: Union[WeightSource, Mapping[str, torch.Tensor], str]) -> WeightSource:
    """Wrap a state dict, file, directory or existing source as a :class:`WeightSource`."""
    if isinstance(weights, WeightSource):
        return weights
    if isinstance(weights, Mapping):
        return DictWeightSource(weights)
    if isinstance(weights, (str, os.PathLike)):
        path = os.fspath(weights)
        if os.path.isdir(path):
            return hf_checkpoint(path)
        if path.endswith(".safetensors"):
            return SafetensorsSource(path)
        return TorchFileSource(path)
    raise TypeError(f"cannot interpret {type(weights).__name__} as model weights")


__all__ = [
    "DictWeightSource",
    "SafetensorsSource",
    "TorchFileSource",
    "WeightSource",
    "hf_checkpoint",
    "open_weights",
]
