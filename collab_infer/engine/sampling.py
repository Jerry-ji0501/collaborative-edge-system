"""Token sampling (greedy, temperature, top-k, top-p)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SamplingParams:
    """``temperature <= 0`` means greedy decoding."""

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0


class Sampler:
    """Draws the next token from logits.

    In the engine only one rank (the leader of the last pipeline stage)
    samples and broadcasts the result, so all ranks agree on the tokens even
    if heterogeneous hardware produces slightly different logits.
    """

    def __init__(self, params: Optional[SamplingParams] = None) -> None:
        self.params = params or SamplingParams()
        self.generator = torch.Generator()
        if self.params.seed is not None:
            self.generator.manual_seed(self.params.seed)
        else:
            self.generator.seed()

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """``logits``: ``[B, V]`` -> token ids ``[B]`` (int64, same device)."""
        p = self.params
        if p.greedy:
            return logits.argmax(dim=-1)
        scores = logits.float() / p.temperature
        if p.top_k > 0 and p.top_k < scores.shape[-1]:
            kth = torch.topk(scores, p.top_k, dim=-1).values[..., -1:]
            scores = scores.masked_fill(scores < kth, float("-inf"))
        if p.top_p < 1.0:
            sorted_scores, order = torch.sort(scores, dim=-1, descending=True)
            probs = torch.softmax(sorted_scores, dim=-1)
            # drop tokens once the probability mass before them exceeds top_p
            drop = (torch.cumsum(probs, dim=-1) - probs) > p.top_p
            sorted_scores = sorted_scores.masked_fill(drop, float("-inf"))
            scores = torch.full_like(scores, float("-inf")).scatter(-1, order, sorted_scores)
        probs = torch.softmax(scores, dim=-1).cpu()
        tokens = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
        return tokens.to(logits.device)


__all__ = ["Sampler", "SamplingParams"]
