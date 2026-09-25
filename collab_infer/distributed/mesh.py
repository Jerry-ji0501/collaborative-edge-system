"""The 3-D device mesh ``(pp, sp, tp)`` and its communicators."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.distributed as dist

from ..config import NetworkConfig, ParallelConfig
from .comm import CommStats, Communicator, NetworkEmulator, dist_ready


class ParallelContext:
    """Per-process view of the device mesh.

    Global rank ``r`` has coordinates ``(pp_rank, sp_rank, tp_rank)`` with
    ``r = (pp_rank * sp_size + sp_rank) * tp_size + tp_rank``.  The context owns
    one :class:`Communicator` per mesh dimension plus two helpers:

    * ``tp``: ranks with equal ``(pp, sp)`` – tensor-parallel group,
    * ``sp``: ranks with equal ``(pp, tp)`` – sequence-parallel group,
    * ``pp``: ranks with equal ``(sp, tp)`` – one rank per stage, ordered by
      stage, used for point-to-point activation transfer,
    * ``stage``: all ``sp * tp`` ranks of this pipeline stage,
    * ``world``: every rank.

    With a single process no ``torch.distributed`` initialisation is needed
    and all communicators are trivial.
    """

    def __init__(
        self,
        pp_size: int = 1,
        sp_size: int = 1,
        tp_size: int = 1,
        *,
        device: Optional[torch.device] = None,
        network: Optional[NetworkConfig] = None,
        force_comm_fallback: bool = False,
        small_message_bytes: Optional[int] = None,
    ) -> None:
        self.pp_size, self.sp_size, self.tp_size = int(pp_size), int(sp_size), int(tp_size)
        expected = self.pp_size * self.sp_size * self.tp_size
        if dist_ready():
            self.world_size, self.rank = dist.get_world_size(), dist.get_rank()
        else:
            self.world_size, self.rank = 1, 0
        if self.world_size != expected:
            raise ValueError(
                f"world size {self.world_size} does not match pp*sp*tp = "
                f"{self.pp_size}*{self.sp_size}*{self.tp_size} = {expected}"
            )
        self.pp_rank, self.sp_rank, self.tp_rank = self.coords(self.rank)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.stats = CommStats()
        self.emulator = NetworkEmulator.from_config(network)
        self._force_fallback = force_comm_fallback
        self._small_message_bytes = small_message_bytes

        pp, sp, tp = self.pp_size, self.sp_size, self.tp_size
        # new_group must be called by every rank for every group, in the same order
        self.tp = self._build("tp", [[self.rank_of(p, s, t) for t in range(tp)] for p in range(pp) for s in range(sp)])
        self.sp = self._build("sp", [[self.rank_of(p, s, t) for s in range(sp)] for p in range(pp) for t in range(tp)])
        self.pp = self._build("pp", [[self.rank_of(p, s, t) for p in range(pp)] for s in range(sp) for t in range(tp)])
        self.stage = self._build(
            "stage", [[self.rank_of(p, s, t) for s in range(sp) for t in range(tp)] for p in range(pp)]
        )
        world_group = dist.group.WORLD if self.world_size > 1 else None
        self.world = self._communicator("world", list(range(self.world_size)), world_group)

    @classmethod
    def from_config(
        cls, config: ParallelConfig, device: Optional[torch.device] = None, **kwargs
    ) -> "ParallelContext":
        return cls(
            config.pp_size,
            config.sp_size,
            config.tp_size,
            device=device,
            network=config.network,
            **kwargs,
        )

    # ---------------------------------------------------------------- groups
    def _communicator(self, name: str, ranks: List[int], group) -> Communicator:
        return Communicator(
            ranks,
            group,
            name=name,
            device=self.device,
            stats=self.stats,
            emulator=self.emulator,
            force_fallback=self._force_fallback,
            small_message_bytes=self._small_message_bytes,
        )

    def _build(self, name: str, all_groups: Sequence[List[int]]) -> Communicator:
        mine = None
        for ranks in all_groups:
            group = dist.new_group(ranks) if len(ranks) > 1 else None
            if self.rank in ranks:
                mine = (ranks, group)
        assert mine is not None, f"rank {self.rank} missing from {name} groups"
        return self._communicator(name, *mine)

    # ------------------------------------------------------------ coordinates
    def rank_of(self, pp_rank: int, sp_rank: int, tp_rank: int) -> int:
        return (pp_rank * self.sp_size + sp_rank) * self.tp_size + tp_rank

    def coords(self, rank: int):
        tp_rank = rank % self.tp_size
        sp_rank = (rank // self.tp_size) % self.sp_size
        pp_rank = rank // (self.tp_size * self.sp_size)
        return pp_rank, sp_rank, tp_rank

    @property
    def is_first_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    @property
    def is_stage_leader(self) -> bool:
        return self.sp_rank == 0 and self.tp_rank == 0

    def describe(self) -> str:
        return (
            f"rank {self.rank}/{self.world_size} -> (pp={self.pp_rank}, sp={self.sp_rank}, "
            f"tp={self.tp_rank}) on {self.device}"
        )

    def __repr__(self) -> str:
        return (
            f"ParallelContext(pp={self.pp_size}, sp={self.sp_size}, tp={self.tp_size}, "
            f"rank={self.rank}, coords=({self.pp_rank}, {self.sp_rank}, {self.tp_rank}))"
        )


__all__ = ["ParallelContext"]
