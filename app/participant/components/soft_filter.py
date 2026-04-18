from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig

from app.participant.components.utils import _instantiate

_MODULE = "app.participant.components.soft_filter"


def build_soft_filter(cfg: DictConfig) -> SoftFilter:
    return _instantiate(cfg.soft_filter, _MODULE)


class SoftFilter(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg=cfg

    @abstractmethod
    def run(
        self,
        candidates: list[dict[str, Any]],
        soft_facts: dict[str, Any],
    ) -> list[dict[str, Any]]:
        pass


class DumbSoftFilter(SoftFilter):
    def run(
        self,
        candidates: list[dict[str, Any]],
        soft_facts: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return candidates
