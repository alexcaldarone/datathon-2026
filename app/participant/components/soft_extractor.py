from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Any

from omegaconf import DictConfig

from app.participant.components.utils import _instantiate

_MODULE = "app.participant.components.soft_extractor"


def build_soft_extractor(cfg: DictConfig) -> SoftFactExtractor:
    return _instantiate(cfg.soft_extractor, _MODULE)


class SoftFactExtractor(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg=cfg

    @abstractmethod
    def run(self, query: str) -> dict[str, Any]:
        pass


class DumbSoftExtractor(SoftFactExtractor):
    def run(self, query: str) -> dict[str, Any]:
        return {"query": query}
