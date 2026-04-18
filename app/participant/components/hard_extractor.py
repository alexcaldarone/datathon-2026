from __future__ import annotations

from abc import abstractmethod, ABC

from omegaconf import DictConfig

from app.models.schemas import HardFilters
from app.participant.components.utils import _instantiate

_MODULE = "app.participant.components.hard_extractor"


def build_hard_extractor(cfg: DictConfig) -> HardFactExtractor:
    return _instantiate(cfg.hard_extractor, _MODULE)


class HardFactExtractor(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def run(self, query: str) -> HardFilters:
        pass


class DumbHardExtractor(HardFactExtractor):
    def run(self, _query: str) -> HardFilters:
        return HardFilters()
