from __future__ import annotations

import os
from abc import abstractmethod, ABC
from pathlib import Path

from pydantic_ai import Agent

from app.models.schemas import HardFilters

from omegaconf import DictConfig

from app.models.schemas import HardFilters
from app.participant.components.utils import _instantiate, read_system_prompt

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

class LLMHardFactExtractor(HardFactExtractor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        system_prompt = read_system_prompt(self.__class__.__name__)
        self._agent = Agent(
            f"bedrock:{cfg.model_id}",
            system_prompt=system_prompt,
            output_type=HardFilters,
        )

    async def run(self, query: str) -> HardFilters:
        result = await self._agent.run(query)
        return result.output

class DumbHardExtractor(HardFactExtractor):
    def run(self, _query: str) -> HardFilters:
        return HardFilters()
