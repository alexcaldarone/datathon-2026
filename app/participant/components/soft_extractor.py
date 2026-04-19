from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Any

from omegaconf import DictConfig
from pydantic_ai import Agent

from app.models.schemas import SoftFacts
from app.participant.components.utils import _instantiate, read_system_prompt

_MODULE = "app.participant.components.soft_extractor"


def build_soft_extractor(cfg: DictConfig) -> SoftFactExtractor:
    return _instantiate(cfg.soft_extractor, _MODULE)


class SoftFactExtractor(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def run(self, query: str) -> dict[str, Any]:
        pass


class DumbSoftExtractor(SoftFactExtractor):
    def run(self, query: str) -> dict[str, Any]:
        return {"query": query}


class LLMSoftExtractor(SoftFactExtractor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        system_prompt = read_system_prompt(self.__class__.__name__)
        self._agent = Agent(
            f"bedrock:{cfg.model_id}",
            system_prompt=system_prompt,
            output_type=SoftFacts,
        )

    def run(self, query: str) -> dict[str, Any]:
        if not query.strip():
            return {"query": query, "preferences": []}
        result = self._agent.run_sync(query)
        soft_facts = result.output
        return {"query": query, "preferences": soft_facts.preferences}
