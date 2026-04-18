from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

from omegaconf import DictConfig
from pydantic_ai import Agent

from app.models.schemas import ValidationResult
from app.participant.components.utils import _instantiate, read_system_prompt

_MODULE = "app.participant.components.query_validator"


def build_query_validator(cfg: DictConfig) -> QueryValidator:
    return _instantiate(cfg.query_validator, _MODULE)


class QueryValidator(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def run(self, query: str) -> ValidationResult:
        pass


class LLMQueryValidator(QueryValidator):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        system_prompt = read_system_prompt(self.__class__.__name__)
        self._agent = Agent(
            f"bedrock:{cfg.model_id}",
            system_prompt=system_prompt,
            output_type=ValidationResult,
        )

    def run(self, query: str) -> ValidationResult:
        result = self._agent.run_sync(query)
        return result.output


class DumbQueryValidator(QueryValidator):
    def run(self, query: str) -> ValidationResult:
        return ValidationResult(is_valid=True)
