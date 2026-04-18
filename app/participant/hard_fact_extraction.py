from __future__ import annotations

from app.models.schemas import HardFilters
from app.participant.components.hard_extractor import LLMHardFactExtractor

_extractor = LLMHardFactExtractor()


def extract_hard_facts(query: str) -> HardFilters:
    return _extractor.run(query)
