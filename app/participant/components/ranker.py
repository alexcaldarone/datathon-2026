from abc import ABC, abstractmethod
from typing import Any

from models.schemas import RankedListingResult

class Ranker(ABC):
    def __init__(self):
        pass

    def run(
        self,
        candidates: list[dict[str, Any]],
        soft_facts: dict[str, Any]
    ) -> list[RankedListingResult]:
        pass
