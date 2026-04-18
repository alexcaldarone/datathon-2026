from abc import ABC, abstractmethod
from typing import Any

class SoftFilter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(
        self,
        candidates: list[dict[str, Any]],
        soft_facts: dict[str, Any]
    ):
        pass
