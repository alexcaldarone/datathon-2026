from abc import abstractmethod, ABC
from typing import Any

class SoftFactExtractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self, query: str) -> list[dict[str, Any]]:
        pass
