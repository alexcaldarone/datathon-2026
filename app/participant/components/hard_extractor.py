from abc import abstractmethod, ABC

from app.models.schemas import HardFilters

class HardFactExtractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self, query: str) -> HardFilters:
        pass
