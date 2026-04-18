from __future__ import annotations

from app.models.schemas import HardFilters
from app.participant.components import Config, build_hard_extractor


def extract_hard_facts(query: str) -> HardFilters:
    
    cfg = Config.get_cfg()

    hard_extractor = build_hard_extractor(cfg)

    return hard_extractor.run(query)
