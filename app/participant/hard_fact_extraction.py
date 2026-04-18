from __future__ import annotations

from app.models.schemas import HardFilters
from app.participant.components import build_hard_extractor, Config


def extract_hard_facts(query: str) -> HardFilters:
    
    cfg = Config.get_cfg()

    hard_extractor = build_hard_extractor(cfg)

    return hard_extractor.run(query)
