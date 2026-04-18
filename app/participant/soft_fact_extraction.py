from __future__ import annotations
from typing import Any
from app.participant.components.soft_extractor import LLMSoftFactExtractor
from app.participant.components import build_soft_extractor, Config

def extract_soft_facts(query: str) -> dict[str, Any]:

    cfg = Config.get_cfg()
    soft_extractor = build_soft_extractor(cfg)
    soft_facts = soft_extractor.run(query)
    # always include the original query so retrieval-based filters can use it
    soft_facts["_query"] = query
    return soft_facts
