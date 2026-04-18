from __future__ import annotations

from typing import Any

from app.models.schemas import RankedListingResult
from app.participant.components import build_reranker, Config

def rank_listings(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
) -> list[RankedListingResult]:

    cfg = Config.get_cfg()
    reranker = build_reranker(cfg)

    return reranker.run(candidates, soft_facts)