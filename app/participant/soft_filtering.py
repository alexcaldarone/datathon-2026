from __future__ import annotations

from typing import Any
from app.participant.components import build_soft_filter, Config

def filter_soft_facts(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
    target: int,
) -> list[dict[str, Any]]:

    cfg = Config.get_cfg()
    soft_filter = build_soft_filter(cfg)

    return soft_filter.run(candidates, soft_facts, target)
