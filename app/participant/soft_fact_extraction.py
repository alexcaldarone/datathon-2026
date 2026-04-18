from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from app.participant.components import build_soft_extractor, Config

# maps anchor dimension names to OpenSearch boost fields (VLM + geo features)
_ANCHOR_TO_BOOST: dict[str, str] = {
    "daylight_score": "vlm_features.brightness",
    "spaciousness_perception": "vlm_features.spaciousness",
    "interior_modernity": "vlm_features.modernity",
    "view_quality_score": "vlm_features.view_quality",
    "green_view_ratio": "vlm_features.greenery",
    "noise_level": "vlm_features.noise_impression",
    "neighborhood_safety_score": "geo_features.family_score",
    "public_transport_score": "geo_features.transit_score",
    "walkability_score": "geo_features.walkability_score",
}

_anchors: dict[str, list[str]] | None = None


def _load_anchors() -> dict[str, list[str]]:
    global _anchors
    if _anchors is not None:
        return _anchors
    anchor_path = Path(__file__).parents[1] / "configs" / "soft_extractor_anchors.yaml"
    if not anchor_path.exists():
        # fall back to project root
        anchor_path = Path(__file__).parents[2] / "configs" / "soft_extractor_anchors.yaml"
    with open(anchor_path) as f:
        _anchors = yaml.safe_load(f) or {}
    return _anchors


def _detect_boost_fields(query: str) -> list[str]:
    query_lower = query.lower()
    anchors = _load_anchors()
    matched: list[str] = []
    for dimension, phrases in anchors.items():
        boost_field = _ANCHOR_TO_BOOST.get(dimension)
        if not boost_field:
            continue
        for phrase in phrases:
            # check if any significant word from the anchor phrase appears in the query
            words = re.findall(r"[a-zäöüéèàâêîôû]+", phrase.lower())
            # match if at least one distinctive word (>3 chars) appears
            if any(w in query_lower for w in words if len(w) > 3):
                matched.append(boost_field)
                break
    return matched


def extract_soft_facts(query: str) -> dict[str, Any]:

    cfg = Config.get_cfg()
    soft_extractor = build_soft_extractor(cfg)
    soft_facts = soft_extractor.run(query)
    # always include the original query so retrieval-based filters can use it
    soft_facts["_query"] = query
    # detect which VLM/geo fields to boost based on anchor matching
    soft_facts["boost_fields"] = _detect_boost_fields(query)
    return soft_facts
