from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import boto3
from omegaconf import DictConfig

from app.models.schemas import ListingData, RankedListingResult
from app.participant.components.utils import _instantiate

_MODULE = "app.participant.components.reranker"


def build_reranker(cfg: DictConfig) -> ReRanker:
    return _instantiate(cfg.reranker, _MODULE)


class ReRanker(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg=cfg

    @abstractmethod
    def run(
        self,
        candidates: list[dict[str, Any]],
        soft_facts: dict[str, Any],
    ) -> list[RankedListingResult]:
        pass


class DumbReRanker(ReRanker):
    def run(
        self,
        candidates: list[dict[str, Any]],
        soft_facts: dict[str, Any],
    ) -> list[RankedListingResult]:
        return [
            RankedListingResult(
                listing_id=str(c["listing_id"]),
                score=1.0,
                reason="Matched hard filters; soft ranking stub.",
                listing=_to_listing_data(c),
            )
            for c in candidates
        ]


def _to_listing_data(c: dict[str, Any]) -> ListingData:
    return ListingData(
        id=str(c["listing_id"]),
        title=c["title"],
        description=c.get("description"),
        street=c.get("street"),
        city=c.get("city"),
        postal_code=c.get("postal_code"),
        canton=c.get("canton"),
        latitude=c.get("latitude"),
        longitude=c.get("longitude"),
        price_chf=c.get("price"),
        rooms=c.get("rooms"),
        living_area_sqm=_coerce_int(c.get("area")),
        available_from=c.get("available_from"),
        image_urls=_coerce_image_urls(c.get("image_urls")),
        hero_image_url=c.get("hero_image_url"),
        original_listing_url=c.get("original_url"),
        features=c.get("features") or [],
        offer_type=c.get("offer_type"),
        object_category=c.get("object_category"),
        object_type=c.get("object_type"),
    )


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _coerce_image_urls(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return None


def _to_document_text(c: dict[str, Any]) -> str:
    features = c.get("features") or []
    parts = [
        c.get("title", ""),
        c.get("description", ""),
        f"Location: {c.get('city', '')}, {c.get('canton', '')}",
        f"Price: CHF {c.get('price', '')}",
        f"Rooms: {c.get('rooms', '')}",
        f"Area: {c.get('area', '')} sqm",
        f"Features: {', '.join(features)}",
    ]
    return " | ".join(p for p in parts if p.strip())


class CohereReRanker(ReRanker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._client = boto3.client("bedrock-runtime", region_name=cfg.region)

    def run(
        self,
        candidates: list[dict[str, Any]],
        soft_facts: dict[str, Any],
    ) -> list[RankedListingResult]:
        if not candidates:
            return []

        query = soft_facts.get("query", "")
        texts = [_to_document_text(c) for c in candidates]
        top_n = min(int(self.cfg.top_n), len(candidates))

        body = json.dumps({
            "query": query,
            "documents": texts,
            "top_n": top_n,
            "api_version": 2,
        })
        raw = self._client.invoke_model(
            modelId=self.cfg.model_id,
            accept="application/json",
            contentType="application/json",
            body=body,
        )
        response = json.loads(raw["body"].read())

        return [
            RankedListingResult(
                listing_id=str(candidates[r["index"]]["listing_id"]),
                score=float(r["relevance_score"]),
                reason=f"Cohere Rerank score: {r['relevance_score']:.4f}",
                listing=_to_listing_data(candidates[r["index"]]),
            )
            for r in response["results"]
        ]
