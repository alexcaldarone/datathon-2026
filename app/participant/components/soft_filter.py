from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from app.ingestion.components.client import OpenSearchClient
from app.ingestion.components.augmenters import build_augmenters
from app.participant.components.utils import _instantiate

_MODULE = "app.participant.components.soft_filter"


def build_soft_filter(cfg: DictConfig) -> SoftFilter:
    return _instantiate(cfg.soft_filter, _MODULE)


class SoftFilter(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg=cfg

    @abstractmethod
    def run(
        self,
        candidates: list[dict[str, Any]],
        soft_facts: dict[str, Any],
    ) -> list[dict[str, Any]]:
        pass


class DumbSoftFilter(SoftFilter):
    def run(
        self,
        candidates: list[dict[str, Any]],
        soft_facts: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return candidates


class HybridSimilarityFilter(SoftFilter):
    # top-N terms from sparse BM25 weights included in the rank_features query
    _SPARSE_TOP_K: int = 30

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        ingestion_cfg = self._load_ingestion_cfg()
        self._index = ingestion_cfg.index_name
        self._pipeline = ingestion_cfg.pipeline_name
        self._augmenters = build_augmenters(ingestion_cfg)
        self._client = OpenSearchClient()  # singleton — not rebuilt per request

    def run(
        self,
        candidates: list[dict[str, Any]],
        soft_facts: dict[str, Any],
    ) -> list[dict[str, Any]]:
        query_text: str = soft_facts.get("_query", "")
        if not query_text or not candidates:
            return candidates
        # candidates=[{"listing_id":"695fdc5dc12a1d2f41553111"}]
        # soft_facts={"_query":"house in zurich"}
        listing_ids = [c["listing_id"] for c in candidates]
        query_listing = {"full_text": query_text}
        features = {aug.field_name: aug.augment(query_listing).content for aug in self._augmenters}

        resp = self._client.search(
            index=self._index,
            body=self._build_query(
                query_text=query_text,
                dense_vector=features.get("dense_embedding", []),
                sparse_weights=features.get("sparse_embedding", {}),
                listing_ids=listing_ids,
            ),
            pipeline=self._pipeline,
        )

        hits = resp["hits"]["hits"]
        top_k = int(self.cfg.top_k)
        
        padding = []
        if len(hits) < top_k:
            padding = candidates[:top_k-len(hits)]

        return [hit["_source"] for hit in hits][:top_k] + padding

    def _build_query(
        self,
        query_text: str,
        dense_vector: list[float],
        sparse_weights: dict[str, float],
        listing_ids: list[str],
    ) -> dict:
        n = len(listing_ids)
        top_k = int(self.cfg.top_k)
        candidate_filter = {"terms": {"listing_id": listing_ids}}

        # top-N sparse terms by BM25 weight; rank_feature (singular) is the query type
        top_terms = sorted(sparse_weights.items(), key=lambda x: -x[1])[:self._SPARSE_TOP_K]
        sparse_query = {
            "bool": {
                "should": [{"rank_feature": {"field": f"sparse_embedding.{t}"}} for t, _ in top_terms],
                "filter": candidate_filter,
            }
        }

        return {
            # size=n ensures the hybrid+RRF stage sees all candidates before post_filter trims
            "size": n,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "bool": {
                                "must": {
                                    "multi_match": {
                                        "query": query_text,
                                        "fields": ["full_text", "title", "description"],
                                    }
                                },
                                "filter": candidate_filter,
                            }
                        },
                        {
                            "knn": {
                                "dense_embedding": {
                                    "vector": dense_vector,
                                    "k": n,
                                    "filter": candidate_filter,
                                }
                            }
                        },
                        sparse_query,
                    ]
                }
            },
            # post_filter as safety net in case any non-candidate slips through
            "post_filter": candidate_filter,
        }

    @staticmethod
    def _load_ingestion_cfg() -> DictConfig:
        cfg_path = Path(__file__).parents[3] / "configs" / "ingestion" / "config.yaml"
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        return OmegaConf.create(raw)
