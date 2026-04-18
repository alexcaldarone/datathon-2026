from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from app.ingestion.components.client import OpenSearchClient
from app.ingestion.components.augmenters import build_augmenters, ImageEmbeddingAugmenter
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
        # optional image augmenter for multimodal query embedding
        self._image_augmenter: ImageEmbeddingAugmenter | None = None
        if ingestion_cfg.get("enable_image_embeddings", False):
            self._image_augmenter = ImageEmbeddingAugmenter(ingestion_cfg)

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

        # embed query text in multimodal space for image kNN (if enabled)
        image_vector: list[float] | None = None
        if self._image_augmenter:
            image_vector = self._image_augmenter._embed_text(query_text)

        boost_fields: list[str] = soft_facts.get("boost_fields", [])

        resp = self._client.search(
            index=self._index,
            body=self._build_query(
                query_text=query_text,
                dense_vector=features.get("dense_embedding", []),
                sparse_weights=features.get("sparse_embedding", {}),
                listing_ids=listing_ids,
                image_vector=image_vector,
                boost_fields=boost_fields,
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
        image_vector: list[float] | None = None,
        boost_fields: list[str] | None = None,
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

        hybrid_queries = [
            {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["full_text", "title", "description", "full_text_en"],
                        }
                    },
                    "filter": candidate_filter,
                }
            },
            {
                # knn filter inside hybrid silently returns 0 hits — post_filter restricts instead
                "knn": {
                    "dense_embedding": {
                        "vector": dense_vector,
                        # k=n covers the full candidate neighborhood before post_filter
                        "k": n,
                    }
                }
            },
            sparse_query,
        ]

        # add image kNN sub-query when multimodal embeddings are available
        if image_vector:
            hybrid_queries.append({
                "knn": {
                    "image_embedding": {
                        "vector": image_vector,
                        "k": n,
                    }
                }
            })

        # add function_score boosts for VLM / geo feature fields
        if boost_fields:
            functions = [
                {
                    "field_value_factor": {
                        "field": field,
                        "modifier": "log1p",
                        "missing": 5,
                    }
                }
                for field in boost_fields
            ]
            hybrid_queries.append({
                "function_score": {
                    "query": {"bool": {"filter": candidate_filter}},
                    "functions": functions,
                    "boost_mode": "multiply",
                    "score_mode": "sum",
                }
            })

        return {
            # return top_k; post_filter guarantees results come from the candidate set
            "size": top_k,
            "query": {"hybrid": {"queries": hybrid_queries}},
            # restrict final results to the hard-filtered candidate set
            "post_filter": candidate_filter,
        }

    @staticmethod
    def _load_ingestion_cfg() -> DictConfig:
        cfg_path = Path(__file__).parents[3] / "configs" / "ingestion" / "config.yaml"
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        return OmegaConf.create(raw)
