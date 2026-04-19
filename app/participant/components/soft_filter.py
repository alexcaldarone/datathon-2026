from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from app.ingestion.components.client import OpenSearchClient
from app.ingestion.components.augmenters import build_augmenters, ImageEmbeddingAugmenter
from app.participant.components.logger import PipelineLogger
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
        target: int,
    ) -> list[dict[str, Any]]:
        pass


class DumbSoftFilter(SoftFilter):
    def run(
        self,
        candidates: list[dict[str, Any]],
        soft_facts: dict[str, Any],
        target: int,
    ) -> list[dict[str, Any]]:
        return candidates[:target]


class AugmentorFilter(SoftFilter):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.ingestion_cfg = self._load_ingestion_cfg()
        self._index = self.ingestion_cfg.index_name
        self._pipeline = self.ingestion_cfg.pipeline_name
        self._augmenters = build_augmenters(self.ingestion_cfg)
        self._client = OpenSearchClient()  # singleton
        # minimum query-anchor cosine similarity to include a dimension as a boost
        self.anchor_threshold = cfg.anchor_threshold
        # top-N terms from sparse BM25 weights included in the rank_features query
        self.sparse_k_for_scoring: int = 30

    def run(
        self,
        candidates: list[dict[str, Any]],
        soft_facts: dict[str, Any],
        target: int,
    ) -> list[dict[str, Any]]:
        query_text: str = soft_facts.get("_query", "")
        if not query_text or not candidates:
            return candidates
        # candidates=[{"listing_id":"695fdc5dc12a1d2f41553111"}]
        # soft_facts={"_query":"house in zurich"}
        listing_ids = [c["listing_id"] for c in candidates]
        query_listing = {"full_text": query_text}
        features = {aug.field_name: aug.augment(query_listing).content for aug in self._augmenters}

        cfg = self.cfg
        # Dense Embeddings
        dense_vector = features.get("dense_embedding") if cfg.get("enable_dense_vector", True) else None
        
        # Sparse Embeddings
        sparse_weights = features.get("sparse_embedding") if cfg.get("enable_sparse_weights", True) else None
        
        # Image embeddings
        if cfg.get("enable_image_vector", True):
            image_augmenter = ImageEmbeddingAugmenter(self.ingestion_cfg)
            image_vector = image_augmenter._embed_text(query_text)
        else:
            image_vector = None

        # Boost fields
        boost_fields: list[tuple[str, float]] | None = soft_facts.get("boost_fields") if cfg.get("enable_boost_fields", True) else None

        # Translated queries
        query_text_en: str | None = soft_facts.get("_query_en") if cfg.get("enable_query_text_en", True) else None

        # Anchor scores: cosine similarity of query to each anchor dimension
        anchor_scores: dict[str, float] | None = (
            features.get("anchor_features") if cfg.get("enable_anchor_features", True) else None
        )

        _log = PipelineLogger.get()._logger
        active = [name for name, val in [
            ("dense_vector", dense_vector),
            ("sparse_weights", sparse_weights),
            ("image_vector", image_vector),
            ("boost_fields", boost_fields),
            ("query_text_en", query_text_en),
            ("anchor_scores", anchor_scores),
        ] if val]
        _log.info("SOFT_FILTER  active_fields=%s", active)

        resp = self._client.search(
            index=self._index,
            body=self._build_query(
                query_text=query_text,
                dense_vector=dense_vector,
                sparse_weights=sparse_weights,
                listing_ids=listing_ids,
                image_vector=image_vector,
                boost_fields=boost_fields,
                query_text_en=query_text_en,
                anchor_scores=anchor_scores,
                target=target,
            ),
            pipeline=self._pipeline,
        )

        hits = resp["hits"]["hits"]

        padding = []
        if len(hits) < target:
            padding = candidates[:target - len(hits)]

        return [hit["_source"] for hit in hits][:target] + padding

    def _build_query(
        self,
        query_text: str,
        listing_ids: list[str],
        target: int,
        dense_vector: list[float] | None = None,
        sparse_weights: dict[str, float] | None = None,
        image_vector: list[float] | None = None,
        boost_fields: list[tuple[str, float]] | None = None,
        query_text_en: str | None = None,
        anchor_scores: dict[str, float] | None = None,
    ) -> dict:
        n = len(listing_ids)
        candidate_filter = {"terms": {"listing_id": listing_ids}}

        # BM25 on original-language fields
        bm25_fields = ["full_text", "title", "description"]
        if not query_text_en:
            # no translation available — include full_text_en with the original query
            bm25_fields.append("full_text_en")

        hybrid_queries = [
            {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query_text,
                            "fields": bm25_fields,
                        }
                    },
                    "filter": candidate_filter,
                }
            },
        ]

        if dense_vector:
            # knn filter inside hybrid silently returns 0 hits — post_filter restricts instead
            hybrid_queries.append({
                "knn": {
                    "dense_embedding": {
                        "vector": dense_vector,
                        # k=n covers the full candidate neighborhood before post_filter
                        "k": n,
                    }
                }
            })

        if sparse_weights:
            # top-N sparse terms by BM25 weight; rank_feature (singular) is the query type
            top_terms = sorted(sparse_weights.items(), key=lambda x: -x[1])[:self.sparse_k_for_scoring]
            hybrid_queries.append({
                "bool": {
                    "should": [{"rank_feature": {"field": f"sparse_embedding.{t}"}} for t, _ in top_terms],
                    "filter": candidate_filter,
                }
            })

        # add translated-query BM25 sub-query for cross-language matching
        if query_text_en:
            hybrid_queries.append({
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query_text_en,
                            "fields": ["full_text_en", "full_text", "title", "description"],
                        }
                    },
                    "filter": candidate_filter,
                }
            })

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
                    },
                    "weight": weight,
                }
                for field, weight in boost_fields
            ]
            hybrid_queries.append({
                "function_score": {
                    "query": {"bool": {"filter": candidate_filter}},
                    "functions": functions,
                    "boost_mode": "multiply",
                    "score_mode": "sum",
                }
            })

        if anchor_scores:
            boosted = {dim: score for dim, score in anchor_scores.items() if score >= self.anchor_threshold}
            if boosted:
                hybrid_queries.append({
                    "function_score": {
                        "query": {"bool": {"filter": candidate_filter}},
                        "functions": [
                            {
                                "field_value_factor": {
                                    "field": f"anchor_features.{dim}",
                                    "modifier": "none",
                                    "missing": 0,
                                },
                                "weight": score,
                            }
                            for dim, score in boosted.items()
                        ],
                        "boost_mode": "multiply",
                        "score_mode": "sum",
                    }
                })

        return {
            # return target; post_filter guarantees results come from the candidate set
            "size": target,
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
