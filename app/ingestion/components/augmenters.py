import json
import re
import time
import os
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any

import boto3
from omegaconf import DictConfig
from pydantic import BaseModel

from app.models.schemas import SoftFactWeights
from sentence_transformers import SentenceTransformer, util
import yaml
import torch


def build_augmenters(cfg: DictConfig) -> list["Augmenter"]:
    return [DenseEmbeddingAugmenter(cfg), BM25SparseAugmenter(cfg)]


class FeatureType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    IMAGE = "image"


class AugmentedFeature(BaseModel):
    name: str
    type: FeatureType
    content: Any


class Augmenter(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @property
    @abstractmethod
    def field_name(self) -> str: ...

    @property
    @abstractmethod
    def feature_type(self) -> FeatureType: ...

    @property
    @abstractmethod
    def field_mapping(self) -> dict: ...

    @abstractmethod
    def augment(self, listing: dict) -> AugmentedFeature: ...

    # default: sequential; concrete classes may override for efficiency
    def augment_batch(self, listings: list[dict]) -> list[AugmentedFeature]:
        return [self.augment(l) for l in listings]


class DenseEmbeddingAugmenter(Augmenter):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._bedrock = boto3.client("bedrock-runtime", region_name=os.environ["AWS_DEFAULT_REGION"])

    @property
    def field_name(self) -> str:
        return "dense_embedding"

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.DENSE

    @property
    def field_mapping(self) -> dict:
        return {
            "type": "knn_vector",
            "dimension": self.cfg.embed_dim,
            "method": {
                "name": "hnsw",
                "space_type": "l2",
                "engine": "lucene",
                "parameters": {"ef_construction": 256, "m": 16},
            },
        }

    def augment(self, listing: dict) -> AugmentedFeature:
        embedding = self._embed(listing["full_text"])
        return AugmentedFeature(name=self.field_name, type=self.feature_type, content=embedding)

    def augment_batch(self, listings: list[dict]) -> list[AugmentedFeature]:
        texts = [l["full_text"] for l in listings]
        results: list[list[float] | None] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=self.cfg.embed_workers) as pool:
            futures = {pool.submit(self._embed, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return [
            AugmentedFeature(name=self.field_name, type=self.feature_type, content=e)
            for e in results
        ]

    def _embed(self, text: str, retries: int = 3) -> list[float]:
        payload = json.dumps({"inputText": text, "dimensions": self.cfg.embed_dim, "normalize": True})
        for attempt in range(retries):
            try:
                resp = self._bedrock.invoke_model(
                    modelId=self.cfg.model_id,
                    body=payload,
                    contentType="application/json",
                    accept="application/json",
                )
                return json.loads(resp["body"].read())["embedding"]
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"\nEmbedding error (attempt {attempt + 1}): {exc}. Retrying in {wait}s...")
                time.sleep(wait)
        return []

class AnchorsAugmenter(Augmenter):
    _model = None # class level cache

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if AnchorsAugmenter._model is None:
            # Load the multilingual model requested by the user
            AnchorsAugmenter._model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Load semantic "anchors" from YAML
        anchor_path = getattr(self.cfg, "anchor_path", "configs/soft_extractor_anchors.yaml")
        if os.path.exists(anchor_path):
            with open(anchor_path, "r", encoding="utf-8") as f:
                self.anchors = yaml.safe_load(f)
        else:
            self.anchors = {}
        
        # Load importance keywords from YAML
        keywords_path = getattr(self.cfg, "importance_keywords_path", "configs/soft_extractor_importance_keywords.yaml")
        if os.path.exists(keywords_path):
            with open(keywords_path, "r", encoding="utf-8") as f:
                self.importance_keywords = yaml.safe_load(f)
        else:
            self.importance_keywords = {}
        
        # Pre-compute anchor embeddings for positive and negative phrase sets
        self.anchor_embeddings = {
            attr: {
                side: self._model.encode(phrases, convert_to_tensor=True)
                for side, phrases in sides.items()
                if isinstance(phrases, list) and phrases
            }
            for attr, sides in self.anchors.items()
        }
    
    @property
    def field_name(self) -> str:
        return "anchors"

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.DENSE
    
    @property
    def field_mapping(self) -> dict:
        return {
            "type": "knn_vector",
            "dimension": self.cfg.num_anchor,
            "method": {
                "name": "hnsw",
                "space_type": "l2", # would change to cosine similarity
                "engine": "lucene",
                "parameters": {"ef_construction": 256, "m": 16},
            },
        }
    
    def augment(self, listing: dict) -> AugmentedFeature:
        vector = self._process(listing["full_text"])
        return AugmentedFeature(name=self.field_name, type=self.feature_type, content=vector)

    def augment_batch(self, listings: list[dict]) -> list[AugmentedFeature]:
        texts = [l["full_text"] for l in listings]
        results: list[list[float] | None] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=self.cfg.embed_workers) as pool:
            futures = {pool.submit(self._process, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return [
            AugmentedFeature(name=self.field_name, type=self.feature_type, content=e)
            for e in results
        ]

    def _process(self, text: str) -> list[float]:
        return self._weights_to_vector(self._get_weights(text))

    def _get_weights(self, query: str) -> SoftFactWeights:
        query_embedding = self._model.encode(query, convert_to_tensor=True)
        q_lower = query.lower()
        weights_dict = {}

        # Intensity multiplier: 1.0 for neutral queries, > 1.0 when strong importance
        # keywords are detected (e.g. "muss", "wichtig"). Never flips sign.
        intensity = 1.0
        for val, keywords in self.importance_keywords.get("amplify", {}).items():
            if any(k in q_lower for k in keywords):
                intensity = max(intensity, float(val))

        for attr, sides in self.anchor_embeddings.items():
            pos_embeddings = sides.get("positive")
            neg_embeddings = sides.get("negative")

            max_pos_sim = float(torch.max(util.cos_sim(query_embedding, pos_embeddings))) if pos_embeddings is not None else 0.0
            max_neg_sim = float(torch.max(util.cos_sim(query_embedding, neg_embeddings))) if neg_embeddings is not None else 0.0

            # Raw directional score. tanh(k * raw) amplifies clear signals
            # toward ±1 while keeping neutral queries near 0.
            raw = max_pos_sim - max_neg_sim
            weight = float(torch.tanh(torch.tensor(raw * 3.0 * intensity)))

            weights_dict[attr] = round(weight, 2)

        self.weights = SoftFactWeights(**weights_dict)
        return self.weights

    def _weights_to_vector(self, weights: SoftFactWeights) -> list[float]:
        return list(weights.model_dump().values())

        

class BM25SparseAugmenter(Augmenter):
    # BM25 parameters (standard defaults)
    _K1: float = 1.2
    _B: float = 0.75
    # corpus average document length in tokens (approximate; tunes length normalization)
    _AVG_DOC_LEN: int = 100

    @property
    def field_name(self) -> str:
        return "sparse_embedding"

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.SPARSE

    @property
    def field_mapping(self) -> dict:
        # rank_features stores token → weight maps and supports rank_features queries
        return {"type": "rank_features"}

    def augment(self, listing: dict) -> AugmentedFeature:
        weights = self._bm25_weights(listing["full_text"])
        return AugmentedFeature(name=self.field_name, type=self.feature_type, content=weights)

    def _bm25_weights(self, text: str) -> dict[str, float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return {}
        tf = Counter(tokens)
        doc_len = len(tokens)
        return {
            term: round(
                (freq * (self._K1 + 1))
                / (freq + self._K1 * (1 - self._B + self._B * doc_len / self._AVG_DOC_LEN)),
                4,
            )
            for term, freq in tf.items()
        }
