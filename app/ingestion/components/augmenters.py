import base64
import json
import math
import re
import threading
import time
import os
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any
import boto3
import numpy as np
import requests
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from app.ingestion.components.logger import IngestionLogger

_logger = IngestionLogger.get()



def build_augmenters(cfg: DictConfig) -> list["Augmenter"]:
    augmenters: list[Augmenter] = [DenseEmbeddingAugmenter(cfg), BM25SparseAugmenter(cfg)]
    if cfg.get("enable_image_embeddings", False):
        augmenters.append(ImageEmbeddingAugmenter(cfg))
    if cfg.get("enable_vlm_features", False):
        augmenters.append(VLMFeatureAugmenter(cfg))
    if cfg.get("enable_translation", False):
        augmenters.append(TranslationAugmenter(cfg))
    if cfg.get("enable_geo_features", False):
        augmenters.append(GeoFeatureAugmenter(cfg))
    if cfg.get("enable_anchor_features", False):
        augmenters.append(AnchorsAugmenter(cfg))
    return augmenters


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

    @property
    def needs_images(self) -> bool:
        return False

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
                _logger.warning("Embedding error (attempt %d): %s. Retrying in %ds...", attempt + 1, exc, wait)
                time.sleep(wait)
        return []


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


class ImageEmbeddingAugmenter(Augmenter):
    _IMAGE_DIM: int = 1024
    _REQUEST_TIMEOUT: int = 10

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._bedrock = boto3.client("bedrock-runtime", region_name=os.environ["AWS_DEFAULT_REGION"])
        self._model_id = cfg.get("image_model_id", "amazon.titan-embed-image-v1")

    @property
    def needs_images(self) -> bool:
        return True

    @property
    def field_name(self) -> str:
        return "image_embedding"

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.IMAGE

    @property
    def field_mapping(self) -> dict:
        return {
            "type": "knn_vector",
            "dimension": self._IMAGE_DIM,
            "method": {
                "name": "hnsw",
                "space_type": "l2",
                "engine": "lucene",
                "parameters": {"ef_construction": 256, "m": 16},
            },
        }

    def augment(self, listing: dict) -> AugmentedFeature:
        image_bytes = listing.get("_image_bytes")
        if image_bytes:
            embedding = self._embed_image(image_bytes)
        else:
            # fallback: embed text in the same multimodal vector space
            embedding = self._embed_text(listing["full_text"])
        return AugmentedFeature(name=self.field_name, type=self.feature_type, content=embedding)

    def augment_batch(self, listings: list[dict]) -> list[AugmentedFeature]:
        results: list[AugmentedFeature | None] = [None] * len(listings)
        with ThreadPoolExecutor(max_workers=self.cfg.get("embed_workers", 8)) as pool:
            futures = {pool.submit(self.augment, l): i for i, l in enumerate(listings)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results  # type: ignore[return-value]

    def _embed_image(self, image_bytes: bytes, retries: int = 3) -> list[float]:
        payload = json.dumps({"inputImage": base64.b64encode(image_bytes).decode()})
        return self._invoke(payload, retries)

    def _embed_text(self, text: str, retries: int = 3) -> list[float]:
        payload = json.dumps({"inputText": text})
        return self._invoke(payload, retries)

    def _invoke(self, payload: str, retries: int) -> list[float]:
        for attempt in range(retries):
            try:
                resp = self._bedrock.invoke_model(
                    modelId=self._model_id,
                    body=payload,
                    contentType="application/json",
                    accept="application/json",
                )
                return json.loads(resp["body"].read())["embedding"]
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                _logger.warning("Image embedding error (attempt %d): %s. Retrying in %ds...", attempt + 1, exc, wait)
                time.sleep(wait)
        return []


# ---------------------------------------------------------------------------
# VLM Feature Augmenter
# ---------------------------------------------------------------------------

_VLM_FEATURE_KEYS = [
    "brightness", "spaciousness", "modernity", "view_quality",
    "greenery", "kitchen_quality", "condition", "noise_impression",
]


class VLMFeatureAugmenter(Augmenter):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._bedrock = boto3.client("bedrock-runtime", region_name=os.environ["AWS_DEFAULT_REGION"])
        self._model_id = cfg.get("vlm_model_id", "anthropic.claude-3-haiku-20240307-v1:0")
        from app.participant.components.utils import read_system_prompt  # lazy: avoids circular import
        self.system_prompt = read_system_prompt(self.__class__.__name__)

    @property
    def needs_images(self) -> bool:
        return True

    @property
    def field_name(self) -> str:
        return "vlm_features"

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.SPARSE

    @property
    def field_mapping(self) -> dict:
        return {
            "type": "object",
            "properties": {k: {"type": "float"} for k in _VLM_FEATURE_KEYS},
        }

    def augment(self, listing: dict) -> AugmentedFeature:
        image_bytes = listing.get("_image_bytes")
        if not image_bytes:
            return AugmentedFeature(name=self.field_name, type=self.feature_type, content={})
        scores = self._analyze_image(image_bytes)
        return AugmentedFeature(name=self.field_name, type=self.feature_type, content=scores)

    def augment_batch(self, listings: list[dict]) -> list[AugmentedFeature]:
        results: list[AugmentedFeature | None] = [None] * len(listings)
        with ThreadPoolExecutor(max_workers=self.cfg.get("embed_workers", 8)) as pool:
            futures = {pool.submit(self.augment, l): i for i, l in enumerate(listings)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results  # type: ignore[return-value]

    def _analyze_image(self, image_bytes: bytes, retries: int = 3) -> dict:
        b64 = base64.b64encode(image_bytes).decode()
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": self.system_prompt},
                    ],
                }
            ],
        })
        for attempt in range(retries):
            try:
                resp = self._bedrock.invoke_model(
                    modelId=self._model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                raw = json.loads(resp["body"].read())
                text = raw["content"][0]["text"]
                return self._parse_scores(text)
            except Exception as exc:
                if attempt == retries - 1:
                    _logger.error("VLM analysis failed: %s", exc)
                    return {}
                time.sleep(2 ** attempt)
        return {}

    @staticmethod
    def _parse_scores(text: str) -> dict:
        # extract JSON from response (may have markdown fences)
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return {}
        # validate: keep only known keys with numeric values 0-10
        scores = {}
        for key in _VLM_FEATURE_KEYS:
            val = raw.get(key)
            if isinstance(val, (int, float)) and 0 <= val <= 10:
                scores[key] = float(val)
        return scores


# ---------------------------------------------------------------------------
# Translation Augmenter
# ---------------------------------------------------------------------------

_NON_ENGLISH_MARKERS = {
    "wohnung", "zimmer", "miete", "strasse", "nähe", "küche", "schlafzimmer",
    "appartement", "chambre", "loyer", "cuisine", "près", "étage",
    "appartamento", "camera", "affitto", "cucina", "vicino", "piano",
    "zürich", "bern", "luzern", "genève", "lausanne",
}


class TranslationAugmenter(Augmenter):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._bedrock = boto3.client("bedrock-runtime", region_name=os.environ["AWS_DEFAULT_REGION"])
        self._model_id = cfg.get("translation_model_id", "anthropic.claude-3-haiku-20240307-v1:0")

    @property
    def field_name(self) -> str:
        return "full_text_en"

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.SPARSE

    @property
    def field_mapping(self) -> dict:
        return {"type": "text", "analyzer": "standard", "similarity": "custom_bm25"}

    def augment(self, listing: dict) -> AugmentedFeature:
        text = listing["full_text"]
        if self._is_english(text):
            return AugmentedFeature(name=self.field_name, type=self.feature_type, content=text)
        translated = self._translate(text)
        return AugmentedFeature(name=self.field_name, type=self.feature_type, content=translated)

    def augment_batch(self, listings: list[dict]) -> list[AugmentedFeature]:
        results: list[AugmentedFeature | None] = [None] * len(listings)
        with ThreadPoolExecutor(max_workers=self.cfg.get("embed_workers", 8)) as pool:
            futures = {pool.submit(self.augment, l): i for i, l in enumerate(listings)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results  # type: ignore[return-value]

    @staticmethod
    def _is_english(text: str) -> bool:
        tokens = set(re.findall(r"[a-zäöüéèàâêîôû]+", text.lower()))
        hits = tokens & _NON_ENGLISH_MARKERS
        return len(hits) < 3

    def _translate(self, text: str, retries: int = 3) -> str:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Translate the following real estate listing to English. "
                        "Preserve all factual details (numbers, addresses, features). "
                        "Return only the translation, no commentary.\n\n"
                        f"{text}"
                    ),
                }
            ],
        })
        for attempt in range(retries):
            try:
                resp = self._bedrock.invoke_model(
                    modelId=self._model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                raw = json.loads(resp["body"].read())
                return raw["content"][0]["text"].strip()
            except Exception as exc:
                if attempt == retries - 1:
                    _logger.error("Translation failed: %s", exc)
                    return text  # fallback: return original
                time.sleep(2 ** attempt)
        return text


# ---------------------------------------------------------------------------
# Geo Feature Augmenter (transit + amenities via Overpass)
# ---------------------------------------------------------------------------

_GEO_FEATURE_KEYS = [
    "stops_500m", "stations_1km", "nearest_stop_m", "nearest_station_m",
    "schools_1km", "kindergartens_1km", "playgrounds_500m",
    "supermarkets_500m", "pharmacies_500m", "gyms_1km",
    "restaurants_500m", "doctors_1km", "parks_500m",
    "family_score", "transit_score", "walkability_score",
]

_GEO_MAPPING_TYPES = {
    "stops_500m": "integer", "stations_1km": "integer",
    "nearest_stop_m": "float", "nearest_station_m": "float",
    "schools_1km": "integer", "kindergartens_1km": "integer",
    "playgrounds_500m": "integer", "supermarkets_500m": "integer",
    "pharmacies_500m": "integer", "gyms_1km": "integer",
    "restaurants_500m": "integer", "doctors_1km": "integer",
    "parks_500m": "integer",
    "family_score": "float", "transit_score": "float", "walkability_score": "float",
}


class GeoFeatureAugmenter(Augmenter):
    _STOP_RADIUS: int = 500
    _STATION_RADIUS: int = 1000
    _AMENITY_CLOSE_RADIUS: int = 500
    _AMENITY_FAR_RADIUS: int = 1000

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._overpass_url = cfg.get("overpass_url", "https://overpass.osm.ch/api/")
        self._workers = int(cfg.get("overpass_workers", 20))

    @property
    def field_name(self) -> str:
        return "geo_features"

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.SPARSE

    @property
    def field_mapping(self) -> dict:
        return {
            "type": "object",
            "properties": {k: {"type": v} for k, v in _GEO_MAPPING_TYPES.items()},
        }

    def augment(self, listing: dict) -> AugmentedFeature:
        lat, lon = listing.get("latitude"), listing.get("longitude")
        if not lat or not lon:
            return AugmentedFeature(name=self.field_name, type=self.feature_type, content={})
        try:
            lat_f, lon_f = float(lat), float(lon)
        except (TypeError, ValueError):
            return AugmentedFeature(name=self.field_name, type=self.feature_type, content={})
        data = self._query_overpass(lat_f, lon_f)
        features = self._extract_features(lat_f, lon_f, data)
        return AugmentedFeature(name=self.field_name, type=self.feature_type, content=features)

    def augment_batch(self, listings: list[dict]) -> list[AugmentedFeature]:
        results: list[AugmentedFeature | None] = [None] * len(listings)
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {pool.submit(self.augment, l): i for i, l in enumerate(listings)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results  # type: ignore[return-value]

    def _query_overpass(self, lat: float, lon: float, retries: int = 3) -> dict:
        query = f"""
[out:json][timeout:30];
(
  node["public_transport"="stop_position"](around:{self._STOP_RADIUS},{lat},{lon});
  node["public_transport"="platform"](around:{self._STOP_RADIUS},{lat},{lon});
  node["railway"="station"](around:{self._STATION_RADIUS},{lat},{lon});
  node["railway"="halt"](around:{self._STATION_RADIUS},{lat},{lon});
  node["railway"="tram_stop"](around:{self._STOP_RADIUS},{lat},{lon});
  node["amenity"="school"](around:{self._AMENITY_FAR_RADIUS},{lat},{lon});
  node["amenity"="kindergarten"](around:{self._AMENITY_FAR_RADIUS},{lat},{lon});
  node["leisure"="playground"](around:{self._AMENITY_CLOSE_RADIUS},{lat},{lon});
  node["shop"="supermarket"](around:{self._AMENITY_CLOSE_RADIUS},{lat},{lon});
  node["amenity"="pharmacy"](around:{self._AMENITY_CLOSE_RADIUS},{lat},{lon});
  node["leisure"="fitness_centre"](around:{self._AMENITY_FAR_RADIUS},{lat},{lon});
  node["amenity"="restaurant"](around:{self._AMENITY_CLOSE_RADIUS},{lat},{lon});
  node["amenity"="doctors"](around:{self._AMENITY_FAR_RADIUS},{lat},{lon});
  node["leisure"="park"](around:{self._AMENITY_CLOSE_RADIUS},{lat},{lon});
);
out body;
"""
        for attempt in range(retries):
            try:
                resp = requests.post(
                    self._overpass_url,
                    data={"data": query},
                    timeout=60,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                if attempt == retries - 1:
                    _logger.error("Overpass query failed for (%s, %s): %s", lat, lon, exc)
                    return {"elements": []}
                time.sleep(2 ** attempt)
        return {"elements": []}

    def _extract_features(self, lat: float, lon: float, data: dict) -> dict:
        elements = data.get("elements", [])

        # classify elements by tag
        stops: list[dict] = []
        stations: list[dict] = []
        amenity_counts: dict[str, int] = {
            "schools_1km": 0, "kindergartens_1km": 0, "playgrounds_500m": 0,
            "supermarkets_500m": 0, "pharmacies_500m": 0, "gyms_1km": 0,
            "restaurants_500m": 0, "doctors_1km": 0, "parks_500m": 0,
        }

        for el in elements:
            tags = el.get("tags", {})
            if tags.get("railway") in ("station", "halt"):
                stations.append(el)
            elif tags.get("public_transport") in ("stop_position", "platform") or tags.get("railway") == "tram_stop":
                stops.append(el)
            elif tags.get("amenity") == "school":
                amenity_counts["schools_1km"] += 1
            elif tags.get("amenity") == "kindergarten":
                amenity_counts["kindergartens_1km"] += 1
            elif tags.get("leisure") == "playground":
                amenity_counts["playgrounds_500m"] += 1
            elif tags.get("shop") == "supermarket":
                amenity_counts["supermarkets_500m"] += 1
            elif tags.get("amenity") == "pharmacy":
                amenity_counts["pharmacies_500m"] += 1
            elif tags.get("leisure") == "fitness_centre":
                amenity_counts["gyms_1km"] += 1
            elif tags.get("amenity") == "restaurant":
                amenity_counts["restaurants_500m"] += 1
            elif tags.get("amenity") == "doctors":
                amenity_counts["doctors_1km"] += 1
            elif tags.get("leisure") == "park":
                amenity_counts["parks_500m"] += 1

        nearest_stop = self._nearest_distance(lat, lon, stops)
        nearest_station = self._nearest_distance(lat, lon, stations)

        features = {
            "stops_500m": len(stops),
            "stations_1km": len(stations),
            "nearest_stop_m": round(nearest_stop, 1) if nearest_stop is not None else None,
            "nearest_station_m": round(nearest_station, 1) if nearest_station is not None else None,
            **amenity_counts,
        }

        # composite scores (0-10 scale)
        features["family_score"] = self._family_score(amenity_counts)
        features["transit_score"] = self._transit_score(len(stops), len(stations), nearest_stop)
        features["walkability_score"] = self._walkability_score(amenity_counts)

        # strip None values
        return {k: v for k, v in features.items() if v is not None}

    @staticmethod
    def _nearest_distance(lat: float, lon: float, elements: list[dict]) -> float | None:
        if not elements:
            return None
        min_dist = float("inf")
        for el in elements:
            el_lat = el.get("lat")
            el_lon = el.get("lon")
            if el_lat is None or el_lon is None:
                continue
            dist = _haversine(lat, lon, el_lat, el_lon)
            min_dist = min(min_dist, dist)
        return min_dist if min_dist < float("inf") else None

    @staticmethod
    def _family_score(counts: dict[str, int]) -> float:
        raw = (
            counts["schools_1km"] * 2.0
            + counts["kindergartens_1km"] * 2.0
            + counts["playgrounds_500m"] * 1.5
            + counts["parks_500m"] * 1.0
            + counts["supermarkets_500m"] * 0.5
        ) / 3.0
        return round(min(10.0, raw), 2)

    @staticmethod
    def _transit_score(stops: int, stations: int, nearest_stop_m: float | None) -> float:
        # more stops and closer = higher score
        count_score = min(5.0, stops * 0.5 + stations * 2.0)
        if nearest_stop_m is not None and nearest_stop_m > 0:
            # 0m → 5.0, 500m → 0.0
            proximity_score = max(0.0, 5.0 * (1.0 - nearest_stop_m / 500.0))
        else:
            proximity_score = 0.0
        return round(min(10.0, count_score + proximity_score), 2)

    @staticmethod
    def _walkability_score(counts: dict[str, int]) -> float:
        raw = (
            counts["supermarkets_500m"] * 2.0
            + counts["pharmacies_500m"] * 1.5
            + counts["restaurants_500m"] * 0.5
            + counts["doctors_1km"] * 1.0
        ) / 2.0
        return round(min(10.0, raw), 2)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000  # earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Anchors Augmenter
# ---------------------------------------------------------------------------

_ANCHOR_AGGREGATIONS = {
    "max": np.max,
    "min": np.min,
    "mean": np.mean,
    "median": np.median,
}


class AnchorsAugmenter(Augmenter):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._bedrock = boto3.client("bedrock-runtime", region_name=os.environ["AWS_DEFAULT_REGION"])
        self._model_id = cfg.get("anchor_model_id", "amazon.titan-embed-text-v2:0")
        self._embed_dim = int(cfg.get("anchor_embed_dim", 1024))
        self._workers = int(cfg.get("anchor_workers", 10))
        self._agg_fn = cfg.get("anchor_aggregation", "max")
        self._anchor_path = cfg.get("anchor_path", "configs/soft_extractor_anchors.yaml")
        self._cache_path = cfg.get("anchor_cache_path", ".cache/anchors_embeddings.npz")
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        self._anchors: dict[str, list[str]] = OmegaConf.to_container(OmegaConf.load(self._anchor_path))  # type: ignore[assignment]
        self._cache_lock = threading.Lock()
        self._embedding_cache: dict[str, np.ndarray] = self._load_cache()
        # pre-embed all anchor sentences at startup (uses cache when available)
        self._anchor_embeddings: dict[str, np.ndarray] = self._build_anchor_embeddings()

    @property
    def field_name(self) -> str:
        return "anchor_features"

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.SPARSE

    @property
    def field_mapping(self) -> dict:
        return {
            "type": "object",
            "properties": {name: {"type": "float"} for name in self._anchors},
        }

    def augment(self, listing: dict) -> AugmentedFeature:
        text_emb = np.array(self._embed(listing["full_text"]))
        scores = self._score(text_emb)
        return AugmentedFeature(name=self.field_name, type=self.feature_type, content=scores)

    def augment_batch(self, listings: list[dict]) -> list[AugmentedFeature]:
        results: list[AugmentedFeature | None] = [None] * len(listings)
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {pool.submit(self.augment, l): i for i, l in enumerate(listings)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results  # type: ignore[return-value]

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not os.path.exists(self._cache_path):
            return {}
        data = np.load(self._cache_path, allow_pickle=True)
        return dict(zip(data["sentences"].tolist(), data["embeddings"]))

    def _save_cache(self) -> None:
        sentences = list(self._embedding_cache.keys())
        embeddings = np.array([self._embedding_cache[s] for s in sentences])
        np.savez_compressed(
            self._cache_path,
            sentences=np.array(sentences, dtype=object),
            embeddings=embeddings,
        )

    def _build_anchor_embeddings(self) -> dict[str, np.ndarray]:
        all_sentences = [s for sentences in self._anchors.values() for s in sentences]
        missing = [s for s in all_sentences if s not in self._embedding_cache]

        if missing:
            fresh: dict[str, np.ndarray] = {}
            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                futures = {pool.submit(self._embed, s): s for s in missing}
                for fut in as_completed(futures):
                    sentence = futures[fut]
                    fresh[sentence] = np.array(fut.result())
            with self._cache_lock:
                self._embedding_cache.update(fresh)
                self._save_cache()

        return {
            name: np.stack([self._embedding_cache[s] for s in sentences])
            for name, sentences in self._anchors.items()
        }

    def _embed(self, text: str, retries: int = 3) -> list[float]:
        payload = json.dumps({"inputText": text, "dimensions": self._embed_dim, "normalize": True})
        for attempt in range(retries):
            try:
                resp = self._bedrock.invoke_model(
                    modelId=self._model_id,
                    body=payload,
                    contentType="application/json",
                    accept="application/json",
                )
                return json.loads(resp["body"].read())["embedding"]
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                _logger.warning("Anchor embedding error (attempt %d): %s. Retrying in %ds...", attempt + 1, exc, wait)
                time.sleep(wait)
        return []

    def _score(self, text_emb: np.ndarray) -> dict[str, float]:
        # dot product = cosine similarity since Bedrock normalizes embeddings (normalize=True)
        agg = _ANCHOR_AGGREGATIONS.get(self._agg_fn)
        if agg is None:
            raise ValueError(f"Unknown anchor_aggregation '{self._agg_fn}'. Choose from: {list(_ANCHOR_AGGREGATIONS)}")
        return {
            name: float(agg(anchor_embs @ text_emb))
            for name, anchor_embs in self._anchor_embeddings.items()
        }

