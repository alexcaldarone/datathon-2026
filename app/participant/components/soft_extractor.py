from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Any
import json
from sentence_transformers import SentenceTransformer, util
import yaml
import os
import torch
from app.models.schemas import SoftFactWeights

from omegaconf import DictConfig

from app.participant.components.utils import _instantiate

_MODULE = "app.participant.components.soft_extractor"


def build_soft_extractor(cfg: DictConfig) -> SoftFactExtractor:
    return _instantiate(cfg.soft_extractor, _MODULE)


class SoftFactExtractor(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg=cfg

    @abstractmethod
    def run(self, query: str) -> dict[str, Any]:
        pass

class LLMSoftExtractor(SoftFactExtractor):
    """
    Uses an LLM to parse the natural language query and map it to feature weights.
    """
    _model = None  # Class-level model cache

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.weights = SoftFactWeights()
        self.buildings = list[Property]
        if LLMSoftExtractor._model is None:
            # Load the multilingual model requested by the user
            LLMSoftExtractor._model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
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

        # Pre-compute anchor embeddings
        self.anchor_embeddings = {
            attr: self._model.encode(phrases, convert_to_tensor=True)
            for attr, phrases in self.anchors.items()
            if isinstance(phrases, list) and phrases
        }

    # Prompt (not used)
    def _get_system_prompt(self) -> str:
        return """
        You are an expert real estate analyzer. Given a user query, extract the importance of various 'soft' attributes.
        Assign a weight between -1.0 and 1.0 for each attribute.
        - 1.0: Essential / Must have / Explicitly mentioned as a priority.
        - 0.7: Strong preference / Looking for...
        - 0.4: Mentioned but not critical / "Nice to have".
        - 0.0: Not mentioned.
        - -0.5: Avoid / Prefer not to have.
        - -1.0: Strictly avoid / Must not have.

        Attributes to score:
        - daylight_score: Brightness, sunny, large windows.
        - noise_level: Quietness, peaceful neighborhood.
        - view_quality_score: Views of lake, mountains, or city.
        - green_view_ratio: Proximity to parks, trees, gardens.
        - walkability_score: Close to shops, restaurants, central area.
        - public_transport_score: Near tram, bus, or train stations.
        - interior_modernity: Modern finishings, renovated, high-end, dishwasher.
        - work_from_home_fitness: High-speed internet, office space, quiet for work.
        - spaciousness_perception: Perception of space, high ceilings, large kitchen.
        - neighborhood_safety_score: Family friendly, safe area.
        - proximity_to_desired_location_score: Close to specific landmarks mentioned (e.g., 'near Uni', '15 mins from HB').

        IMPORTANT: Ignore hard facts like Price/Budget and Location (Districts/Cities). Focus ONLY on quality of life and feature preferences.
        Return ONLY valid JSON corresponding to the SoftFactWeights schema.
        """

    def get_weights(self, query: str) -> SoftFactWeights:
        query_embedding = self._model.encode(query, convert_to_tensor=True)
        q_lower = query.lower()
        weights_dict = {}

        # Intensity multiplier: 1.0 for neutral queries, > 1.0 when strong importance
        # keywords are detected (e.g. "must", "essential"). Never flips sign.
        intensity = 1.0
        for val, keywords in self.importance_keywords.get("amplify", {}).items():
            if any(k in q_lower for k in keywords):
                intensity = max(intensity, float(val))

        for attr, anchor_embeds in self.anchor_embeddings.items():
            avg_sim = float(torch.mean(util.cos_sim(query_embedding, anchor_embeds)))
            
            # 0.55 is a safer baseline to distinguish from pure noise.
            baseline = 0
            raw_weight = max(0.0, (avg_sim - baseline) / (1.0 - baseline))
            
            # Use tanh to scale the similarity in [0, 1]
            weight = float(torch.tanh(torch.tensor(raw_weight * intensity)))

            weights_dict[attr] = round(weight, 2)

        self.weights = SoftFactWeights(**weights_dict)
        return self.weights

    def weights_to_vector(self, weights: SoftFactWeights) -> list[float]:
        return list(weights.model_dump().values())

    def run(self, query: str) -> list[dict[str, Any]]:
        extracted_weights = self.get_weights(query)
        self.results = []

        # Indicator can be tuned to be more or less restrictive
        indicator = 0.8 

        for building in getattr(self, "buildings", []):
            included = True
            building_weights_dict = building.weights.model_dump()
            query_weights_dict = extracted_weights.model_dump()

            for attr, required_weight in query_weights_dict.items():
                if required_weight > 0.1:  # Only filter if there's a significant requirement
                    building_weight = building_weights_dict.get(attr, 0)
                    if building_weight < (required_weight * indicator):
                        included = False
                        break
            
            if included:
                self.results.append({
                    "id": building.id,
                    "info": building.info,
                    "weights": building.weights
                })

        return self.results

# ???
class Property():
    def __init__(self, id: str, info: dict, weights: SoftFactWeights):
        self.id = id
        self.info = info
        self.weights = weights

        # Compute weight logic here
    

class DumbSoftExtractor(SoftFactExtractor):
    def run(self, _query: str) -> dict[str, Any]:
        return {}
