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
        self.results = {}
        self.weights = SoftFactWeights()
        if LLMSoftExtractor._model is None:
            # Load the multilingual model requested by the user
            LLMSoftExtractor._model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Define extensive semantic "anchors" for generalization
        self.anchors = {
            "daylight_score": [
                "bright apartment with sun", "natural light", "floor to ceiling windows", "sunny balcony",
                "Helle Wohnung mit viel Sonne", "lichtdurchflutet", "direkte Sonneneinstrahlung", "grosse Fenster",
                "vül sunne am Nomittag", "helli Zimmer", "sunnig und schön"
            ],
            "noise_level": [
                "quiet area away from traffic", "peaceful residential neighborhood", "soundproof windows", "calm environment",
                "Ruhiges Quartier ohne Strassenlärm", "schalldichte Fenster", "keine Fluglärm", "seriöse ruhige Lage",
                "ganz ruhig", "kei Lärm idä Nacht", "stilli Umgäbig"
            ],
            "view_quality_score": [
                "apartment with a view", "panoramic mountain view", "lake view from rooftop", "stunning city skyline",
                "Wohnung mit schöner Aussicht", "Seesicht", "Bergblick", "Weitblick über die Stadt",
                "super Ussicht uf d Alpe", "Zürisee Blick", "mer gseht d Berge"
            ],
            "green_view_ratio": [
                "near nature and parks", "overlooking a garden", "green surroundings", "forest nearby",
                "In der Nähe von Parks und Grünflächen", "Gartenanteil", "Blick ins Grüne", "Naturnah",
                "nöch a dr Natur", "mer gseht Bäum und Gras", "bi de Allemände"
            ],
            "walkability_score": [
                "close to shops and groceries", "central location near amenities", "walkable distance to pharmacy",
                "Einkaufsmöglichkeiten zu Fuss erreichbar", "zentrale Lage", "Migros Coop in der Nähe",
                "Gute Inchauschance", "mer cha alles z Fuäss mache", "nöch bim Coop"
            ],
            "public_transport_score": [
                "near public transport tram train", "good connection to city center", "short walk to station",
                "Gute Anbindung an ÖV", "Bahnhof in der Nähe", "Tramhaltestelle vor der Tür",
                "nöch bi de Tramstation", "S-Bahn Aaschluss", "gueti Verbindig"
            ],
            "interior_modernity": [
                "modern high-end interior", "newly renovated finishings", "dishwasher and washing machine", "luxury kitchen",
                "Moderne hochwertige Innenausstattung", "Erstbezug nach Sanierung", "Geschirrspüler vorhanden",
                "modärni Chuchi", "alles neu renoviert", "luxuriösi Usstattig", "Abwäschmaschine"
            ],
            "work_from_home_fitness": [
                "home office with high speed internet", "quiet for focused work", "fiber optic connection",
                "Idealer Arbeitsplatz für Home Office", "Glasfaseranschluss", "schnelles Internet",
                "home office tauglich", "guets WLAN", "schnells Netz"
            ],
            "spaciousness_perception": [
                "large rooms high ceilings", "generous floor plan", "airy atmosphere", "open living surface",
                "Grosszügige Zimmer", "hohe Decken", "offener Grundriss", "viel Platz",
                "groissi Zimmer", "luftig", "grosszuegigs Wohnzimmer"
            ],
            "neighborhood_safety_score": [
                "safe family friendly area", "kids playing outside", "low crime neighborhood", "schools and kindergartens nearby",
                "Sicheres familienfreundliches Wohnquartier", "verkehrsberuhigte Zone", "Schule in der Nähe",
                "guet für Chind", "sicher zum Spiele", "familiefründlich"
            ],
            "proximity_to_desired_location_score": [
                "close to specific destination", "near university", "walking distance to workplace", "15 minutes from city center",
                "In der Nähe vom Zielort", "kurzer Arbeitsweg", "Uni Nähe",
                "nöch bi de Arbet", "schnäll bi de Uni", "öppe 10 minete vo de Stadt"
            ]
        }
        # Pre-compute anchor embeddings
        self.anchor_embeddings = {
            attr: self._model.encode(phrases, convert_to_tensor=True) 
            for attr, phrases in self.anchors.items()
        }

        self.importance_keywords = {
            1.0: ["must", "essential", "needs", "required", "mandatory", "unbedingt", "muss", "notwendig", "zwingend"],
            0.7: ["prefer", "looking for", "searching", "like to have", "bevorzugt", "wäre toll", "suche", "hät gärn"],
            0.4: ["ideally", "maybe", "possible", "nice to have", "wenn möglich", "vielleicht", "wär schön"]
        }

    # Prompt (not used)
    def _get_system_prompt(self) -> str:
        return """
        You are an expert real estate analyzer. Given a user query, extract the importance of various 'soft' attributes.
        Assign a weight between 0.0 and 1.0 for each attribute.
        - 1.0: Essential / Must have / Explicitly mentioned as a priority.
        - 0.7: Strong preference / Looking for...
        - 0.4: Mentioned but not critical / "Nice to have".
        - 0.0: Not mentioned.

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
        """
        Extracts weights using combined semantic similarity and intensity keyword analysis.
        Initial semantic match is refined by importance keywords (must, prefer, ideally).
        """
        query_embedding = self._model.encode(query, convert_to_tensor=True)
        q_lower = query.lower()
        weights_dict = {}

        # Identify global importance context
        global_intensity = 0.5 # Default to moderate interest
        for val, keywords in self.importance_keywords.items():
            if any(k in q_lower for k in keywords):
                global_intensity = max(global_intensity, val)

        for attr, embeddings in self.anchor_embeddings.items():
            # 1. Compute semantic similarity to anchors
            cos_sims = util.cos_sim(query_embedding, embeddings)
            max_sim = float(torch.max(cos_sims))
            
            # 2. Thresholding: only proceed if there is a semi-strong semantic match (> 0.45)
            if max_sim > 0.45:
                # Semantic strength scaled to [0, 1]
                semantic_strength = (max_sim - 0.45) / 0.45 
                
                # Hybrid weighting: combine semantic match with explicitly detected keyword intensity
                # We give slightly more weight (60%) to the keyword intensity if detected
                final_weight = (semantic_strength * 0.4) + (global_intensity * 0.6)
                
                # Minimum threshold for final weight to avoid noise
                if final_weight > 0.3:
                    weights_dict[attr] = round(min(1.0, final_weight), 2)

        self.weights = SoftFactWeights(**weights_dict)
        return self.weights

    def run(self, query: str) -> list[dict[str, Any]]:
        self.get_weights(query)
        return []

# ???
class Property():
    def __init__(self, id: str, info: dict):
        self.id = id
        self.info = info

class DumbSoftExtractor(SoftFactExtractor):
    def run(self, _query: str) -> dict[str, Any]:
        return {}
