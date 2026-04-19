import os
import requests
import yaml
import spacy
from typing import Any
from omegaconf import DictConfig

class POIFeature:
    """
    Component to calculate features related to Points of Interest (POIs) 
    using OpenRouteService for routing and travel times.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # OpenRouteService API settings
        self.api_key_openrouteservice = cfg.get("api_key_openrouteservice")
        self.profile = cfg.get("profile")
        self.url = f"https://api.openrouteservice.org/v2/directions/{self.profile}"
        self.nlp = spacy.load("en_core_web_sm")

    def get_travel_time(self, start_coords: list[float], end_coords: list[float]) -> float:
        """
        Calculates travel duration in minutes between two coordinates.
        Coordinates are [longitude, latitude].
        """
        if not self.api_key:
            print("Warning: No API_KEY found for POIFeature")
            return -1.0
            
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }
        body = {
            "coordinates": [start_coords, end_coords]
        }
        
        try:
            response = requests.post(self.url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
            # Duration is in seconds, convert to minutes
            duration = data["routes"][0]["summary"]["duration"]
            return duration / 60
        except Exception as e:
            print(f"Error fetching routing from OpenRouteService: {e}")
            return -1.0
    
    def get_poi_chords(self, poi: str) -> list[list[float]]:
        doc = self.nlp(poi)
        address = " ".join([ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC", "CARDINAL"]])


        from_chords = []
        to_chords = []
        return None

    
    def run(self, pois: list[str], address_coords: list[float]) -> dict[str, Any]:
        """
        Placeholder for main feature extraction logic.
        """
        result = {
            "filled": 1.0 if pois else -1.0,
            "walk": None,
            "bike": None,
            "train": None,
            "car": None
        }
        if not pois:
            return result
        
        result["filled"] = 1.0
        return result

if __name__ == "__main__":
    # Robust way to load config relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..", "..")
    config_path = os.path.normpath(os.path.join(project_root, "configs", "config.yaml"))
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
            # Wrap in dot-access friendly dict or just use straight dict for diagnostic
            diagnostic_cfg = raw_cfg if raw_cfg else {}
            
        poi = POIFeature(diagnostic_cfg)
        # Test Zuerich to Bern
        dur = poi.get_travel_time([8.5417, 47.3769], [7.4474, 46.9479])
        print(f"Travel time Zuerich -> Bern: {dur:.2f} minutes")
    else:
        print(f"Config file not found at {config_path}")
