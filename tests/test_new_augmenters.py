import io
import json
import os

from omegaconf import OmegaConf

from app.ingestion.components.augmenters import (
    AugmentedFeature,
    FeatureType,
    GeoFeatureAugmenter,
    TranslationAugmenter,
    VLMFeatureAugmenter,
    build_augmenters,
)


def _base_cfg() -> dict:
    return {
        "embed_dim": 1024,
        "embed_workers": 2,
        "model_id": "amazon.titan-embed-text-v2:0",
        "image_model_id": "amazon.titan-embed-image-v1",
        "vlm_model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "translation_model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "overpass_url": "https://overpass.private.coffee/api/interpreter",
        "overpass_workers": 2,
        "enable_image_embeddings": True,
        "enable_vlm_features": True,
        "enable_translation": True,
        "enable_geo_features": True,
        "index_name": "listings",
        "pipeline_name": "hybrid-rrf-pipeline",
        "rrf_rank_constant": 60,
        "default_batch": 200,
    }


# ---------------------------------------------------------------------------
# VLM Augmenter
# ---------------------------------------------------------------------------

class FakeBedrockVLM:
    def invoke_model(self, **kwargs) -> dict:
        scores = {
            "brightness": 7, "spaciousness": 5, "modernity": 8,
            "view_quality": 3, "greenery": 6, "kitchen_quality": 9,
            "condition": 7, "noise_impression": 4,
        }
        body = json.dumps({"content": [{"text": json.dumps(scores)}]})
        return {"body": io.BytesIO(body.encode())}


class FakeImageResponse:
    status_code = 200
    content = b"\x89PNG\r\n\x1a\n fake image bytes"

    def raise_for_status(self) -> None:
        pass


def test_vlm_augmenter_returns_scores(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-2")
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.boto3.client",
        lambda *a, **kw: FakeBedrockVLM(),
    )
    monkeypatch.setattr(
        "app.ingestion.components.utils.requests.get",
        lambda *a, **kw: FakeImageResponse(),
    )

    cfg = OmegaConf.create(_base_cfg())
    augmenter = VLMFeatureAugmenter(cfg)

    listing = {"full_text": "Bright flat", "hero_image_url": "https://example.com/img.jpg"}
    result = augmenter.augment(listing)

    assert isinstance(result, AugmentedFeature)
    assert result.name == "vlm_features"
    assert result.type == FeatureType.SPARSE
    assert result.content["brightness"] == 7
    assert result.content["modernity"] == 8
    assert all(0 <= v <= 10 for v in result.content.values())


def test_vlm_augmenter_handles_no_image(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-2")
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.boto3.client",
        lambda *a, **kw: FakeBedrockVLM(),
    )

    cfg = OmegaConf.create(_base_cfg())
    augmenter = VLMFeatureAugmenter(cfg)

    listing = {"full_text": "No image flat", "hero_image_url": None, "images_json": None}
    result = augmenter.augment(listing)

    assert result.content == {}


def test_vlm_augmenter_field_mapping_has_float_properties(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-2")
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.boto3.client",
        lambda *a, **kw: FakeBedrockVLM(),
    )

    cfg = OmegaConf.create(_base_cfg())
    augmenter = VLMFeatureAugmenter(cfg)
    mapping = augmenter.field_mapping

    assert mapping["type"] == "object"
    assert mapping["properties"]["brightness"]["type"] == "float"
    assert mapping["properties"]["modernity"]["type"] == "float"


# ---------------------------------------------------------------------------
# Translation Augmenter
# ---------------------------------------------------------------------------

class FakeBedrockTranslation:
    def invoke_model(self, **kwargs) -> dict:
        body_input = json.loads(kwargs["body"])
        text = body_input["messages"][0]["content"]
        body = json.dumps({"content": [{"text": "Bright 3-room apartment in Zurich"}]})
        return {"body": io.BytesIO(body.encode())}


def test_translation_augmenter_passes_through_english(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-2")
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.boto3.client",
        lambda *a, **kw: FakeBedrockTranslation(),
    )

    cfg = OmegaConf.create(_base_cfg())
    augmenter = TranslationAugmenter(cfg)

    listing = {"full_text": "Bright apartment with balcony in central location"}
    result = augmenter.augment(listing)

    assert result.name == "full_text_en"
    # english text should pass through unchanged
    assert result.content == listing["full_text"]


def test_translation_augmenter_translates_german(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-2")
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.boto3.client",
        lambda *a, **kw: FakeBedrockTranslation(),
    )

    cfg = OmegaConf.create(_base_cfg())
    augmenter = TranslationAugmenter(cfg)

    listing = {"full_text": "Helle 3-Zimmer Wohnung in Zürich Nähe Bahnhof mit Küche"}
    result = augmenter.augment(listing)

    assert result.name == "full_text_en"
    # should be translated (not the original German)
    assert result.content != listing["full_text"]
    assert isinstance(result.content, str)


def test_translation_augmenter_field_mapping_is_text(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-2")
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.boto3.client",
        lambda *a, **kw: FakeBedrockTranslation(),
    )

    cfg = OmegaConf.create(_base_cfg())
    augmenter = TranslationAugmenter(cfg)
    mapping = augmenter.field_mapping

    assert mapping["type"] == "text"
    assert mapping["similarity"] == "custom_bm25"


# ---------------------------------------------------------------------------
# GeoFeature Augmenter
# ---------------------------------------------------------------------------

_FAKE_OVERPASS_RESPONSE = {
    "elements": [
        {"type": "node", "lat": 47.3770, "lon": 8.5400, "tags": {"public_transport": "stop_position"}},
        {"type": "node", "lat": 47.3775, "lon": 8.5410, "tags": {"public_transport": "stop_position"}},
        {"type": "node", "lat": 47.3780, "lon": 8.5420, "tags": {"railway": "station"}},
        {"type": "node", "lat": 47.3765, "lon": 8.5390, "tags": {"amenity": "school"}},
        {"type": "node", "lat": 47.3768, "lon": 8.5395, "tags": {"amenity": "kindergarten"}},
        {"type": "node", "lat": 47.3772, "lon": 8.5405, "tags": {"leisure": "playground"}},
        {"type": "node", "lat": 47.3760, "lon": 8.5385, "tags": {"shop": "supermarket"}},
        {"type": "node", "lat": 47.3763, "lon": 8.5388, "tags": {"amenity": "restaurant"}},
    ],
}


class FakeOverpassResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return _FAKE_OVERPASS_RESPONSE


def test_geo_augmenter_returns_transit_and_amenity_counts(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.requests.post",
        lambda *a, **kw: FakeOverpassResponse(),
    )

    cfg = OmegaConf.create(_base_cfg())
    augmenter = GeoFeatureAugmenter(cfg)

    listing = {"latitude": 47.3769, "longitude": 8.5399, "full_text": "test"}
    result = augmenter.augment(listing)

    assert result.name == "geo_features"
    c = result.content
    assert c["stops_500m"] == 2
    assert c["stations_1km"] == 1
    assert c["schools_1km"] == 1
    assert c["kindergartens_1km"] == 1
    assert c["playgrounds_500m"] == 1
    assert c["supermarkets_500m"] == 1
    assert c["restaurants_500m"] == 1
    assert isinstance(c["family_score"], float)
    assert isinstance(c["transit_score"], float)
    assert isinstance(c["walkability_score"], float)


def test_geo_augmenter_handles_missing_coordinates(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.requests.post",
        lambda *a, **kw: FakeOverpassResponse(),
    )

    cfg = OmegaConf.create(_base_cfg())
    augmenter = GeoFeatureAugmenter(cfg)

    listing = {"latitude": None, "longitude": None, "full_text": "test"}
    result = augmenter.augment(listing)

    assert result.content == {}


def test_geo_augmenter_computes_family_score() -> None:
    score = GeoFeatureAugmenter._family_score({
        "schools_1km": 2, "kindergartens_1km": 1, "playgrounds_500m": 2,
        "supermarkets_500m": 1, "parks_500m": 1,
        "pharmacies_500m": 0, "gyms_1km": 0, "restaurants_500m": 0, "doctors_1km": 0,
    })
    # (2*2 + 1*2 + 2*1.5 + 1*1.0 + 1*0.5) / 3.0 = (4+2+3+1+0.5)/3 = 10.5/3 = 3.5
    assert score == 3.5


# ---------------------------------------------------------------------------
# build_augmenters
# ---------------------------------------------------------------------------

class FakeBedrockClient:
    def invoke_model(self, **kwargs) -> dict:
        embedding = [0.1] * 1024
        return {"body": io.BytesIO(json.dumps({"embedding": embedding}).encode())}


def test_build_augmenters_all_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-2")
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.boto3.client",
        lambda *a, **kw: FakeBedrockClient(),
    )

    cfg = OmegaConf.create(_base_cfg())
    augmenters = build_augmenters(cfg)

    assert len(augmenters) == 6


def test_build_augmenters_none_extra_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-2")
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.boto3.client",
        lambda *a, **kw: FakeBedrockClient(),
    )

    raw = _base_cfg()
    raw["enable_image_embeddings"] = False
    raw["enable_vlm_features"] = False
    raw["enable_translation"] = False
    raw["enable_geo_features"] = False
    cfg = OmegaConf.create(raw)
    augmenters = build_augmenters(cfg)

    assert len(augmenters) == 2
