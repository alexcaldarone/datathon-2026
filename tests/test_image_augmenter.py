import io
import json
import os

from omegaconf import OmegaConf

from app.ingestion.components.augmenters import (
    AugmentedFeature,
    FeatureType,
    ImageEmbeddingAugmenter,
    build_augmenters,
)


_FAKE_EMBEDDING = [0.1] * 1024


class FakeBedrockClient:
    def invoke_model(self, **kwargs) -> dict:
        return {"body": io.BytesIO(json.dumps({"embedding": _FAKE_EMBEDDING}).encode())}


class FakeResponse:
    status_code = 200
    content = b"\x89PNG\r\n\x1a\n fake image bytes"

    def raise_for_status(self) -> None:
        pass


def _base_cfg() -> dict:
    return {
        "embed_dim": 1024,
        "embed_workers": 2,
        "model_id": "amazon.titan-embed-text-v2:0",
        "image_model_id": "amazon.titan-embed-image-v1",
        "enable_image_embeddings": True,
        "index_name": "listings",
        "pipeline_name": "hybrid-rrf-pipeline",
        "rrf_rank_constant": 60,
        "default_batch": 200,
    }


def test_image_augmenter_returns_augmented_feature(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-2")
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.boto3.client",
        lambda *a, **kw: FakeBedrockClient(),
    )
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.requests.get",
        lambda *a, **kw: FakeResponse(),
    )

    cfg = OmegaConf.create(_base_cfg())
    augmenter = ImageEmbeddingAugmenter(cfg)

    listing = {
        "full_text": "Bright apartment in Zurich",
        "hero_image_url": "https://example.com/image.jpg",
    }
    result = augmenter.augment(listing)

    assert isinstance(result, AugmentedFeature)
    assert result.name == "image_embedding"
    assert result.type == FeatureType.IMAGE
    assert isinstance(result.content, list)
    assert len(result.content) == 1024


def test_image_augmenter_falls_back_to_text_when_no_image(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-2")
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.boto3.client",
        lambda *a, **kw: FakeBedrockClient(),
    )

    cfg = OmegaConf.create(_base_cfg())
    augmenter = ImageEmbeddingAugmenter(cfg)

    listing = {
        "full_text": "Modern studio in Geneva",
        "hero_image_url": None,
        "images_json": None,
    }
    result = augmenter.augment(listing)

    assert isinstance(result, AugmentedFeature)
    assert result.name == "image_embedding"
    assert len(result.content) == 1024


def test_image_augmenter_field_mapping_is_knn_vector(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-2")
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.boto3.client",
        lambda *a, **kw: FakeBedrockClient(),
    )

    cfg = OmegaConf.create(_base_cfg())
    augmenter = ImageEmbeddingAugmenter(cfg)
    mapping = augmenter.field_mapping

    assert mapping["type"] == "knn_vector"
    assert mapping["dimension"] == 1024


def test_build_augmenters_includes_image_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-2")
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.boto3.client",
        lambda *a, **kw: FakeBedrockClient(),
    )

    cfg = OmegaConf.create(_base_cfg())
    augmenters = build_augmenters(cfg)

    assert len(augmenters) == 3
    assert isinstance(augmenters[2], ImageEmbeddingAugmenter)


def test_build_augmenters_excludes_image_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-2")
    monkeypatch.setattr(
        "app.ingestion.components.augmenters.boto3.client",
        lambda *a, **kw: FakeBedrockClient(),
    )

    raw = _base_cfg()
    raw["enable_image_embeddings"] = False
    cfg = OmegaConf.create(raw)
    augmenters = build_augmenters(cfg)

    assert len(augmenters) == 2
    assert not any(isinstance(a, ImageEmbeddingAugmenter) for a in augmenters)
