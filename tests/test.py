import pytest
from app.ingestion.components.augmenters import AnchorsAugmenter, AugmentedFeature, FeatureType
from app.participant.components.utils import Config


def _vec(augmenter: AnchorsAugmenter, query: str) -> list[float]:
    return augmenter.augment({"full_text": query}).content