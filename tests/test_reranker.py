import json
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from app.models.schemas import RankedListingResult
from app.participant.components.reranker import CohereReRanker, DumbReRanker

_CANDIDATE = {
    "listing_id": "42",
    "title": "Sunny flat in Zurich",
    "description": "Beautiful 3-room flat",
    "city": "Zurich",
    "canton": "ZH",
    "price": 2500,
    "rooms": 3,
    "area": 80,
    "features": ["balcony", "parking"],
    "offer_type": "rent",
    "object_category": "residential",
    "object_type": "flat",
    "available_from": None,
    "image_urls": None,
    "hero_image_url": None,
    "original_url": None,
    "street": None,
    "postal_code": None,
    "latitude": None,
    "longitude": None,
}


def test_dumb_reranker_returns_all_candidates() -> None:
    cfg = OmegaConf.create({"class_name": "DumbReRanker"})
    reranker = DumbReRanker(cfg)
    results = reranker.run([_CANDIDATE], {})

    assert len(results) == 1
    assert isinstance(results[0], RankedListingResult)
    assert results[0].score == 1.0
    assert results[0].listing_id == "42"


def test_cohere_reranker_orders_by_relevance_score() -> None:
    cfg = OmegaConf.create({
        "class_name": "CohereReRanker",
        "model_id": "cohere.rerank-v3-5:0",
        "top_n": 2,
        "region": "us-east-1",
    })

    candidate_a = {**_CANDIDATE, "listing_id": "1", "title": "Flat A"}
    candidate_b = {**_CANDIDATE, "listing_id": "2", "title": "Flat B"}

    fake_body = json.dumps({
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.4},
        ]
    }).encode()

    with patch("app.participant.components.reranker.boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.invoke_model.return_value = {"body": MagicMock(read=lambda: fake_body)}

        reranker = CohereReRanker(cfg)
        results = reranker.run([candidate_a, candidate_b], {"query": "flat in Zurich"})

    assert len(results) == 2
    assert results[0].listing_id == "2"
    assert results[0].score == 0.9
    assert results[1].listing_id == "1"
    assert results[1].score == 0.4


def test_cohere_reranker_empty_candidates() -> None:
    cfg = OmegaConf.create({
        "class_name": "CohereReRanker",
        "model_id": "cohere.rerank-v3-5:0",
        "top_n": 10,
        "region": "us-east-1",
    })

    with patch("app.participant.components.reranker.boto3.client"):
        reranker = CohereReRanker(cfg)
        results = reranker.run([], {"query": "flat"})

    assert results == []
