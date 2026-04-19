from omegaconf import OmegaConf

from app.models.schemas import RankedListingResult
from app.participant.components.reranker import LLMReRanker

_CANDIDATE_A = {
    "listing_id": "1",
    "title": "Sunny 3-room flat in Zurich",
    "description": "Bright flat near the lake",
    "city": "Zurich",
    "canton": "ZH",
    "price": 2400,
    "rooms": 3,
    "area": 75,
    "features": ["balcony"],
    "offer_type": "RENT",
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

_CANDIDATE_B = {
    **_CANDIDATE_A,
    "listing_id": "2",
    "title": "Dark basement studio in Bern",
    "description": "Small studio in the basement",
    "city": "Bern",
    "canton": "BE",
    "price": 900,
    "rooms": 1,
    "area": 25,
    "features": [],
}

_CFG = OmegaConf.create({
    "class_name": "LLMReRanker",
    "model_id": "us.anthropic.claude-3-haiku-20240307-v1:0",
})


def test_llm_reranker_empty_candidates() -> None:
    reranker = LLMReRanker(_CFG)
    assert reranker.run([], {"query": "flat in Zurich"}, target=5) == []


def test_llm_reranker_returns_ranked_results() -> None:
    reranker = LLMReRanker(_CFG)
    results = reranker.run([_CANDIDATE_A, _CANDIDATE_B], {"query": "3-room flat in Zurich"}, target=2)

    assert len(results) == 2
    assert all(isinstance(r, RankedListingResult) for r in results)
    assert {r.listing_id for r in results} == {"1", "2"}
    # listing_id "1" (Zurich, 3-room) should score higher than "2" (Bern, studio)
    scores = {r.listing_id: r.score for r in results}
    assert scores["1"] > scores["2"]
    assert all(0.0 <= r.score <= 1.0 for r in results)
    assert all(r.reason for r in results)
