from omegaconf import OmegaConf
from app.participant.ranking import rank_listings
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

_CFG = OmegaConf.create({
    "class_name": "CohereReRanker",
    "model_id": "cohere.rerank-v3-5:0",
    "top_n": 2,
    "region": "us-east-1",
})


def test_dumb_reranker_returns_all_candidates() -> None:
    cfg = OmegaConf.create({"class_name": "DumbReRanker"})
    reranker = DumbReRanker(cfg)
    results = reranker.run([_CANDIDATE], {})

    assert len(results) == 1
    assert isinstance(results[0], RankedListingResult)
    assert results[0].score == 1.0
    assert results[0].listing_id == "42"


def test_cohere_reranker_orders_by_relevance_score() -> None:
    candidate_a = {**_CANDIDATE, "listing_id": "1", "title": "Flat A"}
    candidate_b = {**_CANDIDATE, "listing_id": "2", "title": "Flat B"}

    reranker = CohereReRanker(_CFG)
    results = reranker.run([candidate_a, candidate_b], {"query": "flat in Zurich"})

    assert len(results) == 2
    assert all(isinstance(r, RankedListingResult) for r in results)
    assert {r.listing_id for r in results} == {"1", "2"}
    # results must be ordered highest score first
    assert results[0].score >= results[1].score


def test_cohere_reranker_empty_candidates() -> None:
    reranker = CohereReRanker(_CFG)
    results = reranker.run([], {"query": "flat"})

    assert results == []

def test_rank_listings_returns_ranked_shape() -> None:
    ranked = rank_listings(
        candidates=[
            {
                "listing_id": "abc",
                "title": "Example",
                "city": "Zurich",
                "price": 2500,
                "rooms": 3.0,
                "latitude": 47.37,
                "longitude": 8.54,
                "street": "Main 1",
                "postal_code": "8000",
                "canton": "ZH",
                "area": 75.0,
                "available_from": "2026-06-01",
                "image_urls": ["https://example.com/1.jpg"],
                "hero_image_url": "https://example.com/1.jpg",
                "original_url": "https://example.com/listing",
                "features": ["balcony", "elevator"],
                "offer_type": "RENT",
                "object_category": "Wohnung",
                "object_type": "Apartment",
            }
        ],
        soft_facts={"query": "bright flat in Zurich"},
    )

    assert len(ranked) == 1
    assert ranked[0].listing_id == "abc"
    assert isinstance(ranked[0].score, float)
    assert isinstance(ranked[0].reason, str)
    assert ranked[0].listing.id == "abc"
    assert ranked[0].listing.title == "Example"
    assert ranked[0].listing.city == "Zurich"
    assert ranked[0].listing.image_urls
