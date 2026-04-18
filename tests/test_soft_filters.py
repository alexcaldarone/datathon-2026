from app.participant.soft_fact_extraction import extract_soft_facts
from app.participant.soft_filtering import filter_soft_facts


def test_extract_soft_facts_returns_stub_structure() -> None:
    result = extract_soft_facts("bright flat near transport")

    assert isinstance(result, dict)


def test_filter_soft_facts_returns_candidate_subset() -> None:
    candidates = [{"listing_id": "1"}, {"listing_id": "2"}]

    filtered = filter_soft_facts(candidates, {"raw_query": "quiet"})

    assert isinstance(filtered, list)
    assert {item["listing_id"] for item in filtered} <= {"1", "2"}
