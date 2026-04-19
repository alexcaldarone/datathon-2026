from app.models.schemas import SoftPreference
from app.participant.soft_fact_extraction import (
    _ANCHOR_TO_BOOST,
    _detect_boost_fields,
    _preferences_to_boost_fields,
)
from app.participant.components.translator import is_english


def test_detect_boost_fields_matches_bright_query() -> None:
    result = _detect_boost_fields("ich suche eine helle Wohnung mit viel Tageslicht")
    fields = [f for f, _ in result]
    assert "vlm_features.brightness" in fields


def test_detect_boost_fields_empty_for_generic_query() -> None:
    result = _detect_boost_fields("3 room apartment in zurich")
    assert result == []


def test_detect_boost_fields_returns_tuples_with_weight() -> None:
    result = _detect_boost_fields("ruhige Wohnung muss unbedingt sein")
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
    # "unbedingt" maps to weight 1.0
    weights = [w for _, w in result]
    assert any(w == 1.0 for w in weights)


def test_preferences_to_boost_fields_maps_dimensions() -> None:
    prefs = [
        SoftPreference(dimension="daylight_score", weight=0.7),
        SoftPreference(dimension="public_transport_score", weight=1.0),
    ]
    result = _preferences_to_boost_fields(prefs)
    assert ("vlm_features.brightness", 0.7) in result
    assert ("geo_features.transit_score", 1.0) in result


def test_preferences_to_boost_fields_skips_unknown_dimensions() -> None:
    prefs = [SoftPreference(dimension="nonexistent_dimension", weight=0.5)]
    result = _preferences_to_boost_fields(prefs)
    assert result == []


def test_is_english_detects_german() -> None:
    assert not is_english("helle Wohnung mit Küche in der Nähe von Zürich Schlafzimmer")


def test_is_english_detects_english() -> None:
    assert is_english("bright apartment with balcony near the train station")
