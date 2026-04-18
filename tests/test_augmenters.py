import pytest
from app.ingestion.components.augmenters import AnchorsAugmenter, AugmentedFeature, FeatureType
from app.participant.components.utils import Config

NUM_ATTRIBUTES = 16

# Attribute Indices
DAYLIGHT       = 0
ORIENTATION    = 1
NOISE          = 2
SOUND_INSUL    = 3
VIEW           = 4
GREEN          = 5
WALKABILITY    = 6
PUBLIC_TRANSP  = 7
CENTRALITY     = 8
SAFETY         = 9
PROXIMITY      = 10
AIR_QUALITY    = 11
SPACIOUSNESS   = 12
MODERNITY      = 13
LIVABILITY     = 14
WFH            = 15

ATTR_NAMES = [
    "DAYLIGHT", "ORIENTATION", "NOISE", "SOUND_INSUL", "VIEW", "GREEN", 
    "WALKABILITY", "PUBLIC_TRANSP", "CENTRALITY", "SAFETY", "PROXIMITY", 
    "AIR_QUALITY", "SPACIOUSNESS", "MODERNITY", "LIVABILITY", "WFH"
]

@pytest.fixture(scope="module")
def augmenter() -> AnchorsAugmenter:
    return AnchorsAugmenter(Config.get_cfg())

def _vec(augmenter: AnchorsAugmenter, query: str) -> list[float]:
    return augmenter.augment({"full_text": query}).content

def _print_results(query: str, vector: list[float]):
    print(f"\nQuery: {query}")
    print("-" * 40)
    for i, val in enumerate(vector):
        if val > 0:
            print(f"{ATTR_NAMES[i]:<15}: {val:.2f}")
    print("-" * 40)

# ── Diagnostic Tests ──────────────────────────────────────────────────────────

def test_diagnostic_complex_queries(augmenter: AnchorsAugmenter) -> None:
    queries = [
        "Zürich Seefeld, 2-Zimmer, modern, zentral, nähe See, bis 2500 CHF",
        "Wir sind eine Familie mit Kind, brauchen viel Tageslicht, ruhig und grün, gute Schulen in der Nähe",
        "Rentnerpaar in Bern, gepflegte Wohnung, Lift, Altstadt fussläufig, Seesicht wäre schön",
        "ETH-Student sucht WG-Zimmer, muss direkt bei der Uni sein, schnelles Internet wichtig",
        "Dunkle gemuetliche Wohnung bevorzugt, ich brauche kein helles Apartment"
    ]
    
    for q in queries:
        vector = _vec(augmenter, q)
        _print_results(q, vector)
        assert all(0.0 <= v <= 1.0 for v in vector)

def test_diagnostic_negation(augmenter: AnchorsAugmenter) -> None:
    # Testing how the system handles explicit negations visually
    q = "Ich hasse Moderne und Glas, lieber eine alte, charmante Wohnung mit Staub."
    vector = _vec(augmenter, q)
    _print_results(q, vector)
    
    # In the current whole-sentence 0-baseline logic, 'modern' will still likely trigger
    # let's see what the print shows.
    assert vector[MODERNITY] >= 0

def test_diagnostic_intensities(augmenter: AnchorsAugmenter) -> None:
    q1 = "Ruhige Lage"
    q2 = "EXTREM ruhige Lage, absolute Stille ist ein MUSS"
    
    v1 = _vec(augmenter, q1)
    v2 = _vec(augmenter, q2)
    
    print(f"\nComparing Intensity:")
    print(f"Normal : {v1[NOISE]:.2f}")
    print(f"Strong : {v2[NOISE]:.2f}")
    
    _print_results(q1, v1)
    _print_results(q2, v2)

def test_diagnostic_swiss_german(augmenter: AnchorsAugmenter) -> None:
    q = "Helli Wohnig mit viel Sunne isch mir sehr wichtig, aber es mues noed zentral si"
    vector = _vec(augmenter, q)
    _print_results(q, vector)

if __name__ == "__main__":
    # Create a local instance of the augmenter for direct script execution
    cfg = Config.get_cfg()
    aug = AnchorsAugmenter(cfg)
    
    # Run the diagnostic logic
    test_queries = [
        "Zürich Seefeld Single: Seenähe, gepflegtes Quartier, kein Lärm, modern",
        "Wir sind eine Familie mit Kind, brauchen viel Tageslicht, ruhig und grün, gute Schulen in der Nähe",
        "Rentnerpaar in Bern, gepflegte Wohnung, Lift, Altstadt fussläufig, Seesicht wäre schön",
        "ETH-Student sucht WG-Zimmer, muss direkt bei der Uni sein, schnelles Internet wichtig",
        "Dunkle gemuetliche Wohnung bevorzugt, ich brauche kein helles Apartment"
    ]
    
    for q in test_queries:
        v = aug.augment({"full_text": q}).content
        _print_results(q, v)
