from app.participant.components.soft_extractor import LLMSoftExtractor
from app.participant.components.utils import Config

extractor = LLMSoftExtractor(Config.get_cfg())

queries = [
    "Ich suche eine Wohnung in Zürich",
    "Ich will eine laute Wohnung nahe am Nachtleben",
    "Ich brauche eine ruhige Wohnung, schnelles Internet ist ein Muss für Home Office",
    "Ich vermeide moderne Wohnungen",
]

for q in queries:
    weights = extractor.get_weights(q)
    print(f"\n{q}")
    print(extractor.weights_to_vector(weights))
