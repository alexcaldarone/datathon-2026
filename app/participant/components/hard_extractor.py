from __future__ import annotations

import os
from abc import abstractmethod, ABC

from pydantic_ai import Agent

from app.models.schemas import HardFilters

from omegaconf import DictConfig

from app.models.schemas import HardFilters
from app.participant.components.utils import _instantiate

_MODULE = "app.participant.components.hard_extractor"


def build_hard_extractor(cfg: DictConfig) -> HardFactExtractor:
    return _instantiate(cfg.hard_extractor, _MODULE)


class HardFactExtractor(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def run(self, query: str) -> HardFilters:
        pass


HARD_EXTRACTION_SYSTEM_PROMPT = """\
You are a Swiss real-estate search assistant. Extract ONLY the hard, \
non-negotiable constraints from the user's query into structured filters.

Rules:
- Only extract constraints the user explicitly states as requirements.
- "max X CHF" / "under X" / "budget X" → max_price
- "at least N rooms" / "N-room" / "N.5 rooms" → min_rooms (and optionally max_rooms if exact)
- City names → city list (lowercase). If user says "Zürich" use "zurich".
- "in canton ZH" / "in Zurich area" → canton (use two-letter code: ZH, BE, LU, etc.)
- Feature requests like "with balcony", "must have parking" → features list
- "for rent" / "to buy" / "purchase" → offer_type (RENT or SALE)
- Vague preferences ("bright", "modern", "nice area", "good location") are NOT hard filters — omit them.
- "not too expensive" is soft — do NOT set max_price.
- "close to public transport" or any other measure of closeness is soft — do NOT set any filter for it.
- If unsure whether something is hard or soft, omit it (err on fewer filters).
- Default offer_type to "RENT" if the user does not explicitly mention buying or purchasing.

Valid feature keys (use ONLY these exact strings):
balcony, elevator, parking, garage, fireplace, child_friendly, pets_allowed, \
temporary, new_build, wheelchair_accessible, private_laundry, minergie_certified

Return only a valid HardFilters object.
If a value is not explicitly specified in the query, use null.
Do not use placeholders like "UNKNOWN", "N/A", or empty strings.
For list fields, use null unless the query explicitly specifies values.
Only set sort_by if the user explicitly requests sorting.

Swiss canton codes:
ZH, BE, LU, UR, SZ, OW, NW, GL, ZG, FR, SO, BS, BL, SH, AR, AI, SG, GR, \
AG, TG, TI, VD, VS, NE, GE, JU

Here is an example of how hard facts would be extracted from a user query:

    Example 1:
        Query: 3-room bright apartment in Zurich under 2800 CHF with balcony, close to public transport
        Hard requirements: 3-room, Zurich, under 2800CHF, balcony
        Explanation: This is a detailed query. Clearly we should respect the fundamental characteristics which the house should have (i.e. 3-room, Zurich, price), as
        otherwise it would not be aligned with what the user is looking for. Closeness to public transport should influence the ranking in a second stage, as this is not information
        we easily have accessible from the database.
    
    Example 2:
        Query: Bright family-friendly flat in Winterthur, not too expensive, ideally with parking
        Hard requirements: Winterthur
        Explanation: This query is quite vague and most of the preferences the user expresses are qualitative rather than quantitative or easy to measure.
        Therefore, in a first step the only hard requirement we can satisfy is the location in Winterthur. All other preferences will be dealt with other data sources later.
    
    Example 3:
        Query: Modern studio in Geneva for June move-in, quiet area, nice views if possible
        Hard requirements: Studio, Geneva, June move-in
        Explanation: This query has some clear hard requirements: studio (as it informs the number of rooms), Geneva (location) and the move-in date.
        The nice views are a soft requirement which require different data sources to answer.

Other important rules to respect:
- If a prompt is in a language different from english, translate it to english and use that to extract the relevant information.
"""

class LLMHardFactExtractor(HardFactExtractor):
    def __init__(self, model_id: str | None = None):
        super().__init__()
        bedrock_model = model_id or os.getenv(
            "BEDROCK_MODEL_ID",
            "anthropic.claude-3-haiku-20240307-v1:0",
        )
        self._agent = Agent(
            f"bedrock:{bedrock_model}",
            system_prompt=HARD_EXTRACTION_SYSTEM_PROMPT,
            output_type=HardFilters,
        )

    async def run(self, query: str) -> HardFilters:
        result = await self._agent.run(query)
        return result.output

class DumbHardExtractor(HardFactExtractor):
    def run(self, _query: str) -> HardFilters:
        return HardFilters()
