from __future__ import annotations

from app.models.schemas import ValidationResult
from app.participant.components import Config, build_query_validator


def validate_query(query: str) -> ValidationResult:
    cfg = Config.get_cfg()
    validator = build_query_validator(cfg)
    return validator.run(query)
