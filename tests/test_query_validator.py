from omegaconf import OmegaConf

from app.models.schemas import ValidationResult
from app.participant.components.query_validator import DumbQueryValidator, LLMQueryValidator

_VAGUE_QUERY = "I want something nice"
_PRECISE_QUERY = "3-room flat in Zurich under 2500 CHF per month, available from June 2026"

_CFG = OmegaConf.create({
    "class_name": "LLMQueryValidator",
    "model_id": "us.anthropic.claude-3-haiku-20240307-v1:0",
})


def test_dumb_validator_always_valid() -> None:
    cfg = OmegaConf.create({"class_name": "DumbQueryValidator"})
    validator = DumbQueryValidator(cfg)
    result = validator.run(_VAGUE_QUERY)

    assert isinstance(result, ValidationResult)
    assert result.is_valid is True


def test_llm_validator_flags_vague_query() -> None:
    validator = LLMQueryValidator(_CFG)
    result = validator.run(_VAGUE_QUERY)

    assert isinstance(result, ValidationResult)
    assert result.is_valid is False


def test_llm_validator_accepts_precise_query() -> None:
    validator = LLMQueryValidator(_CFG)
    result = validator.run(_PRECISE_QUERY)

    assert isinstance(result, ValidationResult)
    assert result.is_valid is True
