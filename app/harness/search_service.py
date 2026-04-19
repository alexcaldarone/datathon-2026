from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.hard_filters import HardFilterParams, search_listings
from app.models.schemas import HardFilters, ListingsResponse
from app.participant.components import Config, PipelineLogger
from app.participant.components.utils import read_system_prompt
from app.participant.hard_fact_extraction import extract_hard_facts
from app.participant.query_validation import validate_query
from app.participant.ranking import rank_listings
from app.participant.soft_fact_extraction import extract_soft_facts
from app.participant.soft_filtering import filter_soft_facts


def _resolve_target(target: int | float, limit: int) -> int:
    if isinstance(target, float):
        return max(1, round(target * limit))
    return int(target)


def filter_hard_facts(db_path: Path, hard_facts: HardFilters) -> list[dict[str, Any]]:
    return search_listings(db_path, to_hard_filter_params(hard_facts))


def _check_empty_candidates(
    candidates: list[dict[str, Any]], logger: PipelineLogger
) -> ListingsResponse | None:
    if candidates:
        return None
    logger.log_pipeline_end(0)
    return ListingsResponse(
        listings=[],
        meta={
            "status": "no_results",
            "message": read_system_prompt("no_candidates"),
        },
    )


def query_from_text(
    *,
    db_path: Path,
    query: str,
    limit: int,
    offset: int,
) -> ListingsResponse:
    logger = PipelineLogger.get()
    logger.log_query_start(query, limit, offset)

    with logger.stage("validate_query"):
        validation = validate_query(query)
        logger.log_validation(validation)

    if not validation.is_valid:
        logger.log_pipeline_end(0)
        return ListingsResponse(
            listings=[],
            meta={
                "status": "clarification_needed",
                "reason": validation.reason,
                "questions": validation.questions,
            },
        )

    cfg = Config.get_cfg()
    hard_filter_limit = _resolve_target(cfg.hard_filter.target_candidates, limit)
    soft_filter_target = _resolve_target(cfg.soft_filter.target_candidates, limit)
    reranker_target = _resolve_target(cfg.reranker.target_candidates, limit)
    logger.log_pipeline_config(cfg, hard_filter_limit, soft_filter_target, reranker_target)

    with logger.stage("extract_hard_facts"):
        hard_facts = extract_hard_facts(query)
        hard_facts.limit = hard_filter_limit
        hard_facts.offset = offset
        logger.log_hard_facts(hard_facts)

    with logger.stage("filter_hard_facts"):
        candidates = filter_hard_facts(db_path, hard_facts)
        logger.log_candidates("hard_filter", len(candidates))

    if (early_stop := _check_empty_candidates(candidates, logger)) is not None:
        return early_stop

    with logger.stage("extract_soft_facts"):
        soft_facts = extract_soft_facts(query)
        logger.log_soft_facts(soft_facts)

    if len(candidates) > soft_filter_target:
        with logger.stage("filter_soft_facts"):
            candidates = filter_soft_facts(candidates, soft_facts, soft_filter_target)
            logger.log_candidates("soft_filter", len(candidates))

        if (early_stop := _check_empty_candidates(candidates, logger)) is not None:
            return early_stop

    with logger.stage("rank_listings"):
        ranked = rank_listings(candidates, soft_facts, reranker_target)
        logger.log_ranked_results(ranked)

    logger.log_pipeline_end(len(ranked))
    return ListingsResponse(listings=ranked, meta={})


def query_from_filters(
    *,
    db_path: Path,
    hard_facts: HardFilters | None,
) -> ListingsResponse:
    logger = PipelineLogger.get()
    structured_hard_facts = hard_facts or HardFilters()
    limit = structured_hard_facts.limit
    logger.log_query_start("<filter_mode>", limit, structured_hard_facts.offset)
    logger.log_hard_facts(structured_hard_facts)

    cfg = Config.get_cfg()
    soft_filter_target = _resolve_target(cfg.soft_filter.target_candidates, limit)
    reranker_target = _resolve_target(cfg.reranker.target_candidates, limit)
    logger.log_pipeline_config(cfg, limit, soft_filter_target, reranker_target)

    with logger.stage("filter_hard_facts"):
        candidates = filter_hard_facts(db_path, structured_hard_facts)
        logger.log_candidates("hard_filter", len(candidates))

    if (early_stop := _check_empty_candidates(candidates, logger)) is not None:
        return early_stop

    with logger.stage("extract_soft_facts"):
        soft_facts = extract_soft_facts("")
        logger.log_soft_facts(soft_facts)

    if len(candidates) > soft_filter_target:
        with logger.stage("filter_soft_facts"):
            candidates = filter_soft_facts(candidates, soft_facts, soft_filter_target)
            logger.log_candidates("soft_filter", len(candidates))

        if (early_stop := _check_empty_candidates(candidates, logger)) is not None:
            return early_stop

    with logger.stage("rank_listings"):
        ranked = rank_listings(candidates, soft_facts, reranker_target)
        logger.log_ranked_results(ranked)

    logger.log_pipeline_end(len(ranked))
    return ListingsResponse(listings=ranked, meta={})


def to_hard_filter_params(hard_facts: HardFilters) -> HardFilterParams:
    return HardFilterParams(
        city=hard_facts.city,
        postal_code=hard_facts.postal_code,
        canton=hard_facts.canton,
        min_price=hard_facts.min_price,
        max_price=hard_facts.max_price,
        min_rooms=hard_facts.min_rooms,
        max_rooms=hard_facts.max_rooms,
        latitude=hard_facts.latitude,
        longitude=hard_facts.longitude,
        radius_km=hard_facts.radius_km,
        features=hard_facts.features,
        offer_type=hard_facts.offer_type,
        object_category=hard_facts.object_category,
        limit=hard_facts.limit,
        offset=hard_facts.offset,
        sort_by=hard_facts.sort_by,
    )
