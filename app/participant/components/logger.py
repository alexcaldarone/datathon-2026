from __future__ import annotations

import fcntl
import logging
import logging.handlers
import sys
import time
import yaml
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from app.models.schemas import HardFilters, ValidationResult

_INGESTION_CFG_PATH = Path(__file__).parents[3] / "configs" / "ingestion" / "config.yaml"

_LOGS_DIR = Path(__file__).parent.parent.parent.parent / "logs"


class _LockedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        if self.stream is None:
            self.stream = self._open()
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
        try:
            super().emit(record)
        finally:
            try:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


class PipelineLogger:
    _instance: "PipelineLogger | None" = None
    _logger: logging.Logger

    @classmethod
    def get(cls) -> "PipelineLogger":
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("pipeline")
        self._logger.setLevel(logging.DEBUG)
        # prevent double-printing via uvicorn root logger
        self._logger.propagate = False

        if self._logger.handlers:
            return

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)

        file_handler = _LockedRotatingFileHandler(
            _LOGS_DIR / "server.log", maxBytes=5_000_000, backupCount=3
        )
        file_handler.setFormatter(fmt)

        self._logger.addHandler(stdout_handler)
        self._logger.addHandler(file_handler)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        self._logger.info("STAGE START  [%s]", name)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._logger.info("STAGE END    [%s]  elapsed=%.3fs", name, elapsed)

    def log_query_start(self, query: str, limit: int, offset: int) -> None:
        self._logger.info("QUERY\n  query: %s\n  limit: %d  offset: %d", query, limit, offset)

    def log_validation(self, result: ValidationResult) -> None:
        self._logger.info(
            "VALIDATION  is_valid=%s  reason=%r  questions=%s",
            result.is_valid, result.reason, result.questions,
        )

    def log_hard_facts(self, hard_facts: HardFilters) -> None:
        fields = "\n  ".join(f"{k}: {v}" for k, v in hard_facts.model_dump().items())
        self._logger.info("HARD_FACTS\n  %s", fields)

    def log_soft_facts(self, soft_facts: dict[str, Any]) -> None:
        prefs = soft_facts.get("preferences", [])
        boosts = soft_facts.get("boost_fields", [])
        pref_lines = "\n  ".join(str(p) for p in prefs) or "(none)"
        boost_lines = "\n  ".join(str(b) for b in boosts) or "(none)"
        self._logger.info(
            "SOFT_FACTS\n  query: %s\n  preferences (%d):\n  %s\n  boost_fields (%d):\n  %s",
            soft_facts.get("query", ""),
            len(prefs), pref_lines,
            len(boosts), boost_lines,
        )

    def log_candidates(self, after: str, count: int) -> None:
        level = logging.WARNING if count == 0 else logging.INFO
        self._logger.log(level, "CANDIDATES  after=%s  count=%d", after, count)

    def log_ranked_results(self, ranked: list[Any]) -> None:
        lines = "\n  ".join(
            f"[{i+1}] id={r.listing_id}  score={r.score:.4f}  reason={r.reason!r}"
            for i, r in enumerate(ranked)
        )
        self._logger.info("RANKED_RESULTS (%d)\n  %s", len(ranked), lines or "(none)")

    def log_pipeline_config(
        self,
        cfg: DictConfig,
        hard_filter_limit: int,
        soft_filter_target: int,
        reranker_target: int,
    ) -> None:
        with open(_INGESTION_CFG_PATH) as f:
            ingestion_cfg = OmegaConf.create(yaml.safe_load(f))
        index_name = ingestion_cfg.get("index_name", "<unknown>")
        components = {
            "query_validator": cfg.query_validator.class_name,
            "hard_extractor": cfg.hard_extractor.class_name,
            "soft_extractor": cfg.soft_extractor.class_name,
            "soft_filter": cfg.soft_filter.class_name,
            "reranker": cfg.reranker.class_name,
        }
        comp_lines = "\n  ".join(f"{k}: {v}" for k, v in components.items())
        self._logger.info(
            "PIPELINE_CONFIG\n  index: %s\n  %s\n  targets: hard_filter=%d  soft_filter=%d  reranker=%d",
            index_name,
            comp_lines,
            hard_filter_limit,
            soft_filter_target,
            reranker_target,
        )

    def log_pipeline_end(self, listing_count: int) -> None:
        self._logger.info("DONE  listings_returned=%d", listing_count)
