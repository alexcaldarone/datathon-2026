from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_LOGS_DIR = Path(__file__).parent.parent.parent.parent / "logs"


class IngestionLogger:
    _instance: "IngestionLogger | None" = None
    _logger: logging.Logger

    @classmethod
    def get(cls) -> "IngestionLogger":
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("ingestion")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        if self._logger.handlers:
            return

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)

        file_handler = logging.handlers.RotatingFileHandler(
            _LOGS_DIR / "ingestion.log", maxBytes=5_000_000, backupCount=3
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

    @contextmanager
    def augmenter(self, name: str):
        start = time.perf_counter()
        self._logger.info("AUGMENTER START  [%s]", name)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._logger.info("AUGMENTER END    [%s]  elapsed=%.3fs", name, elapsed)

    def batch(self, batch_num: int, total_batches: int, offset: int, size: int) -> None:
        self._logger.info(
            "BATCH  [%d/%d]  offset=%d  size=%d", batch_num, total_batches, offset, size
        )

    def info(self, msg: str, *args) -> None:
        self._logger.info(msg, *args)

    def warning(self, msg: str, *args) -> None:
        self._logger.warning(msg, *args)

    def error(self, msg: str, *args) -> None:
        self._logger.error(msg, *args)

    def summary(self, indexed: int, skipped: int, failed: int, elapsed: float) -> None:
        self._logger.info(
            "SUMMARY  indexed=%d  skipped=%d  failed=%d  total=%d  elapsed=%.1fs",
            indexed, skipped, failed, indexed + skipped + failed, elapsed,
        )
