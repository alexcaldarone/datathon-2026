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
        self._step_times: dict[str, list[float]] = {}
        self._batch_times: list[float] = []

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
            self._step_times.setdefault(name, []).append(elapsed)
            self._logger.info("STAGE END    [%s]  elapsed=%.3fs", name, elapsed)

    @contextmanager
    def augmenter(self, name: str):
        start = time.perf_counter()
        self._logger.info("AUGMENTER START  [%s]", name)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._step_times.setdefault(name, []).append(elapsed)
            self._logger.info("AUGMENTER END    [%s]  elapsed=%.3fs", name, elapsed)

    def batch(self, batch_num: int, total_batches: int, offset: int, size: int) -> None:
        self._logger.info(
            "BATCH  [%d/%d]  offset=%d  size=%d", batch_num, total_batches, offset, size
        )

    def batch_done(self, batch_num: int, total_batches: int, elapsed: float) -> None:
        self._batch_times.append(elapsed)
        avg = sum(self._batch_times) / len(self._batch_times)
        remaining = total_batches - batch_num
        eta_s = avg * remaining
        eta_str = f"{int(eta_s // 3600)}h {int(eta_s % 3600 // 60)}m" if eta_s >= 3600 else f"{eta_s / 60:.1f}m" if eta_s >= 60 else f"{eta_s:.1f}s"
        self._logger.info(
            "BATCH DONE   [%d/%d]  elapsed=%.3fs  avg_per_batch=%.3fs  eta=%s",
            batch_num, total_batches, elapsed, avg, eta_str,
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
        if self._step_times:
            self._logger.info("STEP AVERAGES:")
            for step, times in self._step_times.items():
                avg = sum(times) / len(times)
                self._logger.info("  %-30s  avg=%.3fs  runs=%d", step, avg, len(times))
