import copy
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from app.ingestion.components.augmenters import Augmenter
from app.ingestion.components.client import OpenSearchClient
from app.ingestion.components.logger import IngestionLogger
from app.ingestion.components.utils import parse_features, row_to_full_text, parse_images_json, download_images_batch

_logger = IngestionLogger.get()

_QUERY = """
    SELECT listing_id, title, description, city, canton, postal_code,
           offer_type, object_category, object_type, price, rooms, area,
           latitude, longitude, available_from, features_json, images_json
    FROM listings LIMIT ? OFFSET ?
"""


def _is_complete(doc: dict, fields: list[str]) -> bool:
    return all(doc.get(f) not in (None, {}, [], "") for f in fields)


class IngestionManager:
    def __init__(self, cfg: DictConfig, augmenters: list[Augmenter], client: OpenSearchClient):
        self.cfg = cfg
        self.augmenters = augmenters
        self.client = client
        self._index = cfg.index_name
        self._pipeline = cfg.pipeline_name

    def build_index_body(self, base_body: dict) -> dict:
        # deep-copy so the loaded JSON is never mutated
        body = copy.deepcopy(base_body)
        for aug in self.augmenters:
            body["mappings"]["properties"][aug.field_name] = aug.field_mapping
        return body

    def run(
        self,
        db_path: Path,
        index_body: dict,
        pipeline_body: dict,
        limit: int | None,
        reset: bool,
    ) -> None:
        run_start = time.perf_counter()
        full_index_body = self.build_index_body(index_body)
        self.client.setup(self._index, self._pipeline, full_index_body, pipeline_body, reset=reset)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        total: int = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        if limit:
            total = min(total, limit)

        augmenter_fields = [aug.field_name for aug in self.augmenters]
        indexed = failed = skipped = offset = 0
        batch_size = int(self.cfg.default_batch)
        total_batches = math.ceil(total / batch_size)
        batch_num = 0

        while offset < total:
            n = min(batch_size, total - offset)
            rows = conn.execute(_QUERY, (n, offset)).fetchall()
            if not rows:
                break

            batch_num += 1
            batch_start = time.perf_counter()
            _logger.batch(batch_num, total_batches, offset, len(rows))

            ids = [row["listing_id"] for row in rows]
            existing = self.client.fetch_existing(self._index, ids, augmenter_fields)

            # rows missing at least one augmenter field
            rows_to_update = [
                row for row in rows
                if not _is_complete(existing.get(row["listing_id"], {}), augmenter_fields)
            ]
            skipped += len(rows) - len(rows_to_update)

            if rows_to_update:
                listings_by_id: dict[str, dict] = {
                    row["listing_id"]: dict(row) | {"full_text": row_to_full_text(row)}
                    for row in rows_to_update
                }

                # per-augmenter: which rows are missing this specific field
                aug_needed_rows: dict[str, list] = {
                    aug.field_name: [
                        row for row in rows_to_update
                        if not _is_complete(existing.get(row["listing_id"], {}), [aug.field_name])
                    ]
                    for aug in self.augmenters
                }

                # download images only for rows that need an image-dependent augmenter
                rows_needing_images: set[str] = {
                    row["listing_id"]
                    for aug in self.augmenters if aug.needs_images
                    for row in aug_needed_rows[aug.field_name]
                }
                if rows_needing_images:
                    with _logger.stage("image_download"):
                        download_images_batch(
                            [listings_by_id[lid] for lid in rows_needing_images],
                            max_workers=int(self.cfg.get("image_download_workers", 8)),
                            request_timeout_s=self.cfg.image_request_timeout_s,
                            target_width=self.cfg.get("image_width"),
                            target_height=self.cfg.get("image_height"),
                        )
                    _logger.record_step_stats("image_download", len(rows_needing_images), len(rows_to_update) - len(rows_needing_images))

                # run each augmenter only on rows missing its field
                computed: dict[str, dict[str, Any]] = {}  # field_name -> {listing_id -> value}
                for aug in self.augmenters:
                    needed = aug_needed_rows[aug.field_name]
                    if not needed:
                        _logger.record_step_stats(aug.field_name, 0, len(rows_to_update))
                        continue
                    listings_for_aug = [listings_by_id[row["listing_id"]] for row in needed]
                    with _logger.augmenter(aug.field_name):
                        features = aug.augment_batch(listings_for_aug)
                    computed[aug.field_name] = {
                        needed[i]["listing_id"]: features[i].content
                        for i in range(len(needed))
                    }
                    _logger.record_step_stats(aug.field_name, len(needed), len(rows_to_update) - len(needed))

                docs = self._build_docs(rows_to_update, listings_by_id, computed, existing)
                ok, err = self.client.bulk_upsert(docs, num_workers=self.cfg.upsert_workers)
                indexed += ok
                failed += err

            offset += len(rows)
            _logger.batch_done(batch_num, total_batches, time.perf_counter() - batch_start)

        conn.close()
        _logger.summary(indexed, skipped, failed, time.perf_counter() - run_start)

    def _build_docs(
        self,
        rows: list[sqlite3.Row],
        listings_by_id: dict[str, dict],
        computed: dict[str, dict[str, Any]],
        existing: dict[str, dict],
    ) -> list[dict]:
        docs = []
        for row in rows:
            lid = row["listing_id"]
            listing = listings_by_id[lid]
            doc = {
                "_index":          self._index,
                "_id":             lid,
                "listing_id":      lid,
                "full_text":       listing["full_text"],
                "title":           row["title"],
                "description":     row["description"],
                "city":            row["city"],
                "canton":          row["canton"],
                "postal_code":     row["postal_code"],
                "offer_type":      row["offer_type"],
                "object_category": row["object_category"],
                "object_type":     row["object_type"],
                "price":           row["price"],
                "rooms":           row["rooms"],
                "area":            row["area"],
                "latitude":        row["latitude"],
                "longitude":       row["longitude"],
                "available_from":  row["available_from"],
                "images_urls":     parse_images_json(row["images_json"]),
                "features":        parse_features(row["features_json"]),
            }
            for aug in self.augmenters:
                field = aug.field_name
                if lid in computed.get(field, {}):
                    doc[field] = computed[field][lid]
                elif field in existing.get(lid, {}):
                    # preserve existing value to avoid overwriting with null on full re-index
                    doc[field] = existing[lid][field]
            docs.append(doc)
        return docs
