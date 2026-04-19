import copy
import math
import sqlite3
import time
from pathlib import Path

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
            _logger.batch(batch_num, total_batches, offset, len(rows))

            ids = [row["listing_id"] for row in rows]
            existing = self.client.fetch_existing(self._index, ids, augmenter_fields)
            new_rows = [
                row for row in rows
                if not _is_complete(existing.get(row["listing_id"], {}), augmenter_fields)
            ]
            skipped += len(rows) - len(new_rows)

            if new_rows:
                listings = [dict(row) | {"full_text": row_to_full_text(row)} for row in new_rows]

                with _logger.stage("image_download"):
                    download_images_batch(
                        listings,
                        max_workers=int(self.cfg.get("image_download_workers", 8)),
                        request_timeout_s=self.cfg.image_request_timeout_s,
                        target_width=self.cfg.get("image_width"),
                        target_height=self.cfg.get("image_height"),
                    )

                augmented: dict[str, list] = {}
                for aug in self.augmenters:
                    with _logger.augmenter(aug.field_name):
                        features = aug.augment_batch(listings)
                    augmented[aug.field_name] = [f.content for f in features]

                docs = self._build_docs(new_rows, listings, augmented)
                ok, err = self.client.bulk_upsert(docs, num_workers=self.cfg.upsert_workers)
                indexed += ok
                failed += err

            offset += len(rows)

        conn.close()
        _logger.summary(indexed, skipped, failed, time.perf_counter() - run_start)

    def _build_docs(
        self,
        rows: list[sqlite3.Row],
        listings: list[dict],
        augmented: dict[str, list],
    ) -> list[dict]:
        docs = []
        for i, (row, listing) in enumerate(zip(rows, listings)):
            doc = {
                "_index":          self._index,
                "_id":             row["listing_id"],
                "listing_id":      row["listing_id"],
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
            for field_name, contents in augmented.items():
                doc[field_name] = contents[i]
            docs.append(doc)
        return docs
