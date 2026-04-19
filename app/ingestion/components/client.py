from __future__ import annotations

import os

from opensearchpy import OpenSearch, RequestsHttpConnection

from app.ingestion.components.logger import IngestionLogger

_logger = IngestionLogger.get()


class OpenSearchClient:
    """Singleton OpenSearch client shared across ingestion and retrieval."""

    _instance: OpenSearchClient | None = None

    def __new__(cls) -> OpenSearchClient:
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._client = instance._build()
            cls._instance = instance
        return cls._instance

    # --- retrieval ---

    def search(self, index: str, body: dict, pipeline: str | None = None) -> dict:
        params = {"search_pipeline": pipeline} if pipeline else {}
        return self._client.search(index=index, body=body, params=params)

    def fetch_existing(self, index: str, ids: list[str], fields: list[str]) -> dict[str, dict]:
        """Returns {id: source} for docs that exist and have the requested fields."""
        resp = self._client.mget(index=index, body={"ids": ids}, _source_includes=fields)
        return {
            doc["_id"]: doc["_source"]
            for doc in resp["docs"]
            if doc.get("found") and doc.get("_source")
        }

    # --- ingestion ---

    def setup(
        self,
        index: str,
        pipeline: str,
        index_body: dict,
        pipeline_body: dict,
        reset: bool,
    ) -> None:
        if reset and self._client.indices.exists(index=index):
            _logger.info("Deleting existing index '%s'...", index)
            self._client.indices.delete(index=index)
        if not self._client.indices.exists(index=index):
            _logger.info("Creating index '%s'...", index)
            self._client.indices.create(index=index, body=index_body)
        else:
            _logger.info("Index '%s' already exists — updating mapping.", index)
            # Merge augmenter field mappings into the existing index so that
            # fields added after initial creation (e.g. anchor_features) are
            # explicitly mapped. Without this, strict-dynamic indices silently
            # reject documents containing unmapped fields, making the
            # incremental skip logic never recognise those fields as stored.
            try:
                self._client.indices.put_mapping(
                    index=index, body=index_body["mappings"]
                )
            except Exception as exc:
                _logger.warning("Mapping update failed (non-fatal): %s", exc)
        _logger.info("Upserting search pipeline '%s'...", pipeline)
        self._client.transport.perform_request(
            "PUT", f"/_search/pipeline/{pipeline}", body=pipeline_body
        )
        _logger.info("Index and pipeline ready.")

    def bulk_upsert(self, docs: list[dict], num_workers: int = 4) -> tuple[int, int]:
        from opensearchpy import helpers
        from opensearchpy.helpers import BulkIndexError

        ok = err = 0
        try:
            for success, info in helpers.parallel_bulk(
                self._client,
                docs,
                num_workers,
                chunk_size=len(docs),
                raise_on_error=False
            ):
                if success:
                    ok += 1
                else:
                    err += 1
                    _logger.error("Index error: %s", info)
        except BulkIndexError as exc:
            err += len(exc.errors)
            _logger.error("Bulk error: %s", exc)
        return ok, err

    @staticmethod
    def _build() -> OpenSearch:
        endpoint = (
            os.environ["OPENSEARCH_ENDPOINT"]
            .removeprefix("https://")
            .removeprefix("http://")
            .rstrip("/")
        )
        auth = (os.environ["OPENSEARCH_USER"], os.environ["OPENSEARCH_PW"])
        return OpenSearch(
            hosts=[{"host": endpoint, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            pool_maxsize=30,
            timeout=60,
        )
