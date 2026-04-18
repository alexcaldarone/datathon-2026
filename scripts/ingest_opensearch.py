"""
Ingest listings from SQLite into AWS OpenSearch as a hybrid (BM25 + dense kNN) index.

Required env vars:
  OPENSEARCH_ENDPOINT  - host without scheme, e.g. my-domain.eu-central-1.es.amazonaws.com
  AWS_DEFAULT_REGION   - defaults to eu-central-2
  OPENSEARCH_USER
  OPENSEARCH_PW

Optional env var overrides:
  OPENSEARCH_INDEX     - index name (default from config.yaml)
  BATCH_SIZE           - rows per bulk-index call (default from config.yaml)
  EMBED_WORKERS        - parallel Bedrock threads (default from config.yaml)
  RRF_RANK_CONSTANT    - RRF k constant (default from config.yaml)
"""

import argparse
import json
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from omegaconf import OmegaConf

from app.ingestion.components.client import OpenSearchClient
from app.ingestion.components import IngestionManager, build_augmenters

load_dotenv()

_REPO_ROOT = Path(__file__).parent.parent
_CFG_DIR = _REPO_ROOT / "configs" / "ingestion"


def _load_cfg() -> object:
    with open(_CFG_DIR / "config.yaml") as f:
        raw = yaml.safe_load(f)
    raw["region"] = os.environ.get("AWS_DEFAULT_REGION", "eu-central-2")
    raw["index_name"] = os.environ.get("OPENSEARCH_INDEX", raw["index_name"])
    raw["default_batch"] = int(os.environ.get("BATCH_SIZE", raw["default_batch"]))
    raw["embed_workers"] = int(os.environ.get("EMBED_WORKERS", raw["embed_workers"]))
    raw["rrf_rank_constant"] = int(os.environ.get("RRF_RANK_CONSTANT", raw["rrf_rank_constant"]))
    return OmegaConf.create(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest listings into OpenSearch hybrid index.")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the index.")
    parser.add_argument("--dry-run", action="store_true", help="Init index/pipeline only, skip ingestion.")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of rows to ingest (for testing).")
    args = parser.parse_args()

    cfg = _load_cfg()

    with open(_CFG_DIR / "index_body.json") as f:
        index_body = json.load(f)
    with open(_CFG_DIR / "pipeline_body.json") as f:
        pipeline_body = json.load(f)

    # sync rank_constant from config into the pipeline body template
    (pipeline_body["phase_results_processors"][0]["normalization-processor"]
                  ["normalization"]["parameters"]["rank_constant"]) = cfg.rrf_rank_constant

    augmenters = build_augmenters(cfg)
    client = OpenSearchClient()
    manager = IngestionManager(cfg, augmenters, client)

    print(f"Connected to OpenSearch at {os.environ['OPENSEARCH_ENDPOINT']}")

    if args.dry_run:
        full_body = manager.build_index_body(index_body)
        client.setup(full_body, pipeline_body, reset=args.reset)
        print("Dry run — skipping ingestion.")
        return

    start = time.time()
    manager.run(
        db_path=_REPO_ROOT / "data" / "listings.db",
        index_body=index_body,
        pipeline_body=pipeline_body,
        limit=args.limit,
        reset=args.reset,
    )
    print(f"Elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
