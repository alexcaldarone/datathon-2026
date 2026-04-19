# CLAUDE.md

## Project Vision
A minimal harness for a Datathon challenge focused on building a high-quality listing search and ranking system. Features a FastAPI backend, SQLite + OpenSearch storage, and an MCP-compatible React widget.

## Key Technical Commands
* **Install Dependencies:** `uv sync --dev`
* **Run API:** `uv run uvicorn app.main:app --reload --port 8000`
* **Run MCP Server:** `uv run uvicorn apps_sdk.server.main:app --reload --port 8001`
* **Run Tests:** `uv run pytest tests -q`

## Extension Points (Edit These)
Core participant logic resides in `app/participant/`:
* `hard_fact_extraction.py`: NLP query to structured filters.
* `soft_fact_extraction.py`: Extracting nuances/preferences and OpenSearch boost fields.
* `soft_filtering.py`: Post-hard-filter reranking via hybrid OpenSearch query.
* `ranking.py`: Final scoring and result shaping.
* `listing_row_parser.py`: CSV ingestion and feature extraction logic (41-column tuple).
* `query_validation.py`: Query validity check with clarification questions.

## System Architecture

### APIs
* **Primary API:** `POST /listings` — full pipeline (validate → hard extract → hard filter → soft extract → soft filter → rerank).
* **Search API:** `POST /listings/search/filter` — skip extraction, directly apply hard filters via SQLite.
* **MCP Integration:** `apps_sdk/server` (FastAPI bridge) and `apps_sdk/web` (React/Vite frontend).

### Storage
* **SQLite:** Auto-bootstrapped from `raw_data/*.csv` on startup. Indexed on city, postal_code, canton, price, rooms, lat/lon.
* **OpenSearch:** Stores augmented listing documents (dense/sparse embeddings, image embeddings, VLM features, geo features). Populated by the ingestion pipeline.

### Components Pattern
All participant pipeline components in `app/participant/components/` follow a consistent extensibility pattern:
```
build_component(cfg) → ComponentName   # Factory, reads cfg.class_name
ComponentName (ABC)                    # Abstract interface
DumbComponentName(ComponentName)       # Minimal baseline
LLMComponentName(ComponentName)        # Bedrock Claude implementation
AlternativeComponentName(ComponentName) # e.g., CohereReRanker
```

### System Prompts
Prompts for LLM-based components are stored as `configs/prompts/{ClassName}.md` and loaded via `app/participant/components/utils.py:read_system_prompt(class_name)`.

---

## Full Request Pipeline (`query_from_text`)

```
POST /listings
  ↓
SearchService.query_from_text(query, limit, offset)
  ├─ validate_query(query)                    → ValidationResult (LLMQueryValidator)
  ├─ extract_hard_facts(query)                → HardFilters (LLMHardFactExtractor via Bedrock)
  ├─ filter_hard_facts(db, HardFilters)       → candidates (SQLite query builder)
  ├─ extract_soft_facts(query)                → {preferences, boost_fields, _query_en} (LLMSoftExtractor)
  ├─ filter_soft_facts(candidates, soft_facts)→ reranked candidates (HybridSimilarityFilter via OpenSearch)
  │    └─ Hybrid query: BM25 + knn (dense) + rank_features (sparse) + knn (image) + VLM/geo boost
  └─ rank_listings(candidates, soft_facts)    → RankedListingResult[] (LLMReRanker or CohereReRanker)
  ↓
ListingsResponse(listings, meta)
```

---

## Ingestion Pipeline (`IngestionManager`)

Runs on startup (and optionally on-demand) to populate OpenSearch with augmented listing data.

```
bootstrap_database(db_path, raw_data_dir)     # Normalize SRED CSVs → SQLite
  └─ ensure_sred_normalized_csv() → create_schema() → import_csvs() → create_indexes()

IngestionManager.run(db_path, cfg, limit, reset)
  ├─ build_augmenters(cfg)                    # Instantiate enabled augmenters from config
  ├─ client.setup(index, pipeline, reset)     # OpenSearch index + ingest pipeline
  └─ For each batch of listings:
       ├─ Check OpenSearch for already-augmented fields (incremental, skip existing)
       ├─ download_images_batch() if image augmenters are enabled
       ├─ For each augmenter: augment_batch(listings) [thread pool]
       └─ _build_docs() → client.bulk_upsert(docs)
```

### Augmenters (all extend `Augmenter` ABC)
| Class | Output Field | Backend |
|---|---|---|
| `DenseEmbeddingAugmenter` | `dense_embedding` | Bedrock Titan |
| `BM25SparseAugmenter` | `sparse_embedding` | Local BM25 |
| `ImageEmbeddingAugmenter` | `image_embedding` | Bedrock Titan Multimodal |
| `VLMFeatureAugmenter` | `brightness_score`, `spaciousness_score`, `modernity_score`, etc. | Bedrock Claude Haiku |
| `TranslationAugmenter` | `title_en`, `description_en` | Bedrock Claude Haiku |
| `GeoFeatureAugmenter` | `transit_score`, `amenity_score`, `walkability_score` | OSM Overpass API |
| `AnchorsAugmenter` | `{anchor}_score` fields | Bedrock Titan (similarity to anchor phrases) |

---

## Key Files Reference

| Path | Purpose | Key Classes/Functions |
|---|---|---|
| `app/harness/search_service.py` | Query orchestration | `query_from_text`, `query_from_filters` |
| `app/harness/bootstrap.py` | SQLite init on startup | `bootstrap_database` |
| `app/harness/csv_import.py` | SQLite schema + import | `create_schema`, `import_csvs`, `create_indexes` |
| `app/harness/sred_transform.py` | SRED CSV normalization | `ensure_sred_normalized_csv` |
| `app/ingestion/augmenters.py` | Feature augmentation | `Augmenter` (ABC), 7 concrete implementations, `build_augmenters` |
| `app/ingestion/client.py` | OpenSearch singleton | `OpenSearchClient` |
| `app/ingestion/manager.py` | Ingestion orchestration | `IngestionManager` |
| `app/ingestion/logger.py` | Ingestion instrumentation | `IngestionLogger` (singleton) |
| `app/ingestion/utils.py` | Image + text helpers | `row_to_full_text`, `download_images_batch`, `parse_features` |
| `app/participant/components/utils.py` | Config + prompt loading | `Config` (singleton), `read_system_prompt`, `_instantiate` |
| `app/participant/components/logger.py` | Pipeline instrumentation | `PipelineLogger` (singleton) |
| `app/participant/components/hard_extractor.py` | Hard fact extractors | `HardFactExtractor` (ABC), `DumbHardFactExtractor`, `LLMHardFactExtractor` |
| `app/participant/components/soft_extractor.py` | Soft fact extractors | `SoftFactExtractor` (ABC), `DumbSoftExtractor`, `LLMSoftExtractor` |
| `app/participant/components/soft_filter.py` | Soft filters | `SoftFilter` (ABC), `DumbSoftFilter`, `HybridSimilarityFilter` |
| `app/participant/components/reranker.py` | Ranking implementations | `ReRanker` (ABC), `DumbReRanker`, `LLMReRanker`, `CohereReRanker` |
| `app/participant/components/query_validator.py` | Query validators | `QueryValidator` (ABC), `DumbQueryValidator`, `LLMQueryValidator` |
| `app/participant/components/translator.py` | Language detection + translation | `is_english`, `translate_to_english` |
| `app/participant/listing_row_parser.py` | CSV row normalization | `prepare_listing_row` (returns 41-field tuple) |

---

## Shared Singletons

| Singleton | Location | Purpose |
|---|---|---|
| `OpenSearchClient` | `app/ingestion/client.py` | Single connection pool — used by ingestion manager and `HybridSimilarityFilter` |
| `IngestionLogger` | `app/ingestion/logger.py` | Tracks augmentation job stats (timing, batch progress, step counts) |
| `PipelineLogger` | `app/participant/components/logger.py` | Tracks query pipeline execution per stage (timing, candidate counts) |
| `Config` | `app/participant/components/utils.py` | Hydra config loaded once; drives which component implementations are instantiated |

---

## Coding Standard
- Match the style and structure of the existing code inside `app/`.
- Use type hints on all functions.
- Do not use docstrings.
- Write clear, concise comments to explain logic, especially for complex functions. Avoid docstrings; use comments for clarity.
- Keep code simple, modular, and easy to follow.
- Implement the SOLID principles, specifically ensuring classes have only one reason to change and they depend on abstract interfaces rather than concrete implementations.
- Organize code into additional Python files and modules as needed for clarity and consistency.
- Place all test files in the `tests/` directory.
- Update `.claude/CLAUDE.md` to reflect any changes.
- Update `tests/` to reflect any changes. Keep tests minimal, follow the structure and simplicity of already implemented ones.
