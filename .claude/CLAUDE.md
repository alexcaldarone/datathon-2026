# CLAUDE.md

## Project Vision
A minimal harness for a Datathon challenge focused on building a high-quality listing search and ranking system. Features a FastAPI backend, SQLite storage, and an MCP-compatible React widget.

## Key Technical Commands
* **Install Dependencies:** `uv sync --dev`
* **Run API:** `uv run uvicorn app.main:app --reload --port 8000`
* **Run MCP Server:** `uv run uvicorn apps_sdk.server.main:app --reload --port 8001`
* **Build Widget:** `cd apps_sdk/web && npm install && npm run build`
* **Run Tests:** `uv run pytest tests -q`
* **Docker:** `docker compose up --build`
* **Ingest OpenSearch (dry run):** `OPENSEARCH_ENDPOINT=<host> uv run python scripts/ingest_opensearch.py --dry-run`
* **Ingest OpenSearch (full):** `OPENSEARCH_ENDPOINT=<host> uv run python scripts/ingest_opensearch.py --reset`

## Extension Points (Edit These)
Core participant logic resides in `app/participant/`:
* `hard_fact_extraction.py`: NLP query to structured filters.
* `soft_fact_extraction.py`: Extracting nuances/preferences.
* `soft_filtering.py`: Post-candidate set pruning.
* `ranking.py`: Scoring and result shaping.
* `listing_row_parser.py`: CSV ingestion and feature extraction logic.

## System Architecture
* **Primary API:** `POST /listings` (Main entrypoint: extraction -> filtering -> ranking).
* **Search API:** `POST /listings/search/filter` (Direct structured SQLite filtering).
* **MCP Integration:** `apps_sdk/server` (FastAPI bridge) and `apps_sdk/web` (React/Vite frontend).
* **Data Flow:** CSV files in `raw_data/` are automatically bootstrapped into SQLite on startup.
* **Backend:** FastAPI, `uv` for package management, Pydantic for schemas.
* **Frontend:** React + Vite (Apps SDK widget).
* **Storage:** SQLite (auto-generated from `raw_data/*.csv`).
* **Orchestration:** Logic flow is managed in `app/harness/search_service.py`.
* **components:** the different components of the pipeline are contained in `app/participant/components`. At the top there is the build function, then the abstract interface, and eventually all the concrete implementation.

## Environment Variables
```bash
# S3 Credentials for images
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=eu-central-2

# MCP / Tunneling
export APPS_SDK_LISTINGS_API_BASE_URL=http://localhost:8000
export APPS_SDK_PUBLIC_BASE_URL=https://<your-tunnel-url>
```

## Coding Standard
- Match the style and structure of the existing code inside `app/`.
- Use type hints on all functions.
- Do not use docstrings.
- Write clear, concise comments to explain logic, especially for complex functions. Avoid docstrings; use comments for clarity.
- Keep code simple, modular, and easy to follow.
- Implement the SOLID principles, specifically ensuring classes have only one reason to change and they depend on abstract interfaces rather than concrete implementations.
- Organize code into additional Python files and modules as needed for clarity and consistency.
- Place all test files in the `tests/` directory.
- Update `.claude/CLAUDE.md` to reflect any changes
- Update `tests/` to reflect any changes. Keep tests minimal, follow the structure and simplicity of already implemented ones.