# U-Pipeline

Document ingestion and research retrieval pipeline for financial disclosures. Processes bank regulatory filings (Pillar 3, investor slides, financial supplements) into a searchable PostgreSQL database with vector embeddings, then provides an LLM-powered retrieval interface for querying across documents.

## Prerequisites

- Python 3.12+
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension
- OpenAI-compatible LLM endpoint

## Quick Start

```bash
git clone https://github.com/alexwday/u-pipeline.git
cd u-pipeline
python3 scripts/setup.py
```

That single command runs the full setup flow:

1. Creates `.venv/` at the project root
2. Installs `u-ingestion`, `u-retriever`, and `u-debug` in editable mode
3. Attempts an optional install of `rbc_security` (only available on the
   work-internal index — failures are tolerated everywhere else)
4. Copies `.env.example → .env` for `u-ingestion` and `u-retriever`
5. Opens each `.env` in your `$EDITOR` (or the platform default) so you
   can paste DB credentials, OAuth/API key, and LLM endpoint details,
   then waits at the prompt until you confirm the file is saved
6. Validates the database and LLM connections
7. Creates the schema, tables, indexes, and pgvector objects
8. Detects whether the schema was empty before this run and offers to
   load seed data from `scripts/seed-data/` (or to wipe and reload over
   pre-existing data)
9. Launches the `u-debug` Flask server at `http://localhost:5001` and
   opens it in your browser, ready for your first query

### Seed data

The repo ships with a snapshot of every pipeline table under
`scripts/seed-data/` (gzipped TSV via PostgreSQL's text-format `COPY`).
You can re-dump or reload it manually:

```bash
.venv/bin/python -m scripts.seed_data status   # row counts on disk and in DB
.venv/bin/python -m scripts.seed_data dump     # overwrite seed files from DB
.venv/bin/python -m scripts.seed_data load     # wipe DB tables and reload
```

## Project Structure

```
u-pipeline/
├── u-ingestion/          # Document ingestion pipeline
│   ├── src/ingestion/    # Source code (11 pipeline stages)
│   └── tests/            # Test suite
├── u-retriever/          # Research and retrieval pipeline
│   ├── src/retriever/    # Source code (6 retrieval stages)
│   └── tests/            # Test suite
├── u-debug/              # Web-based debug interface
│   ├── src/debug/        # Flask app + API
│   └── tests/            # Test suite
├── input-data/           # Source documents by type/period/bank
├── docs/                 # Config reference, test queries
└── scripts/              # Setup and database scripts
```

## Running the Ingestion Pipeline

From the `u-ingestion/` directory:

```bash
# Process all files through the full pipeline
../.venv/bin/python -m src.ingestion.main

# Process only PDFs
../.venv/bin/python -m src.ingestion.main --glob "*.pdf"

# Process a specific file
../.venv/bin/python -m src.ingestion.main --file-path /path/to/file.xlsx

# Run through a specific stage only
../.venv/bin/python -m src.ingestion.main --to-stage extraction

# Force reprocess everything
../.venv/bin/python -m src.ingestion.main --force-all
```

The pipeline is resumable — if a run is interrupted, re-running picks up where it left off.

### Pipeline Stages

1. **extraction** — Format-specific content extraction (PDF, DOCX, PPTX, XLSX)
2. **tokenization** — Token counting per page
3. **classification** — Page layout classification (table, text, visual, etc.)
4. **chunking** — Split oversized pages for embedding
5. **doc_metadata** — Extract document-level metadata via LLM
6. **section_detection** — Detect logical document sections via LLM
7. **content_extraction** — Extract and normalize body content
8. **section_summary** — Generate section summaries
9. **doc_summary** — Generate document summary
10. **embedding** — Generate vector embeddings for all content
11. **persistence** — Write everything to PostgreSQL

## Running the Retriever

From the `u-retriever/` directory:

```bash
# Run the pre-built test query suite
../.venv/bin/python -m src.retriever.main --test

# Run an ad-hoc query
../.venv/bin/python -m src.retriever.main \
    --query "What is RBC's CET1 ratio?" \
    --bank RBC --period 2026_Q1

# Query multiple banks
../.venv/bin/python -m src.retriever.main \
    --query "Compare CET1 ratios" \
    --bank RBC --bank BMO --period 2026_Q1

# Filter to specific data sources
../.venv/bin/python -m src.retriever.main \
    --query "Revenue breakdown" \
    --bank RBC --period 2026_Q1 --source pillar3
```

## Debug Interface

A web-based tool for running queries and inspecting retrieval traces.

```bash
cd u-debug
../.venv/bin/python -m src.debug.app
# Open http://localhost:5001 (override with DEBUG_PORT)
```

Features:
- Pre-built test query selector
- Custom query builder with bank/period/source selection
- Streaming response display
- Debug panel with timing metrics, token usage, and prepared query details
- Trace viewer for inspecting search, rerank, expand, and research stages

## Configuration

Both subprojects use `.env` files for configuration. Key settings:

| Setting | Description |
|---------|-------------|
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | PostgreSQL connection |
| `DB_SCHEMA` | Schema name (default: `u_pipeline`) |
| `AUTH_MODE` | `api_key` or `oauth` |
| `LLM_ENDPOINT` | OpenAI-compatible API base URL |
| `DATA_SOURCE_PATH` | Path to input documents (ingestion only) |

See `docs/config-reference.md` for the full parameter reference with tuning guidance.

## Development

Quality gate commands (run from each subproject directory):

```bash
# Tests with coverage
../.venv/bin/python -m pytest --cov=src --cov-report=term-missing

# Formatting
../.venv/bin/python -m black --check src/ tests/

# Linting
../.venv/bin/python -m flake8 src/ tests/
../.venv/bin/python -m pylint src/ tests/
```

See `CLAUDE.md` for coding conventions and detailed project guidelines.
