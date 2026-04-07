# U-Pipeline

## Tech Stack

- **Language**: Python 3.12+
- **LLM**: OpenAI SDK (tool calling only, no streaming) — architecture is LLM-agnostic; the connector layer must be swappable
- **Database**: PostgreSQL + pgvector
- **Testing**: pytest (target complete coverage)
- **Embeddings**: pgvector for semantic search

## Project Structure

Three subprojects sharing a common input-data set:

```
u-pipeline/
├── u-ingestion/          # Document ingestion pipeline
│   ├── src/ingestion/    # Source code
│   ├── tests/            # Test suite
│   └── .env              # Environment config
├── u-retriever/          # Research and retrieval pipeline
│   ├── src/retriever/    # Source code
│   ├── tests/            # Test suite
│   └── .env              # Environment config
├── u-debug/              # Web-based debug interface (Flask)
│   ├── src/debug/        # Flask app + API endpoints
│   └── tests/            # Test suite
├── input-data/           # Source documents by type/period/bank
├── docs/                 # Config reference, test queries
└── scripts/              # Setup and database scripts
```

## Commands

```bash
# All commands run from the subproject directory (u-ingestion/, u-retriever/, or u-debug/)
# using the shared venv at the project root

# Run the startup check
../.venv/bin/python -m src.ingestion.main    # from u-ingestion/
../.venv/bin/python -m src.retriever.main    # from u-retriever/

# Launch debug interface
../.venv/bin/python -m src.debug.app         # from u-debug/, opens http://localhost:5000

# Run all tests with coverage
../.venv/bin/python -m pytest --cov=src --cov-report=term-missing

# Run a specific test file
../.venv/bin/python -m pytest tests/test_<module>.py -v

# Code quality (all must pass clean)
../.venv/bin/python -m black --check src/ tests/
../.venv/bin/python -m flake8 src/ tests/
../.venv/bin/python -m pylint src/ tests/
```

## Code Conventions

### Documentation Style

**Main functions** (pipeline stages, public API, orchestrator functions):
```python
def classify_page(page_image: bytes, file_type: str) -> PageClassification:
    """Classify page content into Text, Table, or Visual.

    Params:
        page_image: Raw image bytes of the rendered page
        file_type: Source format ("pdf", "docx", "pptx", "xlsx", "csv", "md")

    Returns:
        PageClassification with content_types list and confidence scores

    Example:
        >>> result = classify_page(img_bytes, "pdf")
        >>> result.content_types
        ["text", "table"]
    """
```

**Smaller/helper functions** — concise single-line docstring with params and return:
```python
def extract_text_from_cell(cell: Cell) -> str:
    """Extract cleaned text from a table cell. Params: cell (Cell). Returns: str."""
```

### General Rules

- Minimal comments — code should be self-explanatory. Only comment non-obvious logic
- Function names: snake_case, verb-first, consistent with existing patterns
- Type hints on all function signatures
- No classes unless there's a clear reason — prefer functions and dataclasses/TypedDicts for data
- LLM calls must go through `LLMClient` in `utils/llm_connector.py`, never call OpenAI directly from pipeline code
- All LLM interactions use tool calling (structured output), not freeform text

### Testing & Code Quality

- Every module gets a corresponding test file: `src/{package}/module.py` -> `tests/test_module.py`
- Mock LLM calls in unit tests; use fixtures for database interactions
- Test edge cases: empty documents, single-page docs, tables with no headers, mixed content pages
- **After any code change**, run the full quality gate from the subproject directory:
  1. `../.venv/bin/python -m pytest --cov=src --cov-report=term-missing` — target 100% coverage
  2. `../.venv/bin/python -m black --check src/ tests/` — must pass with no reformatting needed
  3. `../.venv/bin/python -m flake8 src/ tests/` — must pass with zero warnings
  4. `../.venv/bin/python -m pylint src/ tests/` — must score 10.00/10
- **No suppressions** — do not use `# noqa`, `# pylint: disable`, `# type: ignore`, or any other mechanism to skip checks. Fix the code to satisfy the linter, not the other way around
- **Approved exceptions** — in rare cases where a tool limitation makes 100% impossible (e.g., coverage can't track cross-process execution), a suppression comment may be added only with explicit user approval. Document the reason inline

### Traceability & Logging

Two layers — **debug traces** for drilling into problems, **console logs** for monitoring runs:

**Debug traces** — Per-file, per-stage structured output capturing the full processing detail (inputs, decisions, outputs, errors). Written to a trace store so any file's journey through the pipeline can be inspected after the fact. These are verbose by design.

**Console logging** — Minimal and clean. No per-page spam. Each pipeline stage produces a short **stage summary** at the end showing counts, timings, and any failures. The goal is a scannable run log, not a wall of text.

Pattern for every pipeline stage:
1. Collect per-file debug trace data as the stage runs
2. Log a single summary block when the stage completes
3. Surface errors/warnings inline but keep info-level output to summaries only

### General Patterns

- Keep pipeline stages as separate modules
- Configuration via environment variables (never hardcode connection strings, API keys, etc.)
- Database operations isolated in `utils/postgres_connector.py` — pipeline code never writes raw SQL
