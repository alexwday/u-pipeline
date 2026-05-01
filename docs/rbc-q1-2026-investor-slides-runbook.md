# RBC Work Computer Runbook: Q1 2026 Investor Slides

Use this runbook from the RBC/work computer to set up the database, wipe
existing pipeline tables, and reprocess only the Q1 2026 investor slide
PDFs for the six Canadian banks.

## 1. Pull Latest Code

Run from the repo root:

```bash
cd /path/to/u-pipeline
git checkout main
git pull origin main
```

Create the repo-local virtual environment if it does not already exist:

```bash
cd /path/to/u-pipeline
python3.12 -m venv .venv
```

If `python3.12` is not available on the work computer, use the RBC
approved Python 3.12+ executable instead.

Install the local packages into the shared virtual environment:

```bash
cd /path/to/u-pipeline
.venv/bin/python -m pip install -e u-ingestion -e u-retriever
```

## 2. Configure RBC Environment

The `.env` files are intentionally not committed because they contain
secrets. If this is the first setup on the work computer, create them
from the examples:

```bash
cd /path/to/u-pipeline
cp u-ingestion/.env.example u-ingestion/.env
cp u-retriever/.env.example u-retriever/.env
```

Then edit `u-ingestion/.env` and `u-retriever/.env` with the RBC values.
At minimum, confirm these are set:

```bash
AUTH_MODE=oauth
LLM_ENDPOINT=<RBC OpenAI-compatible endpoint>
OAUTH_TOKEN_ENDPOINT=<RBC OAuth token endpoint>
OAUTH_CLIENT_ID=<client id>
OAUTH_CLIENT_SECRET=<client secret>
OAUTH_SCOPE=<scope if required>

DB_HOST=<postgres host>
DB_PORT=<postgres port>
DB_NAME=<database name>
DB_USER=<database user>
DB_PASSWORD=<database password>
DB_SCHEMA=u_pipeline
DB_SSLMODE=require
TOKENIZER_MODEL=o200k_base
```

If RBC provides explicit database certificate files, also set:

```bash
DB_SSLROOTCERT=<path to root cert>
DB_SSLCERT=<path to client cert>
DB_SSLKEY=<path to client key>
```

`TOKENIZER_MODEL` is only for local token counting and chunk sizing.
It does not control the embedding model. The `o200k_base` tiktoken
cache is bundled in `u-ingestion/tokenizer-cache/` and configured by
ingestion startup, so token counting should not need to download the
encoding file.

Recommended ingestion concurrency for the first RBC run:

```bash
MAX_WORKERS=1
EXTRACTION_PAGE_WORKERS=2
EXTRACTION_REGION_WORKERS=3
VISION_DPI_SCALE=3.0
```

This keeps file-level concurrency conservative while still parallelizing
pages and chart regions within each PDF.

## 3. Verify Or Create Database Objects

Run from the repo root:

```bash
cd /path/to/u-pipeline
.venv/bin/python -m scripts.setup_database
```

Expected result:

```text
Connection verified
Database setup complete
```

This command uses the canonical ingestion schema creator, including
pgvector extension setup, tables, columns, and retrieval indexes.

## 4. Wipe Existing Pipeline Tables

Warning: this deletes the pipeline catalog, document versions, stage
checkpoints, extracted content, embeddings, summaries, and metadata in
the configured `DB_SCHEMA`.

Run from `u-ingestion`. This uses Python to load `.env`, so the file
does not need to be valid shell syntax:

```bash
cd /path/to/u-pipeline/u-ingestion

../.venv/bin/python - <<'PY'
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

env_path = Path(".env")
if not env_path.exists():
    raise SystemExit(
        "Missing u-ingestion/.env. Run this from u-ingestion or create "
        "the .env file first."
    )

load_dotenv(env_path)

required = [
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_SCHEMA",
]
missing = [name for name in required if not os.getenv(name)]
if missing:
    raise SystemExit(f"Missing required env values: {', '.join(missing)}")

conn_kwargs = {
    "host": os.environ["DB_HOST"],
    "port": os.environ["DB_PORT"],
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.getenv("DB_PASSWORD", ""),
}
for env_name, conn_key in (
    ("DB_SSLMODE", "sslmode"),
    ("DB_SSLROOTCERT", "sslrootcert"),
    ("DB_SSLCERT", "sslcert"),
    ("DB_SSLKEY", "sslkey"),
):
    value = os.getenv(env_name, "")
    if value:
        conn_kwargs[conn_key] = value

schema = os.environ["DB_SCHEMA"]
statement = sql.SQL(
    "TRUNCATE TABLE {}.document_catalog, {}.document_versions "
    "RESTART IDENTITY CASCADE;"
).format(sql.Identifier(schema), sql.Identifier(schema))

with psycopg2.connect(**conn_kwargs) as conn:
    with conn.cursor() as cur:
        cur.execute(statement)

print(f"Wiped pipeline tables in schema {schema}")
PY
```

## 5. Reprocess Q1 2026 Investor Slides Only

Run from `u-ingestion`:

```bash
cd /path/to/u-pipeline/u-ingestion
REPO="$(cd .. && pwd)"

DATA_SOURCE_PATH="$REPO/input-data" \
MAX_WORKERS=1 \
EXTRACTION_PAGE_WORKERS=2 \
EXTRACTION_REGION_WORKERS=3 \
VISION_DPI_SCALE=3.0 \
../.venv/bin/python -m src.ingestion.main \
  --glob "*/investor-slides/2026_Q1/*/*.pdf"
```

The glob targets these six PDFs only:

```text
input-data/investor-slides/2026_Q1/BMO/bmo_q1_2026_investor_slides.pdf
input-data/investor-slides/2026_Q1/BNS/bns_q1_2026_investor_slides.pdf
input-data/investor-slides/2026_Q1/CIBC/cibc_q1_2026_investor_slides.pdf
input-data/investor-slides/2026_Q1/NBC/nbc_q1_2026_investor_slides.pdf
input-data/investor-slides/2026_Q1/RBC/rbc_q1_2026_investor_slides.pdf
input-data/investor-slides/2026_Q1/TD/td_q1_2026_investor_slides.pdf
```

If the run is interrupted, rerun the same command. The pipeline is
checkpointed and will resume from the earliest incomplete stage.

## 6. Optional BNS-Only Test Run

To process only the BNS Q1 2026 investor slides before running all six
banks, run from `u-ingestion`:

```bash
cd /path/to/u-pipeline/u-ingestion
REPO="$(cd .. && pwd)"

DATA_SOURCE_PATH="$REPO/input-data" \
MAX_WORKERS=1 \
EXTRACTION_PAGE_WORKERS=2 \
EXTRACTION_REGION_WORKERS=3 \
VISION_DPI_SCALE=3.0 \
../.venv/bin/python -m src.ingestion.main \
  --file-path "$REPO/input-data/investor-slides/2026_Q1/BNS/bns_q1_2026_investor_slides.pdf"
```

`DATA_SOURCE_PATH` is set inline here because the pipeline's
`load_dotenv()` call does not override environment variables that were
already exported in your shell. This prevents an older
`DATA_SOURCE_PATH` value from pointing discovery at the wrong folder.

If you are already in the repo root, the BNS path resolves to:

```bash
$(pwd)/input-data/investor-slides/2026_Q1/BNS/bns_q1_2026_investor_slides.pdf
```

## 7. Smoke Test Retrieval

After ingestion completes, run a targeted retrieval query from
`u-retriever`:

```bash
cd /path/to/u-pipeline/u-retriever

../.venv/bin/python -m src.retriever.main \
  --query "For BNS Q1 2026, what are credit cards PCLs on impaired loans as a percent of average net loans?" \
  --bank BNS \
  --period 2026_Q1
```

Expected BNS slide 37 value:

```text
523 bps, or 5.23%
```

If retrieval returns `518 bps` for that specific question, inspect the
stored BNS page 37 content before continuing broader testing.
