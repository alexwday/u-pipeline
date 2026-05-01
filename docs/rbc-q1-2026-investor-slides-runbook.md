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

Install the local packages into the shared virtual environment:

```bash
.venv/bin/python -m pip install -e u-ingestion -e u-retriever
```

## 2. Configure RBC Environment

Edit `u-ingestion/.env` and `u-retriever/.env` with the RBC values.
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
```

If RBC provides explicit database certificate files, also set:

```bash
DB_SSLROOTCERT=<path to root cert>
DB_SSLCERT=<path to client cert>
DB_SSLKEY=<path to client key>
```

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

Run from `u-ingestion`:

```bash
cd /path/to/u-pipeline/u-ingestion
set -a
source .env
set +a

PGPASSWORD="$DB_PASSWORD" PGSSLMODE="${DB_SSLMODE:-prefer}" \
PGSSLROOTCERT="$DB_SSLROOTCERT" PGSSLCERT="$DB_SSLCERT" PGSSLKEY="$DB_SSLKEY" \
psql -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -U "$DB_USER" \
  -v schema="$DB_SCHEMA" \
  -c 'TRUNCATE TABLE :"schema".document_catalog, :"schema".document_versions RESTART IDENTITY CASCADE;'
```

## 5. Reprocess Q1 2026 Investor Slides Only

Run from `u-ingestion`:

```bash
cd /path/to/u-pipeline/u-ingestion

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

## 6. Smoke Test Retrieval

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
