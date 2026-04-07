"""Dump and load u-pipeline postgres tables as gzipped seed data.

The dump uses PostgreSQL's text-format COPY, which natively serialises
pgvector and tsvector columns through their text representations, so we
do not need any per-column conversion logic. Files are written gzipped
to keep them small enough to live alongside the source.

CLI usage (from project root, inside the venv):
    python -m scripts.seed_data dump   # write seed files from current DB
    python -m scripts.seed_data load   # wipe and load seed files into DB
    python -m scripts.seed_data status # row counts in DB and on disk
"""

import gzip
import os
import sys
from pathlib import Path
from typing import Optional

import psycopg2
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = PROJECT_ROOT / "scripts" / "seed-data"
ENV_PATH = PROJECT_ROOT / "u-ingestion" / ".env"

# Tables in foreign-key load order. Children come after parents so
# COPY FROM can satisfy referential integrity without deferring keys.
TABLES_IN_LOAD_ORDER = [
    "document_catalog",
    "document_versions",
    "stage_checkpoints",
    "document_metadata",
    "document_sections",
    "document_content",
]

# Tables whose BIGSERIAL primary keys must be re-synced after COPY,
# otherwise the next INSERT will collide with seeded ids.
SEQUENCED_TABLES = [
    ("document_versions", "id"),
    ("document_sections", "id"),
    ("document_content", "id"),
]


def seed_dir() -> Path:
    """Return the directory holding gzipped seed files. Returns: Path."""
    return SEED_DIR


def seed_file(table: str) -> Path:
    """Return the seed file path for a table. Params: table. Returns: Path."""
    return SEED_DIR / f"{table}.tsv.gz"


def seed_files_present() -> bool:
    """Return True if every table has a seed file on disk. Returns: bool."""
    return all(seed_file(t).exists() for t in TABLES_IN_LOAD_ORDER)


def db_config_from_env() -> dict:
    """Read DB connection params from environment. Returns: dict."""
    return {
        "host": os.environ["DB_HOST"],
        "port": os.environ["DB_PORT"],
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ.get("DB_PASSWORD", ""),
    }


def schema_from_env() -> str:
    """Return DB_SCHEMA from environment. Returns: str."""
    return os.environ.get("DB_SCHEMA", "u_pipeline")


def schema_has_any_tables(conn, schema: str) -> bool:
    """Check if the schema already contains any pipeline tables.

    Params:
        conn: psycopg2 connection
        schema: PostgreSQL schema name

    Returns:
        True if any of the seed tables already exist in this schema
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = ANY(%s)
            """,
            (schema, TABLES_IN_LOAD_ORDER),
        )
        return cur.fetchone()[0] > 0


def count_rows(conn, schema: str) -> dict[str, int]:
    """Return per-table row counts for the seed tables.

    Params:
        conn: psycopg2 connection
        schema: PostgreSQL schema name

    Returns:
        dict mapping table name to row count (0 if table missing)
    """
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        for table in TABLES_IN_LOAD_ORDER:
            cur.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
                """,
                (schema, table),
            )
            if not cur.fetchone():
                counts[table] = 0
                continue
            cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
            counts[table] = cur.fetchone()[0]
    return counts


def dump(conn, schema: str) -> dict[str, int]:
    """Dump every seed table from the database to gzipped TSV files.

    Params:
        conn: psycopg2 connection
        schema: PostgreSQL schema name

    Returns:
        dict mapping table name to number of rows dumped
    """
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for table in TABLES_IN_LOAD_ORDER:
        path = seed_file(table)
        with gzip.open(path, "wb") as gz:
            with conn.cursor() as cur:
                cur.copy_expert(f"COPY {schema}.{table} TO STDOUT", gz)
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
            written[table] = cur.fetchone()[0]
        size_kb = path.stat().st_size / 1024
        print(
            f"  {table}: {written[table]} rows -> "
            f"{path.name} ({size_kb:.1f} KB)"
        )
    return written


def wipe(conn, schema: str) -> None:
    """TRUNCATE every seed table and reset its identity sequence.

    Params:
        conn: psycopg2 connection
        schema: PostgreSQL schema name
    """
    qualified = ", ".join(f"{schema}.{t}" for t in TABLES_IN_LOAD_ORDER)
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {qualified} RESTART IDENTITY CASCADE;")
    conn.commit()
    print(f"  truncated: {', '.join(TABLES_IN_LOAD_ORDER)}")


def _reset_sequences(conn, schema: str) -> None:
    """Sync BIGSERIAL sequences to MAX(id) after a COPY load.

    Params:
        conn: psycopg2 connection
        schema: PostgreSQL schema name
    """
    with conn.cursor() as cur:
        for table, pk in SEQUENCED_TABLES:
            cur.execute(
                f"SELECT setval("
                f"  pg_get_serial_sequence('{schema}.{table}', '{pk}'),"
                f"  COALESCE((SELECT MAX({pk}) FROM {schema}.{table}), 1),"
                f"  (SELECT MAX({pk}) IS NOT NULL FROM {schema}.{table})"
                f")"
            )
    conn.commit()


def load(conn, schema: str) -> dict[str, int]:
    """Load gzipped seed files into the database.

    Wipes the seed tables first, then COPY-loads each file in FK
    order, and resets sequences. Caller is responsible for ensuring
    the schema and tables already exist.

    Params:
        conn: psycopg2 connection
        schema: PostgreSQL schema name

    Returns:
        dict mapping table name to number of rows loaded

    Example:
        >>> load(conn, 'u_pipeline')
        {'document_catalog': 8, ...}
    """
    if not seed_files_present():
        missing = [
            t for t in TABLES_IN_LOAD_ORDER if not seed_file(t).exists()
        ]
        raise FileNotFoundError(f"Missing seed files for tables: {missing}")
    wipe(conn, schema)
    loaded: dict[str, int] = {}
    for table in TABLES_IN_LOAD_ORDER:
        path = seed_file(table)
        with gzip.open(path, "rb") as gz:
            with conn.cursor() as cur:
                cur.copy_expert(f"COPY {schema}.{table} FROM STDIN", gz)
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
            loaded[table] = cur.fetchone()[0]
        print(f"  {table}: {loaded[table]} rows loaded")
    conn.commit()
    _reset_sequences(conn, schema)
    print("  sequences reset")
    return loaded


def _open_connection() -> psycopg2.extensions.connection:
    """Load env and open a psycopg2 connection. Returns: connection."""
    load_dotenv(ENV_PATH)
    return psycopg2.connect(**db_config_from_env())


def _cli_status() -> None:
    """Print db row counts and on-disk seed file sizes. Returns: None."""
    schema = schema_from_env()
    print(f"Seed dir: {SEED_DIR}")
    print(f"Schema:   {schema}\n")
    print("On disk:")
    for table in TABLES_IN_LOAD_ORDER:
        path = seed_file(table)
        if path.exists():
            kb = path.stat().st_size / 1024
            print(f"  {table}.tsv.gz   {kb:.1f} KB")
        else:
            print(f"  {table}.tsv.gz   (missing)")
    print("\nIn database:")
    try:
        conn = _open_connection()
    except psycopg2.Error as exc:
        print(f"  cannot connect: {exc}")
        return
    try:
        counts = count_rows(conn, schema)
        for table, count in counts.items():
            print(f"  {schema}.{table}: {count} rows")
    finally:
        conn.close()


def _cli_dump() -> None:
    """CLI entry point for dumping seed data. Returns: None."""
    schema = schema_from_env()
    conn = _open_connection()
    try:
        print(f"Dumping {schema} -> {SEED_DIR}")
        dump(conn, schema)
        print("Dump complete")
    finally:
        conn.close()


def _cli_load() -> None:
    """CLI entry point for loading seed data. Returns: None."""
    schema = schema_from_env()
    conn = _open_connection()
    try:
        print(f"Loading seed data into {schema}")
        load(conn, schema)
        print("Load complete")
    finally:
        conn.close()


def main(argv: Optional[list[str]] = None) -> int:
    """Dispatch a seed_data subcommand.

    Params:
        argv: command-line args (defaults to sys.argv[1:])

    Returns:
        process exit code
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = args[0]
    handlers = {
        "dump": _cli_dump,
        "load": _cli_load,
        "status": _cli_status,
    }
    handler = handlers.get(cmd)
    if handler is None:
        print(f"Unknown subcommand: {cmd}")
        print(__doc__)
        return 2
    handler()
    return 0


if __name__ == "__main__":
    sys.exit(main())
