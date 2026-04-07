"""Database setup: create schema, tables, and verify connectivity.

Run from the project root:
    .venv/bin/python -m scripts.setup_database

Loads the u-ingestion .env, connects to PostgreSQL, creates
the configured schema, and ensures all required tables exist.
Safe to run repeatedly — uses IF NOT EXISTS throughout.
"""

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / "u-ingestion" / ".env"


def _require_env(name: str) -> str:
    """Get required env var or raise. Params: name. Returns: str."""
    value = os.getenv(name, "")
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _get_database_config() -> dict:
    """Read DB connection params from environment. Returns: dict."""
    required = {
        "host": _require_env("DB_HOST"),
        "port": _require_env("DB_PORT"),
        "dbname": _require_env("DB_NAME"),
        "user": _require_env("DB_USER"),
    }
    required["password"] = os.getenv("DB_PASSWORD", "")
    return required


def _get_schema() -> str:
    """Read DB_SCHEMA from environment. Returns: str."""
    return _require_env("DB_SCHEMA")


def _table_ddl(schema: str) -> dict:
    """Return table name -> CREATE TABLE DDL for the given schema.

    Params:
        schema: PostgreSQL schema name

    Returns:
        dict mapping table name to DDL string
    """
    return {
        "document_catalog": f"""
            CREATE TABLE IF NOT EXISTS {schema}.document_catalog (
                file_path           TEXT PRIMARY KEY,
                data_source         TEXT NOT NULL,
                filter_1            TEXT NOT NULL DEFAULT '',
                filter_2            TEXT NOT NULL DEFAULT '',
                filter_3            TEXT NOT NULL DEFAULT '',
                filename            TEXT NOT NULL,
                filetype            TEXT NOT NULL,
                file_size           BIGINT NOT NULL,
                date_last_modified  DOUBLE PRECISION NOT NULL,
                file_hash           TEXT NOT NULL DEFAULT ''
            );
        """,
        "document_versions": f"""
            CREATE TABLE IF NOT EXISTS {schema}.document_versions (
                id                  BIGSERIAL PRIMARY KEY,
                file_path           TEXT NOT NULL,
                data_source         TEXT NOT NULL,
                filter_1            TEXT NOT NULL DEFAULT '',
                filter_2            TEXT NOT NULL DEFAULT '',
                filter_3            TEXT NOT NULL DEFAULT '',
                filename            TEXT NOT NULL,
                filetype            TEXT NOT NULL,
                file_size           BIGINT NOT NULL,
                date_last_modified  DOUBLE PRECISION NOT NULL,
                file_hash           TEXT NOT NULL,
                is_current          BOOLEAN NOT NULL DEFAULT TRUE,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (file_path, file_hash)
            );
        """,
        "stage_checkpoints": f"""
            CREATE TABLE IF NOT EXISTS {schema}.stage_checkpoints (
                document_version_id BIGINT NOT NULL
                    REFERENCES {schema}.document_versions (id)
                    ON DELETE CASCADE,
                stage_name          TEXT NOT NULL,
                status              TEXT NOT NULL,
                stage_signature     TEXT NOT NULL,
                artifact_path       TEXT NOT NULL DEFAULT '',
                artifact_checksum   TEXT NOT NULL DEFAULT '',
                error_message       TEXT NOT NULL DEFAULT '',
                created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at        TIMESTAMPTZ,
                PRIMARY KEY (document_version_id, stage_name)
            );
        """,
    }


def setup_schema(conn, schema: str) -> None:
    """Create the pipeline schema if it doesn't exist.

    Params:
        conn: psycopg2 connection
        schema: Schema name to create
    """
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
    conn.commit()
    print(f"  OK: schema '{schema}'")


def setup_tables(conn, schema: str) -> None:
    """Create all pipeline tables if they don't exist.

    Params:
        conn: psycopg2 connection
        schema: PostgreSQL schema name
    """
    tables = _table_ddl(schema)
    with conn.cursor() as cur:
        for table_name, ddl in tables.items():
            cur.execute(ddl)
            print(f"  OK: {schema}.{table_name}")
    conn.commit()


def main() -> None:
    """Load config, connect to database, create schema and tables.

    Returns:
        None

    Example:
        >>> main()
    """
    print(f"Loading config from {ENV_PATH}")
    load_dotenv(ENV_PATH)

    config = _get_database_config()
    schema = _get_schema()
    print(f"Connecting to {config['host']}:{config['port']}/{config['dbname']}")

    conn = psycopg2.connect(**config)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        print("Connection verified")

        print("Creating schema:")
        setup_schema(conn, schema)

        print("Creating tables:")
        setup_tables(conn, schema)

        print("Database setup complete")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
    sys.exit(0)
