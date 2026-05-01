"""Database setup: create schema, tables, indexes, and verify connectivity.

Run from the project root:
    .venv/bin/python -m scripts.setup_database

Loads the u-ingestion .env and delegates schema creation to the
canonical ingestion PostgreSQL connector. Safe to run repeatedly.
"""

import sys
from pathlib import Path
from importlib import import_module

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / "u-ingestion" / ".env"
INGESTION_SRC = PROJECT_ROOT / "u-ingestion" / "src"
sys.path.insert(0, str(INGESTION_SRC))

CONFIG_SETUP = import_module("ingestion.utils.config_setup")
POSTGRES_CONNECTOR = import_module("ingestion.utils.postgres_connector")
SSL_SETUP = import_module("ingestion.utils.ssl_setup")


def main() -> None:
    """Load config, connect to database, and ensure schema objects exist."""
    print(f"Loading config from {ENV_PATH}")
    load_dotenv(ENV_PATH)

    schema = CONFIG_SETUP.get_database_schema()
    print(f"Ensuring database schema '{schema}'")

    SSL_SETUP.setup_ssl()
    conn = POSTGRES_CONNECTOR.get_connection()
    try:
        POSTGRES_CONNECTOR.verify_connection(conn)
        print("Connection verified")
        POSTGRES_CONNECTOR.ensure_schema_objects(conn)
        print("Database setup complete")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
    sys.exit(0)
