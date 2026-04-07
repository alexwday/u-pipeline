"""Phase 2 of the u-pipeline setup wizard.

Runs *inside* the project venv (so it can freely import psycopg2,
dotenv, ingestion.*, retriever.*, and the seed_data module). Phase 1
in scripts/setup.py re-execs this file via the venv interpreter.

Responsibilities:
    - Load configuration from u-ingestion/.env
    - Validate database and LLM connectivity
    - Create the schema/tables/indexes via the ingestion connector
    - Optionally load (or wipe-and-reload) seed data
    - Launch the u-debug Flask server and open the browser
"""

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

from ingestion.utils.postgres_connector import ensure_schema_objects
from retriever.utils.config_setup import load_config
from retriever.utils.llm_connector import LLMClient
from retriever.utils.ssl_setup import setup_ssl
from scripts import seed_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / ".venv"
ENV_PATH = PROJECT_ROOT / "u-ingestion" / ".env"


def _banner(text: str) -> None:
    """Print a section banner. Params: text. Returns: None."""
    width = 64
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}\n")


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Ask a yes/no question. Params: prompt, default. Returns: bool."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        raw = input(prompt + suffix).strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer 'y' or 'n'.")


def _venv_python() -> Path:
    """Return path to the venv Python interpreter. Returns: Path."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _load_env() -> dict:
    """Load u-ingestion/.env into os.environ. Returns: db config dict."""
    load_dotenv(ENV_PATH, override=True)
    return {
        "DB_HOST": os.environ.get("DB_HOST", ""),
        "DB_PORT": os.environ.get("DB_PORT", ""),
        "DB_NAME": os.environ.get("DB_NAME", ""),
        "DB_USER": os.environ.get("DB_USER", ""),
        "DB_PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "DB_SCHEMA": os.environ.get("DB_SCHEMA", "u_pipeline"),
    }


def _open_db(config: dict):
    """Open a psycopg2 connection. Params: config. Returns: connection."""
    return psycopg2.connect(
        host=config["DB_HOST"],
        port=config["DB_PORT"],
        dbname=config["DB_NAME"],
        user=config["DB_USER"],
        password=config["DB_PASSWORD"],
    )


def _test_database(config: dict) -> bool:
    """Test PostgreSQL connectivity. Params: config. Returns: bool."""
    try:
        conn = _open_db(config)
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        conn.close()
        print("  Database connection: OK")
        return True
    except psycopg2.Error as exc:
        print(f"  Database connection FAILED: {exc}")
        return False


def _test_llm() -> bool:
    """Test LLM connectivity via the retriever client.

    Mirrors retriever.stages.startup.run_startup: loads config,
    installs the rbc_security corporate CA bundle (if available)
    via setup_ssl, then exercises the LLMClient health check.
    Without setup_ssl, the OpenAI HTTPS handshake against the
    work LLM endpoint fails on machines that need the corporate
    CA injected.

    Returns:
        True if the connection test succeeds
    """
    load_config()
    setup_ssl()
    try:
        llm = LLMClient()
        llm.test_connection()
        print("  LLM connection: OK")
        return True
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"  LLM connection FAILED: {exc}")
        return False


def _ensure_schema(conn) -> None:
    """Create schema/tables/indexes via the ingestion connector.

    Params:
        conn: psycopg2 connection
    """
    print("Creating schema, tables, and indexes...")
    ensure_schema_objects(conn)
    print("Schema ready")


def _handle_seed_data(conn, schema: str, schema_was_fresh: bool) -> None:
    """Offer to load seed data based on whether tables pre-existed.

    Params:
        conn: psycopg2 connection
        schema: PostgreSQL schema name
        schema_was_fresh: True if no pipeline tables existed before setup
    """
    if not seed_data.seed_files_present():
        print("No seed files in scripts/seed-data — skipping seed step")
        return

    _banner("Seed data")
    counts = seed_data.count_rows(conn, schema)
    total = sum(counts.values())

    if schema_was_fresh or total == 0:
        print("Tables are empty (newly created).")
        if _ask_yes_no("Load seed data into the database?", default=True):
            seed_data.load(conn, schema)
            print("Seed data loaded")
        else:
            print("Skipped seed load")
        return

    print("Tables already contain data:")
    for table, count in counts.items():
        print(f"  {schema}.{table}: {count} rows")
    print(
        "\nYou can wipe these tables and reload them from "
        "scripts/seed-data, or leave them alone."
    )
    if _ask_yes_no(
        "Wipe existing tables and reload from seed data?", default=False
    ):
        seed_data.load(conn, schema)
        print("Seed data reloaded")
    else:
        print("Existing data preserved")


def _launch_debug_server() -> int:
    """Launch the u-debug Flask server and open the browser.

    Returns:
        process exit code from the debug server
    """
    debug_dir = PROJECT_ROOT / "u-debug"
    port = int(os.environ.get("DEBUG_PORT", "5001"))
    url = f"http://localhost:{port}"

    _banner("Launching debug server")
    print(f"Starting Flask server at {url}")
    print("Press Ctrl-C to stop.\n")

    def _open_browser() -> None:
        time.sleep(2.0)
        try:
            webbrowser.open(url)
        except (OSError, RuntimeError):
            pass

    threading.Thread(target=_open_browser, daemon=True).start()
    result = subprocess.run(
        [str(_venv_python()), "-m", "src.debug.app"],
        cwd=str(debug_dir),
        check=False,
    )
    return result.returncode


def main() -> None:
    """Run phase 2 of the setup wizard. Returns: None."""
    _banner("Phase 2: Validation & Database")
    config = _load_env()

    if not _test_database(config):
        print("\nFix database settings in u-ingestion/.env and re-run.")
        sys.exit(1)

    if not _test_llm():
        print("\nLLM test failed — continuing, but the pipeline")
        print("will not work until LLM credentials are fixed.")
        if not _ask_yes_no("Continue anyway?", default=False):
            sys.exit(1)

    schema = config["DB_SCHEMA"]
    conn = _open_db(config)
    try:
        schema_was_fresh = not seed_data.schema_has_any_tables(conn, schema)
        _ensure_schema(conn)
        _handle_seed_data(conn, schema, schema_was_fresh)
    finally:
        conn.close()

    _banner("Setup complete")
    sys.exit(_launch_debug_server())


if __name__ == "__main__":
    main()
