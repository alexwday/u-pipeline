"""Test query and bank/period endpoints."""

from pathlib import Path

import yaml
from flask import Blueprint, jsonify

from retriever.utils.config_setup import get_database_schema
from retriever.utils.postgres_connector import get_connection

queries_bp = Blueprint("queries", __name__)

_YAML_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "docs"
    / "test_queries.yaml"
)


@queries_bp.route("/test-queries")
def get_test_queries():
    """Load test cases from test_queries.yaml.

    Returns:
        JSON list of test case objects
    """
    with open(_YAML_PATH, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return jsonify(data.get("test_cases", []))


@queries_bp.route("/banks-periods")
def get_banks_periods():
    """Query distinct bank/period/source combos from the database.

    Returns:
        JSON with banks, periods, and sources arrays
    """
    schema = get_database_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT filter_2"
                f" FROM {schema}.document_versions"
                f" WHERE is_current = TRUE"
                f" ORDER BY filter_2"
            )
            banks = [row[0] for row in cur.fetchall()]

            cur.execute(
                f"SELECT DISTINCT filter_1"
                f" FROM {schema}.document_versions"
                f" WHERE is_current = TRUE"
                f" ORDER BY filter_1 DESC"
            )
            periods = [row[0] for row in cur.fetchall()]

            cur.execute(
                f"SELECT DISTINCT data_source"
                f" FROM {schema}.document_versions"
                f" WHERE is_current = TRUE"
                f" ORDER BY data_source"
            )
            sources = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

    return jsonify(
        {
            "banks": banks,
            "periods": periods,
            "sources": sources,
        }
    )
