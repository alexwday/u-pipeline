"""Shared fixtures for retriever tests."""

import pytest


@pytest.fixture(autouse=True)
def _set_env(monkeypatch, tmp_path):
    """Set default env vars for all tests."""
    monkeypatch.setenv("DB_SCHEMA", "u_pipeline")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
    monkeypatch.setenv(
        "RETRIEVAL_TRACE_ROOT",
        str(tmp_path / "retrieval-traces"),
    )
