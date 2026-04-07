"""Shared fixtures for the debug test suite."""

import pytest

from debug.app import create_app


@pytest.fixture(name="app")
def fixture_app(monkeypatch):
    """Create a test Flask app with retriever imports mocked."""
    monkeypatch.setenv("RETRIEVAL_TRACE_ROOT", "/tmp/traces")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "test")
    monkeypatch.setenv("DB_USER", "test")
    monkeypatch.setenv("DB_PASSWORD", "")
    monkeypatch.setenv("DB_SCHEMA", "public")
    return create_app()


@pytest.fixture(name="client")
def fixture_client(app):
    """Create a Flask test client. Params: app. Returns: FlaskClient."""
    return app.test_client()
