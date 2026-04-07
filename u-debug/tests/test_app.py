"""Tests for the Flask app factory and root route."""

from debug.app import create_app


def test_create_app_returns_flask_instance(monkeypatch):
    """App factory returns a valid Flask app."""
    monkeypatch.setenv("RETRIEVAL_TRACE_ROOT", "/tmp/traces")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "test")
    monkeypatch.setenv("DB_USER", "test")
    monkeypatch.setenv("DB_PASSWORD", "")
    monkeypatch.setenv("DB_SCHEMA", "public")

    app = create_app()
    assert app is not None
    assert app.name == "debug.app"


def test_index_serves_html(client):
    """Root route serves the index.html static file."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"U-Pipeline Debug Interface" in response.data


def test_static_css_served(client):
    """CSS file is accessible via /static."""
    response = client.get("/static/style.css")
    assert response.status_code == 200
    assert b"--bg-primary" in response.data


def test_static_js_served(client):
    """JS file is accessible via /static."""
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert b"runQuery" in response.data
