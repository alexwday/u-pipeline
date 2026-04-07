"""Tests for PostgreSQL helpers."""

from types import SimpleNamespace

import pytest

from ingestion.utils import postgres_connector


class DummyCursor:
    """Context-managed cursor test double."""

    def __init__(
        self,
        *,
        rows=None,
        row=None,
        error=None,
        executed=None,
    ):
        self._rows = rows or []
        self._row = row
        self._error = error
        self.executed = executed if executed is not None else []

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """Do not suppress exceptions."""
        return False

    def execute(self, sql, params=None):
        """Record executed SQL and raise configured errors."""
        self.executed.append((sql, params))
        if self._error is not None:
            raise self._error

    def fetchall(self):
        """Return preconfigured rows."""
        return self._rows

    def fetchone(self):
        """Return a single preconfigured row."""
        return self._row


def make_connection(cursor_factory):
    """Create a connection object with cursor and commit methods."""
    committed = []
    return SimpleNamespace(
        cursor=cursor_factory,
        commit=lambda: committed.append(True),
        committed=committed,
    )


def test_get_connection(monkeypatch):
    """Pass config through to psycopg2.connect."""
    calls = []
    monkeypatch.setattr(
        postgres_connector,
        "get_database_config",
        lambda: {"host": "localhost"},
    )
    monkeypatch.setattr(
        postgres_connector.psycopg2,
        "connect",
        lambda **kwargs: calls.append(kwargs) or "connection",
    )

    assert postgres_connector.get_connection() == "connection"
    assert calls == [{"host": "localhost"}]


def test_ensure_schema_objects(monkeypatch):
    """Create the schema and all required tables when missing."""
    executed = []
    conn = make_connection(lambda: DummyCursor(executed=executed))
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    postgres_connector.ensure_schema_objects(conn)

    assert "CREATE EXTENSION IF NOT EXISTS vector" in executed[0][0]
    assert "CREATE SCHEMA IF NOT EXISTS u_pipeline;" in executed[1][0]
    assert (
        "CREATE TABLE IF NOT EXISTS u_pipeline.document_catalog"
        in executed[2][0]
    )
    assert (
        "CREATE TABLE IF NOT EXISTS u_pipeline.document_versions"
        in executed[3][0]
    )
    assert (
        "CREATE TABLE IF NOT EXISTS u_pipeline.stage_checkpoints"
        in executed[4][0]
    )
    assert (
        "CREATE TABLE IF NOT EXISTS u_pipeline.document_metadata"
        in executed[5][0]
    )
    assert (
        "has_toc             BOOLEAN NOT NULL DEFAULT FALSE" in executed[5][0]
    )
    assert "source_toc          JSONB NOT NULL" in executed[5][0]
    assert "generated_toc       JSONB NOT NULL" in executed[5][0]
    assert "rationale           TEXT NOT NULL DEFAULT ''" in executed[5][0]
    assert (
        "CREATE TABLE IF NOT EXISTS u_pipeline.document_sections"
        in executed[6][0]
    )
    assert (
        "CREATE TABLE IF NOT EXISTS u_pipeline.document_content"
        in executed[7][0]
    )
    assert "content_unit_id     TEXT NOT NULL" in executed[7][0]
    assert "UNIQUE (document_version_id, content_unit_id)" in executed[7][0]
    assert "ADD COLUMN IF NOT EXISTS started_at" in executed[8][0]
    assert "ADD COLUMN IF NOT EXISTS has_toc" in executed[9][0]
    assert "ADD COLUMN IF NOT EXISTS source_toc" in executed[10][0]
    assert "ADD COLUMN IF NOT EXISTS generated_toc" in executed[11][0]
    assert "ADD COLUMN IF NOT EXISTS rationale" in executed[12][0]
    assert conn.committed == [True]


def test_verify_connection_success_and_failure():
    """Run a trivial query to verify the database connection."""
    executed = []
    success_conn = make_connection(
        lambda: DummyCursor(executed=executed),
    )

    assert postgres_connector.verify_connection(success_conn) is True
    assert executed == [("SELECT 1;", None)]

    failure_conn = make_connection(
        lambda: DummyCursor(error=RuntimeError("db down")),
    )
    with pytest.raises(RuntimeError, match="db down"):
        postgres_connector.verify_connection(failure_conn)


def test_fetch_catalog_records(monkeypatch):
    """Map catalog rows into FileRecord instances."""
    monkeypatch.setenv("ACCEPTED_FILETYPES", "pdf,xlsx")
    executed = []
    rows = [
        (
            "policy",
            "2026",
            "Q1",
            "RBC",
            "deck.pdf",
            "pdf",
            10,
            100.0,
            "hash",
            "/tmp/deck.pdf",
        )
    ]
    conn = make_connection(lambda: DummyCursor(rows=rows, executed=executed))
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    records = postgres_connector.fetch_catalog_records(conn)

    assert len(records) == 1
    assert records[0].filename == "deck.pdf"
    assert records[0].supported is True
    assert "FROM u_pipeline.document_catalog" in executed[0][0]


def test_upsert_catalog_record(monkeypatch, file_record_factory):
    """Write catalog rows with an upsert query and commit."""
    executed = []
    conn = make_connection(lambda: DummyCursor(executed=executed))
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    postgres_connector.upsert_catalog_record(conn, file_record_factory())

    assert "INSERT INTO u_pipeline.document_catalog" in executed[0][0]
    assert conn.committed == [True]


def test_remove_deleted_files(monkeypatch):
    """Deactivate current versions and delete catalog rows together."""
    executed = []
    conn = make_connection(lambda: DummyCursor(executed=executed))
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    postgres_connector.remove_deleted_files(
        conn,
        ["/tmp/b.pdf", "/tmp/a.pdf", "/tmp/b.pdf"],
    )

    assert "UPDATE u_pipeline.document_versions" in executed[0][0]
    assert "DELETE FROM u_pipeline.document_catalog" in executed[1][0]
    assert executed[0][1] == (["/tmp/a.pdf", "/tmp/b.pdf"],)
    assert executed[1][1] == (["/tmp/a.pdf", "/tmp/b.pdf"],)
    assert conn.committed == [True]

    postgres_connector.remove_deleted_files(conn, [])

    assert len(executed) == 2
    assert conn.committed == [True]


def test_register_document_version(monkeypatch, file_record_factory):
    """Create or refresh the current document version row."""
    executed = []
    row = (
        7,
        "/tmp/doc.pdf",
        "source",
        "",
        "",
        "",
        "doc.pdf",
        "pdf",
        10,
        100.0,
        "hash-123",
        True,
    )
    conn = make_connection(
        lambda: DummyCursor(row=row, executed=executed),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    version = postgres_connector.register_document_version(
        conn,
        file_record_factory(file_hash="hash-123"),
    )

    assert version.document_version_id == 7
    assert version.file_hash == "hash-123"
    assert "UPDATE u_pipeline.document_versions" in executed[0][0]
    assert "INSERT INTO u_pipeline.document_versions" in executed[1][0]
    assert conn.committed == [True]


def test_fetch_current_document_versions(monkeypatch):
    """Map current document version rows into dataclasses."""
    executed = []
    rows = [
        (
            3,
            "/tmp/doc.pdf",
            "source",
            "",
            "",
            "",
            "doc.pdf",
            "pdf",
            10,
            100.0,
            "hash-123",
            True,
        )
    ]
    conn = make_connection(lambda: DummyCursor(rows=rows, executed=executed))
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    versions = postgres_connector.fetch_current_document_versions(conn)

    assert len(versions) == 1
    assert versions[0].document_version_id == 3
    assert "FROM u_pipeline.document_versions" in executed[0][0]


def test_fetch_stage_checkpoints(monkeypatch):
    """Map persisted stage checkpoint rows into dataclasses."""
    executed = []
    rows = [
        (
            3,
            "extraction",
            "succeeded",
            "sig",
            "/tmp/extraction.json",
            "checksum",
            "",
        )
    ]
    conn = make_connection(lambda: DummyCursor(rows=rows, executed=executed))
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    checkpoints = postgres_connector.fetch_stage_checkpoints(conn)

    assert len(checkpoints) == 1
    assert checkpoints[0].stage_name == "extraction"
    assert "FROM u_pipeline.stage_checkpoints" in executed[0][0]


def test_fetch_prunable_document_versions(monkeypatch):
    """Group stale version rows and collect their artifact paths."""
    executed = []
    rows = [
        (9, "/tmp/doc.pdf", "/tmp/artifacts/a/extraction.json"),
        (9, "/tmp/doc.pdf", "/tmp/artifacts/a/tokenization.json"),
        (10, "/tmp/other.pdf", ""),
    ]
    conn = make_connection(lambda: DummyCursor(rows=rows, executed=executed))
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    versions = postgres_connector.fetch_prunable_document_versions(conn, 2)

    assert len(versions) == 2
    assert versions[0].document_version_id == 9
    assert versions[0].file_path == "/tmp/doc.pdf"
    assert versions[0].artifact_paths == [
        "/tmp/artifacts/a/extraction.json",
        "/tmp/artifacts/a/tokenization.json",
    ]
    assert versions[1].document_version_id == 10
    assert versions[1].artifact_paths == []
    assert executed[0][1] == (3,)
    assert "ROW_NUMBER() OVER" in executed[0][0]


def test_delete_document_versions(monkeypatch):
    """Delete stale document versions in one statement and commit."""
    executed = []
    conn = make_connection(lambda: DummyCursor(executed=executed))
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    postgres_connector.delete_document_versions(conn, [3, 4])

    assert "DELETE FROM u_pipeline.document_versions" in executed[0][0]
    assert executed[0][1] == ([3, 4],)
    assert conn.committed == [True]

    postgres_connector.delete_document_versions(conn, [])

    assert len(executed) == 1
    assert conn.committed == [True]


def test_clear_stage_checkpoints(monkeypatch):
    """Delete the requested stage and all downstream checkpoints."""
    executed = []
    conn = make_connection(lambda: DummyCursor(executed=executed))
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_downstream_stages",
        lambda _stage_name: ("tokenization",),
    )

    postgres_connector.clear_stage_checkpoints(conn, 9, "tokenization")

    assert "DELETE FROM u_pipeline.stage_checkpoints" in executed[0][0]
    assert executed[0][1] == (9, ["tokenization"])
    assert conn.committed == [True]


def test_mark_stage_checkpoint_started(monkeypatch):
    """Insert a running checkpoint row before a stage begins."""
    executed = []
    conn = make_connection(lambda: DummyCursor(executed=executed))
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    postgres_connector.mark_stage_checkpoint_started(
        conn,
        5,
        "extraction",
        "sig",
    )

    assert "INSERT INTO u_pipeline.stage_checkpoints" in executed[0][0]
    assert "'running'" in executed[0][0]
    assert "started_at" in executed[0][0]
    assert executed[0][1] == (5, "extraction", "sig")
    assert conn.committed == [True]


def test_mark_stage_checkpoint_succeeded_and_failed(monkeypatch):
    """Upsert stage checkpoint success and failure states."""
    executed = []
    conn = make_connection(lambda: DummyCursor(executed=executed))
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    postgres_connector.mark_stage_checkpoint_succeeded(
        conn,
        5,
        "extraction",
        "sig",
        "/tmp/extraction.json",
        "checksum",
    )
    postgres_connector.mark_stage_checkpoint_failed(
        conn,
        5,
        "tokenization",
        "sig-2",
        "boom",
    )

    assert "INSERT INTO u_pipeline.stage_checkpoints" in executed[0][0]
    assert executed[0][1] == (
        5,
        "extraction",
        "sig",
        "/tmp/extraction.json",
        "checksum",
    )
    assert executed[1][1] == (5, "tokenization", "sig-2", "boom")
    assert conn.committed == [True, True]
