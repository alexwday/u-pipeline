"""Tests for the enrichment persistence module."""

import json
from types import SimpleNamespace

import pytest

from ingestion.stages.enrichment import (
    persistence as mod,
)
from ingestion.utils import postgres_connector as pg
from ingestion.utils.file_types import (
    ExtractionResult,
    PageResult,
)

# ------------------------------------------------------------------
# Test doubles
# ------------------------------------------------------------------

FAKE_VERSION_ID = 42


class DummyCursor:
    """Context-managed cursor test double."""

    def __init__(
        self,
        *,
        rows=None,
        row=None,
        executed=None,
    ):
        """Set up preconfigured results and execution log."""
        self._rows = rows or []
        self._row = row
        self.executed = executed if executed is not None else []

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """Do not suppress exceptions."""
        return False

    def execute(self, sql, params=None):
        """Record executed SQL."""
        self.executed.append((sql, params))

    def fetchall(self):
        """Return preconfigured rows."""
        return self._rows

    def fetchone(self):
        """Return a single preconfigured row."""
        return self._row


def _make_connection(cursor_factory):
    """Build a connection with cursor and commit.

    Params: cursor_factory. Returns: SimpleNamespace.
    """
    committed = []
    closed = []
    return SimpleNamespace(
        cursor=cursor_factory,
        commit=lambda: committed.append(True),
        close=lambda: closed.append(True),
        committed=committed,
        closed=closed,
    )


def _make_page(
    page_number=1,
    raw_content="# Heading\nText",
    **overrides,
):
    """Build a PageResult for testing.

    Params: page_number, raw_content. Returns: PageResult.
    """
    page = PageResult(
        page_number=page_number,
        raw_content=raw_content,
    )
    for key, value in overrides.items():
        setattr(page, key, value)
    return page


def _make_result(
    pages=None,
    sections=None,
    metadata=None,
    content_units=None,
    file_path="/tmp/data/source/doc.pdf",
):
    """Build an ExtractionResult for testing.

    Params: pages, sections, metadata, content_units,
        file_path.
    Returns: ExtractionResult.
    """
    if pages is None:
        pages = [_make_page()]
    result = ExtractionResult(
        file_path=file_path,
        filetype="pdf",
        pages=pages,
        total_pages=len(pages),
    )
    if sections is not None:
        result.sections = sections
    if metadata is not None:
        result.document_metadata = metadata
    if content_units is not None:
        result.content_units = content_units
    return result


def _vec(seed=1, dim=4):
    """Build a fake embedding vector.

    Params: seed, dim. Returns: list[float].
    """
    return [float(seed)] * dim


# ------------------------------------------------------------------
# test_persist_enrichment_inserts_metadata
# ------------------------------------------------------------------


def test_persist_enrichment_inserts_metadata(
    monkeypatch,
):
    """Metadata row inserted with correct fields."""
    executed = []
    conn = _make_connection(
        lambda: DummyCursor(row=(FAKE_VERSION_ID,), executed=executed),
    )
    monkeypatch.setattr(mod, "get_connection", lambda: conn)
    monkeypatch.setattr(
        mod,
        "fetch_current_version_id",
        lambda _c, _p: FAKE_VERSION_ID,
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    metadata = {
        "title": "Q4 Report",
        "executive_summary": "Strong quarter.",
        "summary_embedding": _vec(1),
        "keywords": ["CET1"],
        "entities": ["OSFI"],
    }
    result = _make_result(metadata=metadata, content_units=[])

    returned = mod.persist_enrichment(result, None)

    assert returned is result
    insert_calls = [
        (sql, params)
        for sql, params in executed
        if "INSERT INTO" in sql and "document_metadata" in sql
    ]
    assert len(insert_calls) == 1
    sql, params = insert_calls[0]
    assert "summary_embedding" in sql
    assert params[0] == FAKE_VERSION_ID
    assert params[2] == "Q4 Report"


# ------------------------------------------------------------------
# test_persist_enrichment_inserts_sections
# ------------------------------------------------------------------


def test_persist_enrichment_inserts_sections(
    monkeypatch,
):
    """All sections inserted including subsections."""
    executed = []
    conn = _make_connection(
        lambda: DummyCursor(row=(FAKE_VERSION_ID,), executed=executed),
    )
    monkeypatch.setattr(mod, "get_connection", lambda: conn)
    monkeypatch.setattr(
        mod,
        "fetch_current_version_id",
        lambda _c, _p: FAKE_VERSION_ID,
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    sections = [
        {
            "section_id": "1",
            "level": "section",
            "title": "KM1",
            "summary": "Key metrics.",
            "summary_embedding": _vec(1),
            "keyword_embedding": _vec(2),
            "entity_embedding": _vec(3),
            "keywords": ["CET1"],
            "entities": ["OSFI"],
        },
        {
            "section_id": "1.1",
            "level": "subsection",
            "title": "Sub",
        },
    ]
    result = _make_result(sections=sections, content_units=[])

    mod.persist_enrichment(result, None)

    section_inserts = [
        (sql, params)
        for sql, params in executed
        if "INSERT INTO" in sql and "document_sections" in sql
    ]
    assert len(section_inserts) == 2


# ------------------------------------------------------------------
# test_persist_enrichment_inserts_content
# ------------------------------------------------------------------


def test_persist_enrichment_inserts_content(monkeypatch):
    """All content units inserted with embeddings."""
    executed = []
    conn = _make_connection(
        lambda: DummyCursor(row=(FAKE_VERSION_ID,), executed=executed),
    )
    monkeypatch.setattr(mod, "get_connection", lambda: conn)
    monkeypatch.setattr(
        mod,
        "fetch_current_version_id",
        lambda _c, _p: FAKE_VERSION_ID,
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    units = [
        {
            "content_unit_id": "1",
            "page_number": 1,
            "raw_content": "Revenue grew 5%.",
            "content_embedding": _vec(1),
            "keyword_embedding": _vec(2),
            "entity_embedding": _vec(3),
            "keywords": ["revenue"],
            "entities": ["RBC"],
        },
        {
            "content_unit_id": "2",
            "page_number": 2,
            "raw_content": "CET1 ratio 13.5%.",
            "content_embedding": _vec(4),
            "keyword_embedding": _vec(5),
            "entity_embedding": _vec(6),
            "keywords": ["CET1"],
            "entities": [],
        },
    ]
    result = _make_result(content_units=units)

    mod.persist_enrichment(result, None)

    content_inserts = [
        (sql, params)
        for sql, params in executed
        if "INSERT INTO" in sql and "document_content" in sql
    ]
    assert len(content_inserts) == 2
    sql = content_inserts[0][0]
    assert "content_embedding" in sql
    assert "to_tsvector" in sql


# ------------------------------------------------------------------
# test_persist_enrichment_deletes_existing
# ------------------------------------------------------------------


def test_persist_enrichment_deletes_existing(
    monkeypatch,
):
    """Old enrichment deleted before insert for idempotency."""
    executed = []
    conn = _make_connection(
        lambda: DummyCursor(row=(FAKE_VERSION_ID,), executed=executed),
    )
    monkeypatch.setattr(mod, "get_connection", lambda: conn)
    monkeypatch.setattr(
        mod,
        "fetch_current_version_id",
        lambda _c, _p: FAKE_VERSION_ID,
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    result = _make_result(content_units=[])

    mod.persist_enrichment(result, None)

    delete_calls = [
        (sql, params) for sql, params in executed if "DELETE FROM" in sql
    ]
    assert len(delete_calls) == 3
    tables = [sql for sql, _ in delete_calls]
    assert any("document_content" in t for t in tables)
    assert any("document_sections" in t for t in tables)
    assert any("document_metadata" in t for t in tables)


# ------------------------------------------------------------------
# test_persist_enrichment_handles_empty_embeddings
# ------------------------------------------------------------------


def test_persist_enrichment_handles_empty_embeddings(
    monkeypatch,
):
    """NULL embeddings for empty vectors."""
    executed = []
    conn = _make_connection(
        lambda: DummyCursor(row=(FAKE_VERSION_ID,), executed=executed),
    )
    monkeypatch.setattr(mod, "get_connection", lambda: conn)
    monkeypatch.setattr(
        mod,
        "fetch_current_version_id",
        lambda _c, _p: FAKE_VERSION_ID,
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    units = [
        {
            "content_unit_id": "1",
            "page_number": 1,
            "raw_content": "Some text.",
            "content_embedding": [],
            "keyword_embedding": [],
            "entity_embedding": [],
        },
    ]
    result = _make_result(content_units=units)

    mod.persist_enrichment(result, None)

    content_inserts = [
        (sql, params)
        for sql, params in executed
        if "INSERT INTO" in sql and "document_content" in sql
    ]
    assert len(content_inserts) == 1
    params = content_inserts[0][1]
    content_emb_idx = 14
    assert params[content_emb_idx] is None
    assert params[content_emb_idx + 1] is None
    assert params[content_emb_idx + 2] is None


# ------------------------------------------------------------------
# test_persist_enrichment_generates_tsvector
# ------------------------------------------------------------------


def test_persist_enrichment_generates_tsvector(
    monkeypatch,
):
    """content_tsvector generated from raw_content in SQL."""
    executed = []
    conn = _make_connection(
        lambda: DummyCursor(row=(FAKE_VERSION_ID,), executed=executed),
    )
    monkeypatch.setattr(mod, "get_connection", lambda: conn)
    monkeypatch.setattr(
        mod,
        "fetch_current_version_id",
        lambda _c, _p: FAKE_VERSION_ID,
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    units = [
        {
            "content_unit_id": "1",
            "page_number": 1,
            "raw_content": "Revenue grew 5%.",
            "content_embedding": _vec(1),
            "keyword_embedding": [],
            "entity_embedding": [],
        },
    ]
    result = _make_result(content_units=units)

    mod.persist_enrichment(result, None)

    content_inserts = [
        (sql, params)
        for sql, params in executed
        if "INSERT INTO" in sql and "document_content" in sql
    ]
    sql = content_inserts[0][0]
    assert "to_tsvector('english', %s)" in sql
    params = content_inserts[0][1]
    assert params[-1] == "Revenue grew 5%."


# ------------------------------------------------------------------
# test_persist_enrichment_preserves_result
# ------------------------------------------------------------------


def test_persist_enrichment_preserves_result(
    monkeypatch,
):
    """ExtractionResult returned unchanged."""
    executed = []
    conn = _make_connection(
        lambda: DummyCursor(row=(FAKE_VERSION_ID,), executed=executed),
    )
    monkeypatch.setattr(mod, "get_connection", lambda: conn)
    monkeypatch.setattr(
        mod,
        "fetch_current_version_id",
        lambda _c, _p: FAKE_VERSION_ID,
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    result = _make_result(content_units=[])

    returned = mod.persist_enrichment(result, None)

    assert returned is result
    assert returned.file_path == result.file_path
    assert returned.pages == result.pages


# ------------------------------------------------------------------
# test_format_vector_helper
# ------------------------------------------------------------------


def test_format_vector_helper():
    """Converts list[float] to pgvector string format."""
    vec = [0.1, 0.2, 0.3]

    formatted = pg.format_vector(vec)

    assert formatted == "[0.1,0.2,0.3]"


# ------------------------------------------------------------------
# test_format_vector_empty
# ------------------------------------------------------------------


def test_format_vector_empty():
    """Returns None for empty list."""
    assert pg.format_vector([]) is None
    assert pg.format_vector(None) is None


# ------------------------------------------------------------------
# test_verify_enrichment_completeness
# ------------------------------------------------------------------


def test_verify_enrichment_completeness(monkeypatch):
    """Returns correct counts for each table."""
    call_index = {"idx": 0}
    counts = [1, 5, 20, 18, 4]

    class CountCursor(DummyCursor):
        """Cursor returning sequential count values."""

        def fetchone(self):
            """Return next count. Returns: tuple."""
            val = counts[call_index["idx"]]
            call_index["idx"] += 1
            return (val,)

    executed = []
    conn = _make_connection(
        lambda: CountCursor(executed=executed),
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    result = pg.verify_enrichment_completeness(conn, 42)

    assert result["metadata_exists"] is True
    assert result["section_count"] == 5
    assert result["content_count"] == 20
    assert result["content_with_embedding"] == 18
    assert result["sections_with_embedding"] == 4


# ------------------------------------------------------------------
# test_query_content_by_keywords
# ------------------------------------------------------------------


def test_query_content_by_keywords(monkeypatch):
    """Array containment query built correctly."""
    executed = []
    rows = [("1", "Revenue grew.", '["revenue"]')]
    conn = _make_connection(
        lambda: DummyCursor(rows=rows, executed=executed),
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    results = pg.query_content_by_keywords(conn, 42, ["revenue"])

    assert len(results) == 1
    assert results[0]["content_unit_id"] == "1"
    sql = executed[0][0]
    assert "keywords @>" in sql
    params = executed[0][1]
    assert params == (42, json.dumps(["revenue"]))


# ------------------------------------------------------------------
# test_query_content_by_text
# ------------------------------------------------------------------


def test_query_content_by_text(monkeypatch):
    """tsvector query built correctly."""
    executed = []
    rows = [("1", "Revenue grew 5%.", 0.5)]
    conn = _make_connection(
        lambda: DummyCursor(rows=rows, executed=executed),
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    results = pg.query_content_by_text(conn, 42, "revenue")

    assert len(results) == 1
    assert results[0]["rank"] == 0.5
    sql = executed[0][0]
    assert "plainto_tsquery" in sql
    assert "ts_rank" in sql
    params = executed[0][1]
    assert params == ("revenue", 42, "revenue")


# ------------------------------------------------------------------
# test_fetch_current_version_id
# ------------------------------------------------------------------


def test_fetch_current_version_id(monkeypatch):
    """Returns correct version_id from query."""
    executed = []
    conn = _make_connection(
        lambda: DummyCursor(row=(99,), executed=executed),
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    version_id = pg.fetch_current_version_id(conn, "/tmp/doc.pdf")

    assert version_id == 99
    sql = executed[0][0]
    assert "document_versions" in sql
    assert "is_current = TRUE" in sql
    assert executed[0][1] == ("/tmp/doc.pdf",)


# ------------------------------------------------------------------
# test_delete_enrichment
# ------------------------------------------------------------------


def test_delete_enrichment(monkeypatch):
    """Deletes from all 3 enrichment tables."""
    executed = []
    conn = _make_connection(
        lambda: DummyCursor(executed=executed),
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    pg.delete_enrichment(conn, 42)

    assert len(executed) == 3
    deleted_tables = [sql for sql, _ in executed]
    assert any("document_content" in t for t in deleted_tables)
    assert any("document_sections" in t for t in deleted_tables)
    assert any("document_metadata" in t for t in deleted_tables)
    for _, params in executed:
        assert params == (42,)
    assert conn.committed == [True]


# ------------------------------------------------------------------
# test_fetch_current_version_id_missing
# ------------------------------------------------------------------


def test_fetch_current_version_id_missing(monkeypatch):
    """Raises ValueError when no current version exists."""
    executed = []
    conn = _make_connection(
        lambda: DummyCursor(row=None, executed=executed),
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    with pytest.raises(ValueError, match="No current version found"):
        pg.fetch_current_version_id(conn, "/tmp/gone.pdf")


# ------------------------------------------------------------------
# test_query_content_by_vector
# ------------------------------------------------------------------


def test_query_content_by_vector(monkeypatch):
    """Cosine similarity query built correctly."""
    executed = []
    rows = [("1", "Revenue grew.", 0.12)]
    conn = _make_connection(
        lambda: DummyCursor(rows=rows, executed=executed),
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    results = pg.query_content_by_vector(conn, 42, [0.1, 0.2, 0.3], top_k=3)

    assert len(results) == 1
    assert results[0]["content_unit_id"] == "1"
    assert results[0]["distance"] == 0.12
    sql = executed[0][0]
    assert "content_embedding <=>" in sql
    assert "LIMIT" in sql
    params = executed[0][1]
    assert params == ("[0.1,0.2,0.3]", 42, 3)


# ------------------------------------------------------------------
# test_query_sections_by_summary
# ------------------------------------------------------------------


def test_query_sections_by_summary(monkeypatch):
    """Section summary cosine similarity query works."""
    executed = []
    rows = [("1", "KM1", "Key metrics.", 0.08)]
    conn = _make_connection(
        lambda: DummyCursor(rows=rows, executed=executed),
    )
    monkeypatch.setattr(pg, "get_database_schema", lambda: "u_pipeline")

    results = pg.query_sections_by_summary(conn, 42, [0.1, 0.2], top_k=2)

    assert len(results) == 1
    assert results[0]["section_id"] == "1"
    assert results[0]["title"] == "KM1"
    assert results[0]["distance"] == 0.08
    sql = executed[0][0]
    assert "summary_embedding <=>" in sql
    params = executed[0][1]
    assert params == ("[0.1,0.2]", 42, 2)
