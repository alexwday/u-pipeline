"""Tests for PostgreSQL query functions."""

from types import SimpleNamespace

import pytest

from retriever.utils import postgres_connector


class DummyCursor:
    """Context-managed cursor test double."""

    def __init__(
        self,
        *,
        rows=None,
        row=None,
        description=None,
        error=None,
        executed=None,
    ):
        self._rows = rows or []
        self._row = row
        self._description = description
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

    @property
    def description(self):
        """Return column description tuples."""
        return self._description

    def fetchall(self):
        """Return preconfigured rows."""
        return self._rows

    def fetchone(self):
        """Return a single preconfigured row."""
        return self._row


def make_connection(cursor_factory):
    """Create a connection with cursor and commit."""
    committed = []
    return SimpleNamespace(
        cursor=cursor_factory,
        commit=lambda: committed.append(True),
        committed=committed,
    )


# ------------------------------------------------------------------
# Connection
# ------------------------------------------------------------------


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
        lambda **kw: calls.append(kw) or "connection",
    )

    result = postgres_connector.get_connection()
    assert result == "connection"
    assert calls == [{"host": "localhost"}]


def test_verify_connection_success_and_failure():
    """Run a trivial query to verify connectivity."""
    executed = []
    success = make_connection(
        lambda: DummyCursor(executed=executed),
    )
    assert postgres_connector.verify_connection(success)
    assert executed == [("SELECT 1;", None)]

    failure = make_connection(
        lambda: DummyCursor(error=RuntimeError("down")),
    )
    with pytest.raises(RuntimeError, match="down"):
        postgres_connector.verify_connection(failure)


# ------------------------------------------------------------------
# format_vector
# ------------------------------------------------------------------


def test_format_vector():
    """Serialize float list to pgvector literal."""
    assert postgres_connector.format_vector([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"
    assert postgres_connector.format_vector([]) is None
    assert postgres_connector.format_vector(None) is None


# ------------------------------------------------------------------
# Document resolution
# ------------------------------------------------------------------


def _col_desc(*names):
    """Build a cursor description from column names."""
    return tuple((n,) for n in names)


_VERSION_COLS = _col_desc(
    "document_version_id",
    "data_source",
    "filter_1",
    "filter_2",
    "filename",
)


def test_resolve_document_version_ids(monkeypatch):
    """Find matching document versions for bank+period."""
    executed = []
    rows = [(38, "investor-slides", "2026_Q1", "RBC", "f.pdf")]
    conn = make_connection(
        lambda: DummyCursor(
            rows=rows,
            description=_VERSION_COLS,
            executed=executed,
        ),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    results = postgres_connector.resolve_document_version_ids(
        conn, "RBC", "2026_Q1"
    )

    assert len(results) == 1
    assert results[0]["document_version_id"] == 38
    assert "u_pipeline.document_versions" in executed[0][0]
    assert executed[0][1] == ["RBC", "2026_Q1"]


def test_resolve_document_version_ids_with_sources(
    monkeypatch,
):
    """Filter by data_source when sources list is given."""
    executed = []
    conn = make_connection(
        lambda: DummyCursor(
            rows=[],
            description=_VERSION_COLS,
            executed=executed,
        ),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    postgres_connector.resolve_document_version_ids(
        conn, "RBC", "2026_Q1", sources=["pillar3"]
    )

    assert "ANY(%s)" in executed[0][0]
    assert executed[0][1] == [
        "RBC",
        "2026_Q1",
        ["pillar3"],
    ]


# ------------------------------------------------------------------
# Document size
# ------------------------------------------------------------------


def test_get_document_total_tokens(monkeypatch):
    """Sum token counts for a document version."""
    executed = []
    conn = make_connection(
        lambda: DummyCursor(row=(12500,), executed=executed),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    total = postgres_connector.get_document_total_tokens(conn, 38)

    assert total == 12500
    assert "u_pipeline.document_content" in executed[0][0]


# ------------------------------------------------------------------
# Full document load
# ------------------------------------------------------------------

_CONTENT_COLS = _col_desc(
    "content_unit_id",
    "chunk_id",
    "section_id",
    "page_number",
    "parent_page_number",
    "raw_content",
    "chunk_context",
    "chunk_header",
    "keywords",
    "entities",
    "token_count",
)


def test_load_full_document(monkeypatch):
    """Load all content for a document version."""
    executed = []
    rows = [
        (
            "cu_001",
            "ch_001",
            "s_001",
            1,
            None,
            "text",
            "ctx",
            "hdr",
            ["kw"],
            ["ent"],
            25,
        )
    ]
    conn = make_connection(
        lambda: DummyCursor(
            rows=rows,
            description=_CONTENT_COLS,
            executed=executed,
        ),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    results = postgres_connector.load_full_document(conn, 38)

    assert len(results) == 1
    assert results[0]["content_unit_id"] == "cu_001"
    assert executed[0][1] == (38,)


# ------------------------------------------------------------------
# Vector search
# ------------------------------------------------------------------

_SEARCH_COLS = _col_desc(
    "content_unit_id",
    "chunk_id",
    "section_id",
    "page_number",
    "raw_content",
    "chunk_context",
    "chunk_header",
    "keywords",
    "entities",
    "token_count",
    "distance",
)


def test_search_by_content_vector(monkeypatch):
    """Cosine search on content embeddings."""
    executed = []
    rows = [
        (
            "cu_001",
            "ch_001",
            "s_001",
            1,
            "text",
            "ctx",
            "hdr",
            [],
            [],
            25,
            0.15,
        )
    ]
    conn = make_connection(
        lambda: DummyCursor(
            rows=rows,
            description=_SEARCH_COLS,
            executed=executed,
        ),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    results = postgres_connector.search_by_content_vector(
        conn, 38, [0.1, 0.2], top_k=10
    )

    assert len(results) == 1
    assert results[0]["distance"] == 0.15
    assert "content_embedding" in executed[0][0]
    assert executed[0][1] == ("[0.1,0.2]", 38, 10)


def test_search_by_keyword_vector(monkeypatch):
    """Cosine search on keyword embeddings."""
    executed = []
    conn = make_connection(
        lambda: DummyCursor(
            rows=[],
            description=_SEARCH_COLS,
            executed=executed,
        ),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    postgres_connector.search_by_keyword_vector(conn, 38, [0.5], top_k=5)

    assert "keyword_embedding" in executed[0][0]
    assert executed[0][1] == ("[0.5]", 38, 5)


_SECTION_SEARCH_COLS = _col_desc(
    "section_id",
    "title",
    "summary",
    "distance",
)


def test_search_by_section_summary(monkeypatch):
    """Cosine search on section summary embeddings."""
    executed = []
    rows = [("s_003", "Capital", "Summary text", 0.2)]
    conn = make_connection(
        lambda: DummyCursor(
            rows=rows,
            description=_SECTION_SEARCH_COLS,
            executed=executed,
        ),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    results = postgres_connector.search_by_section_summary(
        conn, 38, [0.1], top_k=5
    )

    assert results[0]["section_id"] == "s_003"
    assert "document_sections" in executed[0][0]


# ------------------------------------------------------------------
# BM25
# ------------------------------------------------------------------

_BM25_COLS = _col_desc(
    "content_unit_id",
    "chunk_id",
    "section_id",
    "page_number",
    "raw_content",
    "chunk_context",
    "chunk_header",
    "keywords",
    "entities",
    "token_count",
    "rank",
)


def test_search_by_bm25(monkeypatch):
    """Full-text search using tsvector ranking."""
    executed = []
    rows = [
        (
            "cu_005",
            "ch_005",
            "s_002",
            3,
            "CET1 ratio",
            "ctx",
            "hdr",
            [],
            [],
            15,
            0.8,
        )
    ]
    conn = make_connection(
        lambda: DummyCursor(
            rows=rows,
            description=_BM25_COLS,
            executed=executed,
        ),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    results = postgres_connector.search_by_bm25(
        conn, 38, "CET1 ratio", top_k=10
    )

    assert len(results) == 1
    assert results[0]["rank"] == 0.8
    assert "content_tsvector" in executed[0][0]
    assert executed[0][1] == ("CET1 ratio", 38, "CET1 ratio", 10)


# ------------------------------------------------------------------
# Array containment
# ------------------------------------------------------------------

_CONTAINMENT_COLS = _col_desc(
    "content_unit_id",
    "chunk_id",
    "section_id",
    "page_number",
    "raw_content",
    "chunk_context",
    "chunk_header",
    "keywords",
    "entities",
    "token_count",
)


def test_search_by_keyword_containment(monkeypatch):
    """Find chunks matching keyword array overlap."""
    executed = []
    rows = [
        (
            "cu_010",
            "ch_010",
            "s_005",
            7,
            "text",
            "ctx",
            "hdr",
            ["CET1"],
            [],
            20,
        )
    ]
    conn = make_connection(
        lambda: DummyCursor(
            rows=rows,
            description=_CONTAINMENT_COLS,
            executed=executed,
        ),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    results = postgres_connector.search_by_keyword_containment(
        conn, 38, ["CET1", "Tier 1"], limit=25
    )

    assert len(results) == 1
    assert "jsonb_array_elements_text(keywords)" in (executed[0][0])
    assert "ILIKE" in executed[0][0]
    assert executed[0][1] == (
        38,
        ["CET1", "Tier 1"],
        ["CET1", "Tier 1"],
        25,
    )


def test_search_by_entity_containment(monkeypatch):
    """Find chunks matching entity array overlap."""
    executed = []
    conn = make_connection(
        lambda: DummyCursor(
            rows=[],
            description=_CONTAINMENT_COLS,
            executed=executed,
        ),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    postgres_connector.search_by_entity_containment(
        conn, 38, ["RBC"], limit=10
    )

    assert "jsonb_array_elements_text(entities)" in (executed[0][0])
    assert "ILIKE" in executed[0][0]
    assert executed[0][1] == (38, ["RBC"], ["RBC"], 10)


# ------------------------------------------------------------------
# Expansion
# ------------------------------------------------------------------


def test_load_section_content(monkeypatch):
    """Load all content units in a section."""
    executed = []
    rows = [
        (
            "cu_010",
            "ch_010",
            "s_003",
            5,
            None,
            "text",
            "ctx",
            "hdr",
            [],
            [],
            25,
        )
    ]
    conn = make_connection(
        lambda: DummyCursor(
            rows=rows,
            description=_CONTENT_COLS,
            executed=executed,
        ),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    results = postgres_connector.load_section_content(conn, 38, "s_003")

    assert len(results) == 1
    assert executed[0][1] == (38, "s_003")


_SECTION_INFO_COLS = _col_desc(
    "section_id",
    "level",
    "title",
    "parent_section_id",
    "token_count",
    "page_start",
    "page_end",
    "summary",
)


def test_get_section_info(monkeypatch):
    """Retrieve section metadata by section_id."""
    executed = []
    row = (
        "s_003",
        "section",
        "Capital",
        None,
        500,
        5,
        8,
        "Summary",
    )
    conn = make_connection(
        lambda: DummyCursor(
            row=row,
            description=_SECTION_INFO_COLS,
            executed=executed,
        ),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    result = postgres_connector.get_section_info(conn, 38, "s_003")

    assert result["title"] == "Capital"
    assert "document_sections" in executed[0][0]


def test_get_section_info_not_found(monkeypatch):
    """Return None when section does not exist."""
    conn = make_connection(
        lambda: DummyCursor(
            row=None,
            description=_SECTION_INFO_COLS,
        ),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    assert postgres_connector.get_section_info(conn, 38, "missing") is None


_CU_SECTION_COLS = _col_desc("section_id")


def test_get_section_for_content(monkeypatch):
    """Look up section info via content unit."""
    call_count = [0]

    def cursor_factory():
        call_count[0] += 1
        if call_count[0] == 1:
            return DummyCursor(
                row=("s_003",),
                description=_CU_SECTION_COLS,
            )
        return DummyCursor(
            row=(
                "s_003",
                "section",
                "Capital",
                None,
                500,
                5,
                8,
                "Summary",
            ),
            description=_SECTION_INFO_COLS,
        )

    conn = make_connection(cursor_factory)
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    result = postgres_connector.get_section_for_content(conn, 38, "cu_010")

    assert result["title"] == "Capital"


def test_get_section_for_content_not_found(monkeypatch):
    """Return None when content unit has no section."""
    conn = make_connection(
        lambda: DummyCursor(row=None, description=_CU_SECTION_COLS),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    assert (
        postgres_connector.get_section_for_content(conn, 38, "missing") is None
    )


def test_get_section_for_content_empty_section_id(
    monkeypatch,
):
    """Return None when section_id is empty string."""
    conn = make_connection(
        lambda: DummyCursor(row=("",), description=_CU_SECTION_COLS),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    assert (
        postgres_connector.get_section_for_content(conn, 38, "cu_nosect")
        is None
    )


def test_load_neighbor_chunks(monkeypatch):
    """Load chunks surrounding a content unit."""
    executed = []
    rows = [
        (
            "cu_009",
            "ch_009",
            "s_003",
            4,
            None,
            "before",
            "ctx",
            "hdr",
            [],
            [],
            20,
        ),
        (
            "cu_010",
            "ch_010",
            "s_003",
            5,
            None,
            "center",
            "ctx",
            "hdr",
            [],
            [],
            25,
        ),
        (
            "cu_011",
            "ch_011",
            "s_003",
            5,
            None,
            "after",
            "ctx",
            "hdr",
            [],
            [],
            22,
        ),
    ]
    conn = make_connection(
        lambda: DummyCursor(
            rows=rows,
            description=_CONTENT_COLS,
            executed=executed,
        ),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    results = postgres_connector.load_neighbor_chunks(
        conn, 38, "cu_010", count=1
    )

    assert len(results) == 3
    assert "ROW_NUMBER()" in executed[0][0]
    assert executed[0][1] == (38, "cu_010", 1, "cu_010", 1)


# ------------------------------------------------------------------
# Document metadata
# ------------------------------------------------------------------

_METADATA_COLS = _col_desc(
    "title",
    "authors",
    "executive_summary",
    "keywords",
    "entities",
    "structure_type",
)


def test_get_document_metadata(monkeypatch):
    """Retrieve document-level metadata."""
    row = (
        "RBC Q1 2026 Investor Slides",
        "RBC IR",
        "Executive summary text",
        ["capital", "risk"],
        ["RBC"],
        "presentation",
    )
    conn = make_connection(
        lambda: DummyCursor(row=row, description=_METADATA_COLS),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    result = postgres_connector.get_document_metadata(conn, 38)

    assert result["title"] == "RBC Q1 2026 Investor Slides"
    assert result["structure_type"] == "presentation"


def test_get_document_metadata_not_found(monkeypatch):
    """Return None when metadata does not exist."""
    conn = make_connection(
        lambda: DummyCursor(row=None, description=_METADATA_COLS),
    )
    monkeypatch.setattr(
        postgres_connector,
        "get_database_schema",
        lambda: "u_pipeline",
    )

    assert postgres_connector.get_document_metadata(conn, 999) is None
