"""Tests for context expansion stage."""

from retriever.models import SearchResult
from retriever.stages.expand import (
    _deduplicate_chunks,
    _find_subsection,
    _get_expansion_strategy,
    _numeric_cuid_sort_key,
    _to_expanded_chunk,
    expand_chunks,
)


def _make_search_result(
    cuid: str,
    section_id: str = "s_001",
    page: int = 1,
) -> SearchResult:
    """Build a SearchResult fixture."""
    return SearchResult(
        content_unit_id=cuid,
        raw_content=f"Content of {cuid}",
        chunk_id=f"ch_{cuid}",
        section_id=section_id,
        page_number=page,
        chunk_context="context",
        chunk_header="header",
        keywords=["kw1"],
        entities=["ent1"],
        token_count=50,
        score=0.8,
        strategy_scores={"bm25": 0.8},
    )


def _make_section_info(
    section_id: str,
    token_count: int = 500,
    title: str = "Capital",
    parent_section_id: str | None = None,
) -> dict:
    """Build a mock section info dict."""
    return {
        "section_id": section_id,
        "level": 1,
        "title": title,
        "parent_section_id": parent_section_id,
        "token_count": token_count,
        "page_start": 1,
        "page_end": 3,
        "summary": f"Summary of {section_id}",
    }


def _make_content_row(
    cuid: str,
    section_id: str = "s_001",
    page: int = 1,
    token_count: int = 50,
) -> dict:
    """Build a mock database content row."""
    return {
        "content_unit_id": cuid,
        "chunk_id": f"ch_{cuid}",
        "section_id": section_id,
        "page_number": page,
        "parent_page_number": None,
        "raw_content": f"Content of {cuid}",
        "chunk_context": "context",
        "chunk_header": "header",
        "keywords": ["kw1"],
        "entities": ["ent1"],
        "token_count": token_count,
    }


def _set_expand_env(monkeypatch):
    """Set expansion threshold env vars."""
    monkeypatch.setenv("EXPAND_SECTION_TOKEN_THRESHOLD", "3000")
    monkeypatch.setenv("EXPAND_SUBSECTION_TOKEN_THRESHOLD", "1500")
    monkeypatch.setenv("EXPAND_NEIGHBOR_COUNT", "2")


def test_expand_full_section(monkeypatch):
    """Small section loads all content."""
    _set_expand_env(monkeypatch)

    section = _make_section_info("s_001", token_count=500, title="KM1")
    content_rows = [
        _make_content_row("cu_1", page=1),
        _make_content_row("cu_2", page=1),
        _make_content_row("cu_3", page=2),
    ]

    monkeypatch.setattr(
        "retriever.stages.expand.get_section_for_content",
        lambda conn, dvid, cuid: section,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.load_section_content",
        lambda conn, dvid, sid: content_rows,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.find_child_section_for_content",
        lambda conn, dvid, sid, cuid: None,
    )

    results = [_make_search_result("cu_1")]
    metrics: dict = {}
    expanded = expand_chunks(None, 1, results, metrics=metrics)

    assert len(expanded) == 3
    cuids = [c["content_unit_id"] for c in expanded]
    assert "cu_1" in cuids
    assert "cu_2" in cuids
    assert "cu_3" in cuids
    assert metrics["strategy_counts"]["full_section"] == 1
    assert metrics["chunks_after"] == 3


def test_expand_propagates_score(monkeypatch):
    """Trigger chunk carries fusion score; context chunks get 0.0."""
    _set_expand_env(monkeypatch)

    section = _make_section_info("s_001", token_count=500, title="KM1")
    content_rows = [
        _make_content_row("cu_1", page=1),
        _make_content_row("cu_2", page=1),
        _make_content_row("cu_3", page=2),
    ]

    monkeypatch.setattr(
        "retriever.stages.expand.get_section_for_content",
        lambda conn, dvid, cuid: section,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.load_section_content",
        lambda conn, dvid, sid: content_rows,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.find_child_section_for_content",
        lambda conn, dvid, sid, cuid: None,
    )

    results = [_make_search_result("cu_1")]
    expanded = expand_chunks(None, 1, results)

    trigger = next(c for c in expanded if c["content_unit_id"] == "cu_1")
    context = [c for c in expanded if c["content_unit_id"] != "cu_1"]

    assert trigger["score"] == 0.8
    for chunk in context:
        assert chunk["score"] == 0.0


def test_expand_populates_trace(monkeypatch):
    """Expansion traces retain lineage from input hit to final chunks."""
    _set_expand_env(monkeypatch)

    section = _make_section_info("s_001", token_count=500, title="KM1")
    content_rows = [
        _make_content_row("cu_1", page=1),
        _make_content_row("cu_2", page=1),
    ]

    monkeypatch.setattr(
        "retriever.stages.expand.get_section_for_content",
        lambda conn, dvid, cuid: section,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.load_section_content",
        lambda conn, dvid, sid: content_rows,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.find_child_section_for_content",
        lambda conn, dvid, sid, cuid: None,
    )

    trace: dict = {}
    expanded = expand_chunks(
        None,
        1,
        [_make_search_result("cu_1")],
        trace=trace,
    )

    assert len(expanded) == 2
    assert trace["expansion_steps"][0]["strategy"] == "full_section"
    assert trace["expansion_steps"][0]["loaded_chunk_ids"] == [
        "cu_1",
        "cu_2",
    ]
    assert trace["chunk_origins"]["cu_2"][0]["trigger_content_unit_id"] == (
        "cu_1"
    )
    assert trace["final_chunks"][0]["content_unit_id"] == "cu_1"


def test_expand_subsection(monkeypatch):
    """Large section with small subsection loads subsection."""
    _set_expand_env(monkeypatch)

    parent_section = _make_section_info(
        "s_001", token_count=5000, title="Capital"
    )
    child_section = _make_section_info(
        "s_001_a",
        token_count=800,
        title="CET1 Detail",
        parent_section_id="s_001",
    )
    child_content = [
        _make_content_row("cu_1", section_id="s_001_a"),
        _make_content_row("cu_5", section_id="s_001_a"),
    ]

    def mock_get_section(_conn, _dvid, _cuid):
        return parent_section

    def mock_load_section(_conn, _dvid, sid):
        if sid == "s_001_a":
            return child_content
        return []

    def mock_find_child(_conn, _dvid, _sid, _cuid):
        return child_section

    monkeypatch.setattr(
        "retriever.stages.expand.get_section_for_content",
        mock_get_section,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.load_section_content",
        mock_load_section,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.find_child_section_for_content",
        mock_find_child,
    )

    results = [_make_search_result("cu_1")]
    expanded = expand_chunks(None, 1, results)

    assert len(expanded) == 2
    cuids = [c["content_unit_id"] for c in expanded]
    assert "cu_1" in cuids
    assert "cu_5" in cuids


def test_expand_neighbors(monkeypatch):
    """Large section, large subsection falls back to neighbors."""
    _set_expand_env(monkeypatch)

    parent_section = _make_section_info(
        "s_001", token_count=5000, title="Capital"
    )
    child_section = _make_section_info(
        "s_001_a",
        token_count=3000,
        title="Large Subsection",
        parent_section_id="s_001",
    )
    child_content = [
        _make_content_row("cu_1", section_id="s_001_a"),
    ]
    neighbor_rows = [
        _make_content_row("cu_0", page=1),
        _make_content_row("cu_1", page=1),
        _make_content_row("cu_2", page=2),
    ]

    def mock_get_section(_conn, _dvid, _cuid):
        return parent_section

    def mock_load_section(_conn, _dvid, sid):
        if sid == "s_001_a":
            return child_content
        return []

    def mock_find_child(_conn, _dvid, _sid, _cuid):
        return child_section

    monkeypatch.setattr(
        "retriever.stages.expand.get_section_for_content",
        mock_get_section,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.load_section_content",
        mock_load_section,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.find_child_section_for_content",
        mock_find_child,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.load_neighbor_chunks",
        lambda conn, dvid, cuid, count: neighbor_rows,
    )

    results = [_make_search_result("cu_1")]
    expanded = expand_chunks(None, 1, results)

    assert len(expanded) == 3
    cuids = [c["content_unit_id"] for c in expanded]
    assert "cu_0" in cuids
    assert "cu_1" in cuids
    assert "cu_2" in cuids


def test_expand_no_section(monkeypatch):
    """Content without section falls back to neighbors."""
    _set_expand_env(monkeypatch)

    neighbor_rows = [
        _make_content_row("cu_prev", page=1),
        _make_content_row("cu_1", page=1),
        _make_content_row("cu_next", page=2),
    ]

    monkeypatch.setattr(
        "retriever.stages.expand.get_section_for_content",
        lambda conn, dvid, cuid: None,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.load_neighbor_chunks",
        lambda conn, dvid, cuid, count: neighbor_rows,
    )

    results = [_make_search_result("cu_1")]
    expanded = expand_chunks(None, 1, results)

    assert len(expanded) == 3


def test_expand_deduplicates(monkeypatch):
    """Same content_unit_id from multiple results merged."""
    _set_expand_env(monkeypatch)

    section = _make_section_info("s_001", token_count=500)
    content_rows = [
        _make_content_row("cu_1", page=1),
        _make_content_row("cu_2", page=1),
        _make_content_row("cu_3", page=2),
    ]

    monkeypatch.setattr(
        "retriever.stages.expand.get_section_for_content",
        lambda conn, dvid, cuid: section,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.load_section_content",
        lambda conn, dvid, sid: content_rows,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.find_child_section_for_content",
        lambda conn, dvid, sid, cuid: None,
    )

    results = [
        _make_search_result("cu_1"),
        _make_search_result("cu_2"),
    ]
    expanded = expand_chunks(None, 1, results)

    cuids = [c["content_unit_id"] for c in expanded]
    assert len(cuids) == len(set(cuids))


def test_expand_marks_originals(monkeypatch):
    """Original search results marked is_original=True."""
    _set_expand_env(monkeypatch)

    section = _make_section_info("s_001", token_count=500)
    content_rows = [
        _make_content_row("cu_1", page=1),
        _make_content_row("cu_2", page=1),
        _make_content_row("cu_3", page=2),
    ]

    monkeypatch.setattr(
        "retriever.stages.expand.get_section_for_content",
        lambda conn, dvid, cuid: section,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.load_section_content",
        lambda conn, dvid, sid: content_rows,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.find_child_section_for_content",
        lambda conn, dvid, sid, cuid: None,
    )

    results = [_make_search_result("cu_2")]
    expanded = expand_chunks(None, 1, results)

    for chunk in expanded:
        if chunk["content_unit_id"] == "cu_2":
            assert chunk["is_original"] is True
        else:
            assert chunk["is_original"] is False


def test_expand_sorts_by_page(monkeypatch):
    """Results sorted by page_number then content_unit_id."""
    _set_expand_env(monkeypatch)

    section = _make_section_info("s_001", token_count=500)
    content_rows = [
        _make_content_row("cu_3", page=3),
        _make_content_row("cu_1", page=1),
        _make_content_row("cu_2", page=2),
    ]

    monkeypatch.setattr(
        "retriever.stages.expand.get_section_for_content",
        lambda conn, dvid, cuid: section,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.load_section_content",
        lambda conn, dvid, sid: content_rows,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.find_child_section_for_content",
        lambda conn, dvid, sid, cuid: None,
    )

    results = [_make_search_result("cu_1")]
    expanded = expand_chunks(None, 1, results)

    pages = [c["page_number"] for c in expanded]
    assert pages == sorted(pages)


def test_expand_empty_results(monkeypatch):
    """Empty input returns empty output."""
    _set_expand_env(monkeypatch)

    expanded = expand_chunks(None, 1, [])
    assert not expanded


def test_get_expansion_strategy_full_section(monkeypatch):
    """Small section returns full_section strategy."""
    section = _make_section_info("s_001", token_count=500)
    monkeypatch.setattr(
        "retriever.stages.expand.get_section_for_content",
        lambda conn, dvid, cuid: section,
    )

    result = _make_search_result("cu_1")
    strategy = _get_expansion_strategy(None, 1, result, 3000, 1500)
    assert strategy == "full_section"


def test_get_expansion_strategy_subsection(monkeypatch):
    """Large section with small subsection returns subsection."""
    section = _make_section_info("s_001", token_count=5000)
    child = _make_section_info(
        "s_001_a",
        token_count=800,
        parent_section_id="s_001",
    )

    monkeypatch.setattr(
        "retriever.stages.expand.get_section_for_content",
        lambda conn, dvid, cuid: section,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.find_child_section_for_content",
        lambda conn, dvid, sid, cuid: child,
    )

    result = _make_search_result("cu_1")
    strategy = _get_expansion_strategy(None, 1, result, 3000, 1500)
    assert strategy == "subsection"


def test_get_expansion_strategy_neighbors(monkeypatch):
    """No section returns neighbors strategy."""
    monkeypatch.setattr(
        "retriever.stages.expand.get_section_for_content",
        lambda conn, dvid, cuid: None,
    )

    result = _make_search_result("cu_1")
    strategy = _get_expansion_strategy(None, 1, result, 3000, 1500)
    assert strategy == "neighbors"


def test_deduplicate_prefers_original():
    """Deduplication keeps is_original=True version."""
    chunks = [
        _to_expanded_chunk(
            _make_content_row("cu_1"),
            "Title",
            False,
        ),
        _to_expanded_chunk(
            _make_content_row("cu_1"),
            "Title",
            True,
        ),
    ]
    deduped = _deduplicate_chunks(chunks)
    assert len(deduped) == 1
    assert deduped[0]["is_original"] is True


def test_deduplicate_fuses_scores():
    """Deduplication keeps the maximum score across duplicates."""
    chunks = [
        _to_expanded_chunk(
            _make_content_row("cu_1"),
            "Title",
            True,
            score=0.5,
        ),
        _to_expanded_chunk(
            _make_content_row("cu_1"),
            "Title",
            False,
            score=0.9,
        ),
    ]
    deduped = _deduplicate_chunks(chunks)
    assert len(deduped) == 1
    assert deduped[0]["is_original"] is True
    assert deduped[0]["score"] == 0.9


def test_to_expanded_chunk():
    """Database row converted to ExpandedChunk correctly."""
    row = _make_content_row("cu_1")
    chunk = _to_expanded_chunk(row, "KM1", True)
    assert chunk["content_unit_id"] == "cu_1"
    assert chunk["section_title"] == "KM1"
    assert chunk["is_original"] is True
    assert chunk["token_count"] == 50
    assert chunk["score"] == 0.0


def test_to_expanded_chunk_with_score():
    """Score propagated to ExpandedChunk."""
    row = _make_content_row("cu_1")
    chunk = _to_expanded_chunk(row, "KM1", True, score=0.85)
    assert chunk["score"] == 0.85


def test_find_subsection_no_children(monkeypatch):
    """No child sections returns None."""
    monkeypatch.setattr(
        "retriever.stages.expand.find_child_section_for_content",
        lambda conn, dvid, sid, cuid: None,
    )

    result = _find_subsection(None, 1, "s_001", "cu_1")
    assert result is None


def test_find_subsection_not_in_children(monkeypatch):
    """Content unit not found in any child returns None."""
    monkeypatch.setattr(
        "retriever.stages.expand.find_child_section_for_content",
        lambda conn, dvid, sid, cuid: None,
    )

    result = _find_subsection(None, 1, "s_001", "cu_1")
    assert result is None


def test_numeric_cuid_sort_key():
    """Sort key orders 24.2 before 24.19."""
    chunks = [
        {"page_number": 24, "content_unit_id": "24.19"},
        {"page_number": 24, "content_unit_id": "24.2"},
        {"page_number": 24, "content_unit_id": "24.10"},
        {"page_number": 9, "content_unit_id": "9"},
    ]
    chunks.sort(key=_numeric_cuid_sort_key)
    cuids = [c["content_unit_id"] for c in chunks]
    assert cuids == ["9", "24.2", "24.10", "24.19"]


def test_expand_multi_section_titles(monkeypatch):
    """Rows spanning sections get per-row section titles."""
    _set_expand_env(monkeypatch)
    monkeypatch.setenv("EXPAND_NEIGHBOR_COUNT", "3")

    parent_section = _make_section_info(
        "s_001", token_count=5000, title="Capital"
    )
    neighbor_rows = [
        _make_content_row("cu_a", section_id="s_001", page=1),
        _make_content_row("cu_b", section_id="s_002", page=2),
        _make_content_row("cu_c", section_id="s_003", page=3),
    ]

    monkeypatch.setattr(
        "retriever.stages.expand.get_section_for_content",
        lambda conn, dvid, cuid: parent_section,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.find_child_section_for_content",
        lambda conn, dvid, sid, cuid: None,
    )
    monkeypatch.setattr(
        "retriever.stages.expand.load_neighbor_chunks",
        lambda conn, dvid, cuid, count: neighbor_rows,
    )

    section_lookup = {
        "s_001": {"title": "Capital"},
        "s_002": {"title": "Liquidity"},
        "s_003": {"title": "Credit Risk"},
    }
    monkeypatch.setattr(
        "retriever.stages.expand.get_section_info",
        lambda conn, dvid, sid: section_lookup.get(sid),
    )

    results = [_make_search_result("cu_b", section_id="s_002", page=2)]
    expanded = expand_chunks(None, 1, results)

    title_map = {c["content_unit_id"]: c["section_title"] for c in expanded}
    assert title_map["cu_a"] == "Capital"
    assert title_map["cu_b"] == "Liquidity"
    assert title_map["cu_c"] == "Credit Risk"
