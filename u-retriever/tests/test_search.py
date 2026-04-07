"""Tests for multi-strategy search and score fusion."""

from retriever.models import PreparedQuery, QueryEmbeddings
from retriever.stages.search import (
    _build_search_result,
    _collect_raw_data,
    _combine_strategies,
    _normalize_scores,
    load_full_document_as_results,
    multi_strategy_search,
)


def _set_search_env(monkeypatch):
    """Set search env vars. Params: monkeypatch. Returns: None."""
    monkeypatch.setenv("SEARCH_TOP_K", "20")
    monkeypatch.setenv("SEARCH_BM25_TOP_K", "20")
    monkeypatch.setenv("SEARCH_BM25_TERM_CAP", "6")
    monkeypatch.setenv("SEARCH_KEYWORD_MATCH_LIMIT", "50")
    monkeypatch.setenv("SEARCH_ENTITY_MATCH_LIMIT", "10")


def _make_embedding(val: float = 0.1) -> list[float]:
    """Return a small dummy embedding vector."""
    return [val] * 4


def _make_prepared_query() -> PreparedQuery:
    """Build a PreparedQuery fixture with dummy embeddings."""
    return PreparedQuery(
        original_query="CET1 ratio for RBC",
        rewritten_query="Common Equity Tier 1 ratio RBC",
        sub_queries=["CET1 trend", "capital adequacy"],
        keywords=["CET1", "Tier 1", "capital"],
        entities=["Royal Bank of Canada"],
        hyde_answer="RBC CET1 ratio was 13.2%.",
        embeddings=QueryEmbeddings(
            rewritten=_make_embedding(0.1),
            sub_queries=[
                _make_embedding(0.2),
                _make_embedding(0.3),
            ],
            keywords=_make_embedding(0.4),
            hyde=_make_embedding(0.5),
        ),
    )


def _make_content_row(
    cuid: str,
    distance: float = 0.0,
    rank: float = 0.0,
) -> dict:
    """Build a mock content row from database queries."""
    return {
        "content_unit_id": cuid,
        "chunk_id": f"ch_{cuid}",
        "section_id": f"s_{cuid}",
        "page_number": 1,
        "raw_content": f"Content of {cuid}",
        "chunk_context": "context",
        "chunk_header": "header",
        "keywords": ["kw1"],
        "entities": ["ent1"],
        "token_count": 50,
        "distance": distance,
        "rank": rank,
    }


def _make_section_row(sid: str, distance: float = 0.0) -> dict:
    """Build a mock section row."""
    return {
        "section_id": sid,
        "title": f"Section {sid}",
        "summary": f"Summary of {sid}",
        "distance": distance,
    }


def _default_weights() -> dict[str, float]:
    """Return default score weights."""
    return {
        "content_vector": 0.25,
        "hyde_vector": 0.20,
        "subquery_vector": 0.15,
        "keyword_vector": 0.10,
        "section_summary": 0.10,
        "bm25": 0.10,
        "keyword_array": 0.05,
        "entity_array": 0.05,
    }


def test_normalize_scores_vector():
    """Distance scores inverted: lower distance = higher score."""
    rows = [
        _make_content_row("a", distance=0.2),
        _make_content_row("b", distance=0.5),
        _make_content_row("c", distance=1.0),
    ]
    scores = _normalize_scores(rows, "distance", invert=True)

    assert scores["a"] == 0.8
    assert scores["b"] == 0.5
    assert scores["c"] == 0.0


def test_normalize_scores_bm25():
    """BM25 rank scores normalized: highest rank = 1.0."""
    rows = [
        _make_content_row("a", rank=0.8),
        _make_content_row("b", rank=0.4),
        _make_content_row("c", rank=0.2),
    ]
    scores = _normalize_scores(rows, "rank")

    assert scores["a"] == 1.0
    assert scores["b"] == 0.5
    assert scores["c"] == 0.25


def test_normalize_scores_empty():
    """Empty result list returns empty dict."""
    assert _normalize_scores([], "distance") == {}


def test_normalize_scores_zero_max():
    """When max value is zero, all scores are 1.0."""
    rows = [
        _make_content_row("a", distance=0.0),
        _make_content_row("b", distance=0.0),
    ]
    scores = _normalize_scores(rows, "distance", invert=True)
    assert scores["a"] == 1.0
    assert scores["b"] == 1.0


def test_combine_strategies():
    """Weighted combination accumulates correctly."""
    strategy_results = [
        ("bm25", {"a": 1.0, "b": 0.5}),
        ("content_vector", {"a": 0.8, "c": 0.6}),
    ]
    weights = {"bm25": 0.10, "content_vector": 0.25}
    combined = _combine_strategies(strategy_results, weights)

    # a: bm25=1.0*0.10 + cv=0.8*0.25 = 0.10 + 0.20 = 0.30
    assert abs(combined["a"]["combined"] - 0.30) < 1e-9
    assert combined["a"]["bm25"] == 1.0
    assert combined["a"]["content_vector"] == 0.8

    # b: bm25=0.5*0.10 = 0.05
    assert abs(combined["b"]["combined"] - 0.05) < 1e-9

    # c: cv=0.6*0.25 = 0.15
    assert abs(combined["c"]["combined"] - 0.15) < 1e-9


def test_collect_raw_data():
    """First occurrence of each content_unit_id is kept."""
    batch_1 = [
        _make_content_row("a", distance=0.1),
        _make_content_row("b", distance=0.2),
    ]
    batch_2 = [
        _make_content_row("a", distance=0.9),
        _make_content_row("c", distance=0.3),
    ]
    raw = _collect_raw_data([batch_1, batch_2])

    assert raw["a"]["distance"] == 0.1
    assert "b" in raw
    assert "c" in raw


def test_build_search_result():
    """SearchResult is built from combined scores and row data."""
    raw_data = {
        "cu_1": _make_content_row("cu_1"),
    }
    combined = {
        "combined": 0.75,
        "bm25": 0.8,
        "content_vector": 0.6,
    }
    result = _build_search_result("cu_1", combined, raw_data)

    assert result["content_unit_id"] == "cu_1"
    assert result["score"] == 0.75
    assert result["strategy_scores"]["bm25"] == 0.8
    assert "combined" not in result["strategy_scores"]
    assert result["raw_content"] == "Content of cu_1"


def test_build_search_result_missing_raw_data():
    """Missing raw data produces empty defaults."""
    combined = {"combined": 0.5}
    result = _build_search_result("cu_x", combined, {})

    assert result["content_unit_id"] == "cu_x"
    assert result["raw_content"] == ""
    assert result["token_count"] == 0


def test_multi_strategy_search_combines_scores(
    monkeypatch,
):
    """Multiple strategies produce weighted combined scores."""
    _set_search_env(monkeypatch)
    prepared = _make_prepared_query()
    weights = _default_weights()

    row_a = _make_content_row("a", distance=0.2)
    row_b = _make_content_row("b", distance=0.5)

    monkeypatch.setattr(
        "retriever.stages.search.search_by_content_vector",
        lambda conn, dvid, emb, top_k: [row_a, row_b],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_vector",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_section_summary",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_bm25",
        lambda conn, dvid, q, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_containment",
        lambda conn, dvid, kws, lim: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_entity_containment",
        lambda conn, dvid, ents, lim: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.load_section_content",
        lambda conn, dvid, sid: [],
    )

    metrics: dict = {}
    results = multi_strategy_search(
        None,
        1,
        prepared,
        weights,
        metrics=metrics,
    )

    assert len(results) >= 1
    assert results[0]["score"] > 0
    assert metrics["unique_results"] >= 1
    assert "content_vector" in metrics["strategies"]


def test_multi_strategy_search_populates_trace(monkeypatch):
    """Search traces preserve per-strategy hits and fused ranking."""
    _set_search_env(monkeypatch)
    prepared = _make_prepared_query()
    weights = _default_weights()

    row_a = _make_content_row("a", distance=0.2, rank=0.9)

    monkeypatch.setattr(
        "retriever.stages.search.search_by_content_vector",
        lambda conn, dvid, emb, top_k: [row_a],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_vector",
        lambda conn, dvid, emb, top_k: [row_a],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_section_summary",
        lambda conn, dvid, emb, top_k: [
            _make_section_row("s_a", distance=0.2)
        ],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_bm25",
        lambda conn, dvid, q, top_k: [row_a],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_containment",
        lambda conn, dvid, kws, lim: [row_a],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_entity_containment",
        lambda conn, dvid, ents, lim: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.load_section_content",
        lambda conn, dvid, sid: [row_a],
    )

    trace: dict = {}
    results = multi_strategy_search(
        None,
        1,
        prepared,
        weights,
        trace=trace,
    )

    assert results
    assert trace["query_inputs"]["hyde_answer"] == prepared["hyde_answer"]
    assert (
        trace["strategy_traces"]["content_vector"]["hits"][0][
            "content_unit_id"
        ]
        == "a"
    )
    assert trace["strategy_traces"]["bm25"]["hits"][0]["hit_position"] == 1
    assert trace["strategy_traces"]["bm25"]["hits"][0]["rank"] == 0.9
    assert trace["fused_results"][0]["content_unit_id"] == "a"


def test_multi_strategy_search_deduplicates(monkeypatch):
    """Same content_unit_id from multiple strategies merged."""
    _set_search_env(monkeypatch)
    prepared = _make_prepared_query()
    weights = _default_weights()

    row_a_cv = _make_content_row("a", distance=0.3)
    row_a_bm = _make_content_row("a", rank=0.9)

    monkeypatch.setattr(
        "retriever.stages.search.search_by_content_vector",
        lambda conn, dvid, emb, top_k: [row_a_cv],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_vector",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_section_summary",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_bm25",
        lambda conn, dvid, q, top_k: [row_a_bm],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_containment",
        lambda conn, dvid, kws, lim: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_entity_containment",
        lambda conn, dvid, ents, lim: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.load_section_content",
        lambda conn, dvid, sid: [],
    )

    results = multi_strategy_search(None, 1, prepared, weights)

    cuids = [r["content_unit_id"] for r in results]
    assert cuids.count("a") == 1
    assert results[0]["score"] > 0


def test_multi_strategy_search_normalizes(monkeypatch):
    """Scores are normalized within 0-1 range."""
    _set_search_env(monkeypatch)
    prepared = _make_prepared_query()
    weights = _default_weights()

    rows = [
        _make_content_row("a", distance=0.1),
        _make_content_row("b", distance=0.5),
        _make_content_row("c", distance=1.0),
    ]

    monkeypatch.setattr(
        "retriever.stages.search.search_by_content_vector",
        lambda conn, dvid, emb, top_k: rows,
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_vector",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_section_summary",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_bm25",
        lambda conn, dvid, q, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_containment",
        lambda conn, dvid, kws, lim: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_entity_containment",
        lambda conn, dvid, ents, lim: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.load_section_content",
        lambda conn, dvid, sid: [],
    )

    results = multi_strategy_search(None, 1, prepared, weights)

    for result in results:
        assert 0.0 <= result["score"] <= 1.0


def test_multi_strategy_search_builds_keyword_focused_bm25_query(
    monkeypatch,
):
    """BM25 uses sanitized high-signal facets instead of the full rewrite."""
    _set_search_env(monkeypatch)
    prepared = _make_prepared_query()
    prepared["keywords"] = [
        "CET1",
        "Tier 1",
        "capital ratio (%)",
    ]
    weights = _default_weights()
    captured: dict[str, str] = {}

    def capture_bm25(_conn, _dvid, query_text, _top_k):
        """Record the BM25 query text and return no hits."""
        captured["query"] = query_text
        return []

    monkeypatch.setattr(
        "retriever.stages.search.search_by_content_vector",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_vector",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_section_summary",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_bm25",
        capture_bm25,
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_containment",
        lambda conn, dvid, kws, lim: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_entity_containment",
        lambda conn, dvid, ents, lim: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.load_section_content",
        lambda conn, dvid, sid: [],
    )

    multi_strategy_search(None, 1, prepared, weights)

    assert (
        captured["query"] == 'CET1 OR "Tier 1" OR "capital ratio percentage"'
        ' OR "Royal Bank of Canada"'
    )


def test_multi_strategy_search_sorts_by_score(
    monkeypatch,
):
    """Results are sorted by combined score descending."""
    _set_search_env(monkeypatch)
    prepared = _make_prepared_query()
    weights = _default_weights()

    rows = [
        _make_content_row("low", distance=0.9),
        _make_content_row("mid", distance=0.5),
        _make_content_row("high", distance=0.1),
    ]

    monkeypatch.setattr(
        "retriever.stages.search.search_by_content_vector",
        lambda conn, dvid, emb, top_k: rows,
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_vector",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_section_summary",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_bm25",
        lambda conn, dvid, q, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_containment",
        lambda conn, dvid, kws, lim: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_entity_containment",
        lambda conn, dvid, ents, lim: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.load_section_content",
        lambda conn, dvid, sid: [],
    )

    results = multi_strategy_search(None, 1, prepared, weights)

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_load_full_document_as_results(monkeypatch):
    """All rows loaded with score=1.0 and empty strategies."""
    rows = [
        {
            "content_unit_id": "cu_1",
            "chunk_id": "ch_1",
            "section_id": "s_1",
            "page_number": 1,
            "raw_content": "text 1",
            "chunk_context": "ctx",
            "chunk_header": "hdr",
            "keywords": ["kw"],
            "entities": ["ent"],
            "token_count": 30,
        },
        {
            "content_unit_id": "cu_2",
            "chunk_id": "ch_2",
            "section_id": "s_1",
            "page_number": 2,
            "raw_content": "text 2",
            "chunk_context": "ctx",
            "chunk_header": "hdr",
            "keywords": [],
            "entities": [],
            "token_count": 25,
        },
    ]
    monkeypatch.setattr(
        "retriever.stages.search.load_full_document",
        lambda conn, dvid: rows,
    )

    results = load_full_document_as_results(None, 42)

    assert len(results) == 2
    assert results[0]["score"] == 1.0
    assert results[1]["score"] == 1.0
    assert results[0]["strategy_scores"] == {}
    assert results[0]["content_unit_id"] == "cu_1"
    assert results[1]["content_unit_id"] == "cu_2"


def test_section_summary_maps_to_content(monkeypatch):
    """Section hits map to their content units."""
    _set_search_env(monkeypatch)
    prepared = _make_prepared_query()
    weights = _default_weights()

    section_hit = _make_section_row("s_1", distance=0.3)
    content_from_section = _make_content_row("cu_sec", distance=0.0)
    content_from_section["section_id"] = "s_1"

    monkeypatch.setattr(
        "retriever.stages.search.search_by_content_vector",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_vector",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_section_summary",
        lambda conn, dvid, emb, top_k: [section_hit],
    )
    monkeypatch.setattr(
        "retriever.stages.search.load_section_content",
        lambda conn, dvid, sid: [content_from_section],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_bm25",
        lambda conn, dvid, q, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_containment",
        lambda conn, dvid, kws, lim: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_entity_containment",
        lambda conn, dvid, ents, lim: [],
    )

    results = multi_strategy_search(None, 1, prepared, weights)

    cuids = [r["content_unit_id"] for r in results]
    assert "cu_sec" in cuids
    sec_result = next(r for r in results if r["content_unit_id"] == "cu_sec")
    assert "section_summary" in sec_result["strategy_scores"]


def test_section_summary_zero_distance(monkeypatch):
    """Section hit with distance=0 produces score=1.0."""
    _set_search_env(monkeypatch)
    prepared = _make_prepared_query()
    weights = _default_weights()

    section_hit = _make_section_row("s_z", distance=0.0)
    content_row = _make_content_row("cu_z", distance=0.0)
    content_row["section_id"] = "s_z"

    monkeypatch.setattr(
        "retriever.stages.search.search_by_content_vector",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_vector",
        lambda conn, dvid, emb, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_section_summary",
        lambda conn, dvid, emb, top_k: [section_hit],
    )
    monkeypatch.setattr(
        "retriever.stages.search.load_section_content",
        lambda conn, dvid, sid: [content_row],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_bm25",
        lambda conn, dvid, q, top_k: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_keyword_containment",
        lambda conn, dvid, kws, lim: [],
    )
    monkeypatch.setattr(
        "retriever.stages.search.search_by_entity_containment",
        lambda conn, dvid, ents, lim: [],
    )

    results = multi_strategy_search(None, 1, prepared, weights)

    sec_result = next(r for r in results if r["content_unit_id"] == "cu_z")
    assert sec_result["strategy_scores"]["section_summary"] == 1.0
