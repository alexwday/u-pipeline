"""Tests for orchestrator wiring all retrieval stages."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from retriever.models import (
    ComboSourceResult,
    ComboSpec,
    ConsolidatedResult,
    ExpandedChunk,
    PreparedQuery,
    ResearchIteration,
    SearchResult,
    SourceSpec,
)
from retriever.stages.orchestrator import (
    _resolve_research_units,
    _search_result_to_expanded,
    run_retrieval,
)

_MOD = "retriever.stages.orchestrator"

_VERSION_ROW = {
    "document_version_id": 38,
    "data_source": "pillar3",
    "filename": "rbc_q1_2026_pillar3.xlsx",
}


def _make_combo(
    bank: str = "RBC",
    period: str = "2026_Q1",
) -> ComboSpec:
    """Build a ComboSpec fixture."""
    return ComboSpec(bank=bank, period=period)


def _make_source(
    data_source: str = "pillar3",
    doc_id: int = 38,
    filename: str = "rbc_q1_2026_pillar3.xlsx",
) -> SourceSpec:
    """Build a SourceSpec fixture."""
    return SourceSpec(
        data_source=data_source,
        document_version_id=doc_id,
        filename=filename,
    )


def _make_prepared_query(
    query: str = "What is the CET1 ratio?",
) -> PreparedQuery:
    """Build a PreparedQuery fixture."""
    return PreparedQuery(
        original_query=query,
        rewritten_query="CET1 capital ratio",
        sub_queries=["CET1 details"],
        keywords=["CET1"],
        entities=["RBC"],
        hyde_answer="CET1 ratio was 13.7%.",
        embeddings={
            "rewritten": [0.1] * 10,
            "sub_queries": [[0.2] * 10],
            "keywords": [0.3] * 10,
            "hyde": [0.4] * 10,
        },
    )


def _make_search_result(
    cuid: str = "cu_1",
    page: int = 3,
    token_count: int = 50,
) -> SearchResult:
    """Build a SearchResult fixture."""
    return SearchResult(
        content_unit_id=cuid,
        raw_content="CET1 ratio was 13.7%",
        chunk_id="ch_1",
        section_id="s_001",
        page_number=page,
        chunk_context="context",
        chunk_header="KM1",
        keywords=["CET1"],
        entities=["RBC"],
        token_count=token_count,
        score=0.95,
        strategy_scores={"content_vector": 0.95},
    )


def _make_expanded_chunk(
    cuid: str = "cu_1",
    page: int = 3,
    token_count: int = 50,
) -> ExpandedChunk:
    """Build an ExpandedChunk fixture."""
    return ExpandedChunk(
        content_unit_id=cuid,
        raw_content="CET1 ratio was 13.7%",
        page_number=page,
        section_id="s_001",
        section_title="KM1",
        chunk_context="context",
        is_original=True,
        token_count=token_count,
    )


def _make_combo_result(
    bank: str = "RBC",
    data_source: str = "pillar3",
) -> ComboSourceResult:
    """Build a ComboSourceResult fixture."""
    return ComboSourceResult(
        combo=_make_combo(bank=bank),
        source=_make_source(data_source=data_source),
        research_iterations=[
            ResearchIteration(
                iteration=1,
                additional_queries=[],
                confidence=0.9,
                findings=[],
            ),
        ],
        findings=[],
        chunk_count=5,
        total_tokens=1000,
    )


def _make_consolidated_result(
    query: str = "CET1 ratio",
) -> ConsolidatedResult:
    """Build a ConsolidatedResult fixture."""
    return ConsolidatedResult(
        query=query,
        combo_results=[_make_combo_result()],
        consolidated_response="CET1 ratio was 13.7%.",
        key_findings=["CET1 ratio is 13.7%"],
        data_gaps=[],
    )


def _set_orchestrator_env(monkeypatch):
    """Set required env vars for orchestrator tests."""
    env_vars = {
        "SMALL_DOC_TOKEN_THRESHOLD": "4000",
        "ORCHESTRATOR_MAX_WORKERS": "4",
        "RERANK_CANDIDATE_LIMIT": "30",
        "WEIGHT_CONTENT_VECTOR": "0.25",
        "WEIGHT_HYDE_VECTOR": "0.20",
        "WEIGHT_SUBQUERY_VECTOR": "0.15",
        "WEIGHT_KEYWORD_VECTOR": "0.10",
        "WEIGHT_SECTION_SUMMARY": "0.10",
        "WEIGHT_BM25": "0.12",
        "WEIGHT_KEYWORD_ARRAY": "0.07",
        "WEIGHT_ENTITY_ARRAY": "0.01",
        "CONSOLIDATION_MODEL": "gpt-test",
        "CONSOLIDATION_MAX_TOKENS": "8000",
        "CONSOLIDATION_TEMPERATURE": "",
        "CONSOLIDATION_REASONING_EFFORT": "",
        "RESEARCH_MODEL": "gpt-test",
        "RESEARCH_MAX_TOKENS": "4000",
        "RESEARCH_TEMPERATURE": "",
        "RESEARCH_MAX_ITERATIONS": "1",
        "RESEARCH_ADDITIONAL_SEARCH_TOP_K": "10",
        "RERANK_MODEL": "gpt-test",
        "RERANK_MAX_TOKENS": "2000",
        "RERANK_TEMPERATURE": "",
        "QUERY_PREP_MODEL": "gpt-test",
        "QUERY_PREP_MAX_TOKENS": "2000",
        "QUERY_PREP_TEMPERATURE": "",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)


def _stub_resolve_single(monkeypatch, version=None):
    """Stub resolve_document_version_ids with one row."""
    row = version or _VERSION_ROW

    def resolve(_conn, _bank, _period, _sources):
        """Return a single version row."""
        return [row]

    monkeypatch.setattr(
        f"{_MOD}.resolve_document_version_ids",
        resolve,
    )


def _stub_prepare(monkeypatch):
    """Stub prepare_query."""

    def prepare(_query, _llm, metrics=None, trace=None):
        """Return a fixed prepared query."""
        if metrics is not None:
            metrics["wall_time_seconds"] = 0.1
        if trace is not None:
            trace["original_query"] = _query
        return _make_prepared_query()

    monkeypatch.setattr(f"{_MOD}.prepare_query", prepare)


def _stub_connection(monkeypatch):
    """Stub get_connection."""
    monkeypatch.setattr(
        f"{_MOD}.get_connection",
        MagicMock,
    )


def _stub_tokens(monkeypatch, total=10000):
    """Stub get_document_total_tokens."""

    def get_tokens(_conn, _doc_id):
        """Return fixed token count."""
        return total

    monkeypatch.setattr(
        f"{_MOD}.get_document_total_tokens",
        get_tokens,
    )


def _stub_search_pipeline(monkeypatch):
    """Stub search, rerank, and expand with defaults."""

    def search(_conn, _doc_id, _prep, _weights, metrics=None, **_kw):
        """Return one search result."""
        if metrics is not None:
            metrics["unique_results"] = 1
        return [_make_search_result()]

    def rerank(results, _query, _llm, **_kw):
        """Return results unchanged."""
        metrics = _kw.get("metrics")
        if metrics is not None:
            metrics["kept"] = len(results)
        return results

    def expand(_conn, _doc_id, results, metrics=None, trace=None):
        """Return one expanded chunk."""
        del results
        del trace
        if metrics is not None:
            metrics["chunks_after"] = 1
        return [_make_expanded_chunk()]

    monkeypatch.setattr(
        f"{_MOD}.multi_strategy_search",
        search,
    )
    monkeypatch.setattr(
        f"{_MOD}.rerank_results",
        rerank,
    )
    monkeypatch.setattr(
        f"{_MOD}.expand_chunks",
        expand,
    )


def _stub_research(monkeypatch):
    """Stub research_combo_source."""

    def research(*_args, **_kwargs):
        """Return a fixed combo result."""
        return _make_combo_result()

    monkeypatch.setattr(
        f"{_MOD}.research_combo_source",
        research,
    )


def _stub_consolidate(monkeypatch):
    """Stub consolidate_results."""

    def consolidate(_query, _results, _llm, metrics=None, **_kwargs):
        """Return a fixed consolidated result."""
        if metrics is not None:
            metrics["wall_time_seconds"] = 0.1
        return _make_consolidated_result()

    monkeypatch.setattr(
        f"{_MOD}.consolidate_results",
        consolidate,
    )


def _stub_all(monkeypatch):
    """Apply all default stubs for orchestrator tests."""
    _stub_resolve_single(monkeypatch)
    _stub_prepare(monkeypatch)
    _stub_connection(monkeypatch)
    _stub_tokens(monkeypatch)
    _stub_search_pipeline(monkeypatch)
    _stub_research(monkeypatch)
    _stub_consolidate(monkeypatch)


def test_run_retrieval_full_flow(monkeypatch):
    """End-to-end with all mocked stages."""
    _set_orchestrator_env(monkeypatch)
    _stub_all(monkeypatch)

    result = run_retrieval(
        "CET1 ratio",
        [_make_combo()],
        None,
        MagicMock(),
        MagicMock(),
    )

    assert result["query"] == "CET1 ratio"
    assert result["consolidated_response"] == "CET1 ratio was 13.7%."


def test_run_retrieval_writes_trace_files(monkeypatch):
    """Successful retrieval writes run-level and source-level traces."""
    _set_orchestrator_env(monkeypatch)
    _stub_all(monkeypatch)

    result = run_retrieval(
        "CET1 ratio",
        [_make_combo()],
        None,
        MagicMock(),
        MagicMock(),
    )

    run_trace_path = Path(result["trace_path"])
    assert run_trace_path.exists()

    with open(run_trace_path, encoding="utf-8") as fh:
        run_trace = json.load(fh)

    assert run_trace["query"] == "CET1 ratio"
    assert run_trace["prepared_query"]["original_query"] == "CET1 ratio"
    assert len(run_trace["source_traces"]) == 1
    source_trace_path = Path(run_trace["source_traces"][0]["trace_path"])
    assert source_trace_path.exists()

    outputs = run_trace["outputs"]
    assert outputs["coverage_audit"] == ""
    assert outputs["uncited_ref_ids"] == []
    assert outputs["unincorporated_findings"] == []


def test_run_retrieval_trace_includes_populated_audit_fields(monkeypatch):
    """Audit fields on ConsolidatedResult round-trip into the run trace."""
    _set_orchestrator_env(monkeypatch)
    _stub_resolve_single(monkeypatch)
    _stub_prepare(monkeypatch)
    _stub_connection(monkeypatch)
    _stub_tokens(monkeypatch)
    _stub_search_pipeline(monkeypatch)
    _stub_research(monkeypatch)

    def consolidate(_query, results, _llm, metrics=None, **_kw):
        """Return a consolidated result with populated audit fields."""
        if metrics is not None:
            metrics["wall_time_seconds"] = 0.1
        return ConsolidatedResult(
            query="CET1 ratio",
            combo_results=results,
            consolidated_response="CET1 ratio 13.7%.",
            key_findings=["CET1 ratio 13.7%."],
            data_gaps=[],
            coverage_audit="## Coverage audit\n\n### Uncited refs\n- [REF:7]",
            uncited_ref_ids=[7],
            unincorporated_findings=[
                {
                    "ref_id": 3,
                    "source": "pillar3",
                    "page": 9,
                    "location_detail": "LI1",
                    "metric_name": "Net write-offs",
                    "metric_value": "634",
                    "finding": "Total net write-offs were 634.",
                },
            ],
        )

    monkeypatch.setattr(f"{_MOD}.consolidate_results", consolidate)

    result = run_retrieval(
        "CET1 ratio",
        [_make_combo()],
        None,
        MagicMock(),
        MagicMock(),
    )

    run_trace_path = Path(result["trace_path"])
    with open(run_trace_path, encoding="utf-8") as fh:
        run_trace = json.load(fh)

    outputs = run_trace["outputs"]
    assert "## Coverage audit" in outputs["coverage_audit"]
    assert outputs["uncited_ref_ids"] == [7]
    assert len(outputs["unincorporated_findings"]) == 1
    entry = outputs["unincorporated_findings"][0]
    assert entry["ref_id"] == 3
    assert entry["metric_value"] == "634"


def test_run_retrieval_validates_citations_before_return(monkeypatch):
    """Citation validation runs on consolidated output."""
    _set_orchestrator_env(monkeypatch)
    _stub_resolve_single(monkeypatch)
    _stub_prepare(monkeypatch)
    _stub_connection(monkeypatch)
    _stub_tokens(monkeypatch)
    _stub_search_pipeline(monkeypatch)
    _stub_research(monkeypatch)

    def consolidate(_query, results, _llm, metrics=None, **_kw):
        """Return a response with no [REF:N] patterns."""
        if metrics is not None:
            metrics["wall_time_seconds"] = 0.1
        return ConsolidatedResult(
            query="CET1 ratio",
            combo_results=results,
            consolidated_response="CET1 ratio 13.7% [pillar3, Page 3].",
            key_findings=["CET1 ratio 13.7% [pillar3, Page 3]."],
            data_gaps=[],
        )

    monkeypatch.setattr(
        f"{_MOD}.consolidate_results",
        consolidate,
    )

    result = run_retrieval(
        "CET1 ratio",
        [_make_combo()],
        None,
        MagicMock(),
        MagicMock(),
    )

    assert result["metrics"]["citation_validation"]["warning_count"] == 0
    assert result["metrics"]["citation_validation"]["skipped"] is False


def test_run_retrieval_uses_research_final_chunks_for_citations(
    monkeypatch,
):
    """Citation validation uses reference_index for cited pages."""
    _set_orchestrator_env(monkeypatch)
    _stub_resolve_single(monkeypatch)
    _stub_prepare(monkeypatch)
    _stub_connection(monkeypatch)
    _stub_tokens(monkeypatch)
    _stub_search_pipeline(monkeypatch)

    def research(*_args, **kwargs):
        """Return a fixed result and simulate follow-up trace chunks."""
        kwargs["trace"]["final_chunks"] = [
            {
                "content_unit_id": "cu_37",
                "page_number": 37,
                "section_id": "s_037",
                "section_title": "Capital ratios",
            }
        ]
        return _make_combo_result()

    def consolidate(_query, results, _llm, metrics=None, **_kw):
        """Return a response citing a follow-up page with ref index."""
        if metrics is not None:
            metrics["wall_time_seconds"] = 0.1
        return ConsolidatedResult(
            query="CET1 ratio",
            combo_results=results,
            consolidated_response="Capital ratios [pillar3, Page 37].",
            key_findings=["Capital ratios [pillar3, Page 37]."],
            data_gaps=[],
            reference_index=[
                {
                    "ref_id": 1,
                    "finding": "Capital ratios",
                    "page": 37,
                    "source": "pillar3",
                },
            ],
        )

    monkeypatch.setattr(f"{_MOD}.research_combo_source", research)
    monkeypatch.setattr(f"{_MOD}.consolidate_results", consolidate)

    result = run_retrieval(
        "CET1 ratio",
        [_make_combo()],
        None,
        MagicMock(),
        MagicMock(),
    )

    assert result["metrics"]["citation_validation"]["warning_count"] == 0
    assert result["metrics"]["citation_validation"]["cited_pages_by_source"][
        "pillar3"
    ] == [37]


def test_run_retrieval_resolves_documents(monkeypatch):
    """Correct document resolution from combos."""
    _set_orchestrator_env(monkeypatch)

    resolve_calls = []

    def mock_resolve(_conn, bank, period, sources):
        """Track resolution calls."""
        resolve_calls.append((bank, period, sources))
        return [
            {
                "document_version_id": 38,
                "data_source": "pillar3",
                "filename": "file.xlsx",
            }
        ]

    monkeypatch.setattr(
        f"{_MOD}.resolve_document_version_ids",
        mock_resolve,
    )
    _stub_prepare(monkeypatch)
    _stub_connection(monkeypatch)
    _stub_tokens(monkeypatch)
    _stub_search_pipeline(monkeypatch)
    _stub_research(monkeypatch)
    _stub_consolidate(monkeypatch)

    run_retrieval(
        "query",
        [_make_combo()],
        ["pillar3"],
        MagicMock(),
        MagicMock(),
    )

    assert len(resolve_calls) == 1
    assert resolve_calls[0] == (
        "RBC",
        "2026_Q1",
        ["pillar3"],
    )


def test_run_retrieval_small_doc_skips_search(
    monkeypatch,
):
    """Small doc loads full content, skips search."""
    _set_orchestrator_env(monkeypatch)

    search_called = []
    full_load_called = []

    def mock_full_load(_conn, doc_id):
        """Track full load calls."""
        full_load_called.append(doc_id)
        return [_make_search_result()]

    def mock_search(_conn, doc_id, _prep, _weights, metrics=None, **_kw):
        """Track search calls."""
        del metrics
        search_called.append(doc_id)
        return []

    _stub_resolve_single(monkeypatch)
    _stub_prepare(monkeypatch)
    _stub_connection(monkeypatch)
    _stub_tokens(monkeypatch, total=2000)
    monkeypatch.setattr(
        f"{_MOD}.load_full_document_as_results",
        mock_full_load,
    )
    monkeypatch.setattr(
        f"{_MOD}.multi_strategy_search",
        mock_search,
    )
    _stub_research(monkeypatch)
    _stub_consolidate(monkeypatch)

    run_retrieval(
        "query",
        [_make_combo()],
        None,
        MagicMock(),
        MagicMock(),
    )

    assert len(full_load_called) == 1
    assert not search_called


def test_run_retrieval_large_doc_searches(monkeypatch):
    """Large doc runs search + rerank + expand."""
    _set_orchestrator_env(monkeypatch)

    search_called = []
    rerank_called = []
    expand_called = []

    def mock_search(_conn, doc_id, _prep, _weights, metrics=None, **_kw):
        """Track search calls."""
        del metrics
        search_called.append(doc_id)
        return [_make_search_result()]

    def mock_rerank(results, _query, _llm, **_kw):
        """Track rerank calls."""
        rerank_called.append(len(results))
        return results

    def mock_expand(_conn, doc_id, _results, metrics=None, trace=None):
        """Track expand calls."""
        del metrics
        del trace
        expand_called.append(doc_id)
        return [_make_expanded_chunk()]

    _stub_resolve_single(monkeypatch)
    _stub_prepare(monkeypatch)
    _stub_connection(monkeypatch)
    _stub_tokens(monkeypatch, total=10000)
    monkeypatch.setattr(
        f"{_MOD}.multi_strategy_search",
        mock_search,
    )
    monkeypatch.setattr(
        f"{_MOD}.rerank_results",
        mock_rerank,
    )
    monkeypatch.setattr(
        f"{_MOD}.expand_chunks",
        mock_expand,
    )
    _stub_research(monkeypatch)
    _stub_consolidate(monkeypatch)

    run_retrieval(
        "query",
        [_make_combo()],
        None,
        MagicMock(),
        MagicMock(),
    )

    assert len(search_called) == 1
    assert len(rerank_called) == 1
    assert len(expand_called) == 1


def test_run_retrieval_multiple_combos(monkeypatch):
    """Multiple combos produce multiple research units."""
    _set_orchestrator_env(monkeypatch)

    research_calls = []

    def mock_research(*_args, **_kwargs):
        """Track research calls."""
        research_calls.append(1)
        return _make_combo_result()

    def resolve_multi(_conn, bank, _period, _sources):
        """Return a version per combo."""
        return [
            {
                "document_version_id": 38,
                "data_source": "pillar3",
                "filename": f"{bank}_file.xlsx",
            }
        ]

    monkeypatch.setattr(
        f"{_MOD}.resolve_document_version_ids",
        resolve_multi,
    )
    _stub_prepare(monkeypatch)
    _stub_connection(monkeypatch)
    _stub_tokens(monkeypatch)
    _stub_search_pipeline(monkeypatch)
    monkeypatch.setattr(
        f"{_MOD}.research_combo_source",
        mock_research,
    )
    _stub_consolidate(monkeypatch)

    combos = [
        _make_combo(bank="RBC"),
        _make_combo(bank="BMO"),
    ]

    run_retrieval(
        "query",
        combos,
        None,
        MagicMock(),
        MagicMock(),
    )

    assert len(research_calls) == 2


def test_run_retrieval_source_filter(monkeypatch):
    """Sources filter is passed to document resolution."""
    _set_orchestrator_env(monkeypatch)

    resolve_calls = []

    def mock_resolve(_conn, _bank, _period, sources):
        """Track resolution calls."""
        resolve_calls.append(sources)
        return [
            {
                "document_version_id": 38,
                "data_source": "pillar3",
                "filename": "file.xlsx",
            }
        ]

    monkeypatch.setattr(
        f"{_MOD}.resolve_document_version_ids",
        mock_resolve,
    )
    _stub_prepare(monkeypatch)
    _stub_connection(monkeypatch)
    _stub_tokens(monkeypatch)
    _stub_search_pipeline(monkeypatch)
    _stub_research(monkeypatch)
    _stub_consolidate(monkeypatch)

    run_retrieval(
        "query",
        [_make_combo()],
        ["pillar3"],
        MagicMock(),
        MagicMock(),
    )

    assert resolve_calls[0] == ["pillar3"]


def test_run_retrieval_parallel_execution(monkeypatch):
    """Multiple units run and all results returned."""
    _set_orchestrator_env(monkeypatch)

    consolidate_results_arg = []

    def mock_consolidate(_query, results, _llm, metrics=None, **_kw):
        """Capture combo results passed to consolidate."""
        del metrics
        consolidate_results_arg.extend(results)
        return _make_consolidated_result()

    def resolve_two(_conn, _bank, _period, _sources):
        """Return two version rows."""
        return [
            {
                "document_version_id": 38,
                "data_source": "pillar3",
                "filename": "file.xlsx",
            },
            {
                "document_version_id": 39,
                "data_source": "investor-slides",
                "filename": "slides.pdf",
            },
        ]

    monkeypatch.setattr(
        f"{_MOD}.resolve_document_version_ids",
        resolve_two,
    )
    _stub_prepare(monkeypatch)
    _stub_connection(monkeypatch)
    _stub_tokens(monkeypatch)
    _stub_search_pipeline(monkeypatch)
    _stub_research(monkeypatch)
    monkeypatch.setattr(
        f"{_MOD}.consolidate_results",
        mock_consolidate,
    )

    run_retrieval(
        "query",
        [_make_combo()],
        None,
        MagicMock(),
        MagicMock(),
    )

    assert len(consolidate_results_arg) == 2


def test_resolve_research_units():
    """Combo + source resolution builds correct tuples."""
    conn = MagicMock()
    versions = [
        {
            "document_version_id": 38,
            "data_source": "pillar3",
            "filename": "rbc_pillar3.xlsx",
        },
        {
            "document_version_id": 39,
            "data_source": "investor-slides",
            "filename": "rbc_slides.pdf",
        },
    ]

    target = f"{_MOD}.resolve_document_version_ids"
    with patch(target, return_value=versions):
        units = _resolve_research_units(
            conn,
            [_make_combo()],
            None,
        )

    assert len(units) == 2
    combo_0, source_0 = units[0]
    assert combo_0["bank"] == "RBC"
    assert source_0["data_source"] == "pillar3"
    assert source_0["document_version_id"] == 38

    combo_1, source_1 = units[1]
    assert combo_1["bank"] == "RBC"
    assert source_1["data_source"] == "investor-slides"
    assert source_1["document_version_id"] == 39


def test_search_result_to_expanded():
    """Conversion helper maps fields correctly."""
    result = _make_search_result(
        cuid="cu_5",
        page=10,
        token_count=75,
    )
    expanded = _search_result_to_expanded(result)

    assert expanded["content_unit_id"] == "cu_5"
    assert expanded["raw_content"] == "CET1 ratio was 13.7%"
    assert expanded["page_number"] == 10
    assert expanded["section_id"] == "s_001"
    assert expanded["section_title"] == "KM1"
    assert expanded["is_original"] is True
    assert expanded["token_count"] == 75
