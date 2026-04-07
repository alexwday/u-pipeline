"""Tests for iterative research agent loop."""

import json
from unittest.mock import MagicMock

import pytest

from retriever.models import (
    ComboSpec,
    ExpandedChunk,
    PreparedQuery,
    ResearchFinding,
    ResearchIteration,
    SourceSpec,
)
from retriever.stages.research import (
    _format_chunks,
    _format_previous_research,
    _format_research_input,
    _order_chunks_by_page,
    _parse_research_response,
    _search_additional,
    research_combo_source,
)


def _make_expanded_chunk(
    cuid: str,
    page: int = 1,
    section_id: str = "s_001",
    section_title: str = "KM1",
    content: str = "Sample content",
    token_count: int = 50,
) -> ExpandedChunk:
    """Build an ExpandedChunk fixture."""
    return ExpandedChunk(
        content_unit_id=cuid,
        raw_content=content,
        page_number=page,
        section_id=section_id,
        section_title=section_title,
        chunk_context="context",
        is_original=True,
        token_count=token_count,
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


def _make_combo() -> ComboSpec:
    """Build a ComboSpec fixture."""
    return ComboSpec(bank="RBC", period="2026_Q1")


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


def _make_findings(
    text: str = "CET1 ratio was 13.7%",
    page: int = 3,
    location: str = "Sheet KM1",
) -> list[dict]:
    """Build a single-finding list for test responses."""
    return [
        {
            "finding": text,
            "page": page,
            "location_detail": location,
            "metric_name": "CET1 Ratio",
            "metric_value": "13.7%",
            "period": "Q1 2026",
            "segment": "Enterprise",
        }
    ]


def _make_llm_response(
    findings: list[dict] | None = None,
    additional_queries: list[str] | None = None,
    confidence: float = 0.85,
) -> dict:
    """Build a mock LLM structured findings response."""
    if findings is None:
        findings = _make_findings()
    if additional_queries is None:
        additional_queries = []
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "produce_research",
                                "arguments": json.dumps(
                                    {
                                        "findings": findings,
                                        "additional_queries": (
                                            additional_queries
                                        ),
                                        "confidence": confidence,
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }


def _make_legacy_llm_response(
    findings: list[dict] | None = None,
    additional_queries: list[str] | None = None,
    confidence: float = 0.85,
) -> dict:
    """Build a mock legacy function_call research response."""
    if findings is None:
        findings = _make_findings()
    if additional_queries is None:
        additional_queries = []
    return {
        "choices": [
            {
                "message": {
                    "function_call": {
                        "name": "produce_research",
                        "arguments": json.dumps(
                            {
                                "findings": findings,
                                "additional_queries": additional_queries,
                                "confidence": confidence,
                            }
                        ),
                    }
                }
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }


def _make_no_tool_response() -> dict:
    """Build a mock LLM response with plain text and no tool call."""
    return {
        "choices": [
            {
                "message": {
                    "content": "RBC's CET1 ratio was 13.7%.",
                }
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        },
    }


def _set_research_env(monkeypatch):
    """Set research stage env vars."""
    monkeypatch.setenv("RESEARCH_MODEL", "gpt-test")
    monkeypatch.setenv("RESEARCH_MAX_TOKENS", "4000")
    monkeypatch.setenv("RESEARCH_TEMPERATURE", "")
    monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "3")
    monkeypatch.setenv("RESEARCH_ADDITIONAL_SEARCH_TOP_K", "10")


def test_research_single_iteration(monkeypatch):
    """Query answered in 1 iteration with no extra queries."""
    _set_research_env(monkeypatch)

    chunks = [
        _make_expanded_chunk("cu_1", page=3),
        _make_expanded_chunk("cu_2", page=5),
    ]
    llm = MagicMock()
    llm.call.return_value = _make_llm_response(
        findings=_make_findings(
            text="CET1 ratio was 13.7%", page=3, location="Sheet KM1"
        ),
        additional_queries=[],
        confidence=0.9,
    )

    metrics: dict = {}
    result = research_combo_source(
        conn=None,
        prepared=_make_prepared_query(),
        expanded=chunks,
        combo=_make_combo(),
        source=_make_source(),
        llm=llm,
        metrics=metrics,
    )

    assert len(result["research_iterations"]) == 1
    assert result["research_iterations"][0]["confidence"] == 0.9
    assert result["findings"][0]["finding"] == "CET1 ratio was 13.7%"
    assert result["findings"][0]["page"] == 3
    assert llm.call.call_count == 1
    assert metrics["iterations"] == 1
    assert metrics["llm_calls"] == 1


def test_parse_research_response_accepts_legacy_function_call():
    """Legacy function_call payloads are still accepted."""
    response = _make_legacy_llm_response(
        findings=_make_findings(
            text="CET1 ratio was 13.7%", page=3, location="Sheet KM1"
        ),
        additional_queries=["follow-up"],
        confidence=0.7,
    )

    findings, additional_queries, confidence = _parse_research_response(
        response
    )

    assert len(findings) == 1
    assert findings[0]["finding"] == "CET1 ratio was 13.7%"
    assert findings[0]["page"] == 3
    assert additional_queries == ["follow-up"]
    assert confidence == 0.7


def test_research_multiple_iterations(monkeypatch):
    """Two iterations with additional queries bring new chunks."""
    _set_research_env(monkeypatch)

    chunks = [_make_expanded_chunk("cu_1", page=3)]

    iter1_resp = _make_llm_response(
        findings=_make_findings(
            text="CET1 ratio was 13.7%", page=3, location="Sheet KM1"
        ),
        additional_queries=["CET1 breakdown"],
        confidence=0.6,
    )
    iter2_resp = _make_llm_response(
        findings=_make_findings(
            text="CET1 capital $21.1B", page=10, location="Sheet CC1"
        ),
        additional_queries=[],
        confidence=0.9,
    )
    llm = MagicMock()
    llm.call.side_effect = [iter1_resp, iter2_resp]
    llm.embed.return_value = [[0.5] * 10]

    search_hits = [
        {
            "content_unit_id": "cu_5",
            "raw_content": "CET1 breakdown details",
            "page_number": 10,
            "section_id": "s_002",
            "chunk_context": "ctx",
            "token_count": 40,
        }
    ]
    monkeypatch.setattr(
        "retriever.stages.research.search_by_content_vector",
        lambda conn, dvid, emb, top_k: search_hits,
    )
    monkeypatch.setattr(
        "retriever.stages.research.get_section_info",
        lambda conn, dvid, sid: {
            "section_id": sid,
            "title": "CC1",
        },
    )

    result = research_combo_source(
        conn=None,
        prepared=_make_prepared_query(),
        expanded=chunks,
        combo=_make_combo(),
        source=_make_source(),
        llm=llm,
    )

    assert len(result["research_iterations"]) == 2
    assert result["research_iterations"][0]["confidence"] == 0.6
    assert result["research_iterations"][1]["confidence"] == 0.9
    assert len(result["findings"]) >= 1
    assert llm.call.call_count == 2
    assert llm.embed.call_count == 1


def test_search_additional_populates_trace(monkeypatch):
    """Follow-up search traces retain per-query hits and additions."""
    _set_research_env(monkeypatch)

    llm = MagicMock()
    llm.embed.return_value = [[0.5] * 10]
    search_hits = [
        {
            "content_unit_id": "cu_5",
            "raw_content": "CET1 breakdown details",
            "page_number": 10,
            "section_id": "s_002",
            "chunk_context": "ctx",
            "chunk_header": "CC1",
            "token_count": 40,
            "distance": 0.12,
        }
    ]
    monkeypatch.setattr(
        "retriever.stages.research.search_by_content_vector",
        lambda conn, dvid, emb, top_k: search_hits,
    )
    monkeypatch.setattr(
        "retriever.stages.research.get_section_info",
        lambda conn, dvid, sid: {"section_id": sid, "title": "CC1"},
    )

    metrics: dict = {}
    trace: dict = {}
    new_chunks = _search_additional(
        conn=None,
        doc_version_id=38,
        queries=["CET1 breakdown"],
        llm=llm,
        top_k=10,
        seen_ids=set(),
        metrics=metrics,
        trace=trace,
    )

    assert [chunk["content_unit_id"] for chunk in new_chunks] == ["cu_5"]
    assert trace["queries"][0]["query"] == "CET1 breakdown"
    assert trace["queries"][0]["hits"][0]["content_unit_id"] == "cu_5"
    assert trace["queries"][0]["new_chunk_ids"] == ["cu_5"]
    assert metrics["new_chunks"] == 1


def test_research_populates_trace(monkeypatch):
    """Research traces retain iteration inputs, follow-ups, and lineage."""
    _set_research_env(monkeypatch)

    chunks = [_make_expanded_chunk("cu_1", page=3)]
    llm = MagicMock()
    llm.call.side_effect = [
        _make_llm_response(
            findings=_make_findings(
                text="CET1 ratio was 13.7%", page=3, location="Sheet KM1"
            ),
            additional_queries=["CET1 breakdown"],
            confidence=0.6,
        ),
        _make_llm_response(
            findings=_make_findings(
                text="CET1 capital was $21.1B",
                page=10,
                location="Sheet CC1",
            ),
            additional_queries=[],
            confidence=0.9,
        ),
    ]
    llm.embed.return_value = [[0.5] * 10]
    search_hits = [
        {
            "content_unit_id": "cu_5",
            "raw_content": "CET1 breakdown details",
            "page_number": 10,
            "section_id": "s_002",
            "chunk_context": "ctx",
            "chunk_header": "CC1",
            "token_count": 40,
            "distance": 0.12,
        }
    ]
    monkeypatch.setattr(
        "retriever.stages.research.search_by_content_vector",
        lambda conn, dvid, emb, top_k: search_hits,
    )
    monkeypatch.setattr(
        "retriever.stages.research.get_section_info",
        lambda conn, dvid, sid: {"section_id": sid, "title": "CC1"},
    )

    trace: dict = {}
    result = research_combo_source(
        conn=None,
        prepared=_make_prepared_query(),
        expanded=chunks,
        combo=_make_combo(),
        source=_make_source(),
        llm=llm,
        trace=trace,
        initial_chunk_origins={
            "cu_1": [{"origin_stage": "initial_expand"}],
        },
    )

    assert len(result["research_iterations"]) == 2
    assert trace["iterations"][0]["input_chunk_ids"] == ["cu_1"]
    assert trace["iterations"][0]["follow_up_search"]["queries"][0][
        "query"
    ] == ("CET1 breakdown")
    assert trace["follow_up_chunk_origins"]["cu_5"][0]["iteration"] == 1
    assert trace["final_chunk_lineage"]["cu_1"] == [
        {"origin_stage": "initial_expand"}
    ]
    assert trace["final_chunk_lineage"]["cu_5"][0]["query"] == (
        "CET1 breakdown"
    )


def test_research_retries_uncited_meta_output(monkeypatch):
    """Uncited meta output is retried once with a correction."""
    _set_research_env(monkeypatch)

    chunks = [_make_expanded_chunk("cu_1", page=38)]
    llm = MagicMock()
    llm.call.side_effect = [
        _make_llm_response(
            findings=[],
            additional_queries=[],
            confidence=0.7,
        ),
        _make_llm_response(
            findings=_make_findings(
                text="TLAC ratio was 30.9%", page=38, location="Sheet KM1"
            ),
            additional_queries=[],
            confidence=0.95,
        ),
    ]

    metrics: dict = {}
    result = research_combo_source(
        conn=None,
        prepared=_make_prepared_query(
            query="What is RBC's TLAC ratio and leverage ratio?"
        ),
        expanded=chunks,
        combo=_make_combo(),
        source=_make_source(data_source="rts", doc_id=39),
        llm=llm,
        metrics=metrics,
    )

    assert result["findings"][0]["finding"] == "TLAC ratio was 30.9%"
    assert result["findings"][0]["page"] == 38
    assert metrics["llm_calls"] == 2
    assert llm.call.call_count == 2


def test_research_retries_draft_output(monkeypatch):
    """Draft-note output with question marks is retried once."""
    _set_research_env(monkeypatch)

    chunks = [_make_expanded_chunk("cu_1", page=25)]
    llm = MagicMock()
    llm.call.side_effect = [
        _make_llm_response(
            findings=[],
            additional_queries=[],
            confidence=0.7,
        ),
        _make_llm_response(
            findings=_make_findings(
                text="Total ACL was $7,767 million",
                page=25,
                location="Sheet CC1",
            ),
            additional_queries=[],
            confidence=0.95,
        ),
    ]

    metrics: dict = {}
    result = research_combo_source(
        conn=None,
        prepared=_make_prepared_query(
            query="What are RBC's key credit risk exposures and provisions?"
        ),
        expanded=chunks,
        combo=_make_combo(),
        source=_make_source(data_source="rts", doc_id=39),
        llm=llm,
        metrics=metrics,
    )

    assert result["findings"][0]["finding"] == "Total ACL was $7,767 million"
    assert result["findings"][0]["page"] == 25
    assert metrics["llm_calls"] == 2
    assert llm.call.call_count == 2


def test_research_retries_missing_tool_call(monkeypatch):
    """Plain-text responses without a tool call are retried once."""
    _set_research_env(monkeypatch)

    chunks = [_make_expanded_chunk("cu_1", page=3)]
    llm = MagicMock()
    llm.call.side_effect = [
        _make_no_tool_response(),
        _make_llm_response(
            findings=_make_findings(
                text="CET1 ratio was 13.7%", page=3, location="Sheet KM1"
            ),
            additional_queries=[],
            confidence=0.95,
        ),
    ]

    metrics: dict = {}
    result = research_combo_source(
        conn=None,
        prepared=_make_prepared_query(),
        expanded=chunks,
        combo=_make_combo(),
        source=_make_source(),
        llm=llm,
        metrics=metrics,
    )

    assert result["findings"][0]["finding"] == "CET1 ratio was 13.7%"
    assert result["findings"][0]["page"] == 3
    assert metrics["llm_calls"] == 2
    assert llm.call.call_count == 2
    second_prompt = llm.call.call_args_list[1].kwargs["messages"][1]["content"]
    assert "Use the provided tool" in second_prompt


def test_research_max_iterations_stop(monkeypatch):
    """Loop stops at max_iterations even with queries."""
    monkeypatch.setenv("RESEARCH_MODEL", "gpt-test")
    monkeypatch.setenv("RESEARCH_MAX_TOKENS", "4000")
    monkeypatch.setenv("RESEARCH_TEMPERATURE", "")
    monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "2")
    monkeypatch.setenv("RESEARCH_ADDITIONAL_SEARCH_TOP_K", "10")

    chunks = [_make_expanded_chunk("cu_1")]

    iter1_resp = _make_llm_response(
        findings=_make_findings(
            text="Finding 1", page=1, location="Sheet KM1"
        ),
        additional_queries=["more info"],
        confidence=0.5,
    )
    iter2_resp = _make_llm_response(
        findings=_make_findings(
            text="Finding 2", page=5, location="Sheet CC1"
        ),
        additional_queries=["still more"],
        confidence=0.7,
    )
    llm = MagicMock()
    llm.call.side_effect = [iter1_resp, iter2_resp]
    llm.embed.return_value = [[0.5] * 10]

    search_hits = [
        {
            "content_unit_id": "cu_10",
            "raw_content": "Extra content",
            "page_number": 5,
            "section_id": "s_003",
            "chunk_context": "ctx",
            "token_count": 30,
        }
    ]
    monkeypatch.setattr(
        "retriever.stages.research.search_by_content_vector",
        lambda conn, dvid, emb, top_k: search_hits,
    )
    monkeypatch.setattr(
        "retriever.stages.research.get_section_info",
        lambda conn, dvid, sid: {
            "section_id": sid,
            "title": "Section",
        },
    )

    result = research_combo_source(
        conn=None,
        prepared=_make_prepared_query(),
        expanded=chunks,
        combo=_make_combo(),
        source=_make_source(),
        llm=llm,
    )

    assert len(result["research_iterations"]) == 2
    assert llm.call.call_count == 2


def test_research_no_new_chunks_stops(monkeypatch):
    """Additional query returns nothing, loop stops."""
    _set_research_env(monkeypatch)

    chunks = [_make_expanded_chunk("cu_1")]

    iter1_resp = _make_llm_response(
        findings=_make_findings(
            text="Partial finding", page=1, location="Sheet KM1"
        ),
        additional_queries=["missing info"],
        confidence=0.4,
    )
    llm = MagicMock()
    llm.call.return_value = iter1_resp
    llm.embed.return_value = [[0.5] * 10]

    monkeypatch.setattr(
        "retriever.stages.research.search_by_content_vector",
        lambda conn, dvid, emb, top_k: [
            {
                "content_unit_id": "cu_1",
                "raw_content": "Same content",
                "page_number": 1,
                "section_id": "s_001",
                "chunk_context": "ctx",
                "token_count": 50,
            }
        ],
    )

    result = research_combo_source(
        conn=None,
        prepared=_make_prepared_query(),
        expanded=chunks,
        combo=_make_combo(),
        source=_make_source(),
        llm=llm,
    )

    assert len(result["research_iterations"]) == 1
    assert llm.call.call_count == 1


def test_research_repeated_follow_up_queries_stop(monkeypatch):
    """Exact repeated follow-up queries stop the loop early."""
    _set_research_env(monkeypatch)

    chunks = [_make_expanded_chunk("cu_1")]
    llm = MagicMock()
    llm.call.side_effect = [
        _make_llm_response(
            findings=_make_findings(
                text="CET1 ratio was 13.7%", page=9, location="Sheet KM1"
            ),
            additional_queries=[
                "Provide RBC's Tier 1 capital ratio for Q1/2026.",
                "Provide RBC's Total Capital ratio for Q1/2026.",
            ],
            confidence=0.6,
        ),
        _make_llm_response(
            findings=_make_findings(
                text="CET1 ratio remains 13.7%",
                page=9,
                location="Sheet KM1",
            ),
            additional_queries=[
                "Provide RBC's Tier 1 capital ratio for Q1/2026.",
                "Provide RBC's Total Capital ratio for Q1/2026.",
            ],
            confidence=0.6,
        ),
    ]
    llm.embed.return_value = [[0.5] * 10]

    monkeypatch.setattr(
        "retriever.stages.research.search_by_content_vector",
        lambda conn, dvid, emb, top_k: [
            {
                "content_unit_id": "cu_2",
                "raw_content": "Capital ratio detail",
                "page_number": 9,
                "section_id": "s_002",
                "chunk_context": "ctx",
                "token_count": 40,
            }
        ],
    )
    monkeypatch.setattr(
        "retriever.stages.research.get_section_info",
        lambda conn, dvid, sid: {
            "section_id": sid,
            "title": "Capital: Strong position",
        },
    )

    metrics: dict = {}
    result = research_combo_source(
        conn=None,
        prepared=_make_prepared_query(
            query=(
                "Describe RBC's capital adequacy position including "
                "CET1, Tier 1, and Total Capital ratios."
            )
        ),
        expanded=chunks,
        combo=_make_combo(),
        source=_make_source(data_source="investor-slides"),
        llm=llm,
        metrics=metrics,
    )

    assert len(result["research_iterations"]) == 2
    assert llm.call.call_count == 2
    assert llm.embed.call_count == 1
    assert metrics["iterations"] == 2


def test_format_chunks():
    """Chunks grouped by section with citation page headers."""
    chunks = [
        _make_expanded_chunk(
            "cu_1",
            page=3,
            section_title="KM1",
            content="| Row | CET1 | 13.7% |",
        ),
        _make_expanded_chunk(
            "cu_2",
            page=10,
            section_title="CC1",
            content="Capital composition details",
        ),
    ]
    text = _format_chunks(chunks)

    assert "=== KM1 (Page 3) ===" in text
    assert "| Row | CET1 | 13.7% |" in text
    assert "=== CC1 (Page 10) ===" in text
    assert "Capital composition details" in text
    assert "[Citation Page: 3]" in text
    assert "[Citation Page: 10]" in text


def test_format_chunks_no_section_title():
    """Chunks without section title grouped under Ungrouped Content."""
    chunks = [
        _make_expanded_chunk(
            "cu_1",
            page=5,
            section_title="",
            content="Orphan content",
        ),
    ]
    text = _format_chunks(chunks)

    assert "=== Ungrouped Content (Page 5) ===" in text
    assert "Orphan content" in text


def test_order_chunks_by_page():
    """Chunks ordered by page number then content_unit_id."""
    chunks = [
        _make_expanded_chunk("cu_3", page=10),
        _make_expanded_chunk("cu_1", page=3),
        _make_expanded_chunk("cu_2", page=3),
    ]
    ordered = _order_chunks_by_page(chunks)
    assert [c["content_unit_id"] for c in ordered] == [
        "cu_1",
        "cu_2",
        "cu_3",
    ]


def test_format_chunks_empty():
    """Empty chunk list returns empty string."""
    assert _format_chunks([]) == ""


def test_format_previous_research():
    """Previous iterations formatted with structured findings."""
    iterations = [
        ResearchIteration(
            iteration=1,
            research="CET1 was 13.7% [Page 3, Sheet KM1].",
            additional_queries=["more"],
            confidence=0.7,
            findings=[
                ResearchFinding(
                    finding="CET1 was 13.7%",
                    page=3,
                    location_detail="Sheet KM1",
                ),
            ],
        ),
        ResearchIteration(
            iteration=2,
            research="Capital $21.1B [Page 10, Capital section].",
            additional_queries=[],
            confidence=0.9,
            findings=[
                ResearchFinding(
                    finding="Capital $21.1B",
                    page=10,
                    location_detail="Capital section",
                ),
            ],
        ),
    ]
    text = _format_previous_research(iterations)

    assert "<previous_research>" in text
    assert "</previous_research>" in text
    assert "[Iteration 1, confidence: 0.7]" in text
    assert "CET1 was 13.7%" in text
    assert "(Page 3, Sheet KM1)" in text
    assert "[Iteration 2, confidence: 0.9]" in text
    assert "Capital $21.1B" in text


def test_format_previous_research_empty():
    """No previous iterations returns empty string."""
    assert _format_previous_research([]) == ""


def test_parse_research_response_valid():
    """Valid tool call parsed correctly."""
    response = _make_llm_response(
        findings=_make_findings(text="Finding", page=3, location="Sheet KM1"),
        additional_queries=["query 2"],
        confidence=0.85,
    )
    findings, queries, confidence = _parse_research_response(response)

    assert len(findings) == 1
    assert findings[0]["finding"] == "Finding"
    assert findings[0]["page"] == 3
    assert queries == ["query 2"]
    assert confidence == 0.85


def test_parse_research_response_malformed():
    """Bad response structures raise ValueError."""
    with pytest.raises(ValueError, match="no choices"):
        _parse_research_response({"choices": []})

    with pytest.raises(ValueError, match="did not return"):
        _parse_research_response({"choices": [{"message": {}}]})

    bad_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "produce_research",
                                "arguments": json.dumps({"rationale": "oops"}),
                            }
                        }
                    ]
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="findings"):
        _parse_research_response(bad_response)


def test_parse_research_response_bad_types():
    """Non-array findings raises ValueError."""
    bad_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "produce_research",
                                "arguments": json.dumps(
                                    {
                                        "findings": 123,
                                        "additional_queries": [],
                                        "confidence": 0.5,
                                        "rationale": "x",
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="findings must be an array"):
        _parse_research_response(bad_response)


def test_search_additional_deduplicates(monkeypatch):
    """Already-seen chunks filtered out of results."""
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embed")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "10")

    llm = MagicMock()
    llm.embed.return_value = [[0.5] * 10]

    hits = [
        {
            "content_unit_id": "cu_1",
            "raw_content": "Already seen",
            "page_number": 1,
            "section_id": "s_001",
            "chunk_context": "ctx",
            "token_count": 50,
        },
        {
            "content_unit_id": "cu_new",
            "raw_content": "New content",
            "page_number": 5,
            "section_id": "s_002",
            "chunk_context": "ctx",
            "token_count": 40,
        },
    ]
    monkeypatch.setattr(
        "retriever.stages.research.search_by_content_vector",
        lambda conn, dvid, emb, top_k: hits,
    )
    monkeypatch.setattr(
        "retriever.stages.research.get_section_info",
        lambda conn, dvid, sid: {
            "section_id": sid,
            "title": "Title",
        },
    )

    seen = {"cu_1"}
    new_chunks = _search_additional(
        conn=None,
        doc_version_id=38,
        queries=["query"],
        llm=llm,
        top_k=10,
        seen_ids=seen,
    )

    assert len(new_chunks) == 1
    assert new_chunks[0]["content_unit_id"] == "cu_new"


def test_search_additional_embeds_queries(monkeypatch):
    """Additional queries are embedded in a single batch call."""
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embed")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "10")

    llm = MagicMock()
    llm.embed.return_value = [[0.5] * 10]

    monkeypatch.setattr(
        "retriever.stages.research.search_by_content_vector",
        lambda conn, dvid, emb, top_k: [],
    )

    queries = ["query one", "query two"]
    _search_additional(
        conn=None,
        doc_version_id=38,
        queries=queries,
        llm=llm,
        top_k=10,
        seen_ids=set(),
    )

    llm.embed.assert_called_once_with(
        queries,
        model="test-embed",
        dimensions=10,
    )


def test_research_produces_combo_source_result(monkeypatch):
    """Return type has all required ComboSourceResult fields."""
    _set_research_env(monkeypatch)

    chunks = [
        _make_expanded_chunk("cu_1", token_count=100),
        _make_expanded_chunk("cu_2", token_count=200),
    ]
    llm = MagicMock()
    llm.call.return_value = _make_llm_response(
        findings=_make_findings(
            text="Research text", page=1, location="Sheet KM1"
        ),
        confidence=0.8,
    )

    combo = _make_combo()
    source = _make_source()

    result = research_combo_source(
        conn=None,
        prepared=_make_prepared_query(),
        expanded=chunks,
        combo=combo,
        source=source,
        llm=llm,
    )

    assert result["combo"] == combo
    assert result["source"] == source
    assert isinstance(result["research_iterations"], list)
    assert isinstance(result["findings"], list)
    assert len(result["findings"]) == 1
    assert result["findings"][0]["finding"] == "Research text"
    assert result["chunk_count"] == 2
    assert result["total_tokens"] == 300


def test_format_research_input():
    """Placeholders substituted in user prompt template."""
    prompt = {
        "user_prompt": (
            "Query: {query}\n"
            "Source: {source_label}\n"
            "Bank: {bank}, Period: {period}\n"
            "{previous_research}\n"
            "Content: {chunks}"
        ),
    }
    text = _format_research_input(
        query="CET1 ratio",
        source_label="pillar3 / file.xlsx",
        bank="RBC",
        period="2026_Q1",
        previous_research="<prev>data</prev>",
        chunks="[Page 3]\nTable data",
        prompt=prompt,
    )

    assert "CET1 ratio" in text
    assert "pillar3 / file.xlsx" in text
    assert "RBC" in text
    assert "2026_Q1" in text
    assert "<prev>data</prev>" in text
    assert "[Page 3]" in text


def test_parse_research_response_bad_queries_type():
    """Non-list additional_queries raises ValueError."""
    bad_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "produce_research",
                                "arguments": json.dumps(
                                    {
                                        "findings": _make_findings(),
                                        "additional_queries": "bad",
                                        "confidence": 0.5,
                                        "rationale": "x",
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="must be a list"):
        _parse_research_response(bad_response)


def test_parse_research_response_bad_confidence_type():
    """Non-numeric confidence raises ValueError."""
    bad_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "produce_research",
                                "arguments": json.dumps(
                                    {
                                        "findings": _make_findings(),
                                        "additional_queries": [],
                                        "confidence": "high",
                                        "rationale": "x",
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="must be a number"):
        _parse_research_response(bad_response)


def test_search_additional_empty_embeddings(monkeypatch):
    """Empty embedding result skips that query."""
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embed")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "10")

    llm = MagicMock()
    llm.embed.return_value = []

    monkeypatch.setattr(
        "retriever.stages.research.search_by_content_vector",
        lambda conn, dvid, emb, top_k: [],
    )

    new_chunks = _search_additional(
        conn=None,
        doc_version_id=38,
        queries=["query"],
        llm=llm,
        top_k=10,
        seen_ids=set(),
    )

    assert not new_chunks
