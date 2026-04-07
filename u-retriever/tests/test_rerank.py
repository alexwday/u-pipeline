"""Tests for LLM-based reranking stage."""

import json
from unittest.mock import MagicMock

import httpx
import pytest
from openai import BadRequestError

from retriever.models import SearchResult
from retriever.stages.rerank import (
    _build_candidate_list,
    _format_rerank_input,
    _parse_rerank_response,
    _truncate_preview,
    rerank_results,
)


def _make_search_result(
    cuid: str,
    score: float = 0.5,
    section_id: str = "s_001",
    page: int = 1,
    keywords: list[str] | None = None,
    content: str = "Sample content text",
) -> SearchResult:
    """Build a SearchResult fixture."""
    return SearchResult(
        content_unit_id=cuid,
        raw_content=content,
        chunk_id=f"ch_{cuid}",
        section_id=section_id,
        page_number=page,
        chunk_context="context",
        chunk_header="header",
        keywords=keywords if keywords is not None else ["kw1"],
        entities=["ent1"],
        token_count=50,
        score=score,
        strategy_scores={"bm25": score},
    )


def _make_llm_response(
    remove_indices: list[int],
    rationale: str = "test rationale",
) -> dict:
    """Build a mock LLM rerank response."""
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "filter_chunks",
                                "arguments": json.dumps(
                                    {
                                        "remove_indices": remove_indices,
                                        "rationale": rationale,
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }


def _make_legacy_llm_response(remove_indices: list[int]) -> dict:
    """Build a mock LLM rerank response with function_call."""
    return {
        "choices": [
            {
                "message": {
                    "function_call": {
                        "name": "filter_chunks",
                        "arguments": json.dumps(
                            {"remove_indices": remove_indices}
                        ),
                    }
                }
            }
        ]
    }


def _make_output_limit_error() -> BadRequestError:
    """Build a BadRequestError matching LLM output-limit failures."""
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://example.test/v1/chat"),
    )
    return BadRequestError(
        (
            "Could not finish the message because max_tokens or "
            "model output limit was reached."
        ),
        response=response,
        body=None,
    )


def _make_no_tool_response() -> dict:
    """Build a mock LLM response with plain text and no tool call."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Remove candidate 1 as irrelevant.",
                }
            }
        ]
    }


def test_rerank_removes_irrelevant(monkeypatch):
    """LLM returns indices to remove, those are filtered out."""
    monkeypatch.setenv("RERANK_MODEL", "gpt-test")
    monkeypatch.setenv("RERANK_MAX_TOKENS", "100")
    monkeypatch.setenv("RERANK_TEMPERATURE", "")
    monkeypatch.setenv("RERANK_REASONING_EFFORT", "")
    monkeypatch.setenv("RERANK_PREVIEW_MAX_TOKENS", "200")
    monkeypatch.setenv("RERANK_CANDIDATE_LIMIT", "30")
    monkeypatch.setenv("RERANK_MIN_KEEP", "10")

    results = [
        _make_search_result("cu_1", score=0.9),
        _make_search_result("cu_2", score=0.5),
        _make_search_result("cu_3", score=0.3),
    ]
    llm = MagicMock()
    llm.call.return_value = _make_llm_response([1])

    metrics: dict = {}
    kept = rerank_results(results, "CET1 ratio", llm, metrics=metrics)

    assert len(kept) == 2
    cuids = [r["content_unit_id"] for r in kept]
    assert "cu_1" in cuids
    assert "cu_3" in cuids
    assert "cu_2" not in cuids
    assert metrics["removed"] == 1
    assert metrics["candidates_shown"] == 3


def test_rerank_populates_trace(monkeypatch):
    """Rerank traces retain candidate snapshots and decisions."""
    monkeypatch.setenv("RERANK_MODEL", "gpt-test")
    monkeypatch.setenv("RERANK_MAX_TOKENS", "100")
    monkeypatch.setenv("RERANK_TEMPERATURE", "")
    monkeypatch.setenv("RERANK_REASONING_EFFORT", "")
    monkeypatch.setenv("RERANK_PREVIEW_MAX_TOKENS", "120")
    monkeypatch.setenv("RERANK_CANDIDATE_LIMIT", "30")
    monkeypatch.setenv("RERANK_MIN_KEEP", "10")

    results = [
        _make_search_result("cu_1", score=0.9),
        _make_search_result("cu_2", score=0.5),
    ]
    llm = MagicMock()
    llm.call.return_value = _make_llm_response([1])

    trace: dict = {}
    kept = rerank_results(results, "CET1 ratio", llm, trace=trace)

    assert [result["content_unit_id"] for result in kept] == ["cu_1"]
    assert trace["removed_ids"] == ["cu_2"]
    assert trace["kept_ids"] == ["cu_1"]
    assert trace["remove_indices"] == [1]
    assert trace["attempts"][0]["status"] == "success"
    assert trace["candidates"][0]["index"] == 0


def test_rerank_keeps_all_when_empty_indices(monkeypatch):
    """Empty remove list keeps everything."""
    monkeypatch.setenv("RERANK_MODEL", "gpt-test")
    monkeypatch.setenv("RERANK_MAX_TOKENS", "100")
    monkeypatch.setenv("RERANK_TEMPERATURE", "")
    monkeypatch.setenv("RERANK_REASONING_EFFORT", "")
    monkeypatch.setenv("RERANK_PREVIEW_MAX_TOKENS", "200")
    monkeypatch.setenv("RERANK_CANDIDATE_LIMIT", "30")
    monkeypatch.setenv("RERANK_MIN_KEEP", "10")

    results = [
        _make_search_result("cu_1"),
        _make_search_result("cu_2"),
    ]
    llm = MagicMock()
    llm.call.return_value = _make_llm_response([])

    kept = rerank_results(results, "capital ratio", llm)

    assert len(kept) == 2


def test_rerank_retries_with_smaller_preview_on_output_limit(monkeypatch):
    """Output-limit rerank failures retry with shorter previews."""
    monkeypatch.setenv("RERANK_MODEL", "gpt-test")
    monkeypatch.setenv("RERANK_MAX_TOKENS", "100")
    monkeypatch.setenv("RERANK_TEMPERATURE", "")
    monkeypatch.setenv("RERANK_REASONING_EFFORT", "")
    monkeypatch.setenv("RERANK_PREVIEW_MAX_TOKENS", "120")
    monkeypatch.setenv("RERANK_CANDIDATE_LIMIT", "30")
    monkeypatch.setenv("RERANK_MIN_KEEP", "10")

    results = [
        _make_search_result(
            "cu_1",
            content="Capital ratio details " * 80,
        ),
        _make_search_result(
            "cu_2",
            content="Unrelated glossary entry " * 80,
        ),
    ]
    llm = MagicMock()
    llm.call.side_effect = [
        _make_output_limit_error(),
        _make_llm_response([1]),
    ]

    metrics: dict = {}
    kept = rerank_results(results, "CET1 ratio", llm, metrics=metrics)

    assert [result["content_unit_id"] for result in kept] == ["cu_1"]
    assert llm.call.call_count == 2
    first_prompt = llm.call.call_args_list[0].kwargs["messages"][1]["content"]
    second_prompt = llm.call.call_args_list[1].kwargs["messages"][1]["content"]
    assert len(second_prompt) < len(first_prompt)
    assert metrics["llm_calls"] == 2
    assert metrics["retry_count"] == 1
    assert metrics["preview_max_tokens"] == 60
    assert metrics["fallback_keep_all"] is False


def test_rerank_falls_back_to_all_results_after_output_limit_retries(
    monkeypatch,
):
    """Repeated output-limit rerank failures keep all candidates."""
    monkeypatch.setenv("RERANK_MODEL", "gpt-test")
    monkeypatch.setenv("RERANK_MAX_TOKENS", "100")
    monkeypatch.setenv("RERANK_TEMPERATURE", "")
    monkeypatch.setenv("RERANK_PREVIEW_MAX_TOKENS", "120")

    results = [
        _make_search_result("cu_1", content="Capital ratio details " * 80),
        _make_search_result("cu_2", content="Supplementary detail " * 80),
    ]
    llm = MagicMock()
    llm.call.side_effect = [
        _make_output_limit_error(),
        _make_output_limit_error(),
        _make_output_limit_error(),
    ]

    metrics: dict = {}
    kept = rerank_results(results, "CET1 ratio", llm, metrics=metrics)

    assert kept == results
    assert llm.call.call_count == 3
    assert metrics["kept"] == 2
    assert metrics["removed"] == 0
    assert metrics["llm_calls"] == 3
    assert metrics["retry_count"] == 2
    assert metrics["preview_max_tokens"] == 40
    assert metrics["fallback_keep_all"] is True


def test_rerank_keeps_all_when_response_has_no_tool_call(monkeypatch):
    """Plain-text rerank responses fall back to keeping all candidates."""
    monkeypatch.setenv("RERANK_MODEL", "gpt-test")
    monkeypatch.setenv("RERANK_MAX_TOKENS", "100")
    monkeypatch.setenv("RERANK_TEMPERATURE", "")
    monkeypatch.setenv("RERANK_PREVIEW_MAX_TOKENS", "120")

    results = [
        _make_search_result("cu_1"),
        _make_search_result("cu_2"),
    ]
    llm = MagicMock()
    llm.call.return_value = _make_no_tool_response()

    metrics: dict = {}
    kept = rerank_results(results, "CET1 ratio", llm, metrics=metrics)

    assert kept == results
    assert metrics["kept"] == 2
    assert metrics["removed"] == 0
    assert metrics["fallback_keep_all"] is True
    assert metrics["fallback_reason"] == "missing_tool_call"


def test_rerank_empty_results(monkeypatch):
    """No results returns empty list without LLM call."""
    monkeypatch.setenv("RERANK_PREVIEW_MAX_TOKENS", "200")

    llm = MagicMock()
    metrics: dict = {}
    kept = rerank_results([], "some query", llm, metrics=metrics)

    assert kept == []
    llm.call.assert_not_called()
    assert metrics["llm_calls"] == 0


def test_build_candidate_list():
    """Correct format with section, page, keywords, preview."""
    results = [
        _make_search_result(
            "cu_1",
            score=0.85,
            section_id="KM1",
            page=3,
            keywords=["CET1", "Tier 1", "capital"],
            content="KM1: Key Capital metrics",
        ),
        _make_search_result(
            "cu_2",
            score=0.42,
            section_id="TLAC3",
            page=49,
            keywords=["creditor", "ranking"],
            content="Resolution entity creditor ranking",
        ),
    ]
    text = _build_candidate_list(results, 200)

    assert "[0]" in text
    assert "[1]" in text
    assert "=== Section: KM1 ===" in text
    assert "Page 3" in text
    assert "CET1" in text
    assert "0.85" in text
    assert "=== Section: TLAC3 ===" in text
    assert "Page 49" in text
    assert "0.42" in text


def test_truncate_preview_short():
    """Short content returned unchanged."""
    text = "Short text"
    assert _truncate_preview(text, 200) == text


def test_truncate_preview_long():
    """Long content truncated with ellipsis."""
    text = "a" * 1000
    result = _truncate_preview(text, 10)
    assert len(result) == 43
    assert result.endswith("...")


def test_parse_rerank_response_valid():
    """Valid response parsed correctly."""
    response = _make_llm_response([0, 2])
    indices = _parse_rerank_response(response)
    assert indices == [0, 2]


def test_parse_rerank_response_accepts_legacy_function_call():
    """Legacy function_call payloads are still accepted."""
    response = _make_legacy_llm_response([1])

    indices = _parse_rerank_response(response)

    assert indices == [1]


def test_parse_rerank_response_malformed():
    """Bad response structures raise ValueError."""
    with pytest.raises(ValueError, match="no choices"):
        _parse_rerank_response({"choices": []})

    with pytest.raises(ValueError, match="did not return"):
        _parse_rerank_response({"choices": [{"message": {}}]})

    bad_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "filter_chunks",
                                "arguments": json.dumps({"rationale": "oops"}),
                            }
                        }
                    ]
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="remove_indices"):
        _parse_rerank_response(bad_response)

    not_list_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "filter_chunks",
                                "arguments": json.dumps(
                                    {
                                        "remove_indices": "bad",
                                        "rationale": "oops",
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
        _parse_rerank_response(not_list_response)


def test_rerank_preserves_order(monkeypatch):
    """Kept results maintain original order."""
    monkeypatch.setenv("RERANK_MODEL", "gpt-test")
    monkeypatch.setenv("RERANK_MAX_TOKENS", "100")
    monkeypatch.setenv("RERANK_TEMPERATURE", "")
    monkeypatch.setenv("RERANK_REASONING_EFFORT", "")
    monkeypatch.setenv("RERANK_PREVIEW_MAX_TOKENS", "200")
    monkeypatch.setenv("RERANK_CANDIDATE_LIMIT", "30")
    monkeypatch.setenv("RERANK_MIN_KEEP", "10")

    results = [
        _make_search_result("cu_1", score=0.9),
        _make_search_result("cu_2", score=0.7),
        _make_search_result("cu_3", score=0.5),
        _make_search_result("cu_4", score=0.3),
    ]
    llm = MagicMock()
    llm.call.return_value = _make_llm_response([1, 3])

    kept = rerank_results(results, "query", llm)

    cuids = [r["content_unit_id"] for r in kept]
    assert cuids == ["cu_1", "cu_3"]


def test_format_rerank_input():
    """User prompt assembled with query and candidates."""
    prompt = {
        "user_prompt": ("Query: {user_input}\n" "Candidates:\n{candidates}"),
    }
    text = _format_rerank_input("CET1 ratio", "candidate list", prompt)
    assert "CET1 ratio" in text
    assert "candidate list" in text
