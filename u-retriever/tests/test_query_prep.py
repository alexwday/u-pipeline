"""Tests for query preparation stage."""

import json
from unittest.mock import MagicMock

import pytest

from retriever.stages.query_prep import (
    _generate_query_embeddings,
    _parse_query_response,
    _soften_hyde_answer,
    prepare_query,
)


def _make_llm_response(arguments: dict) -> dict:
    """Build a mock LLM response with tool call arguments."""
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "prepare_query",
                                "arguments": json.dumps(arguments),
                            }
                        }
                    ]
                }
            }
        ]
    }


def _make_legacy_llm_response(arguments: dict) -> dict:
    """Build a mock LLM response with a legacy function_call payload."""
    return {
        "choices": [
            {
                "message": {
                    "function_call": {
                        "name": "prepare_query",
                        "arguments": json.dumps(arguments),
                    }
                }
            }
        ]
    }


def _valid_arguments() -> dict:
    """Return a complete set of valid tool arguments."""
    return {
        "rewritten_query": (
            "Common Equity Tier 1 capital ratio for " "Royal Bank of Canada"
        ),
        "sub_queries": [
            "CET1 ratio trend for RBC",
            "RBC capital adequacy regulatory minimums",
        ],
        "keywords": [
            "CET1",
            "Tier 1",
            "capital ratio",
            "Basel III",
        ],
        "entities": [
            "Royal Bank of Canada",
            "OSFI",
        ],
        "hyde_answer": (
            "RBC reported the CET1 ratio in its Q1 2026 "
            "capital disclosures and investor materials."
        ),
        "rationale": "Decomposed into ratio trend and regs.",
    }


def _make_embedding(dims: int = 4) -> list[float]:
    """Return a small dummy embedding vector."""
    return [0.1 * i for i in range(dims)]


def test_prepare_query_returns_prepared_query(
    monkeypatch,
):
    """Full flow produces a PreparedQuery with all fields."""
    monkeypatch.setenv("QUERY_PREP_MODEL", "gpt-test")
    monkeypatch.setenv("QUERY_PREP_MAX_TOKENS", "100")
    monkeypatch.setenv("QUERY_PREP_TEMPERATURE", "")

    args = _valid_arguments()
    response = _make_llm_response(args)

    llm = MagicMock()
    llm.call.return_value = response
    # 2 sub-queries => 5 texts total
    llm.embed.return_value = [_make_embedding() for _ in range(5)]

    metrics: dict = {}
    result = prepare_query("CET1 ratio for RBC", llm, metrics=metrics)

    assert result["original_query"] == "CET1 ratio for RBC"
    assert result["rewritten_query"] == args["rewritten_query"]
    assert result["sub_queries"] == args["sub_queries"]
    assert result["keywords"] == args["keywords"]
    assert result["entities"] == args["entities"]
    assert result["hyde_answer"] == args["hyde_answer"]
    assert len(result["embeddings"]["sub_queries"]) == 2
    assert result["embeddings"]["rewritten"] == _make_embedding()
    assert metrics["llm_calls"] == 1
    assert metrics["embed_calls"] == 1
    assert metrics["sub_queries"] == 2


def test_prepare_query_populates_trace(monkeypatch):
    """Prepared-query traces retain the generated retrieval facets."""
    monkeypatch.setenv("QUERY_PREP_MODEL", "gpt-test")
    monkeypatch.setenv("QUERY_PREP_MAX_TOKENS", "100")
    monkeypatch.setenv("QUERY_PREP_TEMPERATURE", "")

    args = _valid_arguments()
    llm = MagicMock()
    llm.call.return_value = _make_llm_response(args)
    llm.embed.return_value = [_make_embedding() for _ in range(5)]

    trace: dict = {}
    prepare_query("CET1 ratio for RBC", llm, trace=trace)

    assert trace["rewritten_query"] == args["rewritten_query"]
    assert trace["sub_queries"] == args["sub_queries"]
    assert trace["keywords"] == args["keywords"]
    assert trace["entities"] == args["entities"]
    assert trace["hyde_answer"] == args["hyde_answer"]
    assert trace["timing"]["llm_calls"] == 1
    assert trace["embedding_inputs"]["keywords_text"] == " ".join(
        args["keywords"]
    )


def test_prepare_query_limits_and_deduplicates_facets(monkeypatch):
    """Duplicate and excess query facets are trimmed deterministically."""
    monkeypatch.setenv("QUERY_PREP_MODEL", "gpt-test")
    monkeypatch.setenv("QUERY_PREP_MAX_TOKENS", "100")
    monkeypatch.setenv("QUERY_PREP_TEMPERATURE", "")

    args = _valid_arguments()
    args["sub_queries"] = ["a", "a", "b", "c", "d"]
    args["keywords"] = [
        "k1",
        "k1",
        "k2",
        "k3",
        "k4",
        "k5",
        "k6",
        "k7",
        "k8",
        "k9",
        "k10",
        "k11",
    ]
    args["entities"] = [
        "e1",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "e6",
        "e7",
        "e8",
        "e9",
    ]

    llm = MagicMock()
    llm.call.return_value = _make_llm_response(args)
    llm.embed.return_value = [_make_embedding() for _ in range(13)]

    result = prepare_query("query", llm)

    assert result["sub_queries"] == ["a", "b", "c"]
    assert result["keywords"] == [
        "k1",
        "k2",
        "k3",
        "k4",
        "k5",
        "k6",
        "k7",
        "k8",
    ]
    assert result["entities"] == [
        "e1",
        "e2",
        "e3",
        "e4",
    ]


def test_prepare_query_generates_embeddings(monkeypatch):
    """Verify embed() is called with all texts batched."""
    monkeypatch.setenv("QUERY_PREP_MODEL", "gpt-test")
    monkeypatch.setenv("QUERY_PREP_MAX_TOKENS", "100")
    monkeypatch.setenv("QUERY_PREP_TEMPERATURE", "")

    args = _valid_arguments()
    response = _make_llm_response(args)

    llm = MagicMock()
    llm.call.return_value = response
    llm.embed.return_value = [_make_embedding() for _ in range(5)]

    prepare_query("CET1 ratio for RBC", llm)

    llm.embed.assert_called_once()
    call_args = llm.embed.call_args
    texts = call_args[0][0] if call_args[0] else call_args[1]["texts"]
    assert len(texts) == 5
    assert texts[0] == args["rewritten_query"]
    assert texts[1] == args["sub_queries"][0]
    assert texts[2] == args["sub_queries"][1]
    assert texts[3] == " ".join(args["keywords"])
    assert texts[4] == args["hyde_answer"]


def test_prepare_query_handles_single_sub_query(
    monkeypatch,
):
    """One sub-query produces one sub-query embedding."""
    monkeypatch.setenv("QUERY_PREP_MODEL", "gpt-test")
    monkeypatch.setenv("QUERY_PREP_MAX_TOKENS", "100")
    monkeypatch.setenv("QUERY_PREP_TEMPERATURE", "")

    args = _valid_arguments()
    args["sub_queries"] = ["single sub-query"]
    response = _make_llm_response(args)

    llm = MagicMock()
    llm.call.return_value = response
    # 1 sub-query => 4 texts total
    llm.embed.return_value = [_make_embedding() for _ in range(4)]

    result = prepare_query("simple query", llm)

    assert len(result["embeddings"]["sub_queries"]) == 1
    texts = llm.embed.call_args[0][0]
    assert len(texts) == 4


def test_parse_query_response_valid():
    """Valid tool call response is parsed correctly."""
    args = _valid_arguments()
    response = _make_llm_response(args)
    parsed = _parse_query_response(response)

    assert parsed["rewritten_query"] == args["rewritten_query"]
    assert parsed["sub_queries"] == args["sub_queries"]
    assert parsed["keywords"] == args["keywords"]
    assert parsed["entities"] == args["entities"]
    assert parsed["hyde_answer"] == args["hyde_answer"]


def test_parse_query_response_accepts_legacy_function_call():
    """Legacy function_call payloads are still accepted."""
    args = _valid_arguments()
    response = _make_legacy_llm_response(args)

    parsed = _parse_query_response(response)

    assert parsed["rewritten_query"] == args["rewritten_query"]


def test_parse_query_response_malformed():
    """Bad response structures raise ValueError."""
    with pytest.raises(ValueError, match="no choices"):
        _parse_query_response({"choices": []})

    with pytest.raises(ValueError, match="did not return"):
        _parse_query_response({"choices": [{"message": {}}]})

    incomplete = _make_llm_response({"rewritten_query": "q"})
    with pytest.raises(ValueError, match="missing fields"):
        _parse_query_response(incomplete)


def test_generate_query_embeddings_batches(monkeypatch):
    """All texts sent to embed() in a single batch call."""
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "4")

    llm = MagicMock()
    vecs = [_make_embedding() for _ in range(6)]
    llm.embed.return_value = vecs

    result = _generate_query_embeddings(
        llm,
        "rewritten",
        ["sq1", "sq2", "sq3"],
        ["kw1", "kw2"],
        "hyde text",
    )

    llm.embed.assert_called_once_with(
        [
            "rewritten",
            "sq1",
            "sq2",
            "sq3",
            "kw1 kw2",
            "hyde text",
        ],
        model="text-embedding-3-large",
        dimensions=4,
    )
    assert result["rewritten"] == vecs[0]
    assert result["sub_queries"] == vecs[1:4]
    assert result["keywords"] == vecs[4]
    assert result["hyde"] == vecs[5]


def test_soften_hyde_answer_removes_precise_numeric_claims():
    """Numeric HYDE anchors are converted into qualitative placeholders."""
    softened = _soften_hyde_answer(
        "RBC reported 13.7%, up 20 bps, with CAD 6.8 billion "
        "of income and a 26-28% TLAC range."
    )

    assert "13.7%" not in softened
    assert "20 bps" not in softened
    assert "CAD 6.8 billion" not in softened
    assert "26-28%" not in softened
    assert "reported percentage" in softened
