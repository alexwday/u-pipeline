"""Tests for the query execution and SSE streaming endpoint."""

import json

from unittest.mock import patch

from debug.api import retrieval as retrieval_mod
from debug.api.retrieval import _serialize_result, _serialize_combo_results


def _swap_state(mod, overrides):
    """Swap module state dict values, returning originals.

    Params:
        mod: retrieval module
        overrides: dict of keys to override

    Returns:
        dict of original values for restoration
    """
    originals = {}
    state = getattr(mod, "_state")
    for key, value in overrides.items():
        originals[key] = state[key]
        state[key] = value
    return originals


def _restore_state(mod, originals):
    """Restore module state to original values.

    Params:
        mod: retrieval module
        originals: dict from _swap_state
    """
    state = getattr(mod, "_state")
    state.update(originals)


def test_ensure_initialized_calls_startup():
    """_ensure_initialized calls run_startup on first invocation."""
    originals = _swap_state(
        retrieval_mod,
        {"initialized": False, "conn": None, "llm": None},
    )
    init_fn = getattr(retrieval_mod, "_ensure_initialized")
    try:
        with patch.object(
            retrieval_mod, "run_startup", return_value=("conn", "llm")
        ) as mock_startup:
            init_fn()
            mock_startup.assert_called_once()
            state = getattr(retrieval_mod, "_state")
            assert state["conn"] == "conn"
            assert state["llm"] == "llm"
            assert state["initialized"] is True
    finally:
        _restore_state(retrieval_mod, originals)


def test_ensure_initialized_skips_when_ready():
    """_ensure_initialized is a no-op when already initialized."""
    originals = _swap_state(retrieval_mod, {"initialized": True})
    init_fn = getattr(retrieval_mod, "_ensure_initialized")
    try:
        with patch.object(retrieval_mod, "run_startup") as mock_startup:
            init_fn()
            mock_startup.assert_not_called()
    finally:
        _restore_state(retrieval_mod, originals)


def test_serialize_result_extracts_fields():
    """Serializer extracts expected fields from a result dict."""
    result = {
        "query": "What is CET1?",
        "consolidated_response": "CET1 is 13.2%",
        "key_findings": ["finding 1"],
        "data_gaps": ["gap 1"],
        "citation_warnings": [],
        "metrics": {"wall_time_seconds": 5.0},
        "trace_id": "trace_abc",
        "trace_path": "/tmp/traces/trace_abc/run_trace.json",
        "combo_results": [],
    }
    serialized = _serialize_result(result)
    assert serialized["query"] == "What is CET1?"
    assert serialized["consolidated_response"] == "CET1 is 13.2%"
    assert serialized["key_findings"] == ["finding 1"]
    assert serialized["trace_id"] == "trace_abc"
    assert not serialized["combo_results"]


def test_serialize_result_handles_missing_fields():
    """Serializer returns defaults for missing fields."""
    serialized = _serialize_result({})
    assert serialized["query"] == ""
    assert not serialized["key_findings"]
    assert not serialized["metrics"]
    assert serialized["trace_id"] == ""


def test_serialize_combo_results():
    """Combo result serializer extracts key fields."""
    combo_results = [
        {
            "combo": {"bank": "RBC", "period": "2026_Q1"},
            "source": {
                "data_source": "pillar3",
                "document_version_id": 42,
                "filename": "rbc.xlsx",
            },
            "research_iterations": [{"iteration": 1}],
            "chunk_count": 10,
            "total_tokens": 5000,
            "findings": [{"finding": "CET1 13.2%", "page": 5}],
            "metrics": {"research_seconds": 3.0},
        }
    ]
    serialized = _serialize_combo_results(combo_results)
    assert len(serialized) == 1
    assert serialized[0]["combo"]["bank"] == "RBC"
    assert serialized[0]["chunk_count"] == 10
    assert len(serialized[0]["findings"]) == 1


@patch("debug.api.retrieval._ensure_initialized")
@patch("debug.api.retrieval.run_retrieval")
def test_query_endpoint_streams_result(mock_retrieval, _mock_init, client):
    """POST /api/query returns SSE events with chunks and result."""
    mock_retrieval.return_value = {
        "query": "test",
        "consolidated_response": "answer",
        "key_findings": [],
        "data_gaps": [],
        "citation_warnings": [],
        "metrics": {},
        "trace_id": "",
        "trace_path": "",
        "combo_results": [],
    }

    def side_effect(*_args, on_chunk=None, **_kwargs):
        if on_chunk:
            on_chunk("chunk1")
            on_chunk("chunk2")
        return mock_retrieval.return_value

    mock_retrieval.side_effect = side_effect

    response = client.post(
        "/api/query",
        json={
            "query": "What is CET1?",
            "combos": [{"bank": "RBC", "period": "2026_Q1"}],
        },
    )
    assert response.status_code == 200
    assert response.content_type == "text/event-stream"

    text = response.get_data(as_text=True)
    assert "event: chunk" in text
    assert "event: result" in text

    events = _parse_sse(text)
    chunks = [e for e in events if e[0] == "chunk"]
    results = [e for e in events if e[0] == "result"]
    assert len(chunks) == 2
    assert json.loads(chunks[0][1]) == "chunk1"
    assert len(results) == 1

    result_data = json.loads(results[0][1])
    assert result_data["query"] == "test"


@patch("debug.api.retrieval._ensure_initialized")
@patch("debug.api.retrieval.run_retrieval")
def test_query_endpoint_handles_error(mock_retrieval, _mock_init, client):
    """POST /api/query sends error event on retrieval failure."""
    mock_retrieval.side_effect = RuntimeError("LLM down")

    response = client.post(
        "/api/query",
        json={
            "query": "fail",
            "combos": [{"bank": "RBC", "period": "2026_Q1"}],
        },
    )
    assert response.status_code == 200

    text = response.get_data(as_text=True)
    assert "event: error" in text

    events = _parse_sse(text)
    errors = [e for e in events if e[0] == "error"]
    assert len(errors) == 1
    assert "LLM down" in json.loads(errors[0][1])


@patch("debug.api.retrieval._ensure_initialized")
@patch("debug.api.retrieval.run_retrieval")
def test_query_with_sources(mock_retrieval, _mock_init, client):
    """POST /api/query passes sources to run_retrieval."""
    mock_retrieval.return_value = {
        "query": "q",
        "consolidated_response": "",
        "key_findings": [],
        "data_gaps": [],
        "citation_warnings": [],
        "metrics": {},
        "combo_results": [],
    }

    client.post(
        "/api/query",
        json={
            "query": "test",
            "combos": [{"bank": "RBC", "period": "2026_Q1"}],
            "sources": ["pillar3"],
        },
    )
    call_args = mock_retrieval.call_args
    assert call_args[0][2] == ["pillar3"]


def _parse_sse(text):
    """Parse SSE text into (event_type, data) tuples.

    Params:
        text: Raw SSE response text

    Returns:
        list of (event_type, data) tuples
    """
    events = []
    for block in text.strip().split("\n\n"):
        event_type = ""
        data = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        if event_type:
            events.append((event_type, data))
    return events
