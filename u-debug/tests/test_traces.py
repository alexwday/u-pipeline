"""Tests for trace loading API endpoints."""

import json

from unittest.mock import patch


def _write_trace(tmp_path, trace_id, filename, content):
    """Write a trace JSON file for testing.

    Params:
        tmp_path: pytest tmp_path fixture
        trace_id: Trace directory name
        filename: JSON filename
        content: dict to serialize

    Returns:
        Path to the trace root directory
    """
    trace_dir = tmp_path / "traces" / trace_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / filename).write_text(json.dumps(content), encoding="utf-8")
    return tmp_path / "traces"


@patch("debug.api.traces.get_trace_root")
def test_get_run_trace(mock_root, client, tmp_path):
    """GET /api/trace/<id> returns run_trace.json content."""
    trace_root = _write_trace(
        tmp_path, "abc123", "run_trace.json", {"trace_id": "abc123"}
    )
    mock_root.return_value = trace_root

    response = client.get("/api/trace/abc123")
    assert response.status_code == 200
    data = response.get_json()
    assert data["trace_id"] == "abc123"


@patch("debug.api.traces.get_trace_root")
def test_get_run_trace_not_found(mock_root, client, tmp_path):
    """GET /api/trace/<id> returns 404 for missing trace."""
    mock_root.return_value = tmp_path / "traces"

    response = client.get("/api/trace/nonexistent")
    assert response.status_code == 404


@patch("debug.api.traces.get_trace_root")
def test_list_source_traces(mock_root, client, tmp_path):
    """GET /api/trace/<id>/sources lists source trace files."""
    trace_root = _write_trace(tmp_path, "abc123", "run_trace.json", {})
    trace_dir = trace_root / "abc123"
    (trace_dir / "source_01_rbc_q1_pillar3_42.json").write_text(
        "{}", encoding="utf-8"
    )
    (trace_dir / "source_02_rbc_q1_slides_43.json").write_text(
        "{}", encoding="utf-8"
    )
    mock_root.return_value = trace_root

    response = client.get("/api/trace/abc123/sources")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 2
    assert "source_01_rbc_q1_pillar3_42.json" in data


@patch("debug.api.traces.get_trace_root")
def test_get_source_trace(mock_root, client, tmp_path):
    """GET /api/trace/<id>/source/<file> returns source trace."""
    trace_root = _write_trace(
        tmp_path, "abc123", "source_01_test.json", {"stage": "search"}
    )
    mock_root.return_value = trace_root

    response = client.get("/api/trace/abc123/source/source_01_test.json")
    assert response.status_code == 200
    data = response.get_json()
    assert data["stage"] == "search"


@patch("debug.api.traces.get_trace_root")
def test_get_source_trace_not_found(mock_root, client, tmp_path):
    """GET /api/trace/<id>/source/<file> returns 404 for missing."""
    trace_root = _write_trace(tmp_path, "abc123", "run_trace.json", {})
    mock_root.return_value = trace_root

    response = client.get("/api/trace/abc123/source/missing.json")
    assert response.status_code == 404


@patch("debug.api.traces.get_trace_root")
def test_path_traversal_blocked(mock_root, client, tmp_path):
    """Path traversal attempts in trace_id are blocked."""
    mock_root.return_value = tmp_path / "traces"
    (tmp_path / "traces").mkdir(parents=True, exist_ok=True)

    response = client.get("/api/trace/../../etc")
    assert response.status_code == 404
