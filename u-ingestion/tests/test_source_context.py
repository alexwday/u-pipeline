"""Tests for source-context helpers."""

from ingestion.utils import source_context as mod
from ingestion.utils.file_types import ExtractionResult


def test_parse_relative_source_context():
    """Relative paths split into data_source and filters."""
    assert mod.parse_relative_source_context("pillar3/2026/Q1/RBC") == (
        "pillar3",
        "2026",
        "Q1",
        "RBC",
    )


def test_parse_relative_source_context_empty_path():
    """Empty relative paths return blank source context."""
    assert mod.parse_relative_source_context("") == ("", "", "", "")


def test_get_source_context_from_path(monkeypatch, tmp_path):
    """Absolute file paths are resolved relative to DATA_SOURCE_PATH."""
    base = tmp_path / "sources"
    file_dir = base / "pillar3" / "2026" / "Q1"
    file_dir.mkdir(parents=True)
    file_path = file_dir / "report.pdf"

    monkeypatch.setattr(mod, "get_data_source_path", lambda: str(base))

    context = mod.get_source_context_from_path(str(file_path))

    assert context == {
        "data_source": "pillar3",
        "filter_1": "2026",
        "filter_2": "Q1",
        "filter_3": "",
    }


def test_get_source_context_from_path_outside_base(monkeypatch, tmp_path):
    """Paths outside DATA_SOURCE_PATH return blank context."""
    base = tmp_path / "sources"
    outside = tmp_path / "other" / "report.pdf"
    base.mkdir()

    monkeypatch.setattr(mod, "get_data_source_path", lambda: str(base))

    context = mod.get_source_context_from_path(str(outside))

    assert context == {
        "data_source": "",
        "filter_1": "",
        "filter_2": "",
        "filter_3": "",
    }


def test_get_result_source_context_prefers_result_fields():
    """Result fields win over any path-based fallback."""
    result = ExtractionResult(
        file_path="/tmp/ignored/report.pdf",
        filetype="pdf",
        pages=[],
        total_pages=0,
        data_source="pillar3",
        filter_1="2026",
        filter_2="Q1",
        filter_3="RBC",
    )

    context = mod.get_result_source_context(result)

    assert context == {
        "data_source": "pillar3",
        "filter_1": "2026",
        "filter_2": "Q1",
        "filter_3": "RBC",
    }
