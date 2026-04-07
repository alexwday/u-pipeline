"""Tests for the test harness CLI in main.py."""

import json
from unittest.mock import MagicMock, patch

import pytest

from retriever.main import (
    _build_parser,
    _evaluate_assertions,
    _load_test_queries,
    _run_adhoc_query,
    _run_test_suite,
)
from retriever.models import ComboSpec

_MOD = "retriever.main"


def test_load_test_queries():
    """Parses test_queries.yaml and returns test cases."""
    cases = _load_test_queries()

    assert len(cases) == 10
    assert cases[0]["name"] == "RBC CET1 ratio"
    assert "CET1" in cases[0]["query"]
    assert cases[0]["combos"][0]["bank"] == "RBC"
    assert cases[0]["combos"][0]["period"] == "2026_Q1"
    assert cases[0]["sources"] is None
    assert cases[0]["assertions"]["max_citation_warnings"] == 1


def test_load_test_queries_has_source_filters():
    """Some test cases include source filters."""
    cases = _load_test_queries()

    filtered = [c for c in cases if c.get("sources")]
    assert len(filtered) >= 3

    sources_case = next(
        c
        for c in cases
        if c["name"] == "RBC capital adequacy - pillar3 + slides"
    )
    assert "investor-slides" in sources_case["sources"]
    assert "pillar3" in sources_case["sources"]


def test_load_test_queries_has_alternative_response_assertions():
    """Benchmark YAML can declare alternative acceptable terms."""
    cases = _load_test_queries()

    tlac_case = next(c for c in cases if c["name"] == "RBC TLAC and leverage")

    assert tlac_case["assertions"]["response_includes_any"] == [
        ["page 38", "page 69"]
    ]


def test_build_parser_test_flag():
    """--test flag is parsed correctly."""
    parser = _build_parser()
    args = parser.parse_args(["--test"])

    assert args.test is True
    assert args.query == ""
    assert args.bank == []
    assert args.period == []
    assert args.source == []


def test_build_parser_adhoc():
    """--query + --bank + --period parsed correctly."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--query",
            "What is CET1?",
            "--bank",
            "RBC",
            "--period",
            "2026_Q1",
        ]
    )

    assert args.query == "What is CET1?"
    assert args.bank == ["RBC"]
    assert args.period == ["2026_Q1"]
    assert args.test is False


def test_build_parser_multiple_banks_periods():
    """Repeatable --bank and --period flags accumulate."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--query",
            "Compare CET1",
            "--bank",
            "RBC",
            "--bank",
            "BMO",
            "--period",
            "2026_Q1",
            "--period",
            "2025_Q4",
            "--source",
            "pillar3",
        ]
    )

    assert args.bank == ["RBC", "BMO"]
    assert args.period == ["2026_Q1", "2025_Q4"]
    assert args.source == ["pillar3"]


def test_run_adhoc_builds_combos():
    """Multiple banks x periods creates correct combo list."""
    captured_combos = []

    def mock_run_retrieval(
        _query,
        combos,
        _sources,
        _conn,
        _llm,
    ):
        """Capture combos passed to run_retrieval."""
        captured_combos.extend(combos)
        return {
            "consolidated_response": "test",
            "key_findings": [],
            "data_gaps": [],
            "combo_results": [],
        }

    conn = MagicMock()
    llm = MagicMock()
    logger = MagicMock()

    with patch(f"{_MOD}.run_retrieval", mock_run_retrieval):
        _run_adhoc_query(
            conn,
            llm,
            "CET1 ratio?",
            ["RBC", "BMO"],
            ["2026_Q1", "2025_Q4"],
            [],
            logger,
        )

    assert len(captured_combos) == 4

    expected = [
        ComboSpec(bank="RBC", period="2026_Q1"),
        ComboSpec(bank="RBC", period="2025_Q4"),
        ComboSpec(bank="BMO", period="2026_Q1"),
        ComboSpec(bank="BMO", period="2025_Q4"),
    ]
    assert captured_combos == expected


def test_run_adhoc_exits_without_bank():
    """Ad-hoc query without --bank exits with error."""
    conn = MagicMock()
    llm = MagicMock()
    logger = MagicMock()

    with pytest.raises(SystemExit):
        _run_adhoc_query(
            conn,
            llm,
            "CET1?",
            [],
            ["2026_Q1"],
            [],
            logger,
        )


def test_run_adhoc_exits_without_period():
    """Ad-hoc query without --period exits with error."""
    conn = MagicMock()
    llm = MagicMock()
    logger = MagicMock()

    with pytest.raises(SystemExit):
        _run_adhoc_query(
            conn,
            llm,
            "CET1?",
            ["RBC"],
            [],
            [],
            logger,
        )


def test_run_adhoc_passes_source_filter():
    """Non-empty sources list is passed through."""
    captured_sources = []

    def mock_run_retrieval(
        _query,
        _combos,
        sources,
        _conn,
        _llm,
    ):
        """Capture sources passed to run_retrieval."""
        captured_sources.append(sources)
        return {
            "consolidated_response": "test",
            "key_findings": [],
            "data_gaps": [],
            "combo_results": [],
        }

    conn = MagicMock()
    llm = MagicMock()
    logger = MagicMock()

    with patch(f"{_MOD}.run_retrieval", mock_run_retrieval):
        _run_adhoc_query(
            conn,
            llm,
            "CET1?",
            ["RBC"],
            ["2026_Q1"],
            ["pillar3"],
            logger,
        )

    assert captured_sources[0] == ["pillar3"]


def test_evaluate_assertions_accepts_any_of_terms():
    """Alternative acceptable response terms satisfy the assertion."""
    case = {
        "assertions": {
            "response_includes_any": [
                ["page 24", "page 63"],
            ]
        }
    }
    result = {
        "consolidated_response": "Exposure detail [rts, Page 63].",
        "key_findings": [],
        "data_gaps": [],
        "citation_warnings": [],
    }

    failures = _evaluate_assertions(case, result)

    assert not failures


def test_evaluate_assertions_ignores_comma_formatting():
    """Equivalent numeric formatting should still satisfy assertions."""
    case = {"assertions": {"response_includes": ["100,415", "734,693"]}}
    result = {
        "consolidated_response": "CET1 100415 and RWA 734693.",
        "key_findings": [],
        "data_gaps": [],
        "citation_warnings": [],
    }

    failures = _evaluate_assertions(case, result)

    assert not failures


def test_evaluate_assertions_fails_when_no_alternative_matches():
    """Alternative acceptable response terms still fail when absent."""
    case = {
        "assertions": {
            "response_includes_any": [
                ["page 24", "page 63"],
            ]
        }
    }
    result = {
        "consolidated_response": "Exposure detail [rts, Page 48].",
        "key_findings": [],
        "data_gaps": [],
        "citation_warnings": [],
    }

    failures = _evaluate_assertions(case, result)

    assert failures == [
        "response missing one of required terms: page 24, page 63"
    ]


def test_run_adhoc_null_source_when_empty():
    """Empty sources list becomes None."""
    captured_sources = []

    def mock_run_retrieval(
        _query,
        _combos,
        sources,
        _conn,
        _llm,
    ):
        """Capture sources passed to run_retrieval."""
        captured_sources.append(sources)
        return {
            "consolidated_response": "test",
            "key_findings": [],
            "data_gaps": [],
            "combo_results": [],
        }

    conn = MagicMock()
    llm = MagicMock()
    logger = MagicMock()

    with patch(f"{_MOD}.run_retrieval", mock_run_retrieval):
        _run_adhoc_query(
            conn,
            llm,
            "CET1?",
            ["RBC"],
            ["2026_Q1"],
            [],
            logger,
        )

    assert captured_sources[0] is None


def test_run_test_suite_persists_combo_results(tmp_path, monkeypatch):
    """Saved test JSON includes per-unit combo result metrics."""
    case = {
        "name": "RBC CET1 ratio",
        "query": "What is CET1?",
        "combos": [{"bank": "RBC", "period": "2026_Q1"}],
        "sources": ["pillar3"],
        "assertions": {
            "response_includes": ["13.7%", "pillar3"],
            "min_key_findings": 1,
            "max_data_gaps": 0,
            "max_citation_warnings": 0,
        },
    }
    response = {
        "consolidated_response": "CET1 ratio 13.7% [pillar3, Page 3].",
        "key_findings": ["CET1 ratio 13.7% [pillar3, Page 3]."],
        "data_gaps": [],
        "citation_warnings": [],
        "combo_results": [
            {
                "combo": {"bank": "RBC", "period": "2026_Q1"},
                "source": {
                    "data_source": "pillar3",
                    "document_version_id": 40,
                    "filename": "rbc_q1_2026_pillar3.xlsx",
                },
                "research_iterations": [],
                "findings": [],
                "chunk_count": 1,
                "total_tokens": 100,
                "metrics": {
                    "expand": {
                        "evidence_catalog": [
                            {
                                "content_unit_id": "cu_1",
                                "page_number": 3,
                                "section_id": "3",
                                "section_title": "KM1",
                            }
                        ],
                        "cited_page_numbers": [3],
                        "cited_evidence": [
                            {
                                "content_unit_id": "cu_1",
                                "page_number": 3,
                                "section_id": "3",
                                "section_title": "KM1",
                            }
                        ],
                    }
                },
            }
        ],
        "metrics": {
            "citation_validation": {"cited_pages_by_source": {"pillar3": [3]}}
        },
    }
    logger = MagicMock()

    monkeypatch.setattr(f"{_MOD}._LOGS_DIR", tmp_path)
    monkeypatch.setattr(f"{_MOD}._load_test_queries", lambda: [case])
    monkeypatch.setattr(f"{_MOD}.run_retrieval", lambda *_args: response)

    _run_test_suite(MagicMock(), MagicMock(), logger)

    output_files = list(tmp_path.glob("test_results_*.json"))
    assert len(output_files) == 1
    payload = json.loads(output_files[0].read_text(encoding="utf-8"))

    assert payload[0]["combo_results"][0]["source"]["data_source"] == "pillar3"
    assert payload[0]["combo_results"][0]["metrics"]["expand"][
        "cited_page_numbers"
    ] == [3]
    assert payload[0]["assertion_failures"] == []
    assert payload[0]["status"] == "success"


def test_evaluate_assertions_reports_missing_terms():
    """Missing required terms are reported as assertion failures."""
    case = {
        "assertions": {
            "response_includes": ["13.7%", "pillar3"],
            "key_findings_include": ["13.7%"],
            "min_key_findings": 2,
            "max_data_gaps": 0,
            "max_citation_warnings": 0,
        }
    }
    result = {
        "consolidated_response": "CET1 ratio reported.",
        "key_findings": ["Reported value."],
        "data_gaps": ["Missing context."],
        "citation_warnings": ["Unsupported citation."],
    }

    failures = _evaluate_assertions(case, result)

    assert "response missing required term: 13.7%" in failures
    assert "response missing required term: pillar3" in failures
    assert "key findings missing required term: 13.7%" in failures
    assert "key findings count below minimum: 1 < 2" in failures
    assert "data gaps exceeded maximum: 1 > 0" in failures
    assert "citation warnings exceeded maximum: 1 > 0" in failures


def test_run_test_suite_marks_quality_failed(tmp_path, monkeypatch):
    """Assertion failures change benchmark status to quality_failed."""
    case = {
        "name": "RBC CET1 ratio",
        "query": "What is CET1?",
        "combos": [{"bank": "RBC", "period": "2026_Q1"}],
        "sources": ["pillar3"],
        "assertions": {
            "response_includes": ["13.7%"],
            "min_key_findings": 1,
            "max_data_gaps": 0,
            "max_citation_warnings": 0,
        },
    }
    response = {
        "consolidated_response": "CET1 ratio unavailable.",
        "key_findings": [],
        "data_gaps": [],
        "citation_warnings": [],
        "combo_results": [],
        "metrics": {},
    }
    logger = MagicMock()

    monkeypatch.setattr(f"{_MOD}._LOGS_DIR", tmp_path)
    monkeypatch.setattr(f"{_MOD}._load_test_queries", lambda: [case])
    monkeypatch.setattr(f"{_MOD}.run_retrieval", lambda *_args: response)

    _run_test_suite(MagicMock(), MagicMock(), logger)

    output_files = list(tmp_path.glob("test_results_*.json"))
    payload = json.loads(output_files[0].read_text(encoding="utf-8"))

    assert payload[0]["status"] == "quality_failed"
    assert payload[0]["assertion_failures"] == [
        "response missing required term: 13.7%",
        "key findings count below minimum: 0 < 1",
    ]
