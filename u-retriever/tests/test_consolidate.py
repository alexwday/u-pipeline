"""Tests for research consolidation across sources."""

from unittest.mock import MagicMock

from retriever.models import (
    ComboSourceResult,
    ComboSpec,
    ResearchFinding,
    SourceSpec,
)
from retriever.stages.consolidate import (
    _build_reference_index,
    _create_ref_replacer,
    _extract_gap_list,
    _parse_sections,
    consolidate_results,
)


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


def _make_combo_result_with_findings(
    bank: str = "RBC",
    data_source: str = "pillar3",
    filename: str = "rbc_q1_2026_pillar3.xlsx",
    findings: list | None = None,
) -> ComboSourceResult:
    """Build a ComboSourceResult with structured findings."""
    if findings is None:
        findings = [
            ResearchFinding(
                finding="CET1 ratio was 13.7%",
                page=3,
                location_detail="Sheet KM1",
                metric_name="CET1 Ratio",
                metric_value="13.7%",
                period="Q1 2026",
                segment="Enterprise",
            ),
        ]
    return ComboSourceResult(
        combo=_make_combo(bank=bank),
        source=_make_source(
            data_source=data_source,
            filename=filename,
        ),
        research_iterations=[],
        findings=findings,
        chunk_count=1,
        total_tokens=50,
    )


# -- _build_reference_index tests --


def test_build_reference_index_single_source():
    """Single source produces numbered entries and formatted text."""
    results = [_make_combo_result_with_findings()]
    entries, text = _build_reference_index(results)

    assert len(entries) == 1
    assert entries[0]["ref_id"] == 1
    assert entries[0]["finding"] == "CET1 ratio was 13.7%"
    assert entries[0]["page"] == 3
    assert entries[0]["source"] == "pillar3"
    assert entries[0]["entity"] == "RBC"
    assert entries[0]["metric_name"] == "CET1 Ratio"
    assert "[REF:1]" in text
    assert "CET1 ratio was 13.7%" in text
    assert "Sheet KM1" in text


def test_build_reference_index_multi_source():
    """Multiple sources get sequential REF numbering."""
    results = [
        _make_combo_result_with_findings(
            data_source="pillar3",
            findings=[
                ResearchFinding(
                    finding="CET1 ratio was 13.7%",
                    page=3,
                    location_detail="Sheet KM1",
                ),
            ],
        ),
        _make_combo_result_with_findings(
            data_source="investor-slides",
            filename="rbc_q1_2026_slides.pdf",
            findings=[
                ResearchFinding(
                    finding="Tier 1 ratio was 15.2%",
                    page=10,
                    location_detail="Capital Overview",
                ),
                ResearchFinding(
                    finding="Revenue grew 5%",
                    page=12,
                    location_detail="Financial Highlights",
                ),
            ],
        ),
    ]
    entries, text = _build_reference_index(results)

    assert len(entries) == 3
    assert entries[0]["ref_id"] == 1
    assert entries[1]["ref_id"] == 2
    assert entries[2]["ref_id"] == 3
    assert entries[0]["source"] == "pillar3"
    assert entries[1]["source"] == "investor-slides"
    assert "[REF:1]" in text
    assert "[REF:2]" in text
    assert "[REF:3]" in text
    assert "=== Source: pillar3" in text
    assert "=== Source: investor-slides" in text


def test_build_reference_index_empty_findings():
    """Sources with no findings produce empty index."""
    results = [
        _make_combo_result_with_findings(findings=[]),
    ]
    entries, text = _build_reference_index(results)

    assert not entries
    assert text == ""


def test_build_reference_index_qualitative_finding():
    """Qualitative findings omit metric fields in output."""
    results = [
        _make_combo_result_with_findings(
            findings=[
                ResearchFinding(
                    finding="Management indicated strong outlook",
                    page=5,
                    location_detail="CEO Commentary",
                ),
            ],
        ),
    ]
    entries, text = _build_reference_index(results)

    assert len(entries) == 1
    assert "metric_name" not in entries[0]
    assert "Metric:" not in text
    assert "[REF:1]" in text
    assert "Management indicated strong outlook" in text


# -- _create_ref_replacer tests --


def _make_ref_entries():
    """Build sample reference entries for replacer tests."""
    return [
        {
            "ref_id": 1,
            "finding": "CET1 ratio was 13.7%",
            "page": 3,
            "location_detail": "Sheet KM1",
            "source": "pillar3",
        },
        {
            "ref_id": 2,
            "finding": "Tier 1 ratio was 15.2%",
            "page": 10,
            "location_detail": "Capital Overview",
            "source": "investor-slides",
        },
    ]


def test_ref_replacer_basic():
    """Single-chunk replacement of [REF:N] patterns."""
    replacer = _create_ref_replacer(_make_ref_entries())
    result = replacer["replace"](
        "CET1 was 13.7% [REF:1] and Tier 1 was 15.2% [REF:2]."
    )
    assert "[pillar3, Page 3 - Sheet KM1]" in result
    assert "[investor-slides, Page 10 - Capital Overview]" in result
    assert "[REF:" not in result
    assert 1 in replacer["used_refs"]
    assert 2 in replacer["used_refs"]


def test_ref_replacer_split_across_chunks():
    """[REF:N] split across two chunks is correctly buffered."""
    replacer = _create_ref_replacer(_make_ref_entries())

    part1 = replacer["replace"]("CET1 was 13.7% [RE")
    part2 = replacer["replace"]("F:1] end.")

    combined = part1 + part2
    assert "[pillar3, Page 3 - Sheet KM1]" in combined
    assert "[REF:" not in combined


def test_ref_replacer_invalid_ref():
    """Invented [REF:N] stays in text and is tracked."""
    replacer = _create_ref_replacer(_make_ref_entries())
    result = replacer["replace"]("Unknown [REF:999] here.")
    assert "[REF:999]" in result
    assert 999 in replacer["invented_refs"]


def test_ref_replacer_flush():
    """Flush returns buffered partial text."""
    replacer = _create_ref_replacer(_make_ref_entries())
    replacer["replace"]("End with partial [REF:")
    flushed = replacer["flush"]()
    assert "[REF:" in flushed


def test_ref_replacer_no_location():
    """Ref with no location_detail omits dash in citation."""
    entries = [
        {
            "ref_id": 1,
            "finding": "Some finding",
            "page": 5,
            "location_detail": "",
            "source": "pillar3",
        },
    ]
    replacer = _create_ref_replacer(entries)
    result = replacer["replace"]("See [REF:1].")
    assert "[pillar3, Page 5]" in result
    assert " - " not in result


# -- _parse_sections tests --


def test_parse_sections_full():
    """All four sections are extracted correctly."""
    text = (
        "## Summary\nSummary text.\n\n"
        "## Metrics\n| col |\n\n"
        "## Detail\nDetail text.\n\n"
        "## Gaps\nNone identified\n"
    )
    sections = _parse_sections(text)

    assert sections["summary"] == "Summary text."
    assert sections["metrics"] == "| col |"
    assert sections["detail"] == "Detail text."
    assert sections["gaps"] == "None identified"


def test_parse_sections_missing_heading():
    """Missing headings map to empty strings."""
    text = "## Summary\nJust a summary.\n"
    sections = _parse_sections(text)

    assert sections["summary"] == "Just a summary."
    assert sections["metrics"] == ""
    assert sections["detail"] == ""
    assert sections["gaps"] == ""


def test_parse_sections_no_headings():
    """Text without any ## headings returns all empty."""
    sections = _parse_sections("Plain text with no sections.")

    assert sections["summary"] == ""
    assert sections["metrics"] == ""


# -- _extract_gap_list tests --


def test_extract_gap_list_none_identified():
    """'None identified' returns empty list."""
    assert not _extract_gap_list("None identified")
    assert not _extract_gap_list("None identified.")
    assert not _extract_gap_list("  none identified  ")


def test_extract_gap_list_items():
    """Bulleted items are split into a list."""
    text = "- Gap one\n- Gap two\n"
    result = _extract_gap_list(text)
    assert result == ["Gap one", "Gap two"]


def test_extract_gap_list_empty():
    """Empty string returns empty list."""
    assert not _extract_gap_list("")


# -- consolidate_results tests --


def _set_env(monkeypatch):
    """Set consolidation stage env vars."""
    monkeypatch.setenv("CONSOLIDATION_MODEL", "gpt-test")
    monkeypatch.setenv("CONSOLIDATION_MAX_TOKENS", "8000")
    monkeypatch.setenv("CONSOLIDATION_TEMPERATURE", "")
    monkeypatch.setenv("CONSOLIDATION_REASONING_EFFORT", "")


_STREAM_CHUNKS = [
    "## Summary\n",
    "CET1 was 13.7% [REF:1].\n\n",
    "## Detail\n",
    "Details here [REF:1].\n\n",
    "## Gaps\n",
    "None identified\n",
]

_STREAM_USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150,
}


def _mock_stream(*_args, **_kwargs):
    """Mock LLM stream generator yielding chunks."""
    yield from _STREAM_CHUNKS
    return _STREAM_USAGE


def _make_stream_llm():
    """Build a mock LLM with stream() method."""
    llm = MagicMock()
    llm.stream = _mock_stream
    return llm


def test_consolidate_empty_results(monkeypatch):
    """No combo_results produces empty result with gap."""
    _set_env(monkeypatch)
    llm = MagicMock()

    result = consolidate_results("query", [], llm)

    assert result["query"] == "query"
    assert result["consolidated_response"] == ""
    assert result["key_findings"] == []
    assert result["data_gaps"] == ["No sources returned results."]
    assert result["combo_results"] == []


def test_consolidate_no_findings(monkeypatch):
    """Combo results with no findings produce empty ref index gap."""
    _set_env(monkeypatch)
    combo_results = [
        _make_combo_result_with_findings(findings=[]),
    ]
    llm = MagicMock()

    result = consolidate_results("query", combo_results, llm)

    assert result["consolidated_response"] == ""
    assert result["data_gaps"] == ["No findings extracted from sources."]
    assert result["reference_index"] == []


def test_consolidate_streams_to_callback(monkeypatch):
    """Streaming calls on_chunk with processed text."""
    _set_env(monkeypatch)
    combo_results = [_make_combo_result_with_findings()]
    llm = _make_stream_llm()
    chunks_received: list[str] = []

    consolidate_results(
        "CET1 ratio",
        combo_results,
        llm,
        on_chunk=chunks_received.append,
    )

    joined = "".join(chunks_received)
    assert "[pillar3, Page 3 - Sheet KM1]" in joined
    assert "[REF:" not in joined
    assert len(chunks_received) > 0


def test_consolidate_returns_result_without_callback(monkeypatch):
    """on_chunk=None still returns a populated ConsolidatedResult."""
    _set_env(monkeypatch)
    combo_results = [_make_combo_result_with_findings()]
    llm = _make_stream_llm()

    result = consolidate_results("CET1 ratio", combo_results, llm)

    assert result["query"] == "CET1 ratio"
    assert result["consolidated_response"] != ""
    assert result["combo_results"] == combo_results
    assert isinstance(result["key_findings"], list)
    assert isinstance(result["data_gaps"], list)


def test_consolidate_parses_sections(monkeypatch):
    """Summary, metrics, and detail sections are populated."""
    _set_env(monkeypatch)
    combo_results = [_make_combo_result_with_findings()]
    llm = _make_stream_llm()

    result = consolidate_results("CET1 ratio", combo_results, llm)

    assert "13.7%" in result["summary_answer"]
    assert result["detailed_summary"] != ""
    assert result["data_gaps"] == []


def test_consolidate_populates_reference_index(monkeypatch):
    """Reference index entries are attached to the result."""
    _set_env(monkeypatch)
    combo_results = [_make_combo_result_with_findings()]
    llm = _make_stream_llm()

    result = consolidate_results("CET1 ratio", combo_results, llm)

    assert len(result["reference_index"]) == 1
    assert result["reference_index"][0]["ref_id"] == 1
    assert result["reference_index"][0]["source"] == "pillar3"


def test_consolidate_metrics(monkeypatch):
    """Usage metrics are recorded in the metrics dict."""
    _set_env(monkeypatch)
    combo_results = [_make_combo_result_with_findings()]
    llm = _make_stream_llm()
    metrics: dict = {}

    consolidate_results(
        "CET1 ratio",
        combo_results,
        llm,
        metrics=metrics,
    )

    assert metrics["prompt_tokens"] == 100
    assert metrics["completion_tokens"] == 50
    assert metrics["llm_calls"] == 1
    assert metrics["source_results"] == 1
    assert metrics["key_findings"] >= 0
    assert metrics["data_gaps"] == 0


def test_consolidate_fallback_without_stream(monkeypatch):
    """LLM without stream() falls back to call()."""
    _set_env(monkeypatch)
    combo_results = [_make_combo_result_with_findings()]
    llm = MagicMock(spec=["call"])
    llm.call.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        "## Summary\n"
                        "CET1 was 13.7% [REF:1].\n\n"
                        "## Gaps\nNone identified\n"
                    )
                }
            }
        ],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 40,
            "total_tokens": 120,
        },
    }

    result = consolidate_results("CET1 ratio", combo_results, llm)

    assert "[pillar3, Page 3 - Sheet KM1]" in (result["consolidated_response"])
    assert result["data_gaps"] == []
    llm.call.assert_called_once()
