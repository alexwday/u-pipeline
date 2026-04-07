"""Tests for research consolidation across sources."""

from pathlib import Path
from unittest.mock import MagicMock

from retriever.models import (
    ComboSourceResult,
    ComboSpec,
    ResearchFinding,
    SourceSpec,
)
from retriever.stages.consolidate import (
    _build_coverage_audit_section,
    _build_reference_index,
    _compute_coverage_audit,
    _create_ref_replacer,
    _extract_gap_list,
    _extract_metric_value_tokens,
    _finding_token_in_text,
    _format_finding_summary,
    _parse_sections,
    consolidate_results,
)
from retriever.utils.prompt_loader import load_prompt

_CONSOLIDATION_PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "retriever"
    / "stages"
    / "prompts"
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
    assert entries[0]["page"] == 3
    assert entries[0]["source"] == "pillar3"
    assert entries[0]["filename"] == "rbc_q1_2026_pillar3.xlsx"
    assert entries[0]["entity"] == "RBC"
    assert len(entries[0]["findings"]) == 1
    assert entries[0]["findings"][0]["finding"] == "CET1 ratio was 13.7%"
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
    finding = entries[0]["findings"][0]
    assert "metric_name" not in finding
    assert "[REF:1]" in text
    assert "Management indicated strong outlook" in text


def test_build_reference_index_dedupes_same_location():
    """Findings at the same (file, page, location) collapse to one ref."""
    results = [
        _make_combo_result_with_findings(
            findings=[
                ResearchFinding(
                    finding="CET1 ratio was 13.7%",
                    page=10,
                    location_detail="CC1: Composition",
                    metric_name="CET1",
                    metric_value="13.7%",
                    period="Q1/26",
                ),
                ResearchFinding(
                    finding="Tier 1 ratio was 15.2%",
                    page=10,
                    location_detail="CC1: Composition",
                    metric_name="Tier 1",
                    metric_value="15.2%",
                    period="Q1/26",
                ),
                ResearchFinding(
                    finding="CET1 ratio was 13.5% in prior quarter",
                    page=10,
                    location_detail="CC1: Composition",
                    metric_name="CET1",
                    metric_value="13.5%",
                    period="Q4/25",
                ),
            ],
        ),
    ]
    entries, text = _build_reference_index(results)

    assert len(entries) == 1
    assert entries[0]["ref_id"] == 1
    assert len(entries[0]["findings"]) == 3
    assert text.count("[REF:") == 1


# -- _create_ref_replacer tests --


def _make_ref_entries():
    """Build sample reference entries for replacer tests."""
    return [
        {
            "ref_id": 1,
            "page": 3,
            "location_detail": "Sheet KM1",
            "source": "pillar3",
            "filename": "rbc_pillar3.xlsx",
        },
        {
            "ref_id": 2,
            "page": 10,
            "location_detail": "Capital Overview",
            "source": "investor-slides",
            "filename": "rbc_slides.pdf",
        },
    ]


def test_ref_replacer_basic():
    """Single-chunk replacement emits clickable anchor citations."""
    replacer = _create_ref_replacer(_make_ref_entries())
    result = replacer["replace"](
        "CET1 was 13.7% [REF:1] and Tier 1 was 15.2% [REF:2]."
    )
    assert (
        '<a href="https://docs.local/files/rbc_pillar3.xlsx#page=3">[1]</a>'
        in result
    )
    assert (
        '<a href="https://docs.local/files/rbc_slides.pdf#page=10">[2]</a>'
        in result
    )
    assert "[REF:" not in result
    assert 1 in replacer["used_refs"]
    assert 2 in replacer["used_refs"]


def test_ref_replacer_split_across_chunks():
    """[REF:N] split across two chunks is correctly buffered."""
    replacer = _create_ref_replacer(_make_ref_entries())

    part1 = replacer["replace"]("CET1 was 13.7% [RE")
    part2 = replacer["replace"]("F:1] end.")

    combined = part1 + part2
    assert (
        '<a href="https://docs.local/files/rbc_pillar3.xlsx#page=3">[1]</a>'
        in combined
    )
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
    """Ref with no location_detail still produces a clickable anchor."""
    entries = [
        {
            "ref_id": 1,
            "page": 5,
            "location_detail": "",
            "source": "pillar3",
            "filename": "rbc.xlsx",
        },
    ]
    replacer = _create_ref_replacer(entries)
    result = replacer["replace"]("See [REF:1].")
    assert (
        '<a href="https://docs.local/files/rbc.xlsx#page=5">[1]</a>' in result
    )


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
    assert '<a href="https://docs.local/files/' in joined
    assert ">[1]</a>" in joined
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


# -- consolidation.yaml structural tests --


def _load_consolidation_prompt() -> dict:
    """Load the consolidation prompt for structural assertions."""
    return load_prompt(
        "consolidation",
        prompts_dir=_CONSOLIDATION_PROMPTS_DIR,
    )


def test_consolidation_prompt_has_reconciliation_rule():
    """Rule 7 tells the LLM to reconcile component bridges."""
    prompt = _load_consolidation_prompt()
    user_prompt = prompt["user_prompt"]

    assert "Reconciliation check" in user_prompt
    assert "components sum to the stated total" in user_prompt
    assert "movement walk" in user_prompt
    assert "waterfall" in user_prompt


def test_consolidation_prompt_gaps_section_widened():
    """Gaps section accepts internal consistency issues."""
    prompt = _load_consolidation_prompt()
    user_prompt = prompt["user_prompt"]

    assert "internal consistency issue" in user_prompt
    assert "does not reconcile" in user_prompt
    assert "scopes differ" in user_prompt


def test_consolidation_prompt_metrics_one_value_per_cell():
    """Metrics section forbids multi-value cells."""
    prompt = _load_consolidation_prompt()
    user_prompt = prompt["user_prompt"]

    assert "ONE VALUE PER CELL" in user_prompt
    assert "semicolon-separated" in user_prompt


def test_consolidation_prompt_metrics_scope_suffix_rule():
    """Metrics section handles precision, scope, and conflict cases."""
    prompt = _load_consolidation_prompt()
    flat = " ".join(prompt["user_prompt"].split())

    assert "rounding precision" in flat
    assert "different scopes" in flat or "consolidation perimeters" in flat
    assert "genuinely differ" in flat
    assert "suffix in the Line Item column" in flat


def test_consolidation_prompt_metrics_duplicate_row_check():
    """Metrics section requires a source-aware duplicate-row check."""
    prompt = _load_consolidation_prompt()
    flat = " ".join(prompt["user_prompt"].split())

    assert "DUPLICATE ROW CHECK" in flat
    assert "identical Entity + Segment + Line Item + Source" in flat
    assert "MUST be merged into a single row" in flat


def test_consolidation_prompt_metrics_has_source_column():
    """Source column combines source + ref inline (F10 restructure)."""
    prompt = _load_consolidation_prompt()
    flat = " ".join(prompt["user_prompt"].split())

    assert "Source (last column)" in flat
    assert "preserve source transparency" in flat
    assert "ONE ROW PER SOURCE" in flat
    assert "Surface the discrepancy as an item in the Gaps section" in flat
    assert "investor-slides [REF:1]; rts [REF:11]" in flat
    assert "refs live inline in the Source column" in flat
    assert "There is NO separate Ref column" in flat


def test_consolidation_prompt_has_scope_comparability_rule():
    """Rule 8 tells the LLM to verify scope before juxtaposing values."""
    prompt = _load_consolidation_prompt()
    flat = " ".join(prompt["user_prompt"].split())

    assert "Scope comparability" in flat
    assert "population scope or measurement basis" in flat
    assert "not directly additive or comparable" in flat


def test_consolidation_prompt_has_formatting_standards():
    """Metrics section defines Unit column / delta / missing-data rules."""
    prompt = _load_consolidation_prompt()
    flat = " ".join(prompt["user_prompt"].split())

    assert "FORMATTING STANDARDS" in flat
    assert "Unit column" in flat
    assert '"$MM"' in flat
    assert '"$B"' in flat
    assert '"bps"' in flat
    assert "Line Item column contains the metric NAME ONLY" in flat
    assert "do NOT append a parenthesized unit suffix to Line Item" in flat
    assert "no currency symbol" in flat
    assert "no thousands separator" in flat
    assert "SIGN-EXPLICIT" in flat
    assert 'NEVER use the words "flat"' in flat
    assert 'em dash "—"' in flat
    assert "Never leave a cell empty between pipes" in flat


def test_consolidation_prompt_has_segment_residual_rule():
    """Rule 9 tells the LLM to compute segment-to-enterprise residuals."""
    prompt = _load_consolidation_prompt()
    flat = " ".join(prompt["user_prompt"].split())

    assert "Segment-to-enterprise residual" in flat
    assert "same metric, same period, and same scope" in flat
    assert "as large in magnitude" in flat
    assert "opposite direction to the enterprise" in flat
    assert "non-[segment] segments collectively" in flat
    assert "do not invent either value" in flat
    assert "Cite the same [REF:N] that supported the two input values" in flat


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

    assert (
        '<a href="https://docs.local/files/' in result["consolidated_response"]
    )
    assert ">[1]</a>" in result["consolidated_response"]
    assert result["data_gaps"] == []
    llm.call.assert_called_once()


def test_consolidate_passes_aggregated_metrics_to_llm(monkeypatch):
    """The aggregated metrics inventory reaches the LLM user prompt."""
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

    consolidate_results("CET1 ratio", combo_results, llm)

    call_messages = llm.call.call_args.kwargs["messages"]
    user_text = next(
        msg["content"] for msg in call_messages if msg["role"] == "user"
    )
    assert "<aggregated_metrics>" in user_text
    assert "[METRIC] CET1 Ratio" in user_text
    assert "Q1 2026: 13.7" in user_text
    assert "{aggregated_metrics}" not in user_text


def test_consolidation_prompt_references_aggregated_metrics():
    """Prompt instructs the LLM to render rows from the inventory."""
    prompt = _load_consolidation_prompt()
    flat = " ".join(prompt["user_prompt"].split())
    assert "<aggregated_metrics>" in flat
    assert "{aggregated_metrics}" in flat
    assert "rendering each [METRIC] entry" in flat
    assert "Do NOT split a single [METRIC] entry" in flat


# -- F10: unit field in _format_finding_summary --


def test_format_finding_summary_includes_unit():
    """Finding with unit field renders 'metric = value unit' in ref index."""
    finding: ResearchFinding = {
        "finding": "CET1 capital was 100,415 million CAD",
        "page": 3,
        "location_detail": "Sheet KM1",
        "metric_name": "Common Equity Tier 1 (CET1)",
        "metric_value": "100,415",
        "unit": "$MM",
        "period": "Q1 2026",
        "segment": "Enterprise",
    }
    text = _format_finding_summary(finding)
    assert "Common Equity Tier 1 (CET1) = 100,415 $MM" in text
    assert "(Q1 2026)" in text
    assert "[Enterprise]" in text


def test_format_finding_summary_without_unit_legacy():
    """Legacy finding without unit field renders cleanly (backwards compat)."""
    finding: ResearchFinding = {
        "finding": "CET1 ratio was 13.7%",
        "page": 3,
        "location_detail": "Sheet KM1",
        "metric_name": "CET1 Ratio",
        "metric_value": "13.7%",
        "period": "Q1 2026",
        "segment": "Enterprise",
    }
    text = _format_finding_summary(finding)
    assert "CET1 Ratio = 13.7%" in text
    # No extraneous trailing space or empty unit
    assert "13.7% (Q1 2026)" in text


# -- F06 cluster: coverage audit tests --


def test_consolidation_prompt_has_coverage_commitment_rule():
    """Rule 6 (replaced) tells the LLM to walk every ref and cover findings."""
    prompt = _load_consolidation_prompt()
    flat = " ".join(prompt["user_prompt"].split())

    assert "Coverage commitment" in flat
    assert "mentally walk each [REF:N] entry" in flat
    assert "EVERY finding listed under that ref is represented" in flat
    assert "Do NOT treat qualitative findings as background" in flat
    assert "Adjusted-vs-reported pairs" in flat
    assert "each distinct finding must be represented independently" in flat
    assert "Retreat from a topic to avoid scope conflict is a coverage" in flat
    assert "Derived inferences allowed by rule 9" in flat
    assert (
        "Do NOT synthesize Metrics table rows by parsing numeric values"
        in flat
    )
    assert "bypasses research-stage rule 22" in flat
    assert (
        "built ONLY from findings whose metric_name and metric_value "
        "fields are populated"
    ) in flat


def test_consolidation_prompt_rule6g_preserves_qualitative_to_detail_routing():
    """Clause (g) routes qualitative-finding numbers to Detail, not Metrics."""
    prompt = _load_consolidation_prompt()
    flat = " ".join(prompt["user_prompt"].split())

    assert "belong in Detail prose" in flat
    assert "NOT independent metrics" in flat
    assert "reintroduces forbidden component rows" in flat
    assert "evidence inside the qualitative narrative" in flat


def test_extract_metric_value_tokens_basic():
    """Tokens extracted from common metric_value shapes."""
    assert _extract_metric_value_tokens("12%") == ["12%"]
    assert _extract_metric_value_tokens("634") == ["634"]
    assert _extract_metric_value_tokens("73 bps (+2 bps QoQ)") == [
        "73 bps",
        "2 bps",
    ]


def test_extract_metric_value_tokens_strips_separators():
    """Thousands separators are stripped before token extraction."""
    tokens = _extract_metric_value_tokens("9,294 (Allowance 2,238)")
    assert "9294" in tokens
    assert "2238" in tokens


def test_extract_metric_value_tokens_empty():
    """Empty / None metric_value returns empty list."""
    assert _extract_metric_value_tokens("") == []
    assert _extract_metric_value_tokens("no numbers here") == []


def test_finding_token_in_text_qualitative_skipped():
    """Qualitative finding (empty metric_value) is treated as represented."""
    finding = {"metric_name": "", "metric_value": "", "finding": "qualitative"}
    assert _finding_token_in_text(finding, "any text") is True


def test_finding_token_in_text_present():
    """Finding whose token appears in text is considered represented."""
    finding = {"metric_name": "ACL ratio", "metric_value": "73 bps"}
    assert _finding_token_in_text(finding, "ACL was 73 bps in Q1") is True


def test_finding_token_in_text_absent():
    """Finding whose token does not appear is flagged."""
    finding = {"metric_name": "Adjusted PPPT", "metric_value": "12%"}
    text = "Reported PPPT grew 14% YoY"
    assert _finding_token_in_text(finding, text) is False


def test_finding_token_in_text_rounded_match():
    """A bare number finding still matches when wrapped in currency suffix."""
    finding = {"metric_name": "Net write-offs", "metric_value": "634"}
    text = "Net write-offs were $634MM in Q1"
    assert _finding_token_in_text(finding, text) is True


def test_finding_token_in_text_thousands_separator_match():
    """Comma-separated source value matches comma-stripped response token."""
    finding = {"metric_name": "Impaired exposures", "metric_value": "9294"}
    # caller pre-strips commas before searching
    text = "Total impaired exposures of 9294 reported"
    assert _finding_token_in_text(finding, text) is True


def test_compute_coverage_audit_layer1_uncited():
    """Layer 1 detects refs that were never cited."""
    ref_entries = [
        {
            "ref_id": 1,
            "source": "rts",
            "page": 4,
            "location_detail": "Capital",
            "findings": [
                {"metric_name": "CET1", "metric_value": "13.7%"},
            ],
        },
        {
            "ref_id": 2,
            "source": "pillar3",
            "page": 16,
            "location_detail": "CRB_f_b",
            "findings": [
                {"metric_name": "Net write-offs", "metric_value": "634"},
            ],
        },
    ]
    response = "CET1 was 13.7% in Q1."
    audit = _compute_coverage_audit(ref_entries, {1}, response)
    assert audit["uncited_ref_ids"] == [2]
    assert not audit["unincorporated_findings"]


def test_compute_coverage_audit_layer2_unincorporated():
    """Layer 2 detects findings whose tokens are missing from text."""
    ref_entries = [
        {
            "ref_id": 1,
            "source": "investor-slides",
            "page": 8,
            "location_detail": "Q1/26 Highlights",
            "findings": [
                {
                    "metric_name": "Reported PPPT",
                    "metric_value": "8,497",
                },
                {
                    "metric_name": "Adjusted PPPT(2) up YoY",
                    "metric_value": "12%",
                },
            ],
        },
    ]
    response = "PPPT was 8,497 in Q1, up 14% YoY."
    audit = _compute_coverage_audit(ref_entries, {1}, response)
    assert not audit["uncited_ref_ids"]
    assert len(audit["unincorporated_findings"]) == 1
    missed = audit["unincorporated_findings"][0]
    assert missed["metric_name"] == "Adjusted PPPT(2) up YoY"
    assert missed["metric_value"] == "12%"
    assert missed["ref_id"] == 1


def test_compute_coverage_audit_qualitative_not_flagged():
    """Qualitative findings inside cited refs are never layer-2 flagged."""
    ref_entries = [
        {
            "ref_id": 1,
            "source": "investor-slides",
            "page": 18,
            "location_detail": "ACL",
            "findings": [
                {
                    "metric_name": "Performing ACL",
                    "metric_value": "5,500",
                },
                {
                    "metric_name": "",
                    "metric_value": "",
                    "finding": "Release in CNB this quarter.",
                },
            ],
        },
    ]
    response = "Performing ACL was 5,500 in Q1."
    audit = _compute_coverage_audit(ref_entries, {1}, response)
    # Qualitative finding skipped — only relies on prompt rule 6(b)
    assert not audit["unincorporated_findings"]


def test_compute_coverage_audit_clean():
    """All refs cited and all tokens present yields empty audit."""
    ref_entries = [
        {
            "ref_id": 1,
            "source": "rts",
            "page": 4,
            "location_detail": "Capital",
            "findings": [
                {"metric_name": "CET1", "metric_value": "13.7%"},
            ],
        },
    ]
    audit = _compute_coverage_audit(ref_entries, {1}, "CET1 was 13.7%.")
    assert not audit["uncited_ref_ids"]
    assert not audit["unincorporated_findings"]


def test_build_coverage_audit_section_empty():
    """Clean audit produces empty string."""
    audit = {"uncited_ref_ids": [], "unincorporated_findings": []}
    assert _build_coverage_audit_section(audit, []) == ""


def test_build_coverage_audit_section_layer1_only():
    """Layer-1 only audit renders the Uncited refs subsection."""
    ref_entries = [
        {
            "ref_id": 2,
            "source": "pillar3",
            "page": 16,
            "location_detail": "CRB_f_b",
            "findings": [
                {
                    "metric_name": "Net write-offs",
                    "metric_value": "634",
                    "finding": "Total net write-offs were 634.",
                },
            ],
        },
    ]
    audit = {"uncited_ref_ids": [2], "unincorporated_findings": []}
    text = _build_coverage_audit_section(audit, ref_entries)
    assert "## Coverage audit" in text
    assert "### Uncited refs" in text
    assert "[REF:2]" in text
    assert "pillar3 p16" in text
    assert "CRB_f_b" in text
    assert "Net write-offs" in text


def test_build_coverage_audit_section_layer2_only():
    """Layer-2 only audit renders the Unincorporated findings subsection."""
    audit = {
        "uncited_ref_ids": [],
        "unincorporated_findings": [
            {
                "ref_id": 1,
                "source": "investor-slides",
                "page": 8,
                "location_detail": "Q1/26 Highlights",
                "metric_name": "Adjusted PPPT(2) up YoY",
                "metric_value": "12%",
                "finding": "Adjusted PPPT grew 12% YoY.",
            }
        ],
    }
    text = _build_coverage_audit_section(audit, [])
    assert "## Coverage audit" in text
    assert "### Unincorporated findings" in text
    assert "[REF:1] investor-slides p8" in text
    assert "Adjusted PPPT(2) up YoY = 12%" in text


def test_consolidate_audit_surfaces_dropped_finding(monkeypatch):
    """End-to-end: a finding the LLM omits gets flagged in coverage audit."""
    _set_env(monkeypatch)
    combo_results = [
        _make_combo_result_with_findings(
            findings=[
                ResearchFinding(
                    finding="Reported PPPT was 8,497.",
                    page=8,
                    location_detail="Q1/26 Highlights",
                    metric_name="Reported PPPT",
                    metric_value="8497",
                    period="Q1 2026",
                    segment="Enterprise",
                ),
                ResearchFinding(
                    finding="Adjusted PPPT grew 12% YoY.",
                    page=8,
                    location_detail="Q1/26 Highlights",
                    metric_name="Adjusted PPPT(2) up YoY",
                    metric_value="12%",
                    period="Q1 2026",
                    segment="Enterprise",
                ),
            ],
        ),
    ]

    def _stream(*_args, **_kwargs):
        yield from [
            "## Summary\n",
            "PPPT was 8497 [REF:1].\n\n",
            "## Detail\n",
            "Reported only [REF:1].\n\n",
            "## Gaps\nNone identified\n",
        ]
        return _STREAM_USAGE

    llm = MagicMock()
    llm.stream = _stream

    result = consolidate_results("PPPT", combo_results, llm)

    assert result["uncited_ref_ids"] == []
    assert len(result["unincorporated_findings"]) == 1
    missed = result["unincorporated_findings"][0]
    assert missed["metric_name"] == "Adjusted PPPT(2) up YoY"
    assert "## Coverage audit" in result["consolidated_response"]
    assert "Adjusted PPPT(2) up YoY = 12%" in result["consolidated_response"]
    assert result["coverage_audit"] != ""


def test_consolidate_audit_clean_run_has_no_section(monkeypatch):
    """Clean run leaves consolidated_response without a coverage audit."""
    _set_env(monkeypatch)
    combo_results = [_make_combo_result_with_findings()]
    llm = _make_stream_llm()

    result = consolidate_results("CET1 ratio", combo_results, llm)

    assert result["uncited_ref_ids"] == []
    assert result["unincorporated_findings"] == []
    assert result.get("coverage_audit", "") == ""
    assert "## Coverage audit" not in result["consolidated_response"]


def test_consolidate_audit_streams_to_callback(monkeypatch):
    """When audit fires, the audit section is also streamed via on_chunk."""
    _set_env(monkeypatch)
    combo_results = [
        _make_combo_result_with_findings(
            findings=[
                ResearchFinding(
                    finding="Reported PPPT was 8,497.",
                    page=8,
                    location_detail="Q1/26 Highlights",
                    metric_name="Reported PPPT",
                    metric_value="8497",
                    period="Q1 2026",
                    segment="Enterprise",
                ),
                ResearchFinding(
                    finding="Adjusted PPPT grew 12% YoY.",
                    page=8,
                    location_detail="Q1/26 Highlights",
                    metric_name="Adjusted PPPT(2) up YoY",
                    metric_value="12%",
                    period="Q1 2026",
                    segment="Enterprise",
                ),
            ],
        ),
    ]

    def _stream(*_args, **_kwargs):
        yield from [
            "## Summary\nPPPT was 8497 [REF:1].\n\n",
            "## Detail\nReported only [REF:1].\n\n",
            "## Gaps\nNone identified\n",
        ]
        return _STREAM_USAGE

    llm = MagicMock()
    llm.stream = _stream
    chunks: list[str] = []

    consolidate_results("PPPT", combo_results, llm, on_chunk=chunks.append)

    joined = "".join(chunks)
    assert "## Coverage audit" in joined
    assert "Adjusted PPPT(2) up YoY = 12%" in joined


def test_parse_sections_recognizes_coverage_audit():
    """## Coverage audit heading is parsed into the coverage_audit key."""
    text = (
        "## Summary\nFoo [REF:1].\n\n"
        "## Coverage audit\n### Uncited refs\n- [REF:2] item\n"
    )
    sections = _parse_sections(text)
    assert sections["summary"].startswith("Foo")
    assert sections["coverage_audit"].startswith("### Uncited refs")
    assert "[REF:2] item" in sections["coverage_audit"]
