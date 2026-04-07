"""Tests for the deterministic metrics aggregator."""

from retriever.models import (
    ComboSourceResult,
    ComboSpec,
    ResearchFinding,
    SourceSpec,
)
from retriever.utils.metrics_aggregator import (
    _format_source_refs,
    _is_quantitative,
    _pick_display_label,
    _replace_abbreviation_parens,
    _resolve_period_canonical,
    _strip_period_suffixes,
    build_metrics_inventory,
    canonicalize_combo_period,
    canonicalize_metric_name,
    canonicalize_period,
    canonicalize_unit,
)


def _count_metric_blocks(text: str) -> int:
    """Count the number of [METRIC] block headers in inventory text."""
    return sum(1 for line in text.splitlines() if line.startswith("[METRIC]"))


def _make_finding(**overrides) -> ResearchFinding:
    """Build a ResearchFinding with reasonable defaults."""
    base = {
        "finding": "CET1 ratio was 13.7%",
        "page": 4,
        "location_detail": "Selected highlights",
        "metric_name": "CET1 ratio",
        "metric_value": "13.7",
        "unit": "%",
        "period": "Q1 2026",
        "segment": "Enterprise",
    }
    base.update(overrides)
    return ResearchFinding(**base)


def _make_combo_result(
    *,
    bank: str = "RBC",
    period: str = "2026-q1",
    data_source: str = "rts",
    filename: str = "rbc_q1_2026_rts.pdf",
    findings: list | None = None,
) -> ComboSourceResult:
    """Build a minimal ComboSourceResult fixture."""
    if findings is None:
        findings = [_make_finding()]
    return ComboSourceResult(
        combo=ComboSpec(bank=bank, period=period),
        source=SourceSpec(
            data_source=data_source,
            document_version_id=1,
            filename=filename,
        ),
        research_iterations=[],
        findings=findings,
        chunk_count=1,
        total_tokens=10,
    )


# -- canonicalize_metric_name --


def test_canonicalize_metric_name_collapses_long_form_with_abbrev():
    """The verbose form collapses to the abbreviation."""
    assert (
        canonicalize_metric_name("Common Equity Tier 1 (CET1) ratio")
        == "cet1 ratio"
    )


def test_canonicalize_metric_name_matches_bare_abbrev():
    """The bare abbreviation form matches the long-form canonical."""
    assert canonicalize_metric_name("CET1 ratio") == "cet1 ratio"


def test_canonicalize_metric_name_strips_trailing_unit_paren():
    """A trailing (%) suffix is stripped."""
    assert canonicalize_metric_name("CET1 ratio (%)") == "cet1 ratio"


def test_canonicalize_metric_name_strips_prior_quarter_suffix():
    """A trailing '— prior quarter' suffix is stripped."""
    assert (
        canonicalize_metric_name("CET1 ratio — prior quarter") == "cet1 ratio"
    )


def test_canonicalize_metric_name_strips_quarter_suffix():
    """A trailing '— Q4/2025' suffix is stripped."""
    assert canonicalize_metric_name("CET1 ratio — Q4/2025") == "cet1 ratio"


def test_canonicalize_metric_name_strips_yoy_suffix():
    """A trailing '— YoY' suffix is stripped."""
    assert canonicalize_metric_name("PPPT — YoY") == "pppt"


def test_canonicalize_metric_name_strips_fy_suffix():
    """A trailing '— FY2025' suffix is stripped."""
    assert canonicalize_metric_name("Net income — FY2025") == "net income"


def test_canonicalize_metric_name_preserves_qoq_change_prefix():
    """A QoQ Change prefix on a delta finding is preserved."""
    canonical = canonicalize_metric_name(
        "QoQ Change (a-b) — Common Equity Tier 1 (CET1)"
    )
    assert canonical == "qoq change (a-b) — cet1"


def test_canonicalize_metric_name_handles_lowercase_in_long_form():
    """Lowercase tokens like 'for' do not break the chain."""
    assert (
        canonicalize_metric_name("Provision for Credit Losses (PCL)") == "pcl"
    )


def test_canonicalize_metric_name_handles_lowercase_subject_words():
    """Lowercase subject words ('credit losses') still chain to the abbrev."""
    assert (
        canonicalize_metric_name("Provision for credit losses (PCL)") == "pcl"
    )


def test_canonicalize_metric_name_empty_returns_empty():
    """An empty input returns an empty string."""
    assert canonicalize_metric_name("") == ""


def test_canonicalize_metric_name_does_not_collapse_lone_lowercase_prefix():
    """A lowercase prefix like 'the (FOO)' must NOT collapse."""
    assert canonicalize_metric_name("the (FOO)") == "the (foo)"


# -- canonicalize_period --


def test_canonicalize_period_quarter_slash():
    """Q1/26 becomes Q1 2026."""
    assert canonicalize_period("Q1/26") == "Q1 2026"


def test_canonicalize_period_quarter_space():
    """Q1 2026 stays Q1 2026."""
    assert canonicalize_period("Q1 2026") == "Q1 2026"


def test_canonicalize_period_quarter_with_year_only():
    """Q1/2026 normalizes to Q1 2026."""
    assert canonicalize_period("Q1/2026") == "Q1 2026"


def test_canonicalize_period_fy():
    """FY 25 normalizes to FY 2025."""
    assert canonicalize_period("FY 25") == "FY 2025"


def test_canonicalize_period_delta_preserved():
    """A delta period is preserved verbatim."""
    text = "QoQ change Q1 2026 vs Q4 2025"
    assert canonicalize_period(text) == text


def test_canonicalize_period_date_passes_through():
    """A free-form date string is returned unchanged."""
    assert (
        canonicalize_period("As at January 31 2026") == "As at January 31 2026"
    )


def test_canonicalize_period_empty():
    """Empty input returns empty."""
    assert canonicalize_period("") == ""


# -- canonicalize_combo_period --


def test_canonicalize_combo_period_year_dash_quarter():
    """2026-q1 becomes Q1 2026."""
    assert canonicalize_combo_period("2026-q1") == "Q1 2026"


def test_canonicalize_combo_period_quarter_dash_year():
    """q1-2026 becomes Q1 2026."""
    assert canonicalize_combo_period("q1-2026") == "Q1 2026"


def test_canonicalize_combo_period_short_year():
    """q1-26 expands to Q1 2026."""
    assert canonicalize_combo_period("q1-26") == "Q1 2026"


def test_canonicalize_combo_period_empty():
    """Empty input returns empty."""
    assert canonicalize_combo_period("") == ""


def test_canonicalize_combo_period_unrecognized_falls_through():
    """Unrecognized combo period passes through canonicalize_period."""
    assert canonicalize_combo_period("unknown") == "unknown"


# -- canonicalize_unit --


def test_canonicalize_unit_strip():
    """Unit is stripped of whitespace."""
    assert canonicalize_unit(" %") == "%"


def test_canonicalize_unit_empty():
    """Empty unit returns empty."""
    assert canonicalize_unit("") == ""


# -- _strip_period_suffixes --


def test_strip_period_suffixes_idempotent():
    """Running the stripper twice gives the same result."""
    once = _strip_period_suffixes("CET1 ratio — prior quarter")
    twice = _strip_period_suffixes(once)
    assert once == twice == "CET1 ratio"


def test_strip_period_suffixes_chained():
    """Chained suffixes are removed in a single call."""
    text = "CET1 ratio — prior quarter — Q4/2025"
    assert _strip_period_suffixes(text) == "CET1 ratio"


# -- _replace_abbreviation_parens --


def test_replace_abbreviation_parens_basic():
    """Long form before abbreviation is collapsed."""
    assert (
        _replace_abbreviation_parens("Common Equity Tier 1 (CET1) ratio")
        == "CET1 ratio"
    )


def test_replace_abbreviation_parens_preserves_unrelated_prefix():
    """Tokens before the long form are preserved."""
    assert (
        _replace_abbreviation_parens(
            "QoQ Change (a-b) — Common Equity Tier 1 (CET1)"
        )
        == "QoQ Change (a-b) — CET1"
    )


def test_replace_abbreviation_parens_no_match():
    """Text without a parens-abbrev returns unchanged."""
    assert (
        _replace_abbreviation_parens("Plain metric name")
        == "Plain metric name"
    )


# -- _pick_display_label --


def test_pick_display_label_prefers_with_abbrev():
    """A label containing (CET1) wins over a bare label."""
    chosen = _pick_display_label(
        ["CET1 ratio", "Common Equity Tier 1 (CET1) ratio"]
    )
    assert chosen == "Common Equity Tier 1 (CET1) ratio"


def test_pick_display_label_shortest_when_no_abbrev():
    """Falls back to the shortest non-empty label."""
    chosen = _pick_display_label(
        ["A long metric label", "Short", "Medium label"]
    )
    assert chosen == "Short"


def test_pick_display_label_empty():
    """Empty input returns empty string."""
    assert _pick_display_label([]) == ""
    assert _pick_display_label(["", ""]) == ""


# -- _is_quantitative --


def test_is_quantitative_true():
    """A finding with metric_name and metric_value is quantitative."""
    assert _is_quantitative(_make_finding())


def test_is_quantitative_missing_value():
    """Empty metric_value makes the finding qualitative."""
    finding = _make_finding(metric_value="")
    assert not _is_quantitative(finding)


def test_is_quantitative_missing_name():
    """Empty metric_name makes the finding qualitative."""
    finding = _make_finding(metric_name="")
    assert not _is_quantitative(finding)


# -- _resolve_period_canonical --


def test_resolve_period_canonical_empty_finding_uses_combo():
    """An empty finding period falls back to the combo period."""
    assert _resolve_period_canonical("", "Q1 2026") == "Q1 2026"


def test_resolve_period_canonical_date_string_uses_combo():
    """A free-form date string falls back to the combo period."""
    assert (
        _resolve_period_canonical("As at January 31 2026", "Q1 2026")
        == "Q1 2026"
    )


def test_resolve_period_canonical_date_string_no_combo_passes_through():
    """Without a combo period, a free-form date string passes through."""
    assert (
        _resolve_period_canonical("As at January 31 2026", "")
        == "As at January 31 2026"
    )


def test_resolve_period_canonical_quarter_kept_over_combo():
    """A finding's clean quarter token wins over the combo period."""
    assert _resolve_period_canonical("Q4 2025", "Q1 2026") == "Q4 2025"


def test_resolve_period_canonical_delta_kept_over_combo():
    """A delta period string is preserved over the combo period."""
    text = "QoQ change Q1 2026 vs Q4 2025"
    assert _resolve_period_canonical(text, "Q1 2026") == text


# -- _format_source_refs --


def test_format_source_refs_dedupes():
    """Identical (source, ref_id) pairs collapse to one entry."""
    records = [
        {"source": "rts", "ref_id": 1},
        {"source": "rts", "ref_id": 1},
        {"source": "pillar3", "ref_id": 2},
    ]
    assert _format_source_refs(records) == "rts [REF:1]; pillar3 [REF:2]"


def test_format_source_refs_zero_ref_id():
    """A zero ref_id falls back to the bare source label."""
    records = [{"source": "rts", "ref_id": 0}]
    assert _format_source_refs(records) == "rts"


# -- build_metrics_inventory: end-to-end --


def test_build_metrics_inventory_empty():
    """No combo_results returns an empty string."""
    assert build_metrics_inventory([]) == ""


def test_build_metrics_inventory_no_quantitative():
    """Only qualitative findings produce an empty inventory."""
    qualitative = _make_finding(
        metric_name="",
        metric_value="",
        unit="",
        period="",
        segment="",
        finding="Management commentary on capital position",
    )
    combo = _make_combo_result(findings=[qualitative])
    assert build_metrics_inventory([combo]) == ""


def test_build_metrics_inventory_merges_same_metric_across_sources():
    """Same canonical metric across sources collapses to ONE block.

    This is the regression test for the user's primary complaint:
    CET1 ratio reported by 3 sources used to render as 3 rows.
    """
    investor = _make_combo_result(
        data_source="investor-slides",
        filename="rbc_slides.pdf",
        findings=[
            _make_finding(
                metric_name="CET1 ratio",
                page=9,
                location_detail="Capital Overview",
            ),
        ],
    )
    pillar3 = _make_combo_result(
        data_source="pillar3",
        filename="rbc_pillar3.xlsx",
        findings=[
            _make_finding(
                metric_name="CET1 ratio (%)",
                page=3,
                location_detail="KM1",
            ),
        ],
    )
    rts = _make_combo_result(
        data_source="rts",
        filename="rbc_rts.pdf",
        findings=[
            _make_finding(
                metric_name="Common Equity Tier 1 (CET1) ratio",
                page=4,
                location_detail="Selected highlights",
            ),
        ],
    )

    text = build_metrics_inventory([investor, pillar3, rts])

    assert _count_metric_blocks(text) == 1
    assert "investor-slides [REF:1]" in text
    assert "pillar3 [REF:2]" in text
    assert "rts [REF:3]" in text
    assert "Q1 2026: 13.7" in text


def test_build_metrics_inventory_pivots_periods_into_columns():
    """Same metric, two periods → ONE block with two period rows."""
    findings = [
        _make_finding(
            metric_name="CET1 ratio",
            metric_value="13.7",
            period="Q1 2026",
            page=9,
        ),
        _make_finding(
            metric_name="CET1 ratio",
            metric_value="13.5",
            period="Q4 2025",
            page=9,
        ),
    ]
    combo = _make_combo_result(findings=findings)
    text = build_metrics_inventory([combo])

    assert _count_metric_blocks(text) == 1
    assert "Q1 2026: 13.7" in text
    assert "Q4 2025: 13.5" in text


def test_build_metrics_inventory_period_leak_in_metric_name():
    """A period leak in metric_name still merges with the canonical row.

    The research stage sometimes emits a comparative finding with the
    period qualifier baked into metric_name (e.g.
    "CET1 ratio — prior quarter"). The aggregator must strip the
    qualifier and merge it into the same block.
    """
    findings = [
        _make_finding(
            metric_name="CET1 ratio",
            metric_value="13.7",
            period="Q1 2026",
        ),
        _make_finding(
            metric_name="CET1 ratio — prior quarter",
            metric_value="13.5",
            period="Q4 2025",
        ),
    ]
    combo = _make_combo_result(findings=findings)
    text = build_metrics_inventory([combo])

    assert _count_metric_blocks(text) == 1
    assert "Q1 2026: 13.7" in text
    assert "Q4 2025: 13.5" in text


def test_build_metrics_inventory_distinct_units_stay_separate():
    """Same canonical name but different units → separate blocks."""
    findings = [
        _make_finding(
            metric_name="Net internal capital generation",
            metric_value="3.4",
            unit="$B",
        ),
        _make_finding(
            metric_name="Net internal capital generation",
            metric_value="46",
            unit="bps",
        ),
    ]
    combo = _make_combo_result(findings=findings)
    text = build_metrics_inventory([combo])

    assert _count_metric_blocks(text) == 2
    assert "$B" in text
    assert "bps" in text


def test_build_metrics_inventory_distinct_segments_stay_separate():
    """Same canonical name but different segments → separate blocks."""
    findings = [
        _make_finding(
            metric_name="Revenue",
            metric_value="100",
            segment="Enterprise",
        ),
        _make_finding(
            metric_name="Revenue",
            metric_value="40",
            segment="Capital Markets",
        ),
    ]
    combo = _make_combo_result(findings=findings)
    text = build_metrics_inventory([combo])

    assert _count_metric_blocks(text) == 2
    assert "Enterprise" in text
    assert "Capital Markets" in text


def test_build_metrics_inventory_values_disagree_surfaced():
    """Disagreeing values within one period surface explicitly."""
    findings = [
        _make_finding(
            metric_name="Retained earnings",
            metric_value="104,809",
            unit="$MM",
            period="Q1 2026",
        ),
        _make_finding(
            metric_name="Retained earnings",
            metric_value="99,023",
            unit="$MM",
            period="Q1 2026",
        ),
    ]
    combo = _make_combo_result(findings=findings)
    text = build_metrics_inventory([combo])

    assert "VALUES DISAGREE" in text
    assert "104,809" in text
    assert "99,023" in text


def test_build_metrics_inventory_combo_period_fallback():
    """Date-string period falls back to combo's filing period."""
    finding = _make_finding(
        metric_name="CET1 capital",
        metric_value="100,415",
        unit="$MM",
        period="As at January 31 2026",
    )
    combo = _make_combo_result(period="2026-q1", findings=[finding])
    text = build_metrics_inventory([combo])

    assert "Q1 2026: 100,415" in text


def test_build_metrics_inventory_delta_period_kept():
    """Delta period strings are not collapsed into the base period."""
    findings = [
        _make_finding(
            metric_name="CET1 capital",
            metric_value="100,415",
            unit="$MM",
            period="Q1 2026",
        ),
        _make_finding(
            metric_name="QoQ Change (a-b) — Common Equity Tier 1 (CET1)",
            metric_value="1,667",
            unit="$MM",
            period="QoQ change Q1 2026 vs Q4 2025",
        ),
    ]
    combo = _make_combo_result(findings=findings)
    text = build_metrics_inventory([combo])

    assert _count_metric_blocks(text) == 2
    assert "QoQ change Q1 2026 vs Q4 2025: 1,667" in text


def test_build_metrics_inventory_omits_empty_metric_name():
    """A finding with empty canonical metric_name is skipped."""
    finding = _make_finding(
        metric_name="(%)",
        metric_value="13.7",
    )
    combo = _make_combo_result(findings=[finding])
    text = build_metrics_inventory([combo])

    assert text == ""


def test_build_metrics_inventory_ref_ids_match_reference_index():
    """Aggregator ref_ids mirror _build_reference_index numbering.

    Both functions iterate combo_results in order and assign a global
    counter per (filename, page, location_detail). The aggregator
    inventory must produce the same ref ids the LLM sees so the LLM
    can render [REF:N] inline citations correctly.
    """
    # pylint: disable=import-outside-toplevel
    from retriever.stages.consolidate import _build_reference_index

    investor = _make_combo_result(
        data_source="investor-slides",
        filename="rbc_slides.pdf",
        findings=[
            _make_finding(
                metric_name="CET1 ratio",
                page=9,
                location_detail="Capital Overview",
            ),
        ],
    )
    pillar3 = _make_combo_result(
        data_source="pillar3",
        filename="rbc_pillar3.xlsx",
        findings=[
            _make_finding(
                metric_name="CET1 ratio",
                page=3,
                location_detail="KM1",
            ),
        ],
    )

    entries, _ = _build_reference_index([investor, pillar3])
    inventory = build_metrics_inventory([investor, pillar3])

    investor_ref = next(e for e in entries if e["source"] == "investor-slides")
    pillar3_ref = next(e for e in entries if e["source"] == "pillar3")
    assert f"investor-slides [REF:{investor_ref['ref_id']}]" in inventory
    assert f"pillar3 [REF:{pillar3_ref['ref_id']}]" in inventory


def test_build_metrics_inventory_header_present():
    """The inventory text starts with the documented header."""
    combo = _make_combo_result()
    text = build_metrics_inventory([combo])
    assert "=== Aggregated Metrics Inventory ===" in text
    assert "Render rows directly from these blocks" in text
