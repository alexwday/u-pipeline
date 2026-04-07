"""Tests for citation validation in consolidated output."""

from retriever.models import ConsolidatedResult
from retriever.utils.citation_validator import (
    _build_cited_pages,
    _find_unreplaced_refs,
    validate_consolidated_citations,
)


def _make_result(
    response: str = "CET1 was 13.7% [pillar3, Page 3 - Sheet KM1].",
    reference_index: list | None = None,
) -> ConsolidatedResult:
    """Build a ConsolidatedResult fixture."""
    if reference_index is None:
        reference_index = [
            {
                "ref_id": 1,
                "finding": "CET1 was 13.7%",
                "page": 3,
                "source": "pillar3",
                "location_detail": "Sheet KM1",
            },
        ]
    return ConsolidatedResult(
        query="CET1 ratio",
        combo_results=[],
        consolidated_response=response,
        key_findings=["CET1 was 13.7%"],
        data_gaps=[],
        reference_index=reference_index,
    )


def test_find_unreplaced_refs_none():
    """Clean text has no unreplaced refs."""
    result = _find_unreplaced_refs("CET1 was 13.7% [pillar3, Page 3].")
    assert not result


def test_find_unreplaced_refs_found():
    """Unreplaced [REF:N] patterns are detected."""
    result = _find_unreplaced_refs("CET1 was 13.7% [REF:1] and [REF:99].")
    assert result == ["[REF:1]", "[REF:99]"]


def test_build_cited_pages():
    """Pages grouped by source from reference index."""
    index = [
        {"ref_id": 1, "page": 3, "source": "pillar3"},
        {"ref_id": 2, "page": 10, "source": "slides"},
        {"ref_id": 3, "page": 3, "source": "pillar3"},
    ]
    pages = _build_cited_pages(index)
    assert pages == {"pillar3": [3], "slides": [10]}


def test_build_cited_pages_empty():
    """Empty index returns empty dict."""
    assert _build_cited_pages([]) == {}


def test_validate_clean_result():
    """Clean result passes validation with no warnings."""
    result = _make_result()
    metrics: dict = {}
    validated = validate_consolidated_citations(result, metrics=metrics)

    assert "citation_warnings" not in validated
    assert metrics["unreplaced_ref_count"] == 0
    assert metrics["cited_pages_by_source"] == {"pillar3": [3]}


def test_validate_unreplaced_refs():
    """Unreplaced [REF:N] generates warnings."""
    result = _make_result(response="CET1 was [REF:1] and Tier 1 was [REF:99].")
    metrics: dict = {}
    validated = validate_consolidated_citations(result, metrics=metrics)

    assert len(validated["citation_warnings"]) == 2
    assert metrics["unreplaced_ref_count"] == 2


def test_validate_empty_reference_index():
    """Empty reference index produces zero citations_checked."""
    result = _make_result(reference_index=[])
    metrics: dict = {}
    validate_consolidated_citations(result, metrics=metrics)

    assert metrics["citations_checked"] == 0
    assert metrics["catalog_sources"] == 0


def test_validate_preserves_optional_fields():
    """Optional fields from the input result are preserved."""
    result = _make_result()
    result["summary_answer"] = "Summary text"
    result["metrics_table"] = "| col |"
    validated = validate_consolidated_citations(result)

    assert validated["summary_answer"] == "Summary text"
    assert validated["metrics_table"] == "| col |"
