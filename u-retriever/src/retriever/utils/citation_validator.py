"""Validate citations in consolidated research output."""

import re

from ..models import ComboSourceResult, ConsolidatedResult

_UNREPLACED_REF_RE = re.compile(r"\[REF:\d+\]")


def _find_unreplaced_refs(text: str) -> list[str]:
    """Find any [REF:N] patterns that were not replaced."""
    return _UNREPLACED_REF_RE.findall(text)


def _build_cited_pages(
    reference_index: list[dict],
) -> dict[str, list[int]]:
    """Build cited pages by source from the reference index.

    Params: reference_index (list[dict]). Returns: dict.
    """
    pages_by_source: dict[str, set[int]] = {}
    for entry in reference_index:
        source = entry.get("source", "")
        page = entry.get("page", 0)
        if source and page:
            pages_by_source.setdefault(source, set()).add(page)
    return {source: sorted(pages) for source, pages in pages_by_source.items()}


def _annotate_cited_evidence(
    combo_results: list[ComboSourceResult],
    cited_pages_by_source: dict[str, list[int]],
) -> None:
    """Annotate per-source evidence refs cited in the final answer."""
    for combo_result in combo_results:
        source = combo_result["source"]["data_source"]
        expand_metrics = combo_result.get("metrics", {}).get("expand", {})
        evidence_catalog = expand_metrics.get("evidence_catalog", [])
        cited_pages = cited_pages_by_source.get(source, [])
        cited_evidence: list[dict] = []
        for ref in evidence_catalog:
            page = ref.get("page_number")
            if page in cited_pages:
                cited_evidence.append(ref)
        expand_metrics["cited_page_numbers"] = cited_pages
        expand_metrics["cited_evidence"] = cited_evidence


def validate_consolidated_citations(
    result: ConsolidatedResult,
    metrics: dict | None = None,
) -> ConsolidatedResult:
    """Validate citations in the consolidated response.

    With structured findings and programmatic [REF:N] replacement,
    citations are inherently valid. This function checks for any
    unreplaced [REF:N] patterns and builds cited-pages metadata
    for tracing.

    Params:
        result: Consolidated result to validate
        metrics: Optional dict to receive validation metrics

    Returns:
        ConsolidatedResult with citation_warnings if any issues
    """
    reference_index = result.get("reference_index", [])
    cited_pages = _build_cited_pages(reference_index)
    unreplaced = _find_unreplaced_refs(result["consolidated_response"])

    warnings: list[str] = []
    for ref_text in unreplaced:
        warnings.append(f"Unreplaced reference in output: {ref_text}")

    stage_metrics = {
        "catalog_sources": len(cited_pages),
        "citations_checked": len(reference_index),
        "unreplaced_ref_count": len(unreplaced),
        "warning_count": len(warnings),
        "cited_pages_by_source": cited_pages,
        "skipped": False,
    }
    if metrics is not None:
        metrics.update(stage_metrics)

    _annotate_cited_evidence(
        result["combo_results"],
        cited_pages,
    )

    validated = ConsolidatedResult(
        query=result["query"],
        combo_results=result["combo_results"],
        consolidated_response=result["consolidated_response"],
        key_findings=result["key_findings"],
        data_gaps=result["data_gaps"],
    )
    for optional_key in (
        "summary_answer",
        "metrics_table",
        "detailed_summary",
        "reference_index",
        "metrics",
        "coverage_audit",
        "uncited_ref_ids",
        "unincorporated_findings",
    ):
        if optional_key in result:
            validated[optional_key] = result[optional_key]
    if warnings:
        validated["citation_warnings"] = warnings
    return validated
