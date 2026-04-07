"""Aggregate quantitative findings into pre-built metrics rows.

The consolidation LLM is poor at multi-axis pivoting under load:
asking it to GROUP BY (entity, segment, line item, unit) AND pivot
periods to columns AND merge sources, all from a flat reference
index, produces duplicate rows in the Metrics table.

This module performs that aggregation deterministically before the
LLM call. It walks all findings across all combo_results, derives a
canonical key per finding, groups them, and renders a text block
the LLM can read as a list of pre-built rows.
"""

import re
from collections import OrderedDict
from typing import Iterable

from ..models import ComboSourceResult, ResearchFinding

_PERIOD_SUFFIX_PATTERNS = [
    re.compile(
        r"\s*[—–-]\s*prior\s+(?:quarter|year|period)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*[—–-]\s*q[1-4]\s*[/\s]\s*\d{2,4}\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*[—–-]\s*fy\s*\d{2,4}\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*[—–-]\s*(?:yoy|qoq)(?:\s+change)?\s*$",
        re.IGNORECASE,
    ),
]

_PARENTHETICAL_UNIT_RE = re.compile(
    r"\s*\((?:\$mm|\$b|%|bps|x|#|millions?|billions?)\)\s*$",
    re.IGNORECASE,
)

_LONG_FORM_TOKEN = r"(?:[A-Za-z][a-zA-Z0-9]*|\d+)"
_LONG_FORM_ABBREV_RE = re.compile(
    r"(?:[A-Z][a-zA-Z0-9]*\s+)"
    r"(?:" + _LONG_FORM_TOKEN + r"\s+)*"
    r"\(([A-Z][A-Z0-9]+)\)"
)
_INFORMATIVE_ABBREV_RE = re.compile(r"\([A-Z][A-Z0-9]+\)")
_QUARTER_RE = re.compile(
    r"\bq([1-4])\s*[/\s]?\s*(\d{2,4})\b",
    re.IGNORECASE,
)
_FY_RE = re.compile(r"\bfy\s*[/\s]?\s*(\d{2,4})\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_DELTA_PERIOD_RE = re.compile(
    r"\b(?:change|qoq|yoy|vs\.?|versus|delta|growth)\b",
    re.IGNORECASE,
)


def _expand_year(year_token: str) -> str:
    """Expand a 2-digit year to 4-digit (assumes 20XX).

    Params: year_token (str). Returns: str.
    """
    if len(year_token) == 2:
        return f"20{year_token}"
    return year_token


def canonicalize_period(period: str) -> str:
    """Normalize a period label to a canonical comparable form.

    Recognizes quarter formats (Q1 2026, Q1/2026, Q1/26) and
    fiscal-year formats (FY 2025, FY25). Period strings that
    describe a delta or comparison ("QoQ change Q1 2026 vs Q4
    2025") are kept as-is so they do not collapse into the
    base period column. Date strings and other free-form text
    pass through with whitespace collapsed.

    Params: period (str). Returns: canonical period string.

    Example:
        >>> canonicalize_period("Q1/26")
        'Q1 2026'
        >>> canonicalize_period("QoQ change Q1 2026 vs Q4 2025")
        'QoQ change Q1 2026 vs Q4 2025'
    """
    if not period:
        return ""
    text = _WHITESPACE_RE.sub(" ", period.strip())
    if _DELTA_PERIOD_RE.search(text):
        return text
    quarter_match = _QUARTER_RE.search(text)
    if quarter_match:
        quarter = quarter_match.group(1)
        year = _expand_year(quarter_match.group(2))
        return f"Q{quarter} {year}"
    fy_match = _FY_RE.search(text)
    if fy_match:
        year = _expand_year(fy_match.group(1))
        return f"FY {year}"
    return text


def canonicalize_combo_period(combo_period: str) -> str:
    """Normalize a combo's period token (e.g. '2026-q1') to 'Q1 2026'.

    The combo period uses the dash-lowercase form chosen by the
    pipeline. This helper produces a canonical 'Q1 2026' form so
    findings whose `period` field is a date string can fall back
    to the document's filing period.

    Params: combo_period (str). Returns: canonical period string.

    Example:
        >>> canonicalize_combo_period("2026-q1")
        'Q1 2026'
    """
    if not combo_period:
        return ""
    text = combo_period.strip().lower()
    match = re.match(r"(\d{4})[-_/](q[1-4])", text)
    if match:
        year = match.group(1)
        quarter = match.group(2).upper()
        return f"{quarter} {year}"
    match = re.match(r"(q[1-4])[-_/](\d{2,4})", text)
    if match:
        quarter = match.group(1).upper()
        year = _expand_year(match.group(2))
        return f"{quarter} {year}"
    return canonicalize_period(combo_period)


def canonicalize_unit(unit: str) -> str:
    """Normalize a unit token to a canonical form.

    Params: unit (str). Returns: canonical unit string.
    """
    if not unit:
        return ""
    return unit.strip()


def _strip_period_suffixes(text: str) -> str:
    """Remove trailing period qualifiers like '— prior quarter'.

    Params: text (str). Returns: str with suffixes removed.
    """
    result = text
    changed = True
    while changed:
        changed = False
        for pattern in _PERIOD_SUFFIX_PATTERNS:
            new_result = pattern.sub("", result)
            if new_result != result:
                result = new_result
                changed = True
    return result


def _replace_abbreviation_parens(text: str) -> str:
    """Collapse 'Long Form (ABBR) suffix' to 'ABBR suffix'.

    A common bank-document pattern is to spell out a metric and
    immediately give the acronym in parentheses, e.g.
    'Common Equity Tier 1 (CET1) ratio'. The aggregator collapses
    those long-form-plus-acronym sequences so they match the bare
    'CET1 ratio' form used elsewhere.

    Only the spelled-out long form is consumed: if other tokens
    (e.g. a 'QoQ Change (a-b) — ' prefix) precede the long form,
    they are preserved.

    Params: text (str). Returns: str.

    Example:
        >>> _replace_abbreviation_parens("Common Equity Tier 1 (CET1) ratio")
        'CET1 ratio'
        >>> _replace_abbreviation_parens(
        ...     "QoQ Change (a-b) — Common Equity Tier 1 (CET1)"
        ... )
        'QoQ Change (a-b) — CET1'
    """
    return _LONG_FORM_ABBREV_RE.sub(r"\1", text)


def canonicalize_metric_name(metric_name: str) -> str:
    """Build a canonical comparable key from a raw metric_name.

    Strips trailing period qualifiers, trailing parenthesized unit
    suffixes, collapses spelled-out abbreviations to their acronym,
    lowercases, and normalizes whitespace.

    Params: metric_name (str). Returns: canonical key string.

    Example:
        >>> canonicalize_metric_name("Common Equity Tier 1 (CET1) ratio")
        'cet1 ratio'
        >>> canonicalize_metric_name("CET1 ratio (%)")
        'cet1 ratio'
        >>> canonicalize_metric_name("CET1 ratio — prior quarter")
        'cet1 ratio'
    """
    if not metric_name:
        return ""
    text = metric_name.strip()
    text = _strip_period_suffixes(text)
    text = _PARENTHETICAL_UNIT_RE.sub("", text)
    text = _replace_abbreviation_parens(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.lower()


def _is_quantitative(finding: ResearchFinding) -> bool:
    """Check whether a finding has a quantitative metric_value.

    Params: finding (ResearchFinding). Returns: bool.
    """
    return bool(finding.get("metric_name") and finding.get("metric_value"))


def _pick_display_label(labels: Iterable[str]) -> str:
    """Choose the cleanest display label from candidate strings.

    Prefers labels that already contain a parenthesized abbreviation
    (most informative), otherwise the shortest non-empty label.

    Params: labels (iterable of str). Returns: chosen label.
    """
    cleaned = [label for label in labels if label]
    if not cleaned:
        return ""
    with_abbrev = [
        label for label in cleaned if _INFORMATIVE_ABBREV_RE.search(label)
    ]
    pool = with_abbrev or cleaned
    return min(pool, key=lambda label: (len(label), label))


def _make_group_key(
    entity: str,
    segment: str,
    canonical_name: str,
    canonical_unit: str,
) -> tuple[str, str, str, str]:
    """Build the grouping key tuple for a quantitative finding.

    Params: entity, segment, canonical_name, canonical_unit (str).
    Returns: 4-tuple.
    """
    return (
        entity.strip(),
        (segment or "").strip(),
        canonical_name,
        canonical_unit,
    )


def _resolve_period_canonical(
    finding_period: str,
    combo_period_canonical: str,
) -> str:
    """Resolve the canonical period, falling back to combo period.

    When a finding's `period` field is a free-form date string
    (e.g. 'As at January 31 2026') and the combo period is known
    (e.g. 'Q1 2026'), use the combo period as the canonical key
    so it merges with sibling findings that did write a clean
    quarter token. Delta periods (containing 'change', 'vs', etc.)
    are preserved as-is.

    Params: finding_period, combo_period_canonical. Returns: str.
    """
    canonical = canonicalize_period(finding_period)
    if not canonical and combo_period_canonical:
        return combo_period_canonical
    if _DELTA_PERIOD_RE.search(canonical):
        return canonical
    if _QUARTER_RE.search(canonical) or _FY_RE.search(canonical):
        return canonical
    if combo_period_canonical:
        return combo_period_canonical
    return canonical


def _build_finding_record(
    finding: ResearchFinding,
    source_label: str,
    ref_id: int,
    combo_period_canonical: str,
) -> dict:
    """Convert a finding into an aggregator record dict.

    Params: finding, source_label, ref_id, combo_period_canonical.
    Returns: dict.
    """
    return {
        "metric_name": finding.get("metric_name", ""),
        "metric_value": finding.get("metric_value", ""),
        "unit": finding.get("unit", ""),
        "period_raw": finding.get("period", ""),
        "period_canonical": _resolve_period_canonical(
            finding.get("period", ""),
            combo_period_canonical,
        ),
        "segment": finding.get("segment", ""),
        "source": source_label,
        "ref_id": ref_id,
    }


def _iter_quantitative_records(
    combo_results: list[ComboSourceResult],
    ref_lookup_by_filename: dict[str, dict[tuple, int]],
) -> Iterable[tuple[tuple, dict]]:
    """Yield (group_key, record) for every quantitative finding.

    The ref_lookup_by_filename maps filename → ((page, location) →
    ref_id) so each record carries the same ref_id the consolidation
    LLM sees in the reference index.

    Params: combo_results, ref_lookup_by_filename. Returns: generator.
    """
    for combo_result in combo_results:
        entity = combo_result["combo"]["bank"]
        source_label = combo_result["source"]["data_source"]
        filename = combo_result["source"]["filename"]
        combo_period_canonical = canonicalize_combo_period(
            combo_result["combo"].get("period", "")
        )
        per_file_lookup = ref_lookup_by_filename.get(filename, {})
        for finding in combo_result.get("findings", []):
            if not _is_quantitative(finding):
                continue
            canonical_name = canonicalize_metric_name(
                finding.get("metric_name", "")
            )
            if not canonical_name:
                continue
            canonical_unit = canonicalize_unit(finding.get("unit", ""))
            key = _make_group_key(
                entity=entity,
                segment=finding.get("segment", ""),
                canonical_name=canonical_name,
                canonical_unit=canonical_unit,
            )
            location_key = (
                finding.get("page", 0),
                finding.get("location_detail", ""),
            )
            ref_id = per_file_lookup.get(location_key, 0)
            yield key, _build_finding_record(
                finding,
                source_label,
                ref_id,
                combo_period_canonical,
            )


def _build_ref_lookup_by_filename(
    combo_results: list[ComboSourceResult],
) -> dict[str, dict[tuple, int]]:
    """Build the per-filename (page, location) → ref_id lookup.

    Note: ref_id numbering must mirror _build_reference_index.
    That function increments a single global counter as it iterates
    combo_results in order, so this helper does the same.

    Params: combo_results. Returns: dict keyed by filename.
    """
    lookup: dict[str, dict[tuple, int]] = {}
    location_to_ref: dict[tuple, int] = {}
    counter = 0
    for combo_result in combo_results:
        filename = combo_result["source"]["filename"]
        per_file = lookup.setdefault(filename, {})
        for finding in combo_result.get("findings", []):
            file_key = (
                filename,
                finding.get("page", 0),
                finding.get("location_detail", ""),
            )
            if file_key not in location_to_ref:
                counter += 1
                location_to_ref[file_key] = counter
            ref_id = location_to_ref[file_key]
            local_key = (
                finding.get("page", 0),
                finding.get("location_detail", ""),
            )
            per_file[local_key] = ref_id
    return lookup


def _merge_period_value(
    period_values: dict[str, list[dict]],
    record: dict,
) -> None:
    """Append a record to its period bucket.

    Params: period_values (dict), record (dict). Returns: None.
    """
    period_key = record["period_canonical"] or "(unspecified)"
    period_values.setdefault(period_key, []).append(record)


def _summarize_period_bucket(records: list[dict]) -> dict:
    """Collapse all records sharing one period into a display row.

    When multiple records report the same value, sources are merged
    with a semicolon. When values disagree, the disagreement is
    surfaced verbatim so the LLM can split per consolidation rule 3.

    Params: records (list[dict]). Returns: dict.
    """
    by_value: "OrderedDict[str, list[dict]]" = OrderedDict()
    for record in records:
        value = record["metric_value"]
        by_value.setdefault(value, []).append(record)
    return {
        "values_agree": len(by_value) == 1,
        "by_value": by_value,
    }


def _format_source_refs(records: list[dict]) -> str:
    """Render a deduplicated 'source [REF:N]; source [REF:M]' string.

    Params: records (list[dict]). Returns: str.
    """
    seen: set[tuple[str, int]] = set()
    parts: list[str] = []
    for record in records:
        key = (record["source"], record["ref_id"])
        if key in seen:
            continue
        seen.add(key)
        if record["ref_id"]:
            parts.append(f"{record['source']} [REF:{record['ref_id']}]")
        else:
            parts.append(record["source"])
    return "; ".join(parts)


def _format_period_line(
    period_label: str,
    bucket: dict,
) -> str:
    """Render one 'Period: value (sources)' line for a metric.

    Params: period_label (str), bucket (dict from _summarize_period_bucket).
    Returns: str.
    """
    if bucket["values_agree"]:
        value, records = next(iter(bucket["by_value"].items()))
        sources = _format_source_refs(records)
        return f"    {period_label}: {value}  (sources: {sources})"
    lines = [f"    {period_label}: VALUES DISAGREE"]
    for value, records in bucket["by_value"].items():
        sources = _format_source_refs(records)
        lines.append(f"      - {value}  (sources: {sources})")
    return "\n".join(lines)


def _format_metric_block(
    display_label: str,
    entity: str,
    segment: str,
    unit: str,
    period_values: dict[str, list[dict]],
) -> str:
    """Render one [METRIC] block with its pivoted period values.

    Params: display_label, entity, segment, unit, period_values.
    Returns: str.
    """
    lines = [
        f"[METRIC] {display_label}",
        f"  Entity: {entity} | Segment: {segment or '(unspecified)'}"
        f" | Unit: {unit or '(unspecified)'}",
        "  Periods:",
    ]
    for period_label in sorted(period_values.keys()):
        bucket = _summarize_period_bucket(period_values[period_label])
        lines.append(_format_period_line(period_label, bucket))
    return "\n".join(lines)


def _aggregate_groups(
    combo_results: list[ComboSourceResult],
) -> list[dict]:
    """Build the ordered list of aggregated metric groups.

    Params: combo_results. Returns: list of group dicts ready
    for rendering.
    """
    ref_lookup = _build_ref_lookup_by_filename(combo_results)
    groups: "OrderedDict[tuple, dict]" = OrderedDict()
    for key, record in _iter_quantitative_records(combo_results, ref_lookup):
        group = groups.get(key)
        if group is None:
            entity, segment, _canonical_name, canonical_unit = key
            group = {
                "entity": entity,
                "segment": segment,
                "unit": canonical_unit,
                "labels": [],
                "period_values": {},
            }
            groups[key] = group
        group["labels"].append(record["metric_name"])
        _merge_period_value(group["period_values"], record)
    return list(groups.values())


def build_metrics_inventory(
    combo_results: list[ComboSourceResult],
) -> str:
    """Build the aggregated metrics inventory text block.

    Walks all quantitative findings, groups them by (entity,
    segment, canonical metric name, unit), pivots periods into
    a per-group dict, and renders a text block the consolidation
    LLM can render directly into the Metrics table.

    Returns an empty string when no quantitative findings exist
    so the consolidation prompt can omit the section cleanly.

    Params:
        combo_results: Research outputs with structured findings

    Returns:
        str — formatted inventory text, or empty string

    Example:
        >>> text = build_metrics_inventory(results)
        >>> "[METRIC]" in text
        True
    """
    groups = _aggregate_groups(combo_results)
    if not groups:
        return ""
    lines = [
        "=== Aggregated Metrics Inventory ===",
        "Each [METRIC] block below is one pre-built row for the",
        "Metrics table. Period values are already pivoted into",
        "columns and sources with their refs are already merged.",
        "Render rows directly from these blocks — do NOT split a",
        "single block into multiple rows, and do NOT merge two",
        "blocks (their entity/segment/line item/unit differ).",
        "",
    ]
    for group in groups:
        display_label = _pick_display_label(group["labels"])
        lines.append(
            _format_metric_block(
                display_label=display_label,
                entity=group["entity"],
                segment=group["segment"],
                unit=group["unit"],
                period_values=group["period_values"],
            )
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
