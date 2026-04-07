"""Stage 6: Consolidation of research across sources.

Builds a [REF:N] reference index from structured findings,
streams the LLM response with programmatic reference replacement,
and parses the output into Summary/Metrics/Detail/Gaps sections.
"""

import re
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from ..models import ComboSourceResult, ConsolidatedResult, ResearchFinding
from ..utils.llm_connector import (
    LLMClient,
    extract_content_text,
    get_usage_metrics,
)
from ..utils.logging_setup import get_stage_logger
from ..utils.metrics_aggregator import build_metrics_inventory
from ..utils.prompt_loader import load_prompt

STAGE = "6-CONSOLIDATION"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _format_finding_summary(finding: ResearchFinding) -> str:
    """Format a single finding for the prompt's reference index.

    Params: finding (ResearchFinding). Returns: str.
    """
    parts: list[str] = []
    if finding.get("metric_name"):
        metric = finding["metric_name"]
        if finding.get("metric_value"):
            metric += f" = {finding['metric_value']}"
            if finding.get("unit"):
                metric += f" {finding['unit']}"
        parts.append(metric)
    if finding.get("period"):
        parts.append(f"({finding['period']})")
    if finding.get("segment"):
        parts.append(f"[{finding['segment']}]")
    prefix = " ".join(parts)
    text = finding["finding"]
    if prefix:
        return f"  - {prefix} — {text}"
    return f"  - {text}"


def _build_reference_index(
    combo_results: list[ComboSourceResult],
) -> tuple[list[dict], str]:
    """Build a deduplicated reference index from structured findings.

    Findings sharing the same (filename, page, location_detail)
    collapse to a single [REF:N] entry. Each entry groups all
    findings at that evidence location so the LLM still has full
    context to write the response.

    Params:
        combo_results: Research outputs with findings

    Returns:
        tuple of (reference entries, formatted index text)

    Example:
        >>> entries, text = _build_reference_index(results)
        >>> entries[0]["ref_id"]
        1
    """
    entries: list[dict] = []
    location_to_entry: dict[tuple, dict] = {}
    ref_counter = 0

    for result in combo_results:
        source_label = result["source"]["data_source"]
        filename = result["source"]["filename"]
        bank = result["combo"]["bank"]
        period = result["combo"]["period"]
        for finding in result.get("findings", []):
            page = finding["page"]
            location = finding.get("location_detail", "")
            key = (filename, page, location)
            entry = location_to_entry.get(key)
            if entry is None:
                ref_counter += 1
                entry = {
                    "ref_id": ref_counter,
                    "source": source_label,
                    "filename": filename,
                    "page": page,
                    "location_detail": location,
                    "entity": bank,
                    "period": period,
                    "findings": [],
                }
                entries.append(entry)
                location_to_entry[key] = entry
            entry["findings"].append(finding)

    lines: list[str] = []
    by_filename: dict[str, list[dict]] = {}
    for entry in entries:
        by_filename.setdefault(entry["filename"], []).append(entry)
    for filename, file_entries in by_filename.items():
        first = file_entries[0]
        lines.append(f"=== Source: {first['source']} / {filename} ===")
        lines.append(
            f"    Entity: {first['entity']}"
            f" | Document Period: {first['period']}"
        )
        for entry in file_entries:
            location = entry["location_detail"]
            location_str = f" - {location}" if location else ""
            lines.append(
                f"[REF:{entry['ref_id']}] Page {entry['page']}"
                f"{location_str}"
            )
            for finding in entry["findings"]:
                lines.append(_format_finding_summary(finding))
        lines.append("")

    return entries, "\n".join(lines)


_REF_PATTERN = re.compile(r"\[REF:(\d+)\]")
_PARTIAL_REF_TAIL = re.compile(r"\[(?:R(?:E(?:F(?::(?:\d+)?)?)?)?)?$")


def _create_ref_replacer(
    reference_entries: list[dict],
) -> dict:
    """Build a stateful reference replacer for streaming chunks.

    Returns a state dict with a replace() callable and tracking data.

    Params:
        reference_entries: Entries from _build_reference_index()

    Returns:
        dict with keys: replace (callable), flush (callable),
        used_refs (set), invented_refs (set)
    """
    lookup: dict[int, dict] = {
        entry["ref_id"]: entry for entry in reference_entries
    }
    state: dict = {
        "buffer": "",
        "used_refs": set(),
        "invented_refs": set(),
    }

    def _expand_ref(match: re.Match) -> str:
        ref_id = int(match.group(1))
        entry = lookup.get(ref_id)
        if entry is None:
            state["invented_refs"].add(ref_id)
            return match.group(0)
        state["used_refs"].add(ref_id)
        filename = entry.get("filename", "")
        page = entry.get("page", 0)
        href = f"https://docs.local/files/{filename}#page={page}"
        return f'<a href="{href}">[{ref_id}]</a>'

    def replace(chunk: str) -> str:
        """Replace [REF:N] in chunk, buffering partials."""
        text = state["buffer"] + chunk
        state["buffer"] = ""
        partial = _PARTIAL_REF_TAIL.search(text)
        if partial:
            state["buffer"] = text[partial.start() :]
            text = text[: partial.start()]
        return _REF_PATTERN.sub(_expand_ref, text)

    def flush() -> str:
        """Return any buffered trailing text."""
        remaining = state["buffer"]
        state["buffer"] = ""
        return _REF_PATTERN.sub(_expand_ref, remaining)

    return {
        "replace": replace,
        "flush": flush,
        "used_refs": state["used_refs"],
        "invented_refs": state["invented_refs"],
    }


_ADJACENT_DUP_RE = re.compile(r'(<a href="[^"]+">\[(\d+)\]</a>)(?:\1)+')


def _dedup_adjacent_refs(text):
    """Collapse adjacent duplicate citation anchors.

    Params: text (str). Returns: str.
    """
    return _ADJACENT_DUP_RE.sub(r"\1", text)


def _build_reference_appendix(entries, used_refs):
    """Build a markdown reference index from used entries.

    Params:
        entries: Full reference entries list
        used_refs: Set of ref_id ints that were actually cited

    Returns:
        str — formatted markdown section, or empty if no refs
    """
    if not used_refs:
        return ""
    lines = ["\n\n## References\n"]
    lines.append("| # | Source | File | Page | Location |")
    lines.append("|---|--------|------|------|----------|")
    for entry in entries:
        if entry["ref_id"] not in used_refs:
            continue
        ref_id = entry["ref_id"]
        source = entry["source"]
        filename = entry.get("filename", "")
        page = entry["page"]
        location = entry.get("location_detail", "")
        lines.append(
            f"| {ref_id} | {source} | {filename} | {page} | {location} |"
        )
    return "\n".join(lines)


def _extract_metric_value_tokens(metric_value: str) -> list[str]:
    """Extract searchable numeric tokens from a metric_value string.

    Strips thousands separators, then matches digit groups with
    optional % or bps unit. Used by the layer-2 coverage audit
    to check whether a finding's value appears in the rendered
    response.

    Params: metric_value (str). Returns: list of token strings.
    """
    if not metric_value:
        return []
    normalized = metric_value.replace(",", "")
    return _METRIC_TOKEN_RE.findall(normalized)


def _finding_token_in_text(finding: dict, normalized_text: str) -> bool:
    """Check whether any of a finding's metric tokens appear in text.

    Qualitative findings (empty metric_value) return True — layer-2
    cannot audit qualitative content; that responsibility rests with
    the prompt rule.

    Params: finding (dict), normalized_text (str — comma-stripped).
    Returns: bool.
    """
    metric_value = finding.get("metric_value", "") or ""
    tokens = _extract_metric_value_tokens(metric_value)
    if not tokens:
        return True
    return any(tok in normalized_text for tok in tokens)


def _compute_coverage_audit(
    ref_entries: list[dict],
    used_refs: set[int],
    response_text: str,
) -> dict:
    """Run the two-layer coverage audit on a consolidation response.

    Layer 1 detects refs that were never cited in the response.
    Layer 2 detects findings inside cited refs whose numeric tokens
    do not appear anywhere in the rendered response text.

    Params:
        ref_entries: Reference index entries from _build_reference_index
        used_refs: Set of ref ids the LLM cited
        response_text: Rendered response text (after ref replacement
            and adjacent-ref dedup; before audit/appendix appending)

    Returns:
        dict with keys uncited_ref_ids (list[int]),
        unincorporated_findings (list[dict]).

    Example:
        >>> audit = _compute_coverage_audit(entries, {1}, "text [1]")
        >>> audit["uncited_ref_ids"]
        [2]
    """
    uncited: list[int] = []
    unincorporated: list[dict] = []
    normalized_text = response_text.replace(",", "")
    for entry in ref_entries:
        ref_id = entry["ref_id"]
        if ref_id not in used_refs:
            uncited.append(ref_id)
            continue
        for finding in entry.get("findings", []):
            if _finding_token_in_text(finding, normalized_text):
                continue
            unincorporated.append(
                {
                    "ref_id": ref_id,
                    "source": entry.get("source", ""),
                    "page": entry.get("page", 0),
                    "location_detail": entry.get("location_detail", ""),
                    "metric_name": finding.get("metric_name", "") or "",
                    "metric_value": finding.get("metric_value", "") or "",
                    "finding": finding.get("finding", "") or "",
                }
            )
    uncited.sort()
    return {
        "uncited_ref_ids": uncited,
        "unincorporated_findings": unincorporated,
    }


def _build_coverage_audit_section(
    audit: dict,
    ref_entries: list[dict],
) -> str:
    """Render coverage audit results as a markdown section.

    Returns an empty string when there are no audit items so the
    parent does not need to handle the no-op case.

    Params: audit (dict from _compute_coverage_audit),
        ref_entries (list[dict]).
    Returns: str.
    """
    uncited = audit.get("uncited_ref_ids", [])
    unincorporated = audit.get("unincorporated_findings", [])
    if not uncited and not unincorporated:
        return ""
    entry_by_id = {e["ref_id"]: e for e in ref_entries}
    lines = ["\n\n## Coverage audit\n"]
    if uncited:
        lines.append("### Uncited refs (never appeared in response)")
        for ref_id in uncited:
            entry = entry_by_id.get(ref_id, {})
            source = entry.get("source", "")
            page = entry.get("page", 0)
            location = entry.get("location_detail", "")
            loc_str = f" — {location}" if location else ""
            lines.append(f"- [REF:{ref_id}] {source} p{page}{loc_str}")
            for finding in entry.get("findings", []):
                summary = _format_finding_summary(finding).lstrip(" -")
                lines.append(f"    - {summary}")
        lines.append("")
    if unincorporated:
        lines.append(
            "### Unincorporated findings (ref cited but value absent)"
        )
        for item in unincorporated:
            ref_id = item["ref_id"]
            source = item["source"]
            page = item["page"]
            location = item["location_detail"]
            loc_str = f" — {location}" if location else ""
            metric_name = item["metric_name"]
            metric_value = item["metric_value"]
            label = (
                f"{metric_name} = {metric_value}"
                if metric_name
                else (item["finding"][:120])
            )
            lines.append(
                f"- [REF:{ref_id}] {source} p{page}{loc_str} — {label}"
            )
        lines.append("")
    return "\n".join(lines)


def _process_consolidation_response(
    full_text: str,
    ref_entries: list[dict],
    used_refs: set[int],
    on_chunk: Callable[[str], None] | None,
) -> dict:
    """Apply post-LLM-call processing: dedup, audit, appendix, parse.

    Factored out of consolidate_results to keep the parent function
    below pylint's R0914 too-many-locals threshold while adding the
    coverage audit step.

    Params:
        full_text: Raw LLM response with [REF:N] already replaced
        ref_entries: Reference index entries
        used_refs: Set of ref ids the LLM cited
        on_chunk: Optional streaming callback for appended sections

    Returns:
        dict with full_text, sections, gap_list, key_findings,
        uncited_ref_ids, unincorporated_findings keys.
    """
    full_text = _dedup_adjacent_refs(full_text)
    audit = _compute_coverage_audit(ref_entries, used_refs, full_text)
    audit_section = _build_coverage_audit_section(audit, ref_entries)
    if audit_section:
        full_text += audit_section
        if on_chunk is not None:
            on_chunk(audit_section)
    appendix = _build_reference_appendix(ref_entries, used_refs)
    if appendix:
        full_text += appendix
        if on_chunk is not None:
            on_chunk(appendix)
    sections = _parse_sections(full_text)
    return {
        "full_text": full_text,
        "sections": sections,
        "gap_list": _extract_gap_list(sections["gaps"]),
        "key_findings": _extract_cited_sentences(sections["summary"]),
        "uncited_ref_ids": audit["uncited_ref_ids"],
        "unincorporated_findings": audit["unincorporated_findings"],
    }


_SECTION_HEADING_RE = re.compile(r"^##\s+(\S+)[^\n]*\n", re.MULTILINE)
_SECTION_KEYS = {
    "Summary": "summary",
    "Metrics": "metrics",
    "Detail": "detail",
    "Gaps": "gaps",
    "Coverage": "coverage_audit",
}
_NONE_IDENTIFIED_RE = re.compile(
    r"^\s*none\s+identified\.?\s*$", re.IGNORECASE
)
_MAX_KEY_FINDINGS = 5
_METRIC_TOKEN_RE = re.compile(
    r"\d+(?:\.\d+)?(?:%|\s*bps?)?",
    re.IGNORECASE,
)


def _parse_sections(text: str) -> dict[str, str]:
    """Parse response text into sections by ## Heading markers.

    Params: text (str). Returns: dict with summary, metrics,
    detail, gaps, coverage_audit keys.
    """
    result: dict[str, str] = {
        "summary": "",
        "metrics": "",
        "detail": "",
        "gaps": "",
        "coverage_audit": "",
    }
    matches = list(_SECTION_HEADING_RE.finditer(text))
    if not matches:
        return result
    for idx, match in enumerate(matches):
        heading = match.group(1)
        key = _SECTION_KEYS.get(heading)
        if key is None:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else None
        result[key] = text[start:end].strip()
    return result


def _extract_gap_list(gaps_text: str) -> list[str]:
    """Split the Gaps section into individual items.

    Params: gaps_text (str). Returns: list[str].
    """
    if not gaps_text or _NONE_IDENTIFIED_RE.match(gaps_text):
        return []
    items: list[str] = []
    for line in gaps_text.splitlines():
        cleaned = line.strip().lstrip("-*").strip()
        if cleaned:
            items.append(cleaned)
    return items


def _extract_cited_sentences(summary: str) -> list[str]:
    """Extract up to five cited sentences from the summary.

    Params: summary (str). Returns: list[str].
    """
    sentences: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+", summary):
        if "[" in part:
            sentences.append(part.strip())
        if len(sentences) >= _MAX_KEY_FINDINGS:
            break
    return sentences


def _build_messages(
    prompt: dict,
    query: str,
    ref_text: str,
    inventory_text: str,
) -> list[dict]:
    """Build system + user messages for the consolidation call.

    Params: prompt (dict), query (str), ref_text (str),
    inventory_text (str). Returns: list[dict].
    """
    messages: list[dict] = []
    if prompt.get("system_prompt"):
        messages.append({"role": "system", "content": prompt["system_prompt"]})
    user_text = prompt["user_prompt"]
    user_text = user_text.replace("{query}", query)
    user_text = user_text.replace("{reference_index}", ref_text)
    user_text = user_text.replace("{aggregated_metrics}", inventory_text)
    messages.append({"role": "user", "content": user_text})
    return messages


def _stream_and_replace(
    llm: LLMClient,
    messages: list[dict],
    prompt: dict,
    replacer: dict,
    on_chunk: Callable[[str], None] | None,
) -> tuple[str, dict[str, int]]:
    """Stream the LLM response, replacing refs and accumulating text.

    Params:
        llm: LLM client with stream() method
        messages: Chat messages
        prompt: Loaded prompt dict with stage key
        replacer: Ref replacer from _create_ref_replacer
        on_chunk: Optional callback for each processed chunk

    Returns:
        tuple of (accumulated text, usage metrics dict)
    """
    accumulated: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    gen = llm.stream(
        messages=messages, stage=prompt["stage"], context="consolidation"
    )
    try:
        while True:
            chunk_text = next(gen)
            processed = replacer["replace"](chunk_text)
            if processed and on_chunk is not None:
                on_chunk(processed)
            accumulated.append(processed)
    except StopIteration as stop:
        usage = stop.value or usage
    tail = replacer["flush"]()
    if tail:
        if on_chunk is not None:
            on_chunk(tail)
        accumulated.append(tail)
    return "".join(accumulated), usage


def _fallback_call(
    llm: LLMClient,
    messages: list[dict],
    prompt: dict,
    replacer: dict,
) -> tuple[str, dict[str, int]]:
    """Non-streaming fallback for LLM clients without stream().

    Params:
        llm: LLM client
        messages: Chat messages
        prompt: Loaded prompt dict with stage key
        replacer: Ref replacer from _create_ref_replacer

    Returns:
        tuple of (processed text, usage metrics dict)
    """
    response = llm.call(
        messages=messages,
        stage=prompt["stage"],
        context="consolidation",
    )
    raw_text = extract_content_text(response)
    processed = replacer["replace"](raw_text) + replacer["flush"]()
    usage = get_usage_metrics(response)
    return processed, usage


def _build_empty_result(
    query: str,
    combo_results: list[ComboSourceResult],
    gap_message: str,
    metrics: dict | None,
) -> ConsolidatedResult:
    """Build an empty ConsolidatedResult with a data gap.

    Params: query (str), combo_results (list), gap_message (str),
    metrics (dict | None). Returns: ConsolidatedResult.
    """
    if metrics is not None:
        metrics.update(
            {
                "wall_time_seconds": 0.0,
                "source_results": len(combo_results),
                "llm_calls": 0,
                "llm_call_seconds": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "key_findings": 0,
                "data_gaps": 1,
            }
        )
    return ConsolidatedResult(
        query=query,
        combo_results=combo_results,
        consolidated_response="",
        key_findings=[],
        data_gaps=[gap_message],
        summary_answer="",
        metrics_table="",
        detailed_summary="",
        reference_index=[],
    )


def consolidate_results(
    query: str,
    combo_results: list[ComboSourceResult],
    llm: LLMClient,
    metrics: dict | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> ConsolidatedResult:
    """Synthesize research from multiple sources via streaming.

    Builds a reference index from structured findings, streams the
    LLM response with real-time [REF:N] replacement, and parses
    the result into structured sections.

    Params:
        query: Original user research query
        combo_results: Research outputs from all sources
        llm: Configured LLM client
        metrics: Optional dict to receive stage metrics
        on_chunk: Optional callback receiving each processed chunk

    Returns:
        ConsolidatedResult with synthesized response, key findings,
        data gaps, and structured sections

    Example:
        >>> result = consolidate_results(
        ...     "CET1 ratio", combo_results, llm
        ... )
        >>> result["summary_answer"]
        "CET1 ratio was 13.7% ..."
    """
    logger = get_stage_logger(__name__, STAGE)
    start_time = perf_counter()
    logger.info("Consolidating %d source results", len(combo_results))

    if not combo_results:
        logger.info("No results to consolidate")
        return _build_empty_result(
            query, [], "No sources returned results.", metrics
        )

    ref_entries, ref_text = _build_reference_index(combo_results)

    if not ref_entries:
        logger.info("No findings in any source")
        return _build_empty_result(
            query,
            combo_results,
            "No findings extracted from sources.",
            metrics,
        )

    inventory_text = build_metrics_inventory(combo_results)
    prompt = load_prompt("consolidation", prompts_dir=_PROMPTS_DIR)
    messages = _build_messages(prompt, query, ref_text, inventory_text)
    replacer = _create_ref_replacer(ref_entries)

    llm_start = perf_counter()
    if hasattr(llm, "stream"):
        full_text, usage = _stream_and_replace(
            llm, messages, prompt, replacer, on_chunk
        )
    else:
        full_text, usage = _fallback_call(llm, messages, prompt, replacer)
    llm_elapsed = perf_counter() - llm_start

    processed = _process_consolidation_response(
        full_text, ref_entries, replacer["used_refs"], on_chunk
    )
    total_elapsed = perf_counter() - start_time

    _record_metrics(
        metrics,
        total_elapsed,
        llm_elapsed,
        combo_results,
        usage,
        processed["key_findings"],
        processed["gap_list"],
    )
    logger.info(
        "[%s] completed in %.1fs -- sources=%d, llm=%.1fs, "
        "prompt_tokens=%d, key_findings=%d, data_gaps=%d, "
        "uncited_refs=%d, unincorporated=%d",
        STAGE,
        total_elapsed,
        len(combo_results),
        llm_elapsed,
        usage["prompt_tokens"],
        len(processed["key_findings"]),
        len(processed["gap_list"]),
        len(processed["uncited_ref_ids"]),
        len(processed["unincorporated_findings"]),
    )

    return ConsolidatedResult(
        query=query,
        combo_results=combo_results,
        consolidated_response=processed["full_text"],
        key_findings=processed["key_findings"],
        data_gaps=processed["gap_list"],
        summary_answer=processed["sections"]["summary"],
        metrics_table=processed["sections"]["metrics"],
        detailed_summary=processed["sections"]["detail"],
        reference_index=ref_entries,
        coverage_audit=processed["sections"]["coverage_audit"],
        uncited_ref_ids=processed["uncited_ref_ids"],
        unincorporated_findings=processed["unincorporated_findings"],
    )


def _record_metrics(
    metrics: dict | None,
    total_elapsed: float,
    llm_elapsed: float,
    combo_results: list[ComboSourceResult],
    usage: dict[str, int],
    key_findings: list[str],
    gap_list: list[str],
) -> None:
    """Write stage metrics into the caller-provided dict.

    Params: metrics (dict | None), total_elapsed (float),
    llm_elapsed (float), combo_results (list), usage (dict),
    key_findings (list), gap_list (list). Returns: None.
    """
    if metrics is None:
        return
    metrics.update(
        {
            "wall_time_seconds": round(total_elapsed, 3),
            "source_results": len(combo_results),
            "llm_calls": 1,
            "llm_call_seconds": round(llm_elapsed, 3),
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "key_findings": len(key_findings),
            "data_gaps": len(gap_list),
        }
    )
