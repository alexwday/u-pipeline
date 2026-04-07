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
from ..utils.prompt_loader import load_prompt

STAGE = "6-CONSOLIDATION"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _build_reference_index(
    combo_results: list[ComboSourceResult],
) -> tuple[list[dict], str]:
    """Build a numbered reference index from structured findings.

    Assigns each finding a globally unique [REF:N] identifier,
    grouped by source. Returns the entries list and a formatted
    text block for the LLM prompt.

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
    lines: list[str] = []
    ref_counter = 0
    for result in combo_results:
        source_label = result["source"]["data_source"]
        filename = result["source"]["filename"]
        bank = result["combo"]["bank"]
        period = result["combo"]["period"]
        findings: list[ResearchFinding] = result.get("findings", [])
        if not findings:
            continue
        lines.append(f"=== Source: {source_label} / {filename} ===")
        lines.append(f"    Entity: {bank} | Document Period: {period}")
        for finding in findings:
            ref_counter += 1
            entry: dict = {
                "ref_id": ref_counter,
                "finding": finding["finding"],
                "page": finding["page"],
                "location_detail": finding["location_detail"],
                "source": source_label,
                "entity": bank,
            }
            lines.append(f"[REF:{ref_counter}]")
            lines.append(f"  Finding: {finding['finding']}")
            lines.append(
                f"  Page: {finding['page']}"
                f" | Location: {finding['location_detail']}"
            )
            for key in ("metric_name", "metric_value", "period", "segment"):
                value = finding.get(key, "")
                if value:
                    entry[key] = value
            if finding.get("metric_name"):
                lines.append(f"  Metric: {finding['metric_name']}")
                metric_parts: list[str] = []
                if finding.get("metric_value"):
                    metric_parts.append(f"Value: {finding['metric_value']}")
                if finding.get("period"):
                    metric_parts.append(f"Period: {finding['period']}")
                if finding.get("segment"):
                    metric_parts.append(f"Segment: {finding['segment']}")
                if metric_parts:
                    lines.append(f"  {' | '.join(metric_parts)}")
            entries.append(entry)
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
        source = entry["source"]
        page = entry["page"]
        location = entry.get("location_detail", "")
        if location:
            return f"[{source}, Page {page} - {location}]"
        return f"[{source}, Page {page}]"

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


_SECTION_HEADING_RE = re.compile(r"^##\s+(\S+)", re.MULTILINE)
_SECTION_KEYS = {
    "Summary": "summary",
    "Metrics": "metrics",
    "Detail": "detail",
    "Gaps": "gaps",
}
_NONE_IDENTIFIED_RE = re.compile(
    r"^\s*none\s+identified\.?\s*$", re.IGNORECASE
)
_MAX_KEY_FINDINGS = 5


def _parse_sections(text: str) -> dict[str, str]:
    """Parse response text into sections by ## Heading markers.

    Params: text (str). Returns: dict with summary, metrics,
    detail, gaps keys.
    """
    result: dict[str, str] = {
        "summary": "",
        "metrics": "",
        "detail": "",
        "gaps": "",
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


def _build_messages(prompt: dict, query: str, ref_text: str) -> list[dict]:
    """Build system + user messages for the consolidation call.

    Params: prompt (dict), query (str), ref_text (str).
    Returns: list[dict].
    """
    messages: list[dict] = []
    if prompt.get("system_prompt"):
        messages.append({"role": "system", "content": prompt["system_prompt"]})
    user_text = prompt["user_prompt"]
    user_text = user_text.replace("{query}", query)
    user_text = user_text.replace("{reference_index}", ref_text)
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

    prompt = load_prompt("consolidation", prompts_dir=_PROMPTS_DIR)
    messages = _build_messages(prompt, query, ref_text)
    replacer = _create_ref_replacer(ref_entries)

    llm_start = perf_counter()
    if hasattr(llm, "stream"):
        full_text, usage = _stream_and_replace(
            llm, messages, prompt, replacer, on_chunk
        )
    else:
        full_text, usage = _fallback_call(llm, messages, prompt, replacer)
    llm_elapsed = perf_counter() - llm_start

    sections = _parse_sections(full_text)
    gap_list = _extract_gap_list(sections["gaps"])
    key_findings = _extract_cited_sentences(sections["summary"])
    total_elapsed = perf_counter() - start_time

    _record_metrics(
        metrics,
        total_elapsed,
        llm_elapsed,
        combo_results,
        usage,
        key_findings,
        gap_list,
    )
    logger.info(
        "[%s] completed in %.1fs -- sources=%d, llm=%.1fs, "
        "prompt_tokens=%d, key_findings=%d, data_gaps=%d",
        STAGE,
        total_elapsed,
        len(combo_results),
        llm_elapsed,
        usage["prompt_tokens"],
        len(key_findings),
        len(gap_list),
    )

    return ConsolidatedResult(
        query=query,
        combo_results=combo_results,
        consolidated_response=full_text,
        key_findings=key_findings,
        data_gaps=gap_list,
        summary_answer=sections["summary"],
        metrics_table=sections["metrics"],
        detailed_summary=sections["detail"],
        reference_index=ref_entries,
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
