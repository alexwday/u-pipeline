"""Stage 3: Lightweight LLM-based reranking of search results."""

from pathlib import Path
from time import perf_counter

from openai import BadRequestError

from ..models import SearchResult
from ..utils.config_setup import (
    get_rerank_min_keep,
    get_rerank_preview_max_tokens,
)
from ..utils.postgres_connector import get_section_info
from ..utils.llm_connector import (
    LLMClient,
    extract_tool_arguments,
    get_usage_metrics,
)
from ..utils.logging_setup import get_stage_logger
from ..utils.prompt_loader import load_prompt
from ..utils.trace_store import snapshot_search_result

STAGE = "3-RERANK"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_OUTPUT_LIMIT_ERROR_MARKERS = (
    "max_tokens or model output limit was reached",
    "model output limit was reached",
)


def _build_keep_all_metrics(
    results: list[SearchResult],
    attempt_count: int,
    llm_elapsed: float,
    used_preview_max: int,
    total_elapsed: float,
    fallback_reason: str,
) -> dict:
    """Build stage metrics for keep-all fallbacks."""
    return {
        "wall_time_seconds": round(total_elapsed, 3),
        "candidates_shown": len(results),
        "kept": len(results),
        "removed": 0,
        "llm_calls": attempt_count,
        "llm_call_seconds": round(llm_elapsed, 3),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "retry_count": max(0, attempt_count - 1),
        "preview_max_tokens": used_preview_max,
        "fallback_keep_all": True,
        "fallback_reason": fallback_reason,
    }


def _truncate_preview(text: str, max_tokens: int) -> str:
    """Truncate raw content to approximate token limit.

    Params: text (str), max_tokens (int). Returns: str.
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _load_section_summaries(
    conn,
    doc_version_id: int,
    results: list[SearchResult],
) -> dict[str, dict]:
    """Batch-load section info for all unique section_ids.

    Params: conn, doc_version_id, results. Returns: dict mapping
    section_id to section info dict.
    """
    if conn is None:
        return {}
    seen: set[str] = set()
    summaries: dict[str, dict] = {}
    for result in results:
        sid = result.get("section_id", "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        info = get_section_info(conn, doc_version_id, sid)
        if info:
            summaries[sid] = info
    return summaries


def _build_candidate_list(
    results: list[SearchResult],
    preview_max: int,
    section_info: dict[str, dict] | None = None,
) -> str:
    """Format search results grouped by section.

    Groups candidates by section_id. Each section group
    shows its title and summary once, with individual
    candidates nested below. Ungrouped candidates (no
    section_id) appear at the end.

    Params:
        results: Search results to format
        preview_max: Max tokens for content preview
        section_info: Optional section_id to info mapping

    Returns:
        str -- formatted grouped candidate list

    Example:
        >>> text = _build_candidate_list(results, 200)
    """
    if section_info is None:
        section_info = {}
    groups: dict[str, list[tuple[int, SearchResult]]] = {}
    for idx, result in enumerate(results):
        sid = result.get("section_id", "") or ""
        groups.setdefault(sid, []).append((idx, result))
    lines: list[str] = []
    for sid, group in groups.items():
        if sid and sid in section_info:
            info = section_info[sid]
            title = info.get("title", sid)
            summary = info.get("summary", "")
            lines.append(f"=== Section: {title} ===")
            if summary:
                lines.append(f"  Summary: {summary}")
        elif sid:
            lines.append(f"=== Section: {sid} ===")
        else:
            lines.append("=== Ungrouped ===")
        for idx, result in group:
            keywords = result.get("keywords", [])[:5]
            kw_text = ", ".join(keywords) if keywords else "none"
            preview = _truncate_preview(
                result.get("raw_content", ""),
                preview_max,
            )
            header = (
                f"  [{idx}] Page {result.get('page_number', 0)}"
                f" | Keywords: {kw_text}"
                f" | Score: {result.get('score', 0.0):.2f}"
            )
            lines.append(header)
            lines.append(f'    Preview: "{preview}"')
        lines.append("")
    return "\n".join(lines)


def _build_preview_attempts(preview_max: int) -> list[int]:
    """Build progressively smaller preview sizes for rerank retries."""
    attempts = [preview_max]
    attempts.append(max(60, preview_max // 2))
    attempts.append(max(40, preview_max // 3))
    deduped: list[int] = []
    for attempt in attempts:
        if attempt not in deduped:
            deduped.append(attempt)
    return deduped


def _is_output_limit_error(error: Exception) -> bool:
    """Check whether an exception indicates LLM output-token exhaustion."""
    text = str(error).casefold()
    return any(marker in text for marker in _OUTPUT_LIMIT_ERROR_MARKERS)


def _format_rerank_input(
    query: str,
    candidates_text: str,
    prompt: dict,
) -> str:
    """Assemble the user prompt with query and candidates.

    Params:
        query: User research query
        candidates_text: Formatted candidate list
        prompt: Loaded prompt dict

    Returns:
        str -- formatted user prompt text
    """
    text = prompt["user_prompt"]
    text = text.replace("{user_input}", query)
    text = text.replace("{candidates}", candidates_text)
    return text


def _build_rerank_messages(
    query: str,
    results: list[SearchResult],
    preview_max: int,
    prompt: dict,
    section_info: dict[str, dict] | None = None,
) -> list[dict]:
    """Build rerank messages for one preview size attempt."""
    candidates_text = _build_candidate_list(
        results, preview_max, section_info=section_info
    )
    user_text = _format_rerank_input(query, candidates_text, prompt)
    messages = []
    if prompt.get("system_prompt"):
        messages.append(
            {
                "role": "system",
                "content": prompt["system_prompt"],
            }
        )
    messages.append({"role": "user", "content": user_text})
    return messages


def _call_rerank_with_retry(
    results: list[SearchResult],
    query: str,
    llm: LLMClient,
    prompt: dict,
    logger,
    section_info: dict[str, dict] | None = None,
) -> tuple[dict | None, int, float, int, bool, list[dict]]:
    """Call rerank LLM with retries for output-limit failures."""
    preview_max = get_rerank_preview_max_tokens()
    attempts = _build_preview_attempts(preview_max)
    llm_elapsed = 0.0
    attempt_traces: list[dict] = []
    for attempt_index, current_preview_max in enumerate(attempts):
        messages = _build_rerank_messages(
            query,
            results,
            current_preview_max,
            prompt,
            section_info=section_info,
        )
        llm_start = perf_counter()
        try:
            response = llm.call(
                messages=messages,
                stage=prompt["stage"],
                tools=prompt.get("tools"),
                tool_choice=prompt.get("tool_choice"),
                context="rerank",
            )
            attempt_elapsed = perf_counter() - llm_start
            llm_elapsed += attempt_elapsed
            attempt_traces.append(
                {
                    "attempt": attempt_index + 1,
                    "preview_max_tokens": current_preview_max,
                    "llm_call_seconds": round(attempt_elapsed, 3),
                    "status": "success",
                }
            )
            return (
                response,
                attempt_index + 1,
                llm_elapsed,
                current_preview_max,
                False,
                attempt_traces,
            )
        except BadRequestError as error:
            attempt_elapsed = perf_counter() - llm_start
            llm_elapsed += attempt_elapsed
            if not _is_output_limit_error(error):
                raise
            attempt_trace = {
                "attempt": attempt_index + 1,
                "preview_max_tokens": current_preview_max,
                "llm_call_seconds": round(attempt_elapsed, 3),
                "status": "output_limit",
                "error": str(error),
            }
            if attempt_index == len(attempts) - 1:
                attempt_trace["fallback_keep_all"] = True
                attempt_traces.append(attempt_trace)
                logger.warning(
                    "Rerank hit output limit after %d attempts; "
                    "keeping all %d candidates",
                    attempt_index + 1,
                    len(results),
                )
                return (
                    None,
                    attempt_index + 1,
                    llm_elapsed,
                    current_preview_max,
                    True,
                    attempt_traces,
                )
            next_preview_max = attempts[attempt_index + 1]
            attempt_trace["next_preview_max_tokens"] = next_preview_max
            attempt_traces.append(attempt_trace)
            logger.warning(
                "Rerank hit output limit at preview_max=%d; retrying "
                "with preview_max=%d",
                current_preview_max,
                next_preview_max,
            )
    raise RuntimeError("Rerank failed to produce a response")


def _parse_rerank_response(response: dict) -> list[int]:
    """Extract remove_indices from LLM tool call response.

    Params: response (dict). Returns: list[int].
    """
    parsed = extract_tool_arguments(response)
    if "remove_indices" not in parsed:
        raise ValueError("Tool response missing field: remove_indices")
    indices = parsed["remove_indices"]
    if not isinstance(indices, list):
        raise ValueError("remove_indices must be a list")
    return indices


def _build_candidate_trace(
    results: list[SearchResult],
    preview_max: int,
) -> list[dict]:
    """Build traced candidate rows with preview text."""
    candidate_trace: list[dict] = []
    for index, result in enumerate(results):
        entry = snapshot_search_result(result)
        entry["index"] = index
        entry["preview"] = _truncate_preview(
            result.get("raw_content", ""),
            preview_max,
        )
        candidate_trace.append(entry)
    return candidate_trace


def _update_rerank_trace(
    trace: dict | None,
    payload: dict,
) -> None:
    """Populate the structured rerank trace."""
    if trace is None:
        return
    trace.update(payload)
    trace["fallback_keep_all"] = bool(payload.get("fallback_reason"))


def _handle_keep_all_fallback(details: dict) -> list[SearchResult]:
    """Apply keep-all fallback metrics, trace, and logging."""
    results = details["results"]
    total_elapsed = perf_counter() - details["start_time"]
    stage_metrics = _build_keep_all_metrics(
        results,
        details["attempt_count"],
        details["llm_elapsed"],
        details["used_preview_max"],
        total_elapsed,
        details["fallback_reason"],
    )
    if details["metrics"] is not None:
        details["metrics"].update(stage_metrics)
    _update_rerank_trace(
        details["trace"],
        {
            "query": details["query"],
            "candidates": details["candidate_trace"],
            "attempts": details["attempt_traces"],
            "remove_indices": [],
            "removed_ids": [],
            "kept_ids": [result["content_unit_id"] for result in results],
            "fallback_reason": details["fallback_reason"],
        },
    )
    details["logger"].warning(
        details["warning_message"],
        details["attempt_count"],
        len(results),
    )
    details["logger"].info(
        "[%s] completed in %.1fs — shown=%d, kept=%d, "
        "removed=0, llm=%.1fs, prompt_tokens=0, "
        "retries=%d, preview_max=%d, fallback=keep_all",
        STAGE,
        total_elapsed,
        len(results),
        len(results),
        details["llm_elapsed"],
        max(0, details["attempt_count"] - 1),
        details["used_preview_max"],
    )
    return list(results)


def _apply_min_keep_floor(
    results: list[SearchResult],
    remove_set: set[int],
    logger,
) -> set[int]:
    """Restore best-scoring removals to satisfy the minimum keep floor."""
    min_keep = get_rerank_min_keep()
    would_keep = len(results) - len(remove_set)
    if not would_keep < min_keep <= len(results):
        return remove_set
    scored_removals = sorted(
        remove_set,
        key=lambda index: results[index].get("score", 0.0),
    )
    restore_count = min_keep - would_keep
    restored = set(scored_removals[-restore_count:])
    logger.info(
        "Rerank min-keep floor: restored %d of %d removals "
        "(floor=%d, would_keep=%d)",
        len(restored),
        len(remove_set),
        min_keep,
        would_keep,
    )
    return remove_set - restored


def rerank_results(
    results: list[SearchResult],
    query: str,
    llm: LLMClient,
    conn=None,
    doc_version_id: int = 0,
    metrics: dict | None = None,
    trace: dict | None = None,
) -> list[SearchResult]:
    """Filter search results by LLM relevance judgment.

    Groups candidates by section with summaries, then asks
    the LLM to identify clearly irrelevant entries. This is
    conservative — only removes chunks the LLM flags as
    unquestionably unrelated.

    Params:
        results: Search results to rerank
        query: Original user query text
        llm: Configured LLM client
        conn: Optional psycopg2 connection for section summaries
        doc_version_id: Document version for section lookups

    Returns:
        list[SearchResult] with irrelevant entries removed,
        original order preserved

    Example:
        >>> kept = rerank_results(results, "CET1 ratio", llm)
    """
    logger = get_stage_logger(__name__, STAGE)
    start_time = perf_counter()

    if not results:
        logger.info("No results to rerank")
        if metrics is not None:
            metrics.update(
                {
                    "wall_time_seconds": 0.0,
                    "candidates_shown": 0,
                    "kept": 0,
                    "removed": 0,
                    "llm_calls": 0,
                    "llm_call_seconds": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            )
        return []

    prompt = load_prompt("rerank", prompts_dir=_PROMPTS_DIR)
    section_info = _load_section_summaries(conn, doc_version_id, results)
    call_result = _call_rerank_with_retry(
        results, query, llm, prompt, logger, section_info=section_info
    )
    candidate_trace = _build_candidate_trace(results, call_result[3])
    if call_result[4]:
        return _handle_keep_all_fallback(
            {
                "results": results,
                "query": query,
                "metrics": metrics,
                "trace": trace,
                "logger": logger,
                "start_time": start_time,
                "attempt_count": call_result[1],
                "llm_elapsed": call_result[2],
                "used_preview_max": call_result[3],
                "candidate_trace": candidate_trace,
                "attempt_traces": call_result[5],
                "fallback_reason": "output_limit",
                "warning_message": (
                    "Rerank hit output limit after %d attempts; "
                    "keeping all %d candidates"
                ),
            }
        )

    try:
        remove_indices = _parse_rerank_response(call_result[0])
    except ValueError as error:
        if "tool call" not in str(error).casefold():
            raise
        return _handle_keep_all_fallback(
            {
                "results": results,
                "query": query,
                "metrics": metrics,
                "trace": trace,
                "logger": logger,
                "start_time": start_time,
                "attempt_count": call_result[1],
                "llm_elapsed": call_result[2],
                "used_preview_max": call_result[3],
                "candidate_trace": candidate_trace,
                "attempt_traces": call_result[5],
                "fallback_reason": "missing_tool_call",
                "warning_message": (
                    "Rerank returned no tool call after %d attempt(s); "
                    "keeping all %d candidates"
                ),
            }
        )
    remove_set = _apply_min_keep_floor(results, set(remove_indices), logger)
    kept = [
        result for idx, result in enumerate(results) if idx not in remove_set
    ]
    usage = get_usage_metrics(call_result[0])
    total_elapsed = perf_counter() - start_time
    stage_metrics = {
        "wall_time_seconds": round(total_elapsed, 3),
        "candidates_shown": len(results),
        "kept": len(kept),
        "removed": len(remove_set),
        "llm_calls": call_result[1],
        "llm_call_seconds": round(call_result[2], 3),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "retry_count": max(0, call_result[1] - 1),
        "preview_max_tokens": call_result[3],
        "fallback_keep_all": False,
    }
    if metrics is not None:
        metrics.update(stage_metrics)
    _update_rerank_trace(
        trace,
        {
            "query": query,
            "candidates": candidate_trace,
            "attempts": call_result[5],
            "remove_indices": sorted(remove_set),
            "removed_ids": [
                result["content_unit_id"]
                for idx, result in enumerate(results)
                if idx in remove_set
            ],
            "kept_ids": [result["content_unit_id"] for result in kept],
        },
    )

    logger.info(
        "[%s] completed in %.1fs — shown=%d, kept=%d, removed=%d, "
        "llm=%.1fs, prompt_tokens=%d, retries=%d, preview_max=%d",
        STAGE,
        total_elapsed,
        len(results),
        len(kept),
        len(remove_set),
        call_result[2],
        usage["prompt_tokens"],
        max(0, call_result[1] - 1),
        call_result[3],
    )

    return kept
