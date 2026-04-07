"""Stage 5: Iterative research agent loop."""

import re
from pathlib import Path
from time import perf_counter

from ..models import (
    ComboSourceResult,
    ComboSpec,
    ExpandedChunk,
    PreparedQuery,
    ResearchFinding,
    ResearchIteration,
    SourceSpec,
)
from ..utils.config_setup import (
    get_embedding_dimensions,
    get_embedding_model,
    get_research_additional_top_k,
    get_research_max_iterations,
)
from ..utils.llm_connector import (
    LLMClient,
    extract_tool_arguments,
    get_usage_metrics,
)
from ..utils.logging_setup import get_stage_logger
from ..utils.postgres_connector import (
    get_section_info,
    search_by_content_vector,
)
from ..utils.prompt_loader import load_prompt
from ..utils.trace_store import snapshot_content_row, snapshot_expanded_chunk

STAGE = "5-RESEARCH"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_MAX_ADDITIONAL_QUERIES = 3
_SHEET_NAME_RE = re.compile(r"#\s*Sheet:\s*(.+?)(?:\n|$)")


def _extract_sheet_name(chunk_header: str) -> str:
    """Parse sheet name from a chunk header prefix.

    Params: chunk_header (str). Returns: str or empty.
    """
    if not chunk_header:
        return ""
    match = _SHEET_NAME_RE.search(chunk_header)
    return match.group(1).strip() if match else ""


def _order_chunks_by_page(chunks: list[ExpandedChunk]) -> list[ExpandedChunk]:
    """Sort chunks by page number then content_unit_id."""
    return sorted(
        chunks,
        key=lambda c: (c["page_number"], c["content_unit_id"]),
    )


def _queries_are_exact_repeats(
    previous_queries: list[str],
    current_queries: list[str],
) -> bool:
    """Check whether follow-up queries are exact repeats of prior ones."""
    if not previous_queries or not current_queries:
        return False
    previous_set = {q.strip().casefold() for q in previous_queries}
    return all(q.strip().casefold() in previous_set for q in current_queries)


def _needs_findings_retry(findings: list[ResearchFinding]) -> bool:
    """Check whether structured findings need a retry."""
    if not findings:
        return True
    for finding in findings:
        if not finding.get("finding"):
            return True
        if not finding.get("page"):
            return True
        if not finding.get("location_detail"):
            return True
    return False


def _should_retry_parse_error(error: ValueError) -> bool:
    """Check whether a parse failure should trigger one retry."""
    return "tool call" in str(error).casefold()


def _build_retry_user_text(
    user_text: str,
    remind_tool_call: bool = False,
) -> str:
    """Append a short correction when research output is not usable."""
    correction = (
        "Return structured findings using the tool.\n"
        "Each finding must have: finding (text), page (integer), "
        "and location_detail (string).\n"
        "One discrete fact per finding.\n"
        "Do not return an empty findings array if the chunks "
        "contain relevant information.\n"
    )
    if remind_tool_call:
        correction += (
            "Use the provided tool for your response.\n"
            "Do not answer in plain text.\n"
            "Your response must be a tool call.\n"
        )
    return user_text + "\n\nCorrection:\n" + correction


def _format_chunks(chunks: list[ExpandedChunk]) -> str:
    """Format expanded chunks grouped by section for LLM context.

    Groups chunks by section title, ordered by page within
    each section. Each section gets a header; each chunk
    shows its citation page, optional sheet name, and
    fusion score when available.

    Params:
        chunks: Expanded chunks to format

    Returns:
        str -- formatted text with section/page headers

    Example:
        >>> text = _format_chunks(expanded)
        >>> "=== KM1" in text
        True
    """
    if not chunks:
        return ""
    groups: dict[str, list[ExpandedChunk]] = {}
    for chunk in chunks:
        key = chunk["section_title"] or "Ungrouped Content"
        groups.setdefault(key, []).append(chunk)
    lines: list[str] = []
    for section_title, section_chunks in groups.items():
        pages = sorted({c["page_number"] for c in section_chunks})
        if len(pages) == 1:
            page_range = f"Page {pages[0]}"
        else:
            page_range = f"Pages {pages[0]}-{pages[-1]}"
        lines.append(f"=== {section_title} ({page_range}) ===")
        lines.append("")
        for chunk in section_chunks:
            parts = [f"Citation Page: {chunk['page_number']}"]
            sheet_name = _extract_sheet_name(chunk.get("chunk_header", ""))
            if sheet_name:
                parts.append(f"Sheet: {sheet_name}")
            score = chunk.get("score", 0.0)
            if score > 0.0:
                parts.append(f"Score: {score:.2f}")
            header = "[" + " | ".join(parts) + "]"
            lines.append(header)
            lines.append(chunk["raw_content"].strip())
            lines.append("")
    return "\n".join(lines)


def _format_previous_research(
    iterations: list[ResearchIteration],
) -> str:
    """Format prior iteration results for LLM context.

    Wraps each iteration in a header showing iteration
    number and confidence, inside XML tags so the LLM
    does not repeat findings.

    Params:
        iterations: Completed research iterations

    Returns:
        str -- formatted previous research block,
        or empty string if no iterations

    Example:
        >>> text = _format_previous_research(iterations)
        >>> "<previous_research>" in text
        True
    """
    if not iterations:
        return ""
    lines: list[str] = ["<previous_research>"]
    for iteration in iterations:
        header = (
            f"[Iteration {iteration['iteration']}, "
            f"confidence: {iteration['confidence']}]"
        )
        lines.append(header)
        for finding in iteration["findings"]:
            page = finding["page"]
            loc = finding.get("location_detail", "")
            ref = f"(Page {page}, {loc})" if loc else f"(Page {page})"
            lines.append(f"- {finding['finding']} {ref}")
        lines.append("")
    lines.append("</previous_research>")
    return "\n".join(lines)


def _format_research_input(
    query: str,
    source_label: str,
    bank: str,
    period: str,
    previous_research: str,
    chunks: str,
    prompt: dict,
) -> str:
    """Substitute placeholders in the user prompt template.

    Params:
        query: Original user research query
        source_label: Document source description
        bank: Bank identifier
        period: Period identifier
        previous_research: Formatted prior iterations
        chunks: Formatted chunk content
        prompt: Loaded prompt dict with user_prompt

    Returns:
        str -- fully populated user prompt
    """
    text = prompt["user_prompt"]
    text = text.replace("{query}", query)
    text = text.replace("{source_label}", source_label)
    text = text.replace("{bank}", bank)
    text = text.replace("{period}", period)
    text = text.replace("{previous_research}", previous_research)
    text = text.replace("{chunks}", chunks)
    return text


def _build_messages(
    prompt: dict,
    user_text: str,
) -> list[dict]:
    """Assemble system and user messages for the LLM.

    Params:
        prompt: Loaded prompt dict with optional system_prompt
        user_text: Fully populated user prompt text

    Returns:
        list[dict] -- message dicts for the LLM call
    """
    messages: list[dict] = []
    if prompt.get("system_prompt"):
        messages.append(
            {
                "role": "system",
                "content": prompt["system_prompt"],
            }
        )
    messages.append({"role": "user", "content": user_text})
    return messages


def _parse_research_response(
    response: dict,
) -> tuple[list[ResearchFinding], list[str], float]:
    """Extract structured findings from LLM tool call response.

    Params:
        response: Raw LLM API response dict

    Returns:
        tuple of (findings list, additional queries,
        confidence score)

    Example:
        >>> findings, q, c = _parse_research_response(resp)
    """
    parsed = extract_tool_arguments(response)
    if "findings" not in parsed:
        raise ValueError("Tool response missing field: findings")
    raw_findings = parsed["findings"]
    if not isinstance(raw_findings, list):
        raise ValueError("findings must be an array")
    findings: list[ResearchFinding] = []
    for item in raw_findings:
        if not isinstance(item, dict):
            raise ValueError("each finding must be an object")
        if "finding" not in item or not isinstance(item["finding"], str):
            raise ValueError("finding.finding must be a string")
        if "page" not in item or not isinstance(item["page"], (int, float)):
            raise ValueError("finding.page must be an integer")
        if "location_detail" not in item or not isinstance(
            item["location_detail"], str
        ):
            raise ValueError("finding.location_detail must be a string")
        entry = ResearchFinding(
            finding=item["finding"],
            page=int(item["page"]),
            location_detail=item["location_detail"],
        )
        for optional_key in (
            "metric_name",
            "metric_value",
            "period",
            "segment",
        ):
            value = item.get(optional_key, "")
            if isinstance(value, str) and value:
                entry[optional_key] = value
        findings.append(entry)
    additional_queries = parsed.get("additional_queries", [])
    if not isinstance(additional_queries, list):
        raise ValueError("additional_queries must be a list")
    confidence = parsed.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    return findings, _limit_queries(additional_queries), float(confidence)


def _limit_queries(queries: list[str]) -> list[str]:
    """Deduplicate and cap follow-up queries."""
    limited: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        limited.append(normalized)
        if len(limited) >= _MAX_ADDITIONAL_QUERIES:
            break
    return limited


def _build_additional_chunk(
    conn,
    doc_version_id: int,
    hit: dict,
) -> ExpandedChunk:
    """Convert an additional-search hit into an expanded chunk."""
    section_title = ""
    section_id = hit.get("section_id", "")
    if section_id:
        sec_info = get_section_info(conn, doc_version_id, section_id)
        if sec_info:
            section_title = sec_info.get("title", "")
    return ExpandedChunk(
        content_unit_id=hit.get("content_unit_id", ""),
        raw_content=hit.get("raw_content", ""),
        page_number=hit.get("page_number", 0),
        section_id=section_id,
        section_title=section_title,
        chunk_context=hit.get("chunk_context", ""),
        chunk_header=hit.get("chunk_header", ""),
        sheet_passthrough_content=hit.get(
            "sheet_passthrough_content",
            "",
        ),
        section_passthrough_content=hit.get(
            "section_passthrough_content",
            "",
        ),
        is_original=False,
        token_count=hit.get("token_count", 0),
    )


def _coerce_additional_search_context(
    context: dict | None,
    legacy_kwargs: dict,
) -> dict:
    """Normalize additional-search call arguments into one context dict."""
    search_context = dict(context or {})
    for key in ("llm", "top_k", "seen_ids", "metrics", "trace"):
        if key in legacy_kwargs and key not in search_context:
            search_context[key] = legacy_kwargs.pop(key)
    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")
    required = ("llm", "top_k", "seen_ids")
    missing = [key for key in required if key not in search_context]
    if missing:
        names = ", ".join(missing)
        raise TypeError(f"Missing required additional-search context: {names}")
    return search_context


def _update_empty_additional_telemetry(
    metrics: dict | None,
    trace: dict | None,
) -> None:
    """Populate empty metrics and trace payloads for no-op searches."""
    if metrics is not None:
        metrics.update(
            {
                "embed_calls": 0,
                "embed_call_seconds": 0.0,
                "search_seconds": 0.0,
                "new_chunks": 0,
            }
        )
    if trace is not None:
        trace.update(
            {
                "queries": [],
                "embed_call_seconds": 0.0,
                "search_seconds": 0.0,
                "new_chunk_ids": [],
            }
        )


def _collect_additional_hits(
    conn,
    doc_version_id: int,
    hits: list[dict],
    search_state: dict,
) -> tuple[list[ExpandedChunk], dict]:
    """Filter follow-up hits and convert unseen rows to chunks."""
    new_chunks: list[ExpandedChunk] = []
    query_trace = {
        "hits": [],
        "new_chunk_ids": [],
        "skipped_existing_chunk_ids": [],
        "skipped_batch_duplicate_chunk_ids": [],
    }
    for hit in hits:
        chunk_id = hit.get("content_unit_id", "")
        trace_hit = snapshot_content_row(hit)
        trace_hit["distance"] = round(float(hit.get("distance", 0.0)), 6)
        query_trace["hits"].append(trace_hit)
        if chunk_id in search_state["seen_ids"]:
            query_trace["skipped_existing_chunk_ids"].append(chunk_id)
            continue
        if chunk_id in search_state["seen_in_batch"]:
            query_trace["skipped_batch_duplicate_chunk_ids"].append(chunk_id)
            continue
        search_state["seen_in_batch"].add(chunk_id)
        query_trace["new_chunk_ids"].append(chunk_id)
        new_chunks.append(_build_additional_chunk(conn, doc_version_id, hit))
    return new_chunks, query_trace


def _search_additional(
    conn,
    doc_version_id: int,
    queries: list[str],
    context: dict | None = None,
    **legacy_kwargs,
) -> list[ExpandedChunk]:
    """Embed additional queries and search for new chunks.

    For each query, generates an embedding via the LLM
    client, runs a vector search, filters already-seen
    content units, and converts results to ExpandedChunks
    with section metadata.

    Params:
        conn: psycopg2 connection
        doc_version_id: Target document version
        queries: Additional search query texts
        llm: Configured LLM client for embeddings
        top_k: Maximum results per query
        seen_ids: Content unit IDs already in context

    Returns:
        list[ExpandedChunk] -- new chunks not in seen_ids

    Example:
        >>> new = _search_additional(
        ...     conn, 38, ["CET1 details"], llm, 10, seen
        ... )
    """
    search_context = _coerce_additional_search_context(
        context,
        legacy_kwargs,
    )
    runtime = {
        "llm": search_context["llm"],
        "metrics": search_context.get("metrics"),
        "trace": search_context.get("trace"),
        "embed_model": get_embedding_model(),
        "embed_dims": get_embedding_dimensions(),
        "new_chunks": [],
        "search_state": {
            "seen_ids": search_context["seen_ids"],
            "seen_in_batch": set(),
        },
        "search_elapsed": 0.0,
        "query_traces": [],
    }
    if not queries:
        _update_empty_additional_telemetry(
            runtime["metrics"],
            runtime["trace"],
        )
        return runtime["new_chunks"]

    embed_start = perf_counter()
    embeddings = runtime["llm"].embed(
        queries,
        model=runtime["embed_model"],
        dimensions=runtime["embed_dims"],
    )
    embed_elapsed = perf_counter() - embed_start

    for query_text, embedding in zip(queries, embeddings):
        search_start = perf_counter()
        hits = search_by_content_vector(
            conn,
            doc_version_id,
            embedding,
            search_context["top_k"],
        )
        query_search_elapsed = perf_counter() - search_start
        runtime["search_elapsed"] += query_search_elapsed
        query_new_chunks, query_trace = _collect_additional_hits(
            conn,
            doc_version_id,
            hits,
            runtime["search_state"],
        )
        runtime["new_chunks"].extend(query_new_chunks)
        query_trace["query"] = query_text
        query_trace["search_seconds"] = round(query_search_elapsed, 3)
        runtime["query_traces"].append(query_trace)
    if runtime["metrics"] is not None:
        runtime["metrics"].update(
            {
                "embed_calls": 1,
                "embed_call_seconds": round(embed_elapsed, 3),
                "search_seconds": round(runtime["search_elapsed"], 3),
                "new_chunks": len(runtime["new_chunks"]),
                "queries": len(queries),
            }
        )
    if runtime["trace"] is not None:
        runtime["trace"].update(
            {
                "queries": runtime["query_traces"],
                "embed_call_seconds": round(embed_elapsed, 3),
                "search_seconds": round(runtime["search_elapsed"], 3),
                "new_chunk_ids": [
                    chunk["content_unit_id"] for chunk in runtime["new_chunks"]
                ],
            }
        )
    return runtime["new_chunks"]


def _combine_findings(
    iterations: list[ResearchIteration],
) -> list[ResearchFinding]:
    """Merge structured findings from all iterations, deduping by page+text."""
    seen: set[tuple[str, int]] = set()
    combined: list[ResearchFinding] = []
    for iteration in iterations:
        for finding in iteration.get("findings", []):
            key = (finding["finding"], finding["page"])
            if key not in seen:
                seen.add(key)
                combined.append(finding)
    return combined


def _add_usage_totals(totals: dict[str, int], usage: dict[str, int]) -> None:
    """Accumulate token-usage counters into the running totals."""
    for key, value in usage.items():
        totals[key] += value


def _build_research_attempt_trace(
    attempt: int,
    attempt_elapsed: float,
    usage: dict[str, int],
) -> dict:
    """Create the base trace row for one LLM research attempt."""
    return {
        "attempt": attempt,
        "llm_call_seconds": round(attempt_elapsed, 3),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
    }


def _call_llm_research(
    prompt: dict,
    user_text: str,
    iteration_num: int,
    llm: LLMClient,
) -> tuple[ResearchIteration, dict, list[dict]]:
    """Call the LLM and parse into a ResearchIteration.

    Params:
        prompt: Loaded research prompt dict
        user_text: Fully populated user prompt
        iteration_num: Current 1-based iteration number
        llm: Configured LLM client

    Returns:
        ResearchIteration with parsed LLM output

    Example:
        >>> it = _call_llm_research(prompt, text, 1, llm)
    """
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    state: dict = {
        "llm_elapsed": 0.0,
        "llm_calls": 0,
        "retry_text": user_text,
        "findings": [],
        "additional_queries": [],
        "confidence": 0.0,
    }
    attempt_traces: list[dict] = []

    for attempt in range(3):
        messages = _build_messages(prompt, state["retry_text"])
        start_time = perf_counter()
        response = llm.call(
            messages=messages,
            stage=prompt["stage"],
            tools=prompt.get("tools"),
            tool_choice=prompt.get("tool_choice"),
            context="research",
        )
        attempt_elapsed = perf_counter() - start_time
        state["llm_elapsed"] += attempt_elapsed
        state["llm_calls"] += 1
        usage = get_usage_metrics(response)
        _add_usage_totals(total_usage, usage)
        attempt_trace = _build_research_attempt_trace(
            attempt + 1,
            attempt_elapsed,
            usage,
        )
        try:
            (
                state["findings"],
                state["additional_queries"],
                state["confidence"],
            ) = _parse_research_response(response)
        except ValueError as error:
            attempt_trace["status"] = "parse_error"
            attempt_trace["error"] = str(error)
            attempt_traces.append(attempt_trace)
            if attempt < 2 and _should_retry_parse_error(error):
                state["retry_text"] = _build_retry_user_text(
                    user_text,
                    remind_tool_call=True,
                )
                continue
            raise
        attempt_trace["findings"] = state["findings"]
        attempt_trace["additional_queries"] = state["additional_queries"]
        attempt_trace["confidence"] = state["confidence"]
        if attempt == 0 and _needs_findings_retry(state["findings"]):
            attempt_trace["status"] = "rejected_retry"
            attempt_trace["retry_reason"] = "empty_or_incomplete_findings"
            attempt_traces.append(attempt_trace)
            state["retry_text"] = _build_retry_user_text(user_text)
            continue
        attempt_trace["status"] = "accepted"
        attempt_traces.append(attempt_trace)
        break

    return (
        ResearchIteration(
            iteration=iteration_num,
            additional_queries=state["additional_queries"],
            confidence=state["confidence"],
            findings=state["findings"],
        ),
        {
            "llm_calls": state["llm_calls"],
            "llm_call_seconds": round(state["llm_elapsed"], 3),
            "prompt_tokens": total_usage["prompt_tokens"],
            "completion_tokens": total_usage["completion_tokens"],
            "total_tokens": total_usage["total_tokens"],
        },
        attempt_traces,
    )


def _build_iteration_metric(
    iteration_num: int,
    chunk_count: int,
    context_tokens: int,
    iteration: ResearchIteration,
    llm_metrics: dict,
) -> dict:
    """Build a metrics payload for one research iteration."""
    metric = {
        "iteration": iteration_num,
        "chunk_count": chunk_count,
        "context_tokens": context_tokens,
        "confidence": iteration["confidence"],
        "additional_queries": len(iteration["additional_queries"]),
    }
    metric.update(llm_metrics)
    return metric


def _run_follow_up_search(
    conn,
    source: SourceSpec,
    iteration: ResearchIteration,
    llm: LLMClient,
    additional_top_k: int,
    seen_ids: set[str],
) -> tuple[list[ExpandedChunk], dict, dict]:
    """Run follow-up retrieval for any additional queries."""
    additional_metrics: dict = {}
    additional_trace: dict = {}
    new_chunks = _search_additional(
        conn,
        source["document_version_id"],
        iteration["additional_queries"],
        {
            "llm": llm,
            "top_k": additional_top_k,
            "seen_ids": seen_ids,
            "metrics": additional_metrics,
            "trace": additional_trace,
        },
    )
    return new_chunks, additional_metrics, additional_trace


def _build_research_stage_metrics(
    iterations: list[ResearchIteration],
    all_chunks: list[ExpandedChunk],
    llm_calls: int,
    embed_calls: int,
    iteration_metrics: list[dict],
    total_elapsed: float,
) -> dict:
    """Build the final metrics payload for the research stage."""
    final_confidence = iterations[-1]["confidence"] if iterations else 0.0
    total_context_tokens = sum(chunk["token_count"] for chunk in all_chunks)
    return {
        "wall_time_seconds": round(total_elapsed, 3),
        "iterations": len(iterations),
        "llm_calls": llm_calls,
        "embed_calls": embed_calls,
        "chunk_count": len(all_chunks),
        "context_tokens": total_context_tokens,
        "final_confidence": final_confidence,
        "iteration_metrics": iteration_metrics,
    }


def _build_iteration_inputs(
    prepared: PreparedQuery,
    combo: ComboSpec,
    source: SourceSpec,
    prompt: dict,
    state: dict,
) -> tuple[list[ExpandedChunk], str, int]:
    """Build ordered chunks, prompt text, and context-token count."""
    ordered_chunks = _order_chunks_by_page(state["chunks"])
    context_tokens = sum(chunk["token_count"] for chunk in state["chunks"])
    user_text = _format_research_input(
        query=prepared["original_query"],
        source_label=f"{source['data_source']} / {source['filename']}",
        bank=combo["bank"],
        period=combo["period"],
        previous_research=_format_previous_research(state["iterations"]),
        chunks=_format_chunks(ordered_chunks),
        prompt=prompt,
    )
    return ordered_chunks, user_text, context_tokens


def _build_iteration_trace(
    iteration_num: int,
    ordered_chunks: list[ExpandedChunk],
    context_tokens: int,
    llm_attempts: list[dict],
    iteration: ResearchIteration,
) -> dict:
    """Build structured trace payload for one research iteration."""
    return {
        "iteration": iteration_num,
        "input_chunk_ids": [
            chunk["content_unit_id"] for chunk in ordered_chunks
        ],
        "input_chunks": [
            snapshot_expanded_chunk(chunk) for chunk in ordered_chunks
        ],
        "context_tokens": context_tokens,
        "llm_attempts": llm_attempts,
        "findings": iteration["findings"],
        "confidence": iteration["confidence"],
        "additional_queries": iteration["additional_queries"],
    }


def _record_follow_up_origins(
    follow_up_origins: dict[str, list[dict]],
    additional_trace: dict,
    iteration_num: int,
) -> None:
    """Append chunk lineage for newly added follow-up hits."""
    for query_trace in additional_trace.get("queries", []):
        for chunk_id in query_trace.get("new_chunk_ids", []):
            follow_up_origins.setdefault(chunk_id, []).append(
                {
                    "origin_stage": "follow_up_search",
                    "iteration": iteration_num,
                    "query": query_trace.get("query", ""),
                }
            )


def _finalize_research_trace(
    trace: dict | None,
    expanded: list[ExpandedChunk],
    state: dict,
) -> None:
    """Write the final iterative-research trace payload."""
    if trace is None:
        return
    trace.update(
        {
            "initial_chunk_ids": [
                chunk["content_unit_id"] for chunk in expanded
            ],
            "iterations": state["iteration_traces"],
            "follow_up_chunk_origins": state["follow_up_origins"],
            "final_chunk_ids": [
                chunk["content_unit_id"] for chunk in state["chunks"]
            ],
            "final_chunks": [
                snapshot_expanded_chunk(chunk) for chunk in state["chunks"]
            ],
            "stopping_reason": state["stopping_reason"],
        }
    )


def _run_research_iterations(
    conn,
    prepared: PreparedQuery,
    combo: ComboSpec,
    source: SourceSpec,
    context: dict,
    expanded: list[ExpandedChunk],
    trace: dict | None = None,
) -> tuple[list[ResearchIteration], list[ExpandedChunk], list[dict], int, int]:
    """Run the iterative research loop and collect metrics."""
    logger = get_stage_logger(__name__, STAGE)
    state = {
        "seen_ids": {chunk["content_unit_id"] for chunk in expanded},
        "chunks": list(expanded),
        "iterations": [],
        "iteration_metrics": [],
        "llm_calls": 0,
        "embed_calls": 0,
        "previous_queries": [],
        "previous_confidence": 0.0,
        "iteration_traces": [],
        "follow_up_origins": {},
        "stopping_reason": "max_iterations",
    }

    for iteration_num in range(1, get_research_max_iterations() + 1):
        ordered_chunks, user_text, context_tokens = _build_iteration_inputs(
            prepared,
            combo,
            source,
            context["prompt"],
            state,
        )
        iteration, llm_metrics, llm_attempts = _call_llm_research(
            context["prompt"],
            user_text,
            iteration_num,
            context["llm"],
        )
        state["llm_calls"] += llm_metrics["llm_calls"]
        state["iterations"].append(iteration)
        iteration_metric = _build_iteration_metric(
            iteration_num,
            len(state["chunks"]),
            context_tokens,
            iteration,
            llm_metrics,
        )
        iteration_trace = _build_iteration_trace(
            iteration_num,
            ordered_chunks,
            context_tokens,
            llm_attempts,
            iteration,
        )
        logger.info(
            "Iteration %d completed in %.1fs — chunks=%d, "
            "prompt_tokens=%d, confidence=%.2f, additional_queries=%d",
            iteration_num,
            llm_metrics["llm_call_seconds"],
            len(state["chunks"]),
            llm_metrics["prompt_tokens"],
            iteration["confidence"],
            len(iteration["additional_queries"]),
        )
        if llm_metrics["llm_calls"] > 1:
            logger.info(
                "Iteration %d retried due to unusable model output",
                iteration_num,
            )

        if (
            not iteration["additional_queries"]
            or iteration_num >= get_research_max_iterations()
        ):
            state["stopping_reason"] = (
                "no_additional_queries"
                if not iteration["additional_queries"]
                else "max_iterations"
            )
            state["iteration_metrics"].append(iteration_metric)
            state["iteration_traces"].append(iteration_trace)
            break
        if (
            _queries_are_exact_repeats(
                state["previous_queries"],
                iteration["additional_queries"],
            )
            and iteration["confidence"] <= state["previous_confidence"]
        ):
            state["stopping_reason"] = "repeated_follow_up_queries"
            logger.info(
                "Iteration %d stopping on repeated follow-up queries",
                iteration_num,
            )
            state["iteration_metrics"].append(iteration_metric)
            state["iteration_traces"].append(iteration_trace)
            break

        follow_up = _run_follow_up_search(
            conn,
            source,
            iteration,
            context["llm"],
            get_research_additional_top_k(),
            state["seen_ids"],
        )
        state["embed_calls"] += follow_up[1].get("embed_calls", 0)
        iteration_metric.update(follow_up[1])
        iteration_trace["follow_up_search"] = follow_up[2]
        logger.info(
            "Iteration %d follow-up search — queries=%d, embed=%.1fs, "
            "new_chunks=%d",
            iteration_num,
            follow_up[1].get("queries", 0),
            follow_up[1].get("embed_call_seconds", 0.0),
            follow_up[1].get("new_chunks", 0),
        )

        if not follow_up[0]:
            state["stopping_reason"] = "follow_up_found_no_new_chunks"
            logger.info("No new chunks found, stopping research")
            state["iteration_metrics"].append(iteration_metric)
            state["iteration_traces"].append(iteration_trace)
            break

        state["chunks"].extend(follow_up[0])
        state["seen_ids"].update(
            chunk["content_unit_id"] for chunk in follow_up[0]
        )
        _record_follow_up_origins(
            state["follow_up_origins"],
            follow_up[2],
            iteration_num,
        )
        state["previous_confidence"] = iteration["confidence"]
        state["previous_queries"] = iteration["additional_queries"]
        iteration_metric["total_chunks_after"] = len(state["chunks"])
        state["iteration_metrics"].append(iteration_metric)
        iteration_trace["new_chunks"] = [
            snapshot_expanded_chunk(chunk) for chunk in follow_up[0]
        ]
        iteration_trace["total_chunks_after"] = len(state["chunks"])
        state["iteration_traces"].append(iteration_trace)
        logger.info(
            "Added %d new chunks (total: %d)",
            len(follow_up[0]),
            len(state["chunks"]),
        )

    _finalize_research_trace(trace, expanded, state)
    return (
        state["iterations"],
        state["chunks"],
        state["iteration_metrics"],
        state["llm_calls"],
        state["embed_calls"],
    )


def _validate_finding_pages(
    findings: list[ResearchFinding],
    chunks: list[ExpandedChunk],
    logger_instance,
) -> list[str]:
    """Check finding pages against available chunk pages.

    Params:
        findings: Structured findings from research
        chunks: Expanded chunks used as evidence
        logger_instance: Logger for warning messages

    Returns:
        List of citation mismatch warning strings
    """
    available_pages = {chunk["page_number"] for chunk in chunks}
    cited_pages = {finding["page"] for finding in findings}

    warnings: list[str] = []
    for page in sorted(cited_pages - available_pages):
        msg = f"Research cites Page {page} not in evidence chunks"
        warnings.append(msg)
        logger_instance.warning(msg)

    return warnings


def _coerce_research_call(
    context: dict | None,
    legacy_kwargs: dict,
) -> tuple[dict, dict]:
    """Normalize research stage call inputs and instrumentation."""
    call_context = dict(context or {})
    if "llm" in legacy_kwargs and "llm" not in call_context:
        call_context["llm"] = legacy_kwargs.pop("llm")
    required = ("llm",)
    missing = [key for key in required if key not in call_context]
    if missing:
        names = ", ".join(missing)
        raise TypeError(f"Missing required research context: {names}")
    instrumentation = {
        "metrics": legacy_kwargs.pop("metrics", None),
        "trace": legacy_kwargs.pop("trace", None),
        "initial_chunk_origins": legacy_kwargs.pop(
            "initial_chunk_origins",
            None,
        ),
    }
    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")
    return call_context, instrumentation


def _build_final_chunk_lineage(
    all_chunks: list[ExpandedChunk],
    follow_up_origins: dict[str, list[dict]],
    initial_chunk_origins: dict[str, list[dict]] | None,
) -> dict[str, list[dict]]:
    """Merge initial expansion lineage with follow-up lineage."""
    final_chunk_lineage = {}
    for chunk in all_chunks:
        chunk_id = chunk["content_unit_id"]
        final_chunk_lineage[chunk_id] = follow_up_origins.get(
            chunk_id,
            (
                initial_chunk_origins.get(chunk_id, [])
                if initial_chunk_origins is not None
                else []
            ),
        )
    return final_chunk_lineage


def research_combo_source(
    conn,
    prepared: PreparedQuery,
    expanded: list[ExpandedChunk],
    combo: ComboSpec,
    source: SourceSpec,
    context: dict | None = None,
    **legacy_kwargs,
) -> ComboSourceResult:
    """Run the iterative research agent for one source.

    Sends expanded chunks and the research query to the
    LLM, collects findings with page citations, and
    optionally searches for additional content across
    multiple iterations. Research from every iteration is
    preserved and concatenated in the final output.

    Params:
        conn: psycopg2 connection
        prepared: Decomposed query with embeddings
        expanded: Initial expanded chunks from search
        combo: Bank and period combination
        source: Resolved document source
        llm: Configured LLM client

    Returns:
        ComboSourceResult with all iterations and
        structured findings

    Example:
        >>> result = research_combo_source(
        ...     conn, prepared, expanded, combo, source, llm,
        ... )
        >>> result["findings"][0]["finding"]
        "CET1 ratio was 13.7%"
    """
    logger = get_stage_logger(__name__, STAGE)
    start_time = perf_counter()
    coerced_call = _coerce_research_call(context, legacy_kwargs)
    runtime = {
        "call_context": coerced_call[0],
        "instrumentation": coerced_call[1],
        "prompt": load_prompt("research", prompts_dir=_PROMPTS_DIR),
        "source_label": f"{source['data_source']} / {source['filename']}",
    }
    run_result = _run_research_iterations(
        conn,
        prepared,
        combo,
        source,
        {
            "llm": runtime["call_context"]["llm"],
            "prompt": runtime["prompt"],
        },
        expanded,
        trace=runtime["instrumentation"]["trace"],
    )

    combined_findings = _combine_findings(run_result[0])
    citation_warnings = _validate_finding_pages(
        combined_findings,
        run_result[1],
        logger,
    )
    total_elapsed = perf_counter() - start_time
    stage_metrics = _build_research_stage_metrics(
        run_result[0],
        run_result[1],
        run_result[3],
        run_result[4],
        run_result[2],
        total_elapsed,
    )
    stage_metrics["citation_cross_ref_warnings"] = len(citation_warnings)
    if runtime["instrumentation"]["metrics"] is not None:
        runtime["instrumentation"]["metrics"].update(stage_metrics)
    if runtime["instrumentation"]["trace"] is not None:
        follow_up_origins = runtime["instrumentation"]["trace"].get(
            "follow_up_chunk_origins",
            {},
        )
        runtime["instrumentation"]["trace"].update(
            {
                "findings": [dict(f) for f in combined_findings],
                "citation_cross_ref_warnings": citation_warnings,
                "final_chunk_lineage": _build_final_chunk_lineage(
                    run_result[1],
                    follow_up_origins,
                    runtime["instrumentation"]["initial_chunk_origins"],
                ),
            }
        )

    logger.info(
        "[%s] completed in %.1fs — iterations=%d, llm_calls=%d, "
        "embed_calls=%d, chunks=%d, context_tokens=%d, "
        "final_confidence=%.2f (%s)",
        STAGE,
        total_elapsed,
        len(run_result[0]),
        run_result[3],
        run_result[4],
        len(run_result[1]),
        stage_metrics["context_tokens"],
        stage_metrics["final_confidence"],
        runtime["source_label"],
    )

    return ComboSourceResult(
        combo=combo,
        source=source,
        research_iterations=run_result[0],
        findings=combined_findings,
        chunk_count=len(run_result[1]),
        total_tokens=sum(chunk["token_count"] for chunk in run_result[1]),
    )
