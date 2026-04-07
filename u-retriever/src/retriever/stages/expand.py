"""Stage 4: Context expansion around search results."""

from time import perf_counter

from ..models import ExpandedChunk, SearchResult
from ..utils.config_setup import (
    get_expand_neighbor_count,
    get_expand_section_threshold,
    get_expand_subsection_threshold,
)
from ..utils.logging_setup import get_stage_logger
from ..utils.postgres_connector import (
    find_child_section_for_content,
    get_section_for_content,
    get_section_info,
    load_neighbor_chunks,
    load_section_content,
)
from ..utils.trace_store import snapshot_expanded_chunk

STAGE = "4-EXPAND"


def _numeric_cuid_sort_key(
    chunk: dict,
) -> tuple[int, int, int]:
    """Sort key for numeric content_unit_id ordering.

    Parses "24.2" into (24, 24, 2) and "9" into (9, 9, 0)
    so that "24.2" sorts before "24.19".

    Params: chunk (dict). Returns: tuple[int, int, int].
    """
    page = chunk.get("page_number", 0)
    cuid = chunk.get("content_unit_id", "0")
    parts = cuid.split(".", 1)
    try:
        major = int(parts[0])
    except (ValueError, IndexError):
        major = 0
    try:
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        minor = 0
    return (page, major, minor)


def _get_expansion_strategy(
    conn,
    doc_version_id: int,
    result: SearchResult,
    section_thresh: int,
    subsection_thresh: int,
) -> str:
    """Determine expansion strategy for a search result.

    Checks section size against thresholds. If the section
    is small enough, returns "full_section". If the section
    has subsections within the threshold, returns
    "subsection". Otherwise returns "neighbors".

    Params:
        conn: psycopg2 connection
        doc_version_id: Target document version
        result: Search result to expand
        section_thresh: Max tokens for full section load
        subsection_thresh: Max tokens for subsection load

    Returns:
        str -- "full_section", "subsection", or "neighbors"

    Example:
        >>> _get_expansion_strategy(
        ...     conn, 38, result, 3000, 1500
        ... )
        "full_section"
    """
    section = get_section_for_content(
        conn,
        doc_version_id,
        result["content_unit_id"],
    )
    if section is None:
        return "neighbors"

    token_count = section.get("token_count", 0)
    if token_count <= section_thresh:
        return "full_section"

    subsection = _find_subsection(
        conn,
        doc_version_id,
        section["section_id"],
        result["content_unit_id"],
    )
    if subsection is not None:
        sub_tokens = subsection.get("token_count", 0)
        if sub_tokens <= subsection_thresh:
            return "subsection"

    return "neighbors"


def _load_section_chunks(
    conn,
    doc_version_id: int,
    section_id: str,
) -> list[dict]:
    """Load all content units for a section.

    Params:
        conn: psycopg2 connection
        doc_version_id: Target document version
        section_id: Section identifier

    Returns:
        list[dict] -- content rows ordered by position
    """
    return load_section_content(conn, doc_version_id, section_id)


def _find_subsection(
    conn,
    doc_version_id: int,
    section_id: str,
    content_unit_id: str,
) -> dict | None:
    """Find the subsection containing a content unit.

    Uses a single JOIN query instead of loading each child
    section's content separately.

    Params:
        conn: psycopg2 connection
        doc_version_id: Target document version
        section_id: Parent section identifier
        content_unit_id: Content unit to locate

    Returns:
        dict or None -- subsection info if found
    """
    return find_child_section_for_content(
        conn,
        doc_version_id,
        section_id,
        content_unit_id,
    )


def _to_expanded_chunk(
    row: dict,
    section_title: str,
    is_original: bool,
    score: float = 0.0,
) -> ExpandedChunk:
    """Convert a database row to an ExpandedChunk TypedDict.

    Params:
        row: Database content row
        section_title: Title of the parent section
        is_original: True if from original search results
        score: Fusion score from search (0.0 for context chunks)

    Returns:
        ExpandedChunk
    """
    return ExpandedChunk(
        content_unit_id=row.get("content_unit_id", ""),
        raw_content=row.get("raw_content", ""),
        page_number=row.get("page_number", 0),
        section_id=row.get("section_id", ""),
        section_title=section_title,
        chunk_context=row.get("chunk_context", ""),
        chunk_header=row.get("chunk_header", ""),
        sheet_passthrough_content=row.get("sheet_passthrough_content", ""),
        section_passthrough_content=row.get("section_passthrough_content", ""),
        is_original=is_original,
        token_count=row.get("token_count", 0),
        score=score,
    )


def _resolve_row_titles(
    conn,
    doc_version_id: int,
    rows: list[dict],
    fallback: str,
) -> dict[str, str]:
    """Build a section_id to title mapping for a set of rows.

    Params:
        conn: psycopg2 connection
        doc_version_id: Target document version
        rows: Content rows with section_id fields
        fallback: Default title when lookup fails

    Returns:
        dict mapping section_id to title string
    """
    unique_sids = {row.get("section_id", "") for row in rows}
    unique_sids.discard("")
    title_map: dict[str, str] = {}
    for sid in unique_sids:
        info = get_section_info(conn, doc_version_id, sid)
        title_map[sid] = info.get("title", fallback) if info else fallback
    return title_map


def _deduplicate_chunks(
    chunks: list[ExpandedChunk],
) -> list[ExpandedChunk]:
    """Deduplicate by content_unit_id, preferring originals.

    When duplicates exist, the version with is_original=True
    is kept. If neither is original, the first occurrence
    wins. Scores are fused by taking the maximum across
    all occurrences.

    Params: chunks (list[ExpandedChunk]). Returns: list.
    """
    seen: dict[str, int] = {}
    result: list[ExpandedChunk] = []

    for chunk in chunks:
        cuid = chunk["content_unit_id"]
        new_score = chunk.get("score", 0.0)
        if cuid not in seen:
            seen[cuid] = len(result)
            result.append(chunk)
        else:
            existing_idx = seen[cuid]
            existing = result[existing_idx]
            existing_score = existing.get("score", 0.0)
            best_score = max(existing_score, new_score)
            if chunk["is_original"] and not existing["is_original"]:
                result[existing_idx] = chunk
            if best_score > result[existing_idx].get("score", 0.0):
                result[existing_idx]["score"] = best_score

    return result


def _collect_expansions(
    conn,
    doc_version_id: int,
    results: list[SearchResult],
    settings: dict,
) -> dict:
    """Expand each result and collect per-chunk lineage."""
    all_chunks: list[ExpandedChunk] = []
    expansion_steps: list[dict] = []
    chunk_origins: dict[str, list[dict]] = {}
    strategy_counts = {
        "full_section": 0,
        "subsection": 0,
        "neighbors": 0,
    }
    for result in results:
        expanded, strategy, step_trace = _expand_single_result(
            conn,
            doc_version_id,
            result,
            settings["section_thresh"],
            settings["subsection_thresh"],
            settings["neighbor_count"],
            settings["original_cuids"],
        )
        all_chunks.extend(expanded)
        expansion_steps.append(step_trace)
        strategy_counts[strategy] += 1
        for chunk in expanded:
            chunk_origins.setdefault(chunk["content_unit_id"], []).append(
                {
                    "trigger_content_unit_id": result["content_unit_id"],
                    "trigger_page_number": result["page_number"],
                    "strategy": strategy,
                }
            )
        settings["logger"].debug(
            "Expansion strategy %s for %s (page=%d, section=%s)",
            strategy,
            result["content_unit_id"],
            result["page_number"],
            result["section_id"],
        )
    return {
        "all_chunks": all_chunks,
        "expansion_steps": expansion_steps,
        "chunk_origins": chunk_origins,
        "strategy_counts": strategy_counts,
    }


def _find_duplicate_chunk_ids(chunks: list[ExpandedChunk]) -> list[str]:
    """Return chunk ids that appeared more than once pre-dedup."""
    counts: dict[str, int] = {}
    for chunk in chunks:
        chunk_id = chunk["content_unit_id"]
        counts[chunk_id] = counts.get(chunk_id, 0) + 1
    return sorted(chunk_id for chunk_id, count in counts.items() if count > 1)


def _build_expand_trace(
    results: list[SearchResult],
    expansion_steps: list[dict],
    chunk_origins: dict[str, list[dict]],
    duplicate_chunk_ids: list[str],
    deduped: list[ExpandedChunk],
) -> dict:
    """Build structured trace payload for expansion."""
    return {
        "input_results": [
            {
                "content_unit_id": result["content_unit_id"],
                "page_number": result["page_number"],
                "section_id": result.get("section_id", ""),
                "score": round(result.get("score", 0.0), 6),
            }
            for result in results
        ],
        "expansion_steps": expansion_steps,
        "chunk_origins": chunk_origins,
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "final_chunk_ids": [chunk["content_unit_id"] for chunk in deduped],
        "final_chunks": [snapshot_expanded_chunk(chunk) for chunk in deduped],
    }


def _expand_single_result(
    conn,
    doc_version_id: int,
    result: SearchResult,
    section_thresh: int,
    subsection_thresh: int,
    neighbor_count: int,
    original_cuids: set[str],
) -> tuple[list[ExpandedChunk], str, dict]:
    """Expand a single search result into chunks.

    Params:
        conn: psycopg2 connection
        doc_version_id: Target document version
        result: Search result to expand
        section_thresh: Full section token threshold
        subsection_thresh: Subsection token threshold
        neighbor_count: Neighbors on each side
        original_cuids: Set of original content unit ids

    Returns:
        list[ExpandedChunk] -- expanded chunks
    """
    strategy = _get_expansion_strategy(
        conn,
        doc_version_id,
        result,
        section_thresh,
        subsection_thresh,
    )

    section_title = ""
    rows: list[dict] = []

    if strategy == "full_section":
        section = get_section_for_content(
            conn, doc_version_id, result["content_unit_id"]
        )
        if section:
            section_title = section.get("title", "")
            rows = _load_section_chunks(
                conn,
                doc_version_id,
                section["section_id"],
            )

    elif strategy == "subsection":
        section = get_section_for_content(
            conn, doc_version_id, result["content_unit_id"]
        )
        if section:
            subsection = _find_subsection(
                conn,
                doc_version_id,
                section["section_id"],
                result["content_unit_id"],
            )
            if subsection:
                section_title = subsection.get("title", "")
                rows = _load_section_chunks(
                    conn,
                    doc_version_id,
                    subsection["section_id"],
                )

    if strategy == "neighbors" or not rows:
        rows = load_neighbor_chunks(
            conn,
            doc_version_id,
            result["content_unit_id"],
            neighbor_count,
        )
        section = get_section_for_content(
            conn, doc_version_id, result["content_unit_id"]
        )
        if section:
            section_title = section.get("title", "")

    row_sids = {row.get("section_id", "") for row in rows}
    row_sids.discard("")
    needs_multi_title = len(row_sids) > 1
    title_map = (
        _resolve_row_titles(conn, doc_version_id, rows, section_title)
        if needs_multi_title
        else {}
    )
    chunks = []
    for row in rows:
        cuid = row.get("content_unit_id", "")
        is_orig = cuid in original_cuids
        row_title = (
            title_map.get(row.get("section_id", ""), section_title)
            if needs_multi_title
            else section_title
        )
        chunks.append(
            _to_expanded_chunk(
                row,
                row_title,
                is_orig,
                score=(
                    result.get("score", 0.0)
                    if cuid == result["content_unit_id"]
                    else 0.0
                ),
            )
        )

    return (
        chunks,
        strategy,
        {
            "input_content_unit_id": result["content_unit_id"],
            "input_page_number": result["page_number"],
            "input_section_id": result.get("section_id", ""),
            "strategy": strategy,
            "loaded_chunk_ids": [chunk["content_unit_id"] for chunk in chunks],
            "loaded_chunks": [
                snapshot_expanded_chunk(chunk) for chunk in chunks
            ],
            "added_chunk_ids": [
                chunk["content_unit_id"]
                for chunk in chunks
                if not chunk["is_original"]
            ],
        },
    )


def expand_chunks(
    conn,
    doc_version_id: int,
    results: list[SearchResult],
    metrics: dict | None = None,
    trace: dict | None = None,
) -> list[ExpandedChunk]:
    """Expand search results with surrounding context.

    For each search result, determines the best expansion
    strategy (full section, subsection, or neighbors) based
    on token thresholds, loads the expanded content,
    deduplicates, marks originals, and sorts by reading
    order.

    Params:
        conn: psycopg2 connection
        doc_version_id: Target document version
        results: Search results to expand

    Returns:
        list[ExpandedChunk] sorted by page then content_unit_id

    Example:
        >>> expanded = expand_chunks(conn, 38, results)
    """
    logger = get_stage_logger(__name__, STAGE)
    start_time = perf_counter()

    if not results:
        logger.info("No results to expand")
        if metrics is not None:
            metrics.update(
                {
                    "wall_time_seconds": 0.0,
                    "results_before": 0,
                    "chunks_after": 0,
                    "expanded_context_tokens": 0,
                    "strategy_counts": {},
                }
            )
        return []

    section_thresh = get_expand_section_threshold()
    subsection_thresh = get_expand_subsection_threshold()
    neighbor_count = get_expand_neighbor_count()
    original_cuids = {r["content_unit_id"] for r in results}
    expansion_bundle = _collect_expansions(
        conn,
        doc_version_id,
        results,
        {
            "original_cuids": original_cuids,
            "section_thresh": section_thresh,
            "subsection_thresh": subsection_thresh,
            "neighbor_count": neighbor_count,
            "logger": logger,
        },
    )
    deduped = _deduplicate_chunks(expansion_bundle["all_chunks"])
    deduped.sort(key=_numeric_cuid_sort_key)
    duplicate_chunk_ids = _find_duplicate_chunk_ids(
        expansion_bundle["all_chunks"]
    )
    original_count = sum(1 for c in deduped if c["is_original"])
    expanded_context_tokens = sum(chunk["token_count"] for chunk in deduped)
    total_elapsed = perf_counter() - start_time
    stage_metrics = {
        "wall_time_seconds": round(total_elapsed, 3),
        "results_before": len(results),
        "chunks_after": len(deduped),
        "original_chunks": original_count,
        "expanded_chunks": len(deduped) - original_count,
        "expanded_context_tokens": expanded_context_tokens,
        "strategy_counts": expansion_bundle["strategy_counts"],
    }
    if metrics is not None:
        metrics.update(stage_metrics)
    if trace is not None:
        trace.update(
            _build_expand_trace(
                results,
                expansion_bundle["expansion_steps"],
                expansion_bundle["chunk_origins"],
                duplicate_chunk_ids,
                deduped,
            )
        )

    logger.info(
        "[%s] completed in %.1fs — results=%d, chunks=%d, "
        "original=%d, expanded=%d, context_tokens=%d, "
        "strategies=full_section:%d subsection:%d neighbors:%d",
        STAGE,
        total_elapsed,
        len(results),
        len(deduped),
        original_count,
        len(deduped) - original_count,
        expanded_context_tokens,
        expansion_bundle["strategy_counts"]["full_section"],
        expansion_bundle["strategy_counts"]["subsection"],
        expansion_bundle["strategy_counts"]["neighbors"],
    )

    return deduped
