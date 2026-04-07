"""Orchestrator: wires all retrieval stages together."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

from ..models import (
    ComboSourceResult,
    ComboSpec,
    ConsolidatedResult,
    ExpandedChunk,
    PreparedQuery,
    SearchResult,
    SourceSpec,
)
from ..utils.config_setup import (
    get_orchestrator_max_workers,
    get_rerank_candidate_limit,
    get_score_weights,
    get_small_doc_token_threshold,
)
from ..utils.citation_validator import validate_consolidated_citations
from ..utils.llm_connector import LLMClient
from ..utils.logging_setup import get_stage_logger
from ..utils.postgres_connector import (
    get_connection,
    get_document_total_tokens,
    resolve_document_version_ids,
)
from ..utils.trace_store import (
    TRACE_SCHEMA_VERSION,
    get_run_trace_path,
    get_source_trace_path,
    snapshot_expanded_chunk,
    snapshot_search_result,
    start_trace_session,
    write_trace_json,
)
from .consolidate import consolidate_results
from .expand import expand_chunks
from .query_prep import prepare_query
from .rerank import rerank_results
from .research import research_combo_source
from .search import (
    load_full_document_as_results,
    multi_strategy_search,
)

STAGE = "ORCHESTRATOR"


def _resolve_research_units(
    conn,
    combos: list[ComboSpec],
    sources: list[str] | None,
) -> list[tuple[ComboSpec, SourceSpec]]:
    """Resolve all document versions for each combo.

    For each combo (bank + period), queries the database
    to find matching document versions, optionally filtered
    by source type. Returns a flat list of (combo, source)
    tuples representing individual research units.

    Params:
        conn: psycopg2 connection
        combos: Bank and period combinations
        sources: Optional data_source whitelist

    Returns:
        list of (ComboSpec, SourceSpec) tuples

    Example:
        >>> units = _resolve_research_units(
        ...     conn, [{"bank": "RBC", "period": "2026_Q1"}],
        ...     None,
        ... )
    """
    units: list[tuple[ComboSpec, SourceSpec]] = []
    for combo in combos:
        versions = resolve_document_version_ids(
            conn,
            combo["bank"],
            combo["period"],
            sources,
        )
        for version in versions:
            source = SourceSpec(
                data_source=version["data_source"],
                document_version_id=version["document_version_id"],
                filename=version["filename"],
            )
            units.append((combo, source))
    return units


def _resolve_research_units_threadsafe(
    combos: list[ComboSpec],
    sources: list[str] | None,
) -> tuple[list[tuple[ComboSpec, SourceSpec]], float]:
    """Resolve research units on a dedicated DB connection."""
    start_time = perf_counter()
    conn = get_connection()
    try:
        units = _resolve_research_units(conn, combos, sources)
    finally:
        conn.close()
    return units, perf_counter() - start_time


def _prepare_query_with_metrics(
    query: str,
    llm: LLMClient,
) -> tuple[PreparedQuery, dict, dict]:
    """Prepare the query and return collected stage metrics."""
    metrics: dict = {}
    trace: dict = {}
    prepared = prepare_query(query, llm, metrics=metrics, trace=trace)
    return prepared, metrics, trace


def _summarize_dominant_strategies(
    results: list[SearchResult],
) -> dict[str, int]:
    """Count the highest-scoring search strategy per result."""
    counts: dict[str, int] = {}
    for result in results:
        strategy_scores = result.get("strategy_scores", {})
        if not strategy_scores:
            continue
        dominant = max(strategy_scores, key=strategy_scores.get)
        counts[dominant] = counts.get(dominant, 0) + 1
    return counts


def _format_unit_timings(combo_results: list[ComboSourceResult]) -> str:
    """Render per-unit wall times for the orchestrator summary."""
    parts: list[str] = []
    for result in combo_results:
        metrics = result.get("metrics", {})
        parts.append(
            f"{result['source']['data_source']}="
            f"{metrics.get('wall_time_seconds', 0.0):.1f}s"
        )
    return ", ".join(parts)


def _prepare_small_document_context(
    results: list[SearchResult],
) -> tuple[list[ExpandedChunk], dict, dict, dict, dict, dict, dict]:
    """Build stage metrics for the small-document bypass path."""
    expanded = [_search_result_to_expanded(result) for result in results]
    search_metrics = {
        "mode": "full_document",
        "unique_results": len(results),
        "total_raw_hits": len(results),
        "wall_time_seconds": 0.0,
    }
    rerank_metrics = {
        "candidates_shown": len(results),
        "kept": len(results),
        "removed": 0,
        "llm_calls": 0,
    }
    expand_metrics = {
        "results_before": len(results),
        "chunks_after": len(expanded),
        "expanded_context_tokens": sum(
            chunk["token_count"] for chunk in expanded
        ),
        "strategy_counts": {"full_document": len(results)},
    }
    search_trace = {
        "mode": "full_document",
        "results": [snapshot_search_result(result) for result in results],
    }
    rerank_trace = {
        "mode": "full_document_bypass",
        "kept_ids": [result["content_unit_id"] for result in results],
        "removed_ids": [],
        "remove_indices": [],
        "candidates": [snapshot_search_result(result) for result in results],
    }
    expand_trace = {
        "mode": "full_document_bypass",
        "final_chunk_ids": [chunk["content_unit_id"] for chunk in expanded],
        "final_chunks": [snapshot_expanded_chunk(chunk) for chunk in expanded],
        "chunk_origins": {
            chunk["content_unit_id"]: [
                {
                    "trigger_content_unit_id": chunk["content_unit_id"],
                    "strategy": "full_document",
                }
            ]
            for chunk in expanded
        },
    }
    return (
        expanded,
        search_metrics,
        rerank_metrics,
        expand_metrics,
        search_trace,
        rerank_trace,
        expand_trace,
    )


def _build_evidence_catalog(
    chunks: list[ExpandedChunk],
) -> list[dict]:
    """Build unique page/section references from retrieved chunks."""
    catalog: list[dict] = []
    seen: set[tuple[int, str, str]] = set()
    for chunk in chunks:
        key = (
            chunk["page_number"],
            chunk.get("section_id", ""),
            chunk.get("section_title", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        catalog.append(
            {
                "content_unit_id": chunk["content_unit_id"],
                "page_number": chunk["page_number"],
                "section_id": chunk.get("section_id", ""),
                "section_title": chunk.get("section_title", ""),
                "chunk_header": chunk.get("chunk_header", ""),
            }
        )
    return catalog


def _resolve_evidence_chunks(
    expanded: list[ExpandedChunk],
    research_trace: dict,
) -> list[dict]:
    """Prefer research-final chunks when building evidence refs.

    Params:
        expanded: Chunks passed into the research stage
        research_trace: Mutable research trace payload

    Returns:
        Chunk list representing the evidence that final citations
        are allowed to reference.
    """
    final_chunks = research_trace.get("final_chunks", [])
    if final_chunks:
        return final_chunks
    return expanded


def _run_search_pipeline(
    unit_conn,
    doc_id: int,
    prepared: PreparedQuery,
    query: str,
    llm: LLMClient,
) -> tuple[
    list[SearchResult],
    list[ExpandedChunk],
    dict,
    dict,
    dict,
    dict,
    dict,
    dict,
]:
    """Execute search, rerank, and expand for a large document."""
    search_metrics: dict = {}
    rerank_metrics: dict = {}
    expand_metrics: dict = {}
    search_trace: dict = {}
    rerank_trace: dict = {}
    expand_trace: dict = {}
    results = multi_strategy_search(
        unit_conn,
        doc_id,
        prepared,
        get_score_weights(),
        metrics=search_metrics,
        trace=search_trace,
    )
    candidate_limit = get_rerank_candidate_limit()
    limited_results = (
        results[:candidate_limit] if candidate_limit > 0 else results
    )
    search_metrics["candidates_forwarded"] = len(limited_results)
    search_metrics["candidate_limit"] = candidate_limit
    search_metrics["candidates_trimmed"] = max(
        len(results) - len(limited_results),
        0,
    )
    search_trace["rerank_candidate_selection"] = {
        "candidate_limit": candidate_limit,
        "forwarded_ids": [
            result["content_unit_id"] for result in limited_results
        ],
        "trimmed_ids": (
            [result["content_unit_id"] for result in results[candidate_limit:]]
            if candidate_limit > 0
            else []
        ),
    }
    reranked = rerank_results(
        limited_results,
        query,
        llm,
        conn=unit_conn,
        doc_version_id=doc_id,
        metrics=rerank_metrics,
        trace=rerank_trace,
    )
    expanded = expand_chunks(
        unit_conn,
        doc_id,
        reranked,
        metrics=expand_metrics,
        trace=expand_trace,
    )
    return (
        reranked,
        expanded,
        search_metrics,
        rerank_metrics,
        expand_metrics,
        search_trace,
        rerank_trace,
        expand_trace,
    )


def _build_unit_metrics(
    total_tokens: int,
    results: list[SearchResult],
    search_metrics: dict,
    rerank_metrics: dict,
    expand_metrics: dict,
    research_metrics: dict,
    unit_elapsed: float,
) -> dict:
    """Build the metrics payload for one combo/source unit."""
    return {
        "wall_time_seconds": round(unit_elapsed, 3),
        "document_tokens": total_tokens,
        "search": search_metrics,
        "rerank": rerank_metrics,
        "expand": expand_metrics,
        "research": research_metrics,
        "dominant_strategies": _summarize_dominant_strategies(results),
    }


def _run_prep_and_resolution(
    query: str,
    combos: list[ComboSpec],
    sources: list[str] | None,
    llm: LLMClient,
) -> tuple[
    list[tuple[ComboSpec, SourceSpec]],
    float,
    PreparedQuery,
    dict,
    dict,
    float,
    float,
]:
    """Run query preparation and document resolution in parallel."""
    start_time = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        resolve_future = pool.submit(
            _resolve_research_units_threadsafe,
            combos,
            sources,
        )
        prepare_future = pool.submit(
            _prepare_query_with_metrics,
            query,
            llm,
        )
        research_units, resolve_elapsed = resolve_future.result()
        prepared, prep_metrics, prep_trace = prepare_future.result()
    wall_time = perf_counter() - start_time
    overlap = max(
        prep_metrics.get("wall_time_seconds", 0.0)
        + resolve_elapsed
        - wall_time,
        0.0,
    )
    return (
        research_units,
        resolve_elapsed,
        prepared,
        prep_metrics,
        prep_trace,
        wall_time,
        overlap,
    )


def _run_research_units(
    research_units: list[tuple[ComboSpec, SourceSpec]],
    prepared: PreparedQuery,
    llm: LLMClient,
    query: str,
    prepared_trace: dict,
    trace_session,
) -> tuple[list[ComboSourceResult], float, float, int]:
    """Run all combo/source research units in parallel."""
    if not research_units:
        return [], 0.0, 0.0, 0
    worker_limit = max(1, get_orchestrator_max_workers())
    worker_count = min(len(research_units), worker_limit)
    start_time = perf_counter()
    combo_results: list[ComboSourceResult] = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                _research_single_unit,
                {
                    "combo": combo,
                    "source": source,
                    "prepared": prepared,
                    "llm": llm,
                    "query": query,
                    "prepared_trace": prepared_trace,
                    "trace_session": trace_session,
                    "trace_index": index,
                },
            ): (combo, source)
            for index, (combo, source) in enumerate(research_units, start=1)
        }
        for future in as_completed(futures):
            combo_results.append(future.result())
    wall_time = perf_counter() - start_time
    sequential_unit_time = sum(
        result.get("metrics", {}).get("wall_time_seconds", 0.0)
        for result in combo_results
    )
    return combo_results, wall_time, sequential_unit_time, worker_count


def _build_orchestrator_metrics(
    total_elapsed: float,
    prework_metrics: dict,
    research_units: list[tuple[ComboSpec, SourceSpec]],
    combo_results: list[ComboSourceResult],
    research_metrics: dict,
    consolidation_metrics: dict,
) -> dict:
    """Build the metrics payload for the orchestrator."""
    longest_unit = max(
        (
            result.get("metrics", {}).get("wall_time_seconds", 0.0)
            for result in combo_results
        ),
        default=0.0,
    )
    return {
        "wall_time_seconds": round(total_elapsed, 3),
        "query_prep": prework_metrics["query_prep"],
        "document_resolution_seconds": round(
            prework_metrics["document_resolution_seconds"],
            3,
        ),
        "prep_resolution_wall_seconds": round(
            prework_metrics["prep_resolution_wall_seconds"],
            3,
        ),
        "prep_resolution_overlap_seconds": round(
            prework_metrics["prep_resolution_overlap_seconds"],
            3,
        ),
        "research_units": len(research_units),
        "max_workers": research_metrics["max_workers"],
        "parallel_research_wall_seconds": round(
            research_metrics["parallel_research_wall_seconds"],
            3,
        ),
        "sequential_unit_seconds": round(
            research_metrics["sequential_unit_seconds"],
            3,
        ),
        "parallel_overhead_seconds": round(
            max(
                research_metrics["parallel_research_wall_seconds"]
                - longest_unit,
                0.0,
            ),
            3,
        ),
        "parallel_savings_seconds": round(
            max(
                research_metrics["sequential_unit_seconds"]
                - research_metrics["parallel_research_wall_seconds"],
                0.0,
            ),
            3,
        ),
        "consolidation": consolidation_metrics,
    }


def _search_result_to_expanded(
    result: SearchResult,
) -> ExpandedChunk:
    """Convert a SearchResult to an ExpandedChunk.

    Used for the full-document path where search results
    are already complete content units that do not need
    context expansion. Sets is_original=True and uses
    chunk_context for section_title.

    Params:
        result: Search result to convert

    Returns:
        ExpandedChunk with fields mapped from result
    """
    return ExpandedChunk(
        content_unit_id=result["content_unit_id"],
        raw_content=result["raw_content"],
        page_number=result["page_number"],
        section_id=result.get("section_id", ""),
        section_title=result.get("chunk_header", ""),
        chunk_context=result.get("chunk_context", ""),
        chunk_header=result.get("chunk_header", ""),
        sheet_passthrough_content="",
        section_passthrough_content="",
        is_original=True,
        token_count=result["token_count"],
    )


def _load_unit_stage_bundle(
    unit_conn,
    source: SourceSpec,
    prepared: PreparedQuery,
    query: str,
    llm: LLMClient,
) -> dict:
    """Load retrieval-stage outputs for one combo/source unit."""
    doc_id = source["document_version_id"]
    total_tokens = get_document_total_tokens(unit_conn, doc_id)
    small_threshold = get_small_doc_token_threshold()
    if total_tokens <= small_threshold:
        results = load_full_document_as_results(unit_conn, doc_id)
        (
            expanded,
            search_metrics,
            rerank_metrics,
            expand_metrics,
            search_trace,
            rerank_trace,
            expand_trace,
        ) = _prepare_small_document_context(results)
        retrieval_mode = "full_document"
    else:
        (
            results,
            expanded,
            search_metrics,
            rerank_metrics,
            expand_metrics,
            search_trace,
            rerank_trace,
            expand_trace,
        ) = _run_search_pipeline(
            unit_conn,
            doc_id,
            prepared,
            query,
            llm,
        )
        retrieval_mode = "search_pipeline"
    return {
        "document": {
            "document_tokens": total_tokens,
            "small_document_threshold": small_threshold,
            "retrieval_mode": retrieval_mode,
        },
        "results": results,
        "expanded": expanded,
        "search_metrics": search_metrics,
        "rerank_metrics": rerank_metrics,
        "expand_metrics": expand_metrics,
        "search_trace": search_trace,
        "rerank_trace": rerank_trace,
        "expand_trace": expand_trace,
    }


def _build_unit_trace_base(unit_context: dict) -> dict:
    """Build the static trace envelope for one research unit."""
    return {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "run_trace_id": unit_context["trace_session"].trace_id,
        "combo": unit_context["combo"],
        "source": unit_context["source"],
        "query": unit_context["query"],
        "prepared_query": unit_context["prepared_trace"],
    }


def _finalize_unit_trace(
    unit_trace: dict,
    stage_bundle: dict,
    research_trace: dict,
    result: ComboSourceResult,
    unit_metrics: dict,
) -> None:
    """Attach stage traces, metrics, and outputs to a unit trace."""
    unit_trace["document"] = stage_bundle["document"]
    unit_trace["stages"] = {
        "search": stage_bundle["search_trace"],
        "rerank": stage_bundle["rerank_trace"],
        "expand": stage_bundle["expand_trace"],
        "research": research_trace,
    }
    unit_trace["metrics"] = unit_metrics
    unit_trace["outputs"] = {
        "research_iterations": result["research_iterations"],
        "findings_count": len(result.get("findings", [])),
        "chunk_count": result["chunk_count"],
        "total_tokens": result["total_tokens"],
    }


def _research_single_unit(
    unit_context: dict,
) -> ComboSourceResult:
    """Run the full retrieval pipeline for one unit.

    Opens its own DB connection for thread safety, checks
    document size to decide between full-load and search
    paths, then runs the research agent loop.

    Params:
        unit_context: Dict containing combo, source, prepared
            query, llm, and trace metadata

    Returns:
        ComboSourceResult with research output

    Example:
        >>> result = _research_single_unit(
        ...     combo, source, prepared, llm, query
        ... )
    """
    logger = get_stage_logger(__name__, STAGE)
    start_time = perf_counter()
    unit_trace = _build_unit_trace_base(unit_context)
    unit_conn = get_connection()
    try:
        research_metrics: dict = {}
        research_trace: dict = {}
        stage_bundle = _load_unit_stage_bundle(
            unit_conn,
            unit_context["source"],
            unit_context["prepared"],
            unit_context["query"],
            unit_context["llm"],
        )
        logger.info(
            "Researching %s for %s %s (%d tokens)",
            unit_context["source"]["data_source"],
            unit_context["combo"]["bank"],
            unit_context["combo"]["period"],
            stage_bundle["document"]["document_tokens"],
        )
        result = research_combo_source(
            unit_conn,
            unit_context["prepared"],
            stage_bundle["expanded"],
            unit_context["combo"],
            unit_context["source"],
            context={"llm": unit_context["llm"]},
            metrics=research_metrics,
            trace=research_trace,
            initial_chunk_origins=stage_bundle["expand_trace"].get(
                "chunk_origins",
                {},
            ),
        )
        unit_elapsed = perf_counter() - start_time
        stage_bundle["expand_metrics"]["evidence_catalog"] = (
            _build_evidence_catalog(
                _resolve_evidence_chunks(
                    stage_bundle["expanded"],
                    research_trace,
                )
            )
        )
        unit_metrics = _build_unit_metrics(
            stage_bundle["document"]["document_tokens"],
            stage_bundle["results"],
            stage_bundle["search_metrics"],
            stage_bundle["rerank_metrics"],
            stage_bundle["expand_metrics"],
            research_metrics,
            unit_elapsed,
        )
        result["metrics"] = unit_metrics
        _finalize_unit_trace(
            unit_trace,
            stage_bundle,
            research_trace,
            result,
            unit_metrics,
        )
        trace_path = write_trace_json(
            get_source_trace_path(
                unit_context["trace_session"],
                unit_context["trace_index"],
                unit_context["combo"],
                unit_context["source"],
            ),
            unit_trace,
        )
        result["trace_path"] = trace_path
        logger.info(
            "[%s] unit completed in %.1fs — source=%s, kept=%d, "
            "expanded=%d, research_iterations=%d, final_confidence=%.2f",
            STAGE,
            unit_elapsed,
            unit_context["source"]["data_source"],
            stage_bundle["rerank_metrics"].get(
                "kept",
                len(stage_bundle["results"]),
            ),
            stage_bundle["expand_metrics"].get(
                "chunks_after",
                len(stage_bundle["expanded"]),
            ),
            research_metrics.get("iterations", 0),
            research_metrics.get("final_confidence", 0.0),
        )
        return result
    finally:
        unit_conn.close()


def _build_run_trace_payload(
    runtime: dict,
    consolidated: ConsolidatedResult,
    pipeline_metrics: dict,
) -> dict:
    """Build the run-level retrieval trace document."""
    return {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": runtime["trace_session"].trace_id,
        "created_at": runtime["trace_session"].created_at,
        "query": runtime["query"],
        "combos": runtime["combos"],
        "sources": runtime["sources"] or [],
        "prepared_query": runtime["prepared_trace"],
        "document_resolution": {
            "wall_time_seconds": round(runtime["prework"][1], 3),
            "research_units": [
                {"combo": combo, "source": source}
                for combo, source in runtime["research_units"]
            ],
        },
        "source_traces": [
            {
                "combo": result["combo"],
                "source": result["source"],
                "trace_path": result.get("trace_path", ""),
            }
            for result in runtime["combo_results"]
        ],
        "outputs": {
            "consolidated_response": consolidated["consolidated_response"],
            "key_findings": consolidated["key_findings"],
            "data_gaps": consolidated["data_gaps"],
            "citation_warnings": consolidated.get("citation_warnings", []),
            "coverage_audit": consolidated.get("coverage_audit", ""),
            "uncited_ref_ids": consolidated.get("uncited_ref_ids", []),
            "unincorporated_findings": consolidated.get(
                "unincorporated_findings",
                [],
            ),
        },
        "metrics": pipeline_metrics,
    }


def _finalize_run(
    runtime: dict,
    llm: LLMClient,
    on_chunk: Callable[[str], None] | None = None,
) -> tuple[ConsolidatedResult, dict]:
    """Consolidate results, validate citations, and write the run trace."""
    consolidation_metrics: dict = {}
    citation_metrics: dict = {}
    consolidated = consolidate_results(
        runtime["query"],
        runtime["combo_results"],
        llm,
        metrics=consolidation_metrics,
        on_chunk=on_chunk,
    )
    consolidated = validate_consolidated_citations(
        consolidated,
        metrics=citation_metrics,
    )
    total_elapsed = perf_counter() - runtime["start_time"]
    prework_metrics = {
        "query_prep": runtime["prework"][3],
        "document_resolution_seconds": runtime["prework"][1],
        "prep_resolution_wall_seconds": runtime["prework"][5],
        "prep_resolution_overlap_seconds": runtime["prework"][6],
    }
    research_metrics = {
        "parallel_research_wall_seconds": runtime["research_run"][1],
        "sequential_unit_seconds": runtime["research_run"][2],
        "max_workers": runtime["research_run"][3],
    }
    pipeline_metrics = _build_orchestrator_metrics(
        total_elapsed,
        prework_metrics,
        runtime["research_units"],
        runtime["combo_results"],
        research_metrics,
        consolidation_metrics,
    )
    pipeline_metrics["citation_validation"] = citation_metrics
    consolidated["metrics"] = pipeline_metrics
    run_trace_path = write_trace_json(
        get_run_trace_path(runtime["trace_session"]),
        _build_run_trace_payload(
            runtime,
            consolidated,
            pipeline_metrics,
        ),
    )
    consolidated["trace_id"] = runtime["trace_session"].trace_id
    consolidated["trace_path"] = run_trace_path
    return consolidated, citation_metrics


def run_retrieval(
    query: str,
    combos: list[ComboSpec],
    sources: list[str] | None,
    _conn,
    llm: LLMClient,
    on_chunk: Callable[[str], None] | None = None,
) -> ConsolidatedResult:
    """Execute the full retrieval pipeline end-to-end.

    Resolves document versions for each combo, prepares
    the query once, runs per-document research in parallel
    threads, and consolidates all results into a single
    response.

    Params:
        query: Raw user query text
        combos: Bank and period combinations to search
        sources: Optional data_source whitelist
        _conn: Retained for CLI compatibility; resolution now
            uses dedicated per-thread connections
        llm: Configured LLM client (thread-safe)
        on_chunk: Optional callback for streaming consolidation
            output chunks to the caller

    Returns:
        ConsolidatedResult with synthesized response

    Example:
        >>> result = run_retrieval(
        ...     "CET1 ratio for RBC",
        ...     [{"bank": "RBC", "period": "2026_Q1"}],
        ...     None, conn, llm,
        ... )
    """
    logger = get_stage_logger(__name__, STAGE)
    runtime = {
        "start_time": perf_counter(),
        "trace_session": start_trace_session(),
        "query": query,
        "combos": combos,
        "sources": sources,
    }
    runtime["prework"] = _run_prep_and_resolution(
        query,
        combos,
        sources,
        llm,
    )
    runtime["research_units"] = runtime["prework"][0]
    runtime["prepared"] = runtime["prework"][2]
    runtime["prepared_trace"] = runtime["prework"][4]

    logger.info(
        "Resolved %d research units across %d combos in %.1fs",
        len(runtime["research_units"]),
        len(combos),
        runtime["prework"][1],
    )
    logger.info(
        "Query prepared: %d sub-queries, %d keywords",
        len(runtime["prepared"]["sub_queries"]),
        len(runtime["prepared"]["keywords"]),
    )

    runtime["research_run"] = _run_research_units(
        runtime["research_units"],
        runtime["prepared"],
        llm,
        query,
        runtime["prepared_trace"],
        runtime["trace_session"],
    )
    runtime["combo_results"] = runtime["research_run"][0]

    logger.info(
        "All research complete, consolidating %d results",
        len(runtime["combo_results"]),
    )
    consolidated, citation_metrics = _finalize_run(
        runtime, llm, on_chunk=on_chunk
    )
    warning_count = citation_metrics.get("warning_count", 0)
    if warning_count:
        logger.warning(
            "Citation validation flagged %d issue(s)",
            warning_count,
        )
    logger.info("Trace saved to %s", consolidated["trace_path"])
    logger.info(
        "[%s] completed in %.1fs — prep+resolve=%.1fs "
        "(overlap=%.1fs), research_wall=%.1fs, sequential=%.1fs, "
        "overhead=%.1fs, workers=%d, consolidation=%.1fs, per_unit=%s",
        STAGE,
        consolidated["metrics"]["wall_time_seconds"],
        runtime["prework"][5],
        runtime["prework"][6],
        runtime["research_run"][1],
        runtime["research_run"][2],
        consolidated["metrics"]["parallel_overhead_seconds"],
        consolidated["metrics"]["max_workers"],
        consolidated["metrics"]["consolidation"].get(
            "wall_time_seconds",
            0.0,
        ),
        _format_unit_timings(runtime["combo_results"]),
    )
    return consolidated
