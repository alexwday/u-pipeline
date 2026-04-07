"""Stage 2: Multi-strategy search and score fusion."""

import re
from statistics import fmean
from time import perf_counter

from ..models import PreparedQuery, SearchResult
from ..utils.config_setup import (
    get_bm25_term_cap,
    get_bm25_top_k,
    get_entity_match_limit,
    get_keyword_match_limit,
    get_search_top_k,
)
from ..utils.logging_setup import get_stage_logger
from ..utils.postgres_connector import (
    load_full_document,
    load_section_content,
    search_by_bm25,
    search_by_content_vector,
    search_by_entity_containment,
    search_by_keyword_containment,
    search_by_keyword_vector,
    search_by_section_summary,
)
from ..utils.trace_store import snapshot_content_row, snapshot_search_result

STAGE = "2-SEARCH"

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+", re.IGNORECASE)


def _sanitize_bm25_term(term: str) -> str:
    """Normalize one BM25 term to a tsquery-friendly token string."""
    cleaned = _NON_ALNUM_RE.sub(" ", term.replace("%", " percentage "))
    return " ".join(cleaned.split())


def _build_bm25_query(
    prepared: PreparedQuery,
    filtered_entities: list[str],
) -> str:
    """Build a focused BM25 query from high-signal facets."""
    term_cap = get_bm25_term_cap()
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in [*prepared["keywords"], *filtered_entities]:
        cleaned = _sanitize_bm25_term(raw_term)
        normalized = cleaned.casefold()
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        if " " in cleaned:
            terms.append(f'"{cleaned}"')
        else:
            terms.append(cleaned)
        if len(terms) >= term_cap:
            break
    if terms:
        return " OR ".join(terms)
    fallback = _sanitize_bm25_term(prepared["rewritten_query"])
    if not fallback:
        return prepared["rewritten_query"]
    if " " in fallback:
        return f'"{fallback}"'
    return fallback


def _normalize_scores(
    results: list[dict],
    score_field: str,
    invert: bool = False,
) -> dict[str, float]:
    """Normalize a strategy's raw scores to 0.0-1.0.

    For vector distances (invert=True), lower is better,
    so score = 1 - (value / max_value). For rank-based
    scores (invert=False), higher is better, so
    score = value / max_value.

    Params:
        results: Raw search result dicts
        score_field: Key containing the raw score
        invert: True for distance metrics where
            lower is better

    Returns:
        dict mapping content_unit_id to normalized score

    Example:
        >>> _normalize_scores(
        ...     [{"content_unit_id": "a", "distance": 0.3}],
        ...     "distance", invert=True,
        ... )
        {"a": 0.7}
    """
    if not results:
        return {}
    values = [row[score_field] for row in results]
    max_val = max(values)
    if max_val == 0:
        return {row["content_unit_id"]: 1.0 for row in results}
    normalized = {}
    for row in results:
        raw = row[score_field]
        if invert:
            score = 1.0 - (raw / max_val)
        else:
            score = raw / max_val
        cuid = row["content_unit_id"]
        if cuid not in normalized or score > normalized[cuid]:
            normalized[cuid] = score
    return normalized


def _combine_strategies(
    strategy_results: list[tuple[str, dict[str, float]]],
    weights: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Merge per-strategy normalized scores by content unit.

    For each unique content_unit_id, accumulate weighted
    scores across all strategies.

    Params:
        strategy_results: List of (strategy_name,
            {content_unit_id: score}) tuples
        weights: Strategy name to weight mapping

    Returns:
        dict mapping content_unit_id to
        {"combined": float, strategy_name: float, ...}

    Example:
        >>> _combine_strategies(
        ...     [("bm25", {"a": 0.8})], {"bm25": 0.10}
        ... )
        {"a": {"combined": 0.08, "bm25": 0.8}}
    """
    combined: dict[str, dict[str, float]] = {}
    for strategy_name, scores in strategy_results:
        weight = weights.get(strategy_name, 0.0)
        for cuid, score in scores.items():
            if cuid not in combined:
                combined[cuid] = {"combined": 0.0}
            combined[cuid][strategy_name] = score
            combined[cuid]["combined"] += weight * score
    return combined


def _collect_raw_data(
    all_results: list[list[dict]],
) -> dict[str, dict]:
    """Build content_unit_id to full row data mapping.

    First occurrence of each content_unit_id wins.

    Params: all_results (list of result lists). Returns: dict.
    """
    raw_data: dict[str, dict] = {}
    for result_list in all_results:
        for row in result_list:
            cuid = row.get("content_unit_id", "")
            if cuid and cuid not in raw_data:
                raw_data[cuid] = row
    return raw_data


def _build_search_result(
    content_unit_id: str,
    combined: dict[str, float],
    raw_data: dict[str, dict],
) -> SearchResult:
    """Construct a SearchResult TypedDict from fused data.

    Params:
        content_unit_id: Content unit identifier
        combined: Score dict with "combined" and per-strategy
        raw_data: content_unit_id to row data mapping

    Returns:
        SearchResult TypedDict

    Example:
        >>> _build_search_result("cu_1", {"combined": 0.5}, {})
    """
    row = raw_data.get(content_unit_id, {})
    strategy_scores = {k: v for k, v in combined.items() if k != "combined"}
    return SearchResult(
        content_unit_id=content_unit_id,
        raw_content=row.get("raw_content", ""),
        chunk_id=row.get("chunk_id", ""),
        section_id=row.get("section_id", ""),
        page_number=row.get("page_number", 0),
        chunk_context=row.get("chunk_context", ""),
        chunk_header=row.get("chunk_header", ""),
        keywords=row.get("keywords", []),
        entities=row.get("entities", []),
        token_count=row.get("token_count", 0),
        score=combined.get("combined", 0.0),
        strategy_scores=strategy_scores,
    )


def _round_trace_score(value: float) -> float:
    """Round trace scores for stable JSON output."""
    return round(float(value), 6)


def _trace_hit_rows(
    hits: list[dict],
    normalized_scores: dict[str, float],
    raw_score_field: str = "",
) -> list[dict]:
    """Build structured trace rows for one search strategy."""
    traced_hits: list[dict] = []
    for index, row in enumerate(hits, start=1):
        content_unit_id = row.get("content_unit_id", "")
        entry = snapshot_content_row(row)
        entry["hit_position"] = index
        entry["normalized_score"] = _round_trace_score(
            normalized_scores.get(content_unit_id, 0.0)
        )
        if raw_score_field:
            entry[raw_score_field] = _round_trace_score(
                row.get(raw_score_field, 0.0)
            )
        traced_hits.append(entry)
    return traced_hits


def _trace_section_hits(section_hits: list[dict]) -> list[dict]:
    """Build structured trace rows for section-summary hits."""
    if not section_hits:
        return []
    max_distance = max(hit.get("distance", 0.0) for hit in section_hits)
    traced_hits: list[dict] = []
    for index, hit in enumerate(section_hits, start=1):
        distance = float(hit.get("distance", 0.0))
        normalized = (
            1.0 if max_distance == 0 else 1.0 - distance / max_distance
        )
        traced_hits.append(
            {
                "rank": index,
                "section_id": hit.get("section_id", ""),
                "title": hit.get("title", ""),
                "summary": hit.get("summary", ""),
                "distance": _round_trace_score(distance),
                "normalized_score": _round_trace_score(normalized),
            }
        )
    return traced_hits


def _trace_fused_results(results: list[SearchResult]) -> list[dict]:
    """Build structured trace rows for the fused ranked results."""
    traced_results: list[dict] = []
    for index, result in enumerate(results, start=1):
        entry = snapshot_search_result(result)
        entry["rank"] = index
        traced_results.append(entry)
    return traced_results


def _build_strategy_trace_entry(
    query_field: str,
    prepared: PreparedQuery,
    hits: list[dict],
    scores: dict[str, float],
    raw_score_field: str = "",
) -> dict:
    """Build trace payload for one strategy from a prepared-query field."""
    return {
        "query_text": prepared[query_field],
        "hits": _trace_hit_rows(hits, scores, raw_score_field),
    }


def _run_text_strategy(
    conn,
    doc_version_id: int,
    config: dict,
) -> tuple[list[dict], tuple[str, dict[str, float]], dict, dict]:
    """Execute a text-based strategy and build metrics plus trace."""
    start_time = perf_counter()
    hits = config["search_fn"](
        conn,
        doc_version_id,
        config["query_value"],
        config["limit"],
    )
    elapsed = perf_counter() - start_time
    raw_score_field = config.get("raw_score_field", "")
    if raw_score_field:
        scores = _normalize_scores(hits, raw_score_field)
    else:
        scores = {row["content_unit_id"]: 1.0 for row in hits}
    return (
        hits,
        (config["strategy_name"], scores),
        {
            "hits": len(hits),
            "unique_hits": len(scores),
            "elapsed_seconds": round(elapsed, 3),
        },
        {
            config["trace_key"]: config["trace_value"],
            "hits": _trace_hit_rows(hits, scores, raw_score_field),
        },
    )


def _run_vector_strategy_config(
    conn,
    doc_version_id: int,
    prepared: PreparedQuery,
    top_k: int,
    config: dict,
) -> tuple[list[dict], tuple[str, dict[str, float]], dict, dict]:
    """Execute one vector strategy and build metrics plus trace."""
    hits, strategy_scores, metric = _run_strategy(
        config["search_fn"],
        config["strategy_name"],
        "distance",
        conn,
        doc_version_id,
        config["embedding"],
        top_k,
        invert=True,
    )
    if config.get("query_field"):
        trace_entry = _build_strategy_trace_entry(
            config["query_field"],
            prepared,
            hits,
            strategy_scores[1],
            "distance",
        )
    else:
        trace_entry = {
            "query_terms": prepared["keywords"],
            "hits": _trace_hit_rows(
                hits,
                strategy_scores[1],
                "distance",
            ),
        }
    return hits, strategy_scores, metric, trace_entry


def _run_strategy(
    search_fn,
    strategy_name: str,
    score_field: str | None,
    *args,
    invert: bool = False,
) -> tuple[list[dict], tuple[str, dict[str, float]], dict]:
    """Execute one search strategy and return hits, scores, metrics."""
    start_time = perf_counter()
    hits = search_fn(*args)
    elapsed = perf_counter() - start_time
    if score_field is None:
        scores = {row["content_unit_id"]: 1.0 for row in hits}
    else:
        scores = _normalize_scores(hits, score_field, invert=invert)
    return (
        hits,
        (strategy_name, scores),
        {
            "hits": len(hits),
            "unique_hits": len(scores),
            "elapsed_seconds": round(elapsed, 3),
        },
    )


def _run_vector_strategies(
    conn,
    doc_version_id: int,
    prepared: PreparedQuery,
    top_k: int,
    trace: dict | None = None,
) -> tuple[
    list[list[dict]],
    list[tuple[str, dict[str, float]]],
    dict[str, dict],
]:
    """Execute vector-based search strategies.

    Runs content vector (rewritten, HyDE, sub-queries),
    keyword vector, and section summary searches.

    Params:
        conn: psycopg2 connection
        doc_version_id: Target document version
        prepared: Decomposed query with embeddings
        top_k: Maximum results per strategy

    Returns:
        tuple of (raw result lists, strategy score tuples)

    Example:
        >>> raw, strats = _run_vector_strategies(
        ...     conn, 38, prepared, 20
        ... )
    """
    state = {
        "all_raw": [],
        "strategies": [],
        "strategy_metrics": {},
        "strategy_trace": {},
    }
    configs = (
        {
            "strategy_name": "content_vector",
            "search_fn": search_by_content_vector,
            "embedding": prepared["embeddings"]["rewritten"],
            "query_field": "rewritten_query",
        },
        {
            "strategy_name": "hyde_vector",
            "search_fn": search_by_content_vector,
            "embedding": prepared["embeddings"]["hyde"],
            "query_field": "hyde_answer",
        },
        {
            "strategy_name": "keyword_vector",
            "search_fn": search_by_keyword_vector,
            "embedding": prepared["embeddings"]["keywords"],
        },
    )
    for config in configs:
        strategy_result = _run_vector_strategy_config(
            conn,
            doc_version_id,
            prepared,
            top_k,
            config,
        )
        state["all_raw"].append(strategy_result[0])
        state["strategies"].append(strategy_result[1])
        state["strategy_metrics"][config["strategy_name"]] = strategy_result[2]
        state["strategy_trace"][config["strategy_name"]] = strategy_result[3]

    sub_metrics = _run_subquery_vector_strategy(
        conn,
        doc_version_id,
        prepared["sub_queries"],
        prepared["embeddings"]["sub_queries"],
        top_k,
    )
    state["all_raw"].extend(sub_metrics["raw"])
    state["strategies"].extend(sub_metrics["strategies"])
    state["strategy_metrics"]["subquery_vector"] = sub_metrics["metrics"]
    state["strategy_trace"]["subquery_vector"] = sub_metrics["trace"]

    start_time = perf_counter()
    section_result = _run_section_strategy(
        conn,
        doc_version_id,
        prepared,
        top_k,
    )
    sec_raw, sec_scores, section_hits, sec_rows = section_result
    elapsed = perf_counter() - start_time
    state["all_raw"].append(sec_raw)
    state["strategies"].append(("section_summary", sec_scores))
    state["strategy_metrics"]["section_summary"] = {
        "section_hits": section_hits,
        "hits": len(sec_raw),
        "unique_hits": len(sec_scores),
        "elapsed_seconds": round(elapsed, 3),
    }
    state["strategy_trace"]["section_summary"] = {
        "query_text": prepared["rewritten_query"],
        "section_hits": _trace_section_hits(sec_rows),
        "content_hits": _trace_hit_rows(sec_raw, sec_scores),
    }

    if trace is not None:
        trace.update(state["strategy_trace"])

    return (
        state["all_raw"],
        state["strategies"],
        state["strategy_metrics"],
    )


def _run_subquery_vector_strategy(
    conn,
    doc_version_id: int,
    sub_queries: list[str],
    sub_embeddings: list[list[float]],
    top_k: int,
) -> dict[str, object]:
    """Execute sub-query vector search and aggregate metrics."""
    sub_count = len(sub_embeddings)
    raw_batches: list[list[dict]] = []
    strategies: list[tuple[str, dict[str, float]]] = []
    sub_hits = 0
    sub_elapsed = 0.0
    sub_unique_hits: set[str] = set()
    per_query_trace: list[dict] = []
    for query_text, sub_embedding in zip(sub_queries, sub_embeddings):
        hits, (_, scores), metric = _run_strategy(
            search_by_content_vector,
            "subquery_vector",
            "distance",
            conn,
            doc_version_id,
            sub_embedding,
            top_k,
            invert=True,
        )
        raw_batches.append(hits)
        if sub_count > 1:
            scores = {key: value / sub_count for key, value in scores.items()}
        strategies.append(("subquery_vector", scores))
        sub_hits += len(hits)
        sub_unique_hits.update(scores.keys())
        sub_elapsed += metric["elapsed_seconds"]
        per_query_trace.append(
            {
                "query_text": query_text,
                "hits": _trace_hit_rows(hits, scores, "distance"),
                "elapsed_seconds": round(metric["elapsed_seconds"], 3),
            }
        )
    return {
        "raw": raw_batches,
        "strategies": strategies,
        "metrics": {
            "calls": sub_count,
            "hits": sub_hits,
            "unique_hits": len(sub_unique_hits),
            "elapsed_seconds": round(sub_elapsed, 3),
        },
        "trace": {
            "queries": per_query_trace,
        },
    }


def _run_section_strategy(
    conn,
    doc_version_id: int,
    prepared: PreparedQuery,
    top_k: int,
) -> tuple[list[dict], dict[str, float], int, list[dict]]:
    """Execute section summary search and map to content units.

    Params:
        conn: psycopg2 connection
        doc_version_id: Target document version
        prepared: Decomposed query with embeddings
        top_k: Maximum section results

    Returns:
        tuple of (content rows, content_unit_id to score map,
        section hit count, raw section hits)

    Example:
        >>> rows, scores = _run_section_strategy(
        ...     conn, 38, prepared, 20
        ... )
    """
    section_hits = search_by_section_summary(
        conn,
        doc_version_id,
        prepared["embeddings"]["rewritten"],
        top_k,
    )
    content_rows: list[dict] = []
    for hit in section_hits:
        sid = hit.get("section_id", "")
        if sid:
            rows = load_section_content(conn, doc_version_id, sid)
            content_rows.extend(rows)

    scores: dict[str, float] = {}
    if not section_hits:
        return content_rows, scores, 0, section_hits

    distances = [h["distance"] for h in section_hits]
    max_dist = max(distances)
    sec_norm: dict[str, float] = {}
    for hit in section_hits:
        sid = hit["section_id"]
        if max_dist == 0:
            sec_norm[sid] = 1.0
        else:
            sec_norm[sid] = 1.0 - hit["distance"] / max_dist

    for hit in section_hits:
        sid = hit.get("section_id", "")
        sec_score = sec_norm.get(sid, 0.0)
        for cu_row in content_rows:
            if cu_row.get("section_id") == sid:
                cuid = cu_row["content_unit_id"]
                if cuid not in scores or sec_score > scores[cuid]:
                    scores[cuid] = sec_score

    return content_rows, scores, len(section_hits), section_hits


def _run_text_strategies(
    conn,
    doc_version_id: int,
    prepared: PreparedQuery,
    trace: dict | None = None,
) -> tuple[
    list[list[dict]],
    list[tuple[str, dict[str, float]]],
    dict[str, dict],
]:
    """Execute text-based search strategies.

    Runs BM25, keyword containment, and entity containment.

    Params:
        conn: psycopg2 connection
        doc_version_id: Target document version
        prepared: Decomposed query with embeddings

    Returns:
        tuple of (raw result lists, strategy score tuples)

    Example:
        >>> raw, strats = _run_text_strategies(
        ...     conn, 38, prepared
        ... )
    """
    all_raw: list[list[dict]] = []
    strategies: list[tuple[str, dict[str, float]]] = []
    strategy_metrics: dict[str, dict] = {}
    strategy_trace: dict[str, dict] = {}
    filtered_entities = prepared["entities"]
    bm25_query = _build_bm25_query(prepared, filtered_entities)
    configs = (
        {
            "strategy_name": "bm25",
            "search_fn": search_by_bm25,
            "query_value": bm25_query,
            "limit": get_bm25_top_k(),
            "trace_key": "query_text",
            "trace_value": bm25_query,
            "raw_score_field": "rank",
        },
        {
            "strategy_name": "keyword_array",
            "search_fn": search_by_keyword_containment,
            "query_value": prepared["keywords"],
            "limit": get_keyword_match_limit(),
            "trace_key": "query_terms",
            "trace_value": prepared["keywords"],
        },
        {
            "strategy_name": "entity_array",
            "search_fn": search_by_entity_containment,
            "query_value": filtered_entities,
            "limit": get_entity_match_limit(),
            "trace_key": "query_terms",
            "trace_value": filtered_entities,
        },
    )
    for config in configs:
        hits, strategy_score, metric, trace_entry = _run_text_strategy(
            conn,
            doc_version_id,
            config,
        )
        all_raw.append(hits)
        strategies.append(strategy_score)
        strategy_metrics[config["strategy_name"]] = metric
        strategy_trace[config["strategy_name"]] = trace_entry

    if trace is not None:
        trace.update(strategy_trace)

    return all_raw, strategies, strategy_metrics


def _summarize_score_distribution(
    results: list[SearchResult],
) -> dict[str, float]:
    """Calculate min, max, and mean score across fused results."""
    if not results:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    scores = [result["score"] for result in results]
    return {
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "mean": round(fmean(scores), 4),
    }


def _format_strategy_metrics(strategy_metrics: dict[str, dict]) -> str:
    """Render compact per-strategy hit/time metrics."""
    ordered = (
        "content_vector",
        "hyde_vector",
        "subquery_vector",
        "keyword_vector",
        "section_summary",
        "bm25",
        "keyword_array",
        "entity_array",
    )
    parts: list[str] = []
    for strategy_name in ordered:
        stats = strategy_metrics.get(strategy_name, {})
        if not stats:
            continue
        hits = stats.get("hits", 0)
        elapsed = stats.get("elapsed_seconds", 0.0)
        parts.append(f"{strategy_name}={hits}/{elapsed:.1f}s")
    return ", ".join(parts)


def _build_search_stage_metrics(
    doc_version_id: int,
    top_k: int,
    elapsed: float,
    all_raw: list[list[dict]],
    results: list[SearchResult],
    strategy_metrics: dict[str, dict],
) -> dict:
    """Build the summary metrics payload for the search stage."""
    total_hits = sum(len(result_batch) for result_batch in all_raw)
    return {
        "wall_time_seconds": round(elapsed, 3),
        "doc_version_id": doc_version_id,
        "top_k": top_k,
        "bm25_top_k": get_bm25_top_k(),
        "keyword_match_limit": get_keyword_match_limit(),
        "entity_match_limit": get_entity_match_limit(),
        "total_raw_hits": total_hits,
        "unique_results": len(results),
        "score_distribution": _summarize_score_distribution(results),
        "strategies": strategy_metrics,
    }


def _run_all_strategies(
    conn,
    doc_version_id: int,
    prepared: PreparedQuery,
    top_k: int,
    trace: dict | None = None,
) -> tuple[
    list[list[dict]], list[tuple[str, dict[str, float]]], dict[str, dict]
]:
    """Run all configured search strategies for one document."""
    vector_trace: dict = {}
    text_trace: dict = {}
    vec_raw, vec_strats, vector_metrics = _run_vector_strategies(
        conn,
        doc_version_id,
        prepared,
        top_k,
        trace=vector_trace,
    )
    txt_raw, txt_strats, text_metrics = _run_text_strategies(
        conn,
        doc_version_id,
        prepared,
        trace=text_trace,
    )
    if trace is not None:
        trace.update(vector_trace | text_trace)
    return (
        vec_raw + txt_raw,
        vec_strats + txt_strats,
        vector_metrics | text_metrics,
    )


def multi_strategy_search(
    conn,
    doc_version_id: int,
    prepared: PreparedQuery,
    weights: dict[str, float],
    metrics: dict | None = None,
    trace: dict | None = None,
) -> list[SearchResult]:
    """Run all search strategies and fuse scores.

    Executes 8 search strategies against a single document
    version, normalizes each strategy's scores to 0-1, then
    combines them using weighted fusion. Returns all unique
    content units sorted by combined score.

    Params:
        conn: psycopg2 connection
        doc_version_id: Target document version
        prepared: Decomposed query with embeddings
        weights: Per-strategy score weights

    Returns:
        list[SearchResult] sorted by score descending

    Example:
        >>> results = multi_strategy_search(
        ...     conn, 38, prepared, weights
        ... )
    """
    logger = get_stage_logger(__name__, STAGE)
    start_time = perf_counter()
    top_k = get_search_top_k()

    strategy_trace: dict = {}
    all_raw, all_strats, strategy_metrics = _run_all_strategies(
        conn,
        doc_version_id,
        prepared,
        top_k,
        trace=strategy_trace,
    )
    raw_data = _collect_raw_data(all_raw)
    combined = _combine_strategies(all_strats, weights)

    results = [
        _build_search_result(cuid, scores, raw_data)
        for cuid, scores in combined.items()
    ]
    results.sort(key=lambda r: r["score"], reverse=True)

    elapsed = perf_counter() - start_time
    stage_metrics = _build_search_stage_metrics(
        doc_version_id,
        top_k,
        elapsed,
        all_raw,
        results,
        strategy_metrics,
    )
    if metrics is not None:
        metrics.update(stage_metrics)
    if trace is not None:
        trace.update(
            {
                "doc_version_id": doc_version_id,
                "weights": {
                    key: _round_trace_score(value)
                    for key, value in weights.items()
                },
                "query_inputs": {
                    "rewritten_query": prepared["rewritten_query"],
                    "hyde_answer": prepared["hyde_answer"],
                    "sub_queries": prepared["sub_queries"],
                    "keywords": prepared["keywords"],
                    "entities": prepared["entities"],
                },
                "top_k": top_k,
                "strategy_traces": strategy_trace,
                "fused_results": _trace_fused_results(results),
            }
        )

    score_stats = stage_metrics["score_distribution"]
    logger.info(
        "[%s] completed in %.1fs — unique=%d, hits=%d, "
        "score[min=%.4f max=%.4f mean=%.4f], %s",
        STAGE,
        elapsed,
        len(results),
        stage_metrics["total_raw_hits"],
        score_stats["min"],
        score_stats["max"],
        score_stats["mean"],
        _format_strategy_metrics(strategy_metrics),
    )

    return results


def load_full_document_as_results(
    conn,
    doc_version_id: int,
) -> list[SearchResult]:
    """Load all content units as search results with score=1.0.

    Used for small documents below the token threshold where
    search is unnecessary. Every content unit gets a perfect
    score and an empty strategy_scores dict.

    Params:
        conn: psycopg2 connection
        doc_version_id: Target document version

    Returns:
        list[SearchResult] with score=1.0 for all units

    Example:
        >>> results = load_full_document_as_results(conn, 38)
    """
    logger = get_stage_logger(__name__, STAGE)
    rows = load_full_document(conn, doc_version_id)

    results = [
        SearchResult(
            content_unit_id=row["content_unit_id"],
            raw_content=row.get("raw_content", ""),
            chunk_id=row.get("chunk_id", ""),
            section_id=row.get("section_id", ""),
            page_number=row.get("page_number", 0),
            chunk_context=row.get("chunk_context", ""),
            chunk_header=row.get("chunk_header", ""),
            keywords=row.get("keywords", []),
            entities=row.get("entities", []),
            token_count=row.get("token_count", 0),
            score=1.0,
            strategy_scores={},
        )
        for row in rows
    ]

    logger.info(
        "Full document loaded: %d units (doc_version=%d)",
        len(results),
        doc_version_id,
    )

    return results
