"""Stage 1: Query preparation and embedding generation."""

from pathlib import Path
import re
from time import perf_counter

from ..models import PreparedQuery, QueryEmbeddings
from ..utils.config_setup import (
    get_embedding_dimensions,
    get_embedding_model,
)
from ..utils.llm_connector import (
    LLMClient,
    extract_tool_arguments,
    get_usage_metrics,
)
from ..utils.logging_setup import get_stage_logger
from ..utils.prompt_loader import load_prompt

STAGE = "1-QUERY_PREP"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_MAX_SUB_QUERIES = 3
_MAX_KEYWORDS = 8
_MAX_ENTITIES = 4
_PERCENT_RANGE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*[–-]\s*\d+(?:\.\d+)?%")
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?%")
_BASIS_POINTS_RE = re.compile(r"\b\d+(?:\.\d+)?\s*bps?\b", re.IGNORECASE)
_CURRENCY_AMOUNT_RE = re.compile(
    r"\b(?:CAD|USD|C\$|\$)\s*\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:million|billion|bn|mm))?\b",
    re.IGNORECASE,
)
_SCALED_AMOUNT_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:million|billion|bn|mm)\b",
    re.IGNORECASE,
)
_MULTISPACE_RE = re.compile(r"\s{2,}")


def _parse_query_response(response: dict) -> dict:
    """Extract tool call arguments from LLM response.

    Params: response (dict). Returns: dict.
    """
    parsed = extract_tool_arguments(response)
    required = (
        "rewritten_query",
        "sub_queries",
        "keywords",
        "entities",
        "hyde_answer",
    )
    missing = [f for f in required if f not in parsed]
    if missing:
        raise ValueError(f"Tool response missing fields: {', '.join(missing)}")
    return parsed


def _generate_query_embeddings(
    llm: LLMClient,
    rewritten: str,
    sub_queries: list[str],
    keywords: list[str],
    hyde: str,
) -> QueryEmbeddings:
    """Batch-embed all query facets in a single call.

    Params:
        llm: LLM client with embed capability
        rewritten: Rewritten query text
        sub_queries: Sub-query texts
        keywords: Keyword list to join
        hyde: Hypothetical answer text

    Returns:
        QueryEmbeddings with vectors for each facet

    Example:
        >>> emb = _generate_query_embeddings(
        ...     llm, "query", ["sq1"], ["kw"], "hyde"
        ... )
    """
    texts = [rewritten]
    texts.extend(sub_queries)
    texts.append(" ".join(keywords))
    texts.append(hyde)

    model = get_embedding_model()
    dimensions = get_embedding_dimensions()
    vectors = llm.embed(texts, model=model, dimensions=dimensions)

    rewritten_vec = vectors[0]
    sub_count = len(sub_queries)
    sub_vecs = vectors[1 : 1 + sub_count]
    keywords_vec = vectors[1 + sub_count]
    hyde_vec = vectors[2 + sub_count]

    return QueryEmbeddings(
        rewritten=rewritten_vec,
        sub_queries=sub_vecs,
        keywords=keywords_vec,
        hyde=hyde_vec,
    )


def _limit_unique_texts(
    values: list[str],
    max_items: int,
) -> list[str]:
    """Deduplicate and cap a list of text values."""
    limited: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        limited.append(normalized)
        if len(limited) >= max_items:
            break
    return limited


def _soften_hyde_answer(text: str) -> str:
    """Remove speculative numeric anchors from HYDE text.

    Params: text (str). Returns: str.
    """
    softened = _PERCENT_RANGE_RE.sub(
        "a reported percentage range",
        text,
    )
    softened = _PERCENT_RE.sub("a reported percentage", softened)
    softened = _BASIS_POINTS_RE.sub(
        "a reported basis-point change",
        softened,
    )
    softened = _CURRENCY_AMOUNT_RE.sub("a reported amount", softened)
    softened = _SCALED_AMOUNT_RE.sub("a reported amount", softened)
    return _MULTISPACE_RE.sub(" ", softened).strip()


def prepare_query(
    query: str,
    llm: LLMClient,
    metrics: dict | None = None,
    trace: dict | None = None,
) -> PreparedQuery:
    """Decompose a query and generate search embeddings.

    Loads the query_prep prompt, calls the LLM to rewrite
    and decompose the query, then batch-embeds all facets
    for downstream vector search.

    Params:
        query: Raw user query text
        llm: Configured LLM client

    Returns:
        PreparedQuery with text decomposition and embeddings

    Example:
        >>> result = prepare_query("CET1 ratio for RBC", llm)
        >>> result["rewritten_query"]
        "Common Equity Tier 1 capital ratio for ..."
    """
    logger = get_stage_logger(__name__, STAGE)
    logger.info("Preparing query: %s", query[:80])
    start_time = perf_counter()

    prompt = load_prompt("query_prep", prompts_dir=_PROMPTS_DIR)

    messages = []
    if prompt.get("system_prompt"):
        messages.append(
            {
                "role": "system",
                "content": prompt["system_prompt"],
            }
        )
    user_text = prompt["user_prompt"].replace("{user_input}", query)
    messages.append({"role": "user", "content": user_text})

    llm_start = perf_counter()
    response = llm.call(
        messages=messages,
        stage=prompt["stage"],
        tools=prompt.get("tools"),
        tool_choice=prompt.get("tool_choice"),
        context="query_prep",
    )
    llm_elapsed = perf_counter() - llm_start

    parsed = _parse_query_response(response)
    parsed["sub_queries"] = _limit_unique_texts(
        parsed["sub_queries"],
        _MAX_SUB_QUERIES,
    )
    parsed["keywords"] = _limit_unique_texts(
        parsed["keywords"],
        _MAX_KEYWORDS,
    )
    parsed["entities"] = _limit_unique_texts(
        parsed["entities"],
        _MAX_ENTITIES,
    )
    parsed["hyde_answer"] = _soften_hyde_answer(parsed["hyde_answer"])

    logger.debug(
        "Query decomposed: %d sub-queries, %d keywords, %d entities",
        len(parsed["sub_queries"]),
        len(parsed["keywords"]),
        len(parsed["entities"]),
    )

    embed_start = perf_counter()
    embeddings = _generate_query_embeddings(
        llm=llm,
        rewritten=parsed["rewritten_query"],
        sub_queries=parsed["sub_queries"],
        keywords=parsed["keywords"],
        hyde=parsed["hyde_answer"],
    )
    embed_elapsed = perf_counter() - embed_start
    usage = get_usage_metrics(response)
    total_elapsed = perf_counter() - start_time
    stage_metrics = {
        "wall_time_seconds": round(total_elapsed, 3),
        "llm_calls": 1,
        "embed_calls": 1,
        "llm_call_seconds": round(llm_elapsed, 3),
        "embed_call_seconds": round(embed_elapsed, 3),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "sub_queries": len(parsed["sub_queries"]),
        "keywords": len(parsed["keywords"]),
        "entities": len(parsed["entities"]),
    }
    if metrics is not None:
        metrics.update(stage_metrics)
    if trace is not None:
        trace.update(
            {
                "original_query": query,
                "rewritten_query": parsed["rewritten_query"],
                "sub_queries": parsed["sub_queries"],
                "keywords": parsed["keywords"],
                "entities": parsed["entities"],
                "hyde_answer": parsed["hyde_answer"],
                "timing": stage_metrics,
                "embedding_inputs": {
                    "rewritten_query": parsed["rewritten_query"],
                    "sub_queries": parsed["sub_queries"],
                    "keywords_text": " ".join(parsed["keywords"]),
                    "hyde_answer": parsed["hyde_answer"],
                },
            }
        )

    logger.info(
        "[%s] completed in %.1fs — llm=%.1fs, embed=%.1fs, "
        "sub_queries=%d, keywords=%d, entities=%d, "
        "prompt_tokens=%d, completion_tokens=%d",
        STAGE,
        total_elapsed,
        llm_elapsed,
        embed_elapsed,
        len(parsed["sub_queries"]),
        len(parsed["keywords"]),
        len(parsed["entities"]),
        usage["prompt_tokens"],
        usage["completion_tokens"],
    )

    return PreparedQuery(
        original_query=query,
        rewritten_query=parsed["rewritten_query"],
        sub_queries=parsed["sub_queries"],
        keywords=parsed["keywords"],
        entities=parsed["entities"],
        hyde_answer=parsed["hyde_answer"],
        embeddings=embeddings,
    )
