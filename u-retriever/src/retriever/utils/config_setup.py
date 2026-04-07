"""Pipeline configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".env"
PROJECT_ROOT = ENV_PATH.parent


def load_config() -> None:
    """Load .env file into process environment.

    Safe to call multiple times — load_dotenv does not
    overwrite existing env vars by default.

    Params:
        None

    Returns:
        None

    Example:
        >>> load_config()
    """
    load_dotenv(ENV_PATH)


def _require_env(name: str) -> str:
    """Get required env var or raise. Params: name. Returns: str."""
    value = os.getenv(name, "")
    if not value:
        raise ValueError(f"{name} is required")
    return value


def get_auth_mode() -> str:
    """Get the authentication mode from AUTH_MODE env var.

    Params:
        None

    Returns:
        str — "oauth" or "api_key"

    Example:
        >>> get_auth_mode()
        "api_key"
    """
    mode = os.getenv("AUTH_MODE", "")
    if not mode:
        raise ValueError("AUTH_MODE is required")
    mode = mode.lower()
    if mode not in ("oauth", "api_key"):
        raise ValueError(
            f"AUTH_MODE must be 'oauth' or 'api_key', " f"got '{mode}'"
        )
    return mode


def get_oauth_config() -> dict:
    """Get OAuth configuration from environment variables.

    All fields are required when AUTH_MODE=oauth.

    Params:
        None

    Returns:
        dict with keys: token_endpoint, client_id,
        client_secret, scope (optional)

    Example:
        >>> cfg = get_oauth_config()
        >>> cfg["token_endpoint"]
        "https://auth.example.com/token"
    """
    config = {
        "token_endpoint": os.getenv("OAUTH_TOKEN_ENDPOINT", ""),
        "client_id": os.getenv("OAUTH_CLIENT_ID", ""),
        "client_secret": os.getenv("OAUTH_CLIENT_SECRET", ""),
        "scope": os.getenv("OAUTH_SCOPE", ""),
    }
    missing = [
        key
        for key in (
            "token_endpoint",
            "client_id",
            "client_secret",
        )
        if not config[key]
    ]
    if missing:
        raise ValueError(f"OAuth requires: {', '.join(missing)}")
    return config


def get_api_key() -> str:
    """Get the API key from OPENAI_API_KEY env var.

    Params:
        None

    Returns:
        str — the API key

    Example:
        >>> get_api_key()
        "sk-..."
    """
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY is required")
    return key


def get_llm_endpoint() -> str:
    """Get the LLM API base URL. Returns: str."""
    value = os.getenv("LLM_ENDPOINT", "")
    if not value:
        raise ValueError("LLM_ENDPOINT is required")
    return value


def get_stage_model_config(stage: str) -> dict:
    """Get model config for a pipeline stage.

    Reads {STAGE}_MODEL, {STAGE}_MAX_TOKENS, and
    {STAGE}_TEMPERATURE env vars. Stage name is
    uppercased automatically. MODEL and MAX_TOKENS are
    required. TEMPERATURE is optional — omit or leave
    blank for models that don't support it (e.g. o-series).

    Params:
        stage: Pipeline stage name
            (e.g. "startup", "extraction")

    Returns:
        dict with keys: model, max_tokens,
        temperature (float or None)

    Example:
        >>> get_stage_model_config("startup")
        {"model": "gpt-5-mini", "max_tokens": 50, "temperature": 0.0}
    """
    prefix = stage.upper()
    temp_raw = os.getenv(f"{prefix}_TEMPERATURE", "")
    temperature = float(temp_raw) if temp_raw else None
    reasoning_raw = os.getenv(f"{prefix}_REASONING_EFFORT", "")
    reasoning_effort = reasoning_raw if reasoning_raw else None
    return {
        "model": _require_env(f"{prefix}_MODEL"),
        "max_tokens": int(_require_env(f"{prefix}_MAX_TOKENS")),
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
    }


def get_database_config() -> dict:
    """Get PostgreSQL connection parameters from environment.

    Reads DB_HOST, DB_PORT, DB_NAME, DB_USER, and
    optional DB_PASSWORD.

    Params:
        None

    Returns:
        dict with keys: host, port, dbname, user, password

    Example:
        >>> get_database_config()
        {"host": "localhost", "port": "5432", ...}
    """
    return {
        "host": _require_env("DB_HOST"),
        "port": _require_env("DB_PORT"),
        "dbname": _require_env("DB_NAME"),
        "user": _require_env("DB_USER"),
        "password": os.getenv("DB_PASSWORD", ""),
    }


def get_database_schema() -> str:
    """Get the PostgreSQL schema name. Returns: str."""
    return _require_env("DB_SCHEMA")


def get_trace_root() -> Path:
    """Get the root directory for persisted retrieval traces. Returns: Path."""
    return Path(_require_env("RETRIEVAL_TRACE_ROOT"))


def get_embedding_model() -> str:
    """Get the embedding model name. Returns: str."""
    return _require_env("EMBEDDING_MODEL")


def get_embedding_dimensions() -> int:
    """Get the embedding vector dimensions. Returns: int."""
    return int(_require_env("EMBEDDING_DIMENSIONS"))


def get_search_top_k() -> int:
    """Get the vector search result limit. Returns: int."""
    return int(_require_env("SEARCH_TOP_K"))


def get_bm25_top_k() -> int:
    """Get the BM25 text search result limit. Returns: int."""
    return int(_require_env("SEARCH_BM25_TOP_K"))


def get_bm25_term_cap() -> int:
    """Get the max keywords/entities in a BM25 query. Returns: int."""
    return int(_require_env("SEARCH_BM25_TERM_CAP"))


def get_keyword_match_limit() -> int:
    """Get keyword array containment limit. Returns: int."""
    return int(_require_env("SEARCH_KEYWORD_MATCH_LIMIT"))


def get_entity_match_limit() -> int:
    """Get entity array containment limit. Returns: int."""
    return int(_require_env("SEARCH_ENTITY_MATCH_LIMIT"))


def get_small_doc_token_threshold() -> int:
    """Get token threshold for small-doc bypass. Returns: int."""
    return int(_require_env("SMALL_DOC_TOKEN_THRESHOLD"))


def get_rerank_preview_max_tokens() -> int:
    """Get max tokens for rerank preview snippets. Returns: int."""
    return int(_require_env("RERANK_PREVIEW_MAX_TOKENS"))


def get_rerank_candidate_limit() -> int:
    """Get max search candidates shown to rerank. Returns: int."""
    return int(_require_env("RERANK_CANDIDATE_LIMIT"))


def get_rerank_min_keep() -> int:
    """Get minimum candidates to keep after rerank. Returns: int."""
    return int(_require_env("RERANK_MIN_KEEP"))


def get_expand_section_threshold() -> int:
    """Get section-level expansion token threshold. Returns: int."""
    return int(_require_env("EXPAND_SECTION_TOKEN_THRESHOLD"))


def get_expand_subsection_threshold() -> int:
    """Get subsection expansion token threshold. Returns: int."""
    return int(_require_env("EXPAND_SUBSECTION_TOKEN_THRESHOLD"))


def get_expand_neighbor_count() -> int:
    """Get number of neighbor chunks to expand. Returns: int."""
    return int(_require_env("EXPAND_NEIGHBOR_COUNT"))


def get_research_max_iterations() -> int:
    """Get max iterative research loops. Returns: int."""
    return int(_require_env("RESEARCH_MAX_ITERATIONS"))


def get_research_additional_top_k() -> int:
    """Get top-k for additional research searches. Returns: int."""
    return int(_require_env("RESEARCH_ADDITIONAL_SEARCH_TOP_K"))


def get_orchestrator_max_workers() -> int:
    """Get max worker count for parallel research. Returns: int."""
    return int(_require_env("ORCHESTRATOR_MAX_WORKERS"))


def get_score_weights() -> dict[str, float]:
    """Get per-strategy score weights for fusion ranking.

    Returns:
        dict[str, float] -- strategy name to weight mapping

    Example:
        >>> weights = get_score_weights()
        >>> weights["content_vector"]
        0.25
    """
    return {
        "content_vector": float(_require_env("WEIGHT_CONTENT_VECTOR")),
        "hyde_vector": float(_require_env("WEIGHT_HYDE_VECTOR")),
        "subquery_vector": float(_require_env("WEIGHT_SUBQUERY_VECTOR")),
        "keyword_vector": float(_require_env("WEIGHT_KEYWORD_VECTOR")),
        "section_summary": float(_require_env("WEIGHT_SECTION_SUMMARY")),
        "bm25": float(_require_env("WEIGHT_BM25")),
        "keyword_array": float(_require_env("WEIGHT_KEYWORD_ARRAY")),
        "entity_array": float(_require_env("WEIGHT_ENTITY_ARRAY")),
    }
