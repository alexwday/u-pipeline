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


def _get_int_env(
    name: str,
    default: str | None = None,
    minimum: int | None = None,
) -> int:
    """Parse an integer env var with optional bounds.

    Params: name, default, minimum. Returns: int.
    """
    raw_value = (
        os.getenv(name) if default is None else os.getenv(name, default)
    )
    if raw_value is None:
        raw_value = _require_env(name)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer, got '{raw_value}'"
        ) from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _get_float_env(
    name: str,
    default: str | None = None,
    minimum: float | None = None,
    inclusive: bool = True,
) -> float:
    """Parse a float env var with optional bounds.

    Params: name, default, minimum, inclusive. Returns: float.
    """
    raw_value = (
        os.getenv(name) if default is None else os.getenv(name, default)
    )
    if raw_value is None:
        raw_value = _require_env(name)
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a number, got '{raw_value}'"
        ) from exc
    if minimum is not None:
        is_invalid = value < minimum or (not inclusive and value <= minimum)
        if is_invalid:
            comparator = ">=" if inclusive else ">"
            raise ValueError(
                f"{name} must be {comparator} {minimum}, got {value}"
            )
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
    verbosity_raw = os.getenv(f"{prefix}_VERBOSITY", "")
    verbosity = verbosity_raw if verbosity_raw else None
    return {
        "model": _require_env(f"{prefix}_MODEL"),
        "max_tokens": _get_int_env(f"{prefix}_MAX_TOKENS", minimum=1),
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "verbosity": verbosity,
    }


def get_tokenizer_model() -> str:
    """Get the model name used for token counting.

    Reads optional TOKENIZER_MODEL. Defaults to
    o200k_base so token counting stays stable without
    requiring network-fetched tokenizer assets. Override
    it with an embedding model name when you need exact
    embedding-token parity.

    Returns:
        str — tokenizer model name
    """
    value = os.getenv("TOKENIZER_MODEL", "").strip()
    if value:
        return value
    return "o200k_base"


def get_data_source_path() -> str:
    """Get the base path for data source folders.

    Reads DATA_SOURCE_PATH and validates it exists as a directory.

    Returns:
        str — absolute path to the data sources root

    Example:
        >>> get_data_source_path()
        "/data/sources"
    """
    path = _require_env("DATA_SOURCE_PATH")
    if not Path(path).is_dir():
        raise ValueError(f"DATA_SOURCE_PATH is not a directory: {path}")
    return path


def get_retention_count() -> int:
    """Get the retention count for logs and archives.

    Reads RETENTION_COUNT env var.

    Returns:
        int — number of files to keep

    Example:
        >>> get_retention_count()
        31
    """
    return int(_require_env("RETENTION_COUNT"))


def get_accepted_filetypes() -> frozenset:
    """Get accepted file extensions for ingestion.

    Reads ACCEPTED_FILETYPES as a comma-separated list.

    Returns:
        frozenset of lowercase extensions without dots

    Example:
        >>> get_accepted_filetypes()
        frozenset({'pdf', 'xlsx'})
    """
    raw = _require_env("ACCEPTED_FILETYPES")
    return frozenset(
        ext.strip().lower() for ext in raw.split(",") if ext.strip()
    )


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
    """Get the PostgreSQL schema name for pipeline tables.

    Reads DB_SCHEMA env var.

    Returns:
        str — schema name (e.g. "u_pipeline")

    Example:
        >>> get_database_schema()
        "u_pipeline"
    """
    return _require_env("DB_SCHEMA")


def get_document_cache_root() -> Path:
    """Get the root directory for persisted stage outputs.

    Reads optional DOCUMENT_CACHE_ROOT. Defaults to
    {PROJECT_ROOT}/document-cache when unset.

    Returns:
        Path — directory used for durable stage cache

    Example:
        >>> get_document_cache_root().name
        "document-cache"
    """
    value = os.getenv("DOCUMENT_CACHE_ROOT", "")
    if value:
        return Path(value)
    return PROJECT_ROOT / "document-cache"


def get_non_current_version_retention_count() -> int:
    """Get how many non-current versions to keep per file path.

    Reads optional NON_CURRENT_VERSION_RETENTION_COUNT.
    Defaults to 1 old version per file path.

    Returns:
        int — number of non-current versions to retain
    """
    return _get_int_env(
        "NON_CURRENT_VERSION_RETENTION_COUNT",
        default="1",
        minimum=0,
    )


# -----------------------------------------------------------------
# Pipeline processing
# -----------------------------------------------------------------


def get_max_workers() -> int:
    """Get the max worker threads for parallel file processing.

    Reads MAX_WORKERS env var.

    Returns:
        int — number of worker threads

    Example:
        >>> get_max_workers()
        4
    """
    return _get_int_env("MAX_WORKERS", minimum=1)


def get_extraction_page_workers() -> int:
    """Get max concurrent LLM calls per file during extraction.

    Reads optional EXTRACTION_PAGE_WORKERS. Defaults to 4.

    Returns: int — worker threads for page-level parallelism.
    """
    return _get_int_env(
        "EXTRACTION_PAGE_WORKERS",
        default="4",
        minimum=1,
    )


def get_vision_dpi_scale() -> float:
    """Get the DPI scale factor for vision page rendering.

    Reads VISION_DPI_SCALE env var. Higher values produce
    sharper images but larger payloads.

    Returns:
        float — DPI multiplier (e.g. 2.0)

    Example:
        >>> get_vision_dpi_scale()
        2.0
    """
    return _get_float_env(
        "VISION_DPI_SCALE",
        minimum=0.0,
        inclusive=False,
    )


# -----------------------------------------------------------------
# XLSX-specific config
# -----------------------------------------------------------------


def get_xlsx_vision_max_retries() -> int:
    """Get max retries for XLSX vision extraction calls.

    Returns: int — default 3.
    """
    return _get_int_env(
        "XLSX_VISION_MAX_RETRIES",
        default="3",
        minimum=1,
    )


def get_xlsx_vision_retry_delay() -> float:
    """Get base backoff delay for XLSX vision retries.

    Returns: float — default 2.0.
    """
    return _get_float_env(
        "XLSX_VISION_RETRY_DELAY_SECONDS",
        default="2.0",
        minimum=0.0,
    )


# -----------------------------------------------------------------
# PDF retry config
# -----------------------------------------------------------------


def get_pdf_vision_max_retries() -> int:
    """Get max retries for PDF vision extraction calls.

    Returns: int — default 3.
    """
    return _get_int_env(
        "PDF_VISION_MAX_RETRIES",
        default="3",
        minimum=1,
    )


def get_pdf_vision_retry_delay() -> float:
    """Get base backoff delay for PDF vision retries.

    Returns: float — default 2.0.
    """
    return _get_float_env(
        "PDF_VISION_RETRY_DELAY_SECONDS",
        default="2.0",
        minimum=0.0,
    )


# -----------------------------------------------------------------
# DOCX retry config
# -----------------------------------------------------------------


def get_docx_vision_max_retries() -> int:
    """Get max retries for DOCX vision extraction calls.

    Returns: int — default 3.
    """
    return _get_int_env(
        "DOCX_VISION_MAX_RETRIES",
        default="3",
        minimum=1,
    )


def get_docx_vision_retry_delay() -> float:
    """Get base backoff delay for DOCX vision retries.

    Returns: float — default 2.0.
    """
    return _get_float_env(
        "DOCX_VISION_RETRY_DELAY_SECONDS",
        default="2.0",
        minimum=0.0,
    )


# -----------------------------------------------------------------
# PPTX retry config
# -----------------------------------------------------------------


def get_pptx_vision_max_retries() -> int:
    """Get max retries for PPTX vision extraction calls.

    Returns: int — default 3.
    """
    return _get_int_env(
        "PPTX_VISION_MAX_RETRIES",
        default="3",
        minimum=1,
    )


def get_pptx_vision_retry_delay() -> float:
    """Get base backoff delay for PPTX vision retries.

    Returns: float — default 2.0.
    """
    return _get_float_env(
        "PPTX_VISION_RETRY_DELAY_SECONDS",
        default="2.0",
        minimum=0.0,
    )


# -----------------------------------------------------------------
# Chunking config
# -----------------------------------------------------------------


def get_chunking_embedding_token_limit() -> int:
    """Get the hard token limit per chunk for embedding.

    Defaults to 8192 (text-embedding-3-large limit).

    Returns: int — max tokens per assembled chunk.
    """
    return _get_int_env(
        "CHUNKING_EMBEDDING_TOKEN_LIMIT",
        default="8192",
        minimum=1,
    )


def get_chunking_truncation_token_limit() -> int:
    """Get the truncation threshold for oversized pages.

    Defaults to 80000.

    Returns: int — token count above which content is truncated.
    """
    return _get_int_env(
        "CHUNKING_TRUNCATION_TOKEN_LIMIT",
        default="80000",
        minimum=1,
    )


def get_chunking_max_retries() -> int:
    """Get max re-chunk attempts when chunks exceed token limit.

    Returns: int — default 2.
    """
    return _get_int_env(
        "CHUNKING_MAX_RETRIES",
        default="2",
        minimum=0,
    )


def get_chunking_md_batch_size() -> int:
    """Get lines per markdown chunking LLM batch.

    Returns: int — default 100.
    """
    return _get_int_env(
        "CHUNKING_MD_BATCH_SIZE",
        default="100",
        minimum=1,
    )


def get_chunking_xlsx_batch_size() -> int:
    """Get data rows per XLSX chunking LLM batch.

    Returns: int — default 50.
    """
    return _get_int_env(
        "CHUNKING_XLSX_BATCH_SIZE",
        default="50",
        minimum=1,
    )


def get_chunking_xlsx_header_rows() -> int:
    """Get initial sheet rows always included as context.

    Returns: int — default 5.
    """
    return _get_int_env(
        "CHUNKING_XLSX_HEADER_ROWS",
        default="5",
        minimum=0,
    )


def get_chunking_xlsx_overlap_rows() -> int:
    """Get trailing rows from the previous chunk for overlap context.

    Returns: int — default 3.
    """
    return _get_int_env(
        "CHUNKING_XLSX_OVERLAP_ROWS",
        default="3",
        minimum=0,
    )


# -----------------------------------------------------------------
# Enrichment config
# -----------------------------------------------------------------


def get_doc_metadata_context_budget() -> int:
    """Token budget for doc metadata extraction context.

    Returns: int — default 30000.
    """
    return _get_int_env(
        "DOC_METADATA_CONTEXT_BUDGET",
        default="30000",
        minimum=1000,
    )


def get_section_detection_batch_budget() -> int:
    """Token budget per batch for section detection.

    Returns: int — default 80000.
    """
    return _get_int_env(
        "SECTION_DETECTION_BATCH_BUDGET",
        default="80000",
        minimum=1000,
    )


def get_subsection_token_threshold() -> int:
    """Sections above this trigger subsection detection.

    Returns: int — default 15000.
    """
    return _get_int_env(
        "SUBSECTION_TOKEN_THRESHOLD",
        default="15000",
        minimum=1000,
    )


def get_content_extraction_batch_budget() -> int:
    """Token budget per batch for keyword/entity extraction.

    Lowered from 60000 to 30000 to cap batches at ~20 units,
    which dramatically reduces the rate of LLM positional
    drift producing duplicate unit_ids on large batches.

    Returns: int — default 30000.
    """
    return _get_int_env(
        "CONTENT_EXTRACTION_BATCH_BUDGET",
        default="30000",
        minimum=1000,
    )


def get_content_extraction_max_retries() -> int:
    """Get max retries for content extraction batch calls.

    Returns: int — default 3.
    """
    return _get_int_env(
        "CONTENT_EXTRACTION_MAX_RETRIES",
        default="3",
        minimum=1,
    )


def get_content_extraction_retry_delay() -> float:
    """Get base backoff delay for content extraction retries.

    Returns: float — default 2.0.
    """
    return _get_float_env(
        "CONTENT_EXTRACTION_RETRY_DELAY_SECONDS",
        default="2.0",
        minimum=0.0,
    )


def get_doc_metadata_max_retries() -> int:
    """Get max retries for doc_metadata LLM calls.

    Returns: int — default 3.
    """
    return _get_int_env(
        "DOC_METADATA_MAX_RETRIES",
        default="3",
        minimum=1,
    )


def get_doc_metadata_retry_delay() -> float:
    """Get base backoff delay for doc_metadata retries.

    Returns: float — default 2.0.
    """
    return _get_float_env(
        "DOC_METADATA_RETRY_DELAY_SECONDS",
        default="2.0",
        minimum=0.0,
    )


def get_section_detection_max_retries() -> int:
    """Get max retries for section_detection LLM calls.

    Returns: int — default 3.
    """
    return _get_int_env(
        "SECTION_DETECTION_MAX_RETRIES",
        default="3",
        minimum=1,
    )


def get_section_detection_retry_delay() -> float:
    """Get base backoff delay for section_detection retries.

    Returns: float — default 2.0.
    """
    return _get_float_env(
        "SECTION_DETECTION_RETRY_DELAY_SECONDS",
        default="2.0",
        minimum=0.0,
    )


def get_section_summary_max_retries() -> int:
    """Get max retries for section_summary LLM calls.

    Returns: int — default 3.
    """
    return _get_int_env(
        "SECTION_SUMMARY_MAX_RETRIES",
        default="3",
        minimum=1,
    )


def get_section_summary_retry_delay() -> float:
    """Get base backoff delay for section_summary retries.

    Returns: float — default 2.0.
    """
    return _get_float_env(
        "SECTION_SUMMARY_RETRY_DELAY_SECONDS",
        default="2.0",
        minimum=0.0,
    )


def get_doc_summary_max_retries() -> int:
    """Get max retries for doc_summary LLM calls.

    Returns: int — default 3.
    """
    return _get_int_env(
        "DOC_SUMMARY_MAX_RETRIES",
        default="3",
        minimum=1,
    )


def get_doc_summary_retry_delay() -> float:
    """Get base backoff delay for doc_summary retries.

    Returns: float — default 2.0.
    """
    return _get_float_env(
        "DOC_SUMMARY_RETRY_DELAY_SECONDS",
        default="2.0",
        minimum=0.0,
    )


def get_section_summary_batch_budget() -> int:
    """Token budget per batch for section summarization.

    Returns: int — default 80000.
    """
    return _get_int_env(
        "SECTION_SUMMARY_BATCH_BUDGET",
        default="80000",
        minimum=1000,
    )


def get_embedding_model() -> str:
    """Embedding model name.

    Returns: str — default "text-embedding-3-large".
    """
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")


def get_embedding_dimensions() -> int:
    """Embedding vector dimensions.

    Returns: int — default 3072.
    """
    return _get_int_env(
        "EMBEDDING_DIMENSIONS",
        default="3072",
        minimum=1,
    )


def get_embedding_batch_size() -> int:
    """Texts per embedding API call.

    Returns: int — default 20.
    """
    return _get_int_env(
        "EMBEDDING_BATCH_SIZE",
        default="20",
        minimum=1,
    )
