"""Tests for configuration helpers."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from ingestion.utils import config_setup
from ingestion.utils.config_setup import _require_env as require_env


def test_load_config_calls_load_dotenv(monkeypatch):
    """Delegate .env loading to python-dotenv."""
    load_dotenv = Mock()
    monkeypatch.setattr(config_setup, "load_dotenv", load_dotenv)

    config_setup.load_config()

    load_dotenv.assert_called_once_with(config_setup.ENV_PATH)


def test_require_env_and_auth_mode(monkeypatch):
    """Read required values and validate auth modes."""
    monkeypatch.setenv("AUTH_MODE", "API_KEY")

    assert require_env("AUTH_MODE") == "API_KEY"
    assert config_setup.get_auth_mode() == "api_key"

    monkeypatch.setenv("AUTH_MODE", "oauth")
    assert config_setup.get_auth_mode() == "oauth"

    monkeypatch.setenv("AUTH_MODE", "bad-mode")
    with pytest.raises(ValueError, match="AUTH_MODE must be"):
        config_setup.get_auth_mode()

    monkeypatch.delenv("AUTH_MODE")
    with pytest.raises(ValueError, match="AUTH_MODE is required"):
        config_setup.get_auth_mode()
    with pytest.raises(ValueError, match="MISSING is required"):
        require_env("MISSING")


def test_get_oauth_config(monkeypatch):
    """Require the mandatory OAuth settings."""
    monkeypatch.setenv("OAUTH_TOKEN_ENDPOINT", "https://auth.example.com")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OAUTH_SCOPE", "scope")

    assert config_setup.get_oauth_config() == {
        "token_endpoint": "https://auth.example.com",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "scope": "scope",
    }

    monkeypatch.delenv("OAUTH_CLIENT_SECRET")
    with pytest.raises(ValueError, match="OAuth requires"):
        config_setup.get_oauth_config()


def test_get_api_key_and_endpoint(monkeypatch):
    """Read required LLM credentials."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_ENDPOINT", "https://api.example.com/v1")

    assert config_setup.get_api_key() == "sk-test"
    assert config_setup.get_llm_endpoint() == "https://api.example.com/v1"

    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        config_setup.get_api_key()

    monkeypatch.delenv("LLM_ENDPOINT")
    with pytest.raises(ValueError, match="LLM_ENDPOINT is required"):
        config_setup.get_llm_endpoint()


def test_get_stage_model_config(monkeypatch):
    """Parse optional and required per-stage model settings."""
    monkeypatch.setenv("STARTUP_MODEL", "gpt-5-mini")
    monkeypatch.setenv("STARTUP_MAX_TOKENS", "100")
    monkeypatch.setenv("STARTUP_TEMPERATURE", "0.5")
    monkeypatch.setenv("STARTUP_REASONING_EFFORT", "low")
    monkeypatch.setenv("STARTUP_VERBOSITY", "high")

    assert config_setup.get_stage_model_config("startup") == {
        "model": "gpt-5-mini",
        "max_tokens": 100,
        "temperature": 0.5,
        "reasoning_effort": "low",
        "verbosity": "high",
    }

    monkeypatch.setenv("STARTUP_TEMPERATURE", "")
    monkeypatch.setenv("STARTUP_REASONING_EFFORT", "")
    monkeypatch.setenv("STARTUP_VERBOSITY", "")
    assert config_setup.get_stage_model_config("startup") == {
        "model": "gpt-5-mini",
        "max_tokens": 100,
        "temperature": None,
        "reasoning_effort": None,
        "verbosity": None,
    }

    monkeypatch.setenv("STARTUP_MAX_TOKENS", "bad")
    with pytest.raises(
        ValueError, match="STARTUP_MAX_TOKENS must be an integer"
    ):
        config_setup.get_stage_model_config("startup")


def test_get_tokenizer_model_default_and_override(monkeypatch):
    """Read the tokenizer model with an embedding-oriented default."""
    monkeypatch.delenv("TOKENIZER_MODEL", raising=False)
    assert config_setup.get_tokenizer_model() == "o200k_base"

    monkeypatch.setenv("TOKENIZER_MODEL", "text-embedding-3-small")
    assert config_setup.get_tokenizer_model() == "text-embedding-3-small"

    monkeypatch.setenv("TOKENIZER_MODEL", "   ")
    assert config_setup.get_tokenizer_model() == "o200k_base"


def test_get_data_source_path(monkeypatch, tmp_path):
    """Validate data source directories."""
    data_source = tmp_path / "sources"
    data_source.mkdir()
    monkeypatch.setenv("DATA_SOURCE_PATH", str(data_source))

    assert config_setup.get_data_source_path() == str(data_source)

    missing = tmp_path / "missing"
    monkeypatch.setenv("DATA_SOURCE_PATH", str(missing))
    with pytest.raises(ValueError, match="is not a directory"):
        config_setup.get_data_source_path()


def test_other_config_helpers(monkeypatch):
    """Read the remaining scalar configuration values."""
    monkeypatch.setenv("RETENTION_COUNT", "31")
    monkeypatch.setenv("ACCEPTED_FILETYPES", "pdf, xlsx ,, csv ")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "catalog")
    monkeypatch.setenv("DB_USER", "tester")
    monkeypatch.setenv("DB_PASSWORD", "secret")
    monkeypatch.setenv("DB_SCHEMA", "u_pipeline")

    assert config_setup.get_retention_count() == 31
    assert config_setup.get_accepted_filetypes() == frozenset(
        {"pdf", "xlsx", "csv"}
    )
    assert config_setup.get_database_config() == {
        "host": "localhost",
        "port": "5432",
        "dbname": "catalog",
        "user": "tester",
        "password": "secret",
    }
    assert config_setup.get_database_schema() == "u_pipeline"
    assert config_setup.get_document_cache_root() == (
        config_setup.PROJECT_ROOT / "document-cache"
    )

    monkeypatch.delenv("DB_SCHEMA")
    with pytest.raises(ValueError, match="DB_SCHEMA is required"):
        config_setup.get_database_schema()

    assert isinstance(config_setup.PROJECT_ROOT, Path)


def test_get_document_cache_root_from_env(monkeypatch, tmp_path):
    """Allow DOCUMENT_CACHE_ROOT to override the default location."""
    monkeypatch.setenv("DOCUMENT_CACHE_ROOT", str(tmp_path / "stage-cache"))

    assert config_setup.get_document_cache_root() == tmp_path / "stage-cache"


def test_get_non_current_version_retention_count(monkeypatch):
    """Read retention for stale document versions with a default."""
    monkeypatch.delenv("NON_CURRENT_VERSION_RETENTION_COUNT", raising=False)
    assert config_setup.get_non_current_version_retention_count() == 1

    monkeypatch.setenv("NON_CURRENT_VERSION_RETENTION_COUNT", "3")
    assert config_setup.get_non_current_version_retention_count() == 3


def test_get_max_workers_and_vision_dpi(monkeypatch):
    """Read parallel processing and vision settings."""
    monkeypatch.setenv("MAX_WORKERS", "8")
    monkeypatch.setenv("VISION_DPI_SCALE", "2.5")

    assert config_setup.get_max_workers() == 8
    assert config_setup.get_vision_dpi_scale() == 2.5

    monkeypatch.delenv("MAX_WORKERS")
    with pytest.raises(ValueError, match="MAX_WORKERS is required"):
        config_setup.get_max_workers()

    monkeypatch.delenv("VISION_DPI_SCALE")
    with pytest.raises(ValueError, match="VISION_DPI_SCALE is required"):
        config_setup.get_vision_dpi_scale()


def test_numeric_worker_and_rendering_config_validates_bounds(monkeypatch):
    """Reject non-positive worker counts and invalid rendering settings."""
    monkeypatch.setenv("MAX_WORKERS", "0")
    with pytest.raises(ValueError, match="MAX_WORKERS must be >= 1"):
        config_setup.get_max_workers()

    monkeypatch.setenv("EXTRACTION_PAGE_WORKERS", "0")
    with pytest.raises(
        ValueError,
        match="EXTRACTION_PAGE_WORKERS must be >= 1",
    ):
        config_setup.get_extraction_page_workers()

    monkeypatch.setenv("VISION_DPI_SCALE", "fast")
    with pytest.raises(ValueError, match="VISION_DPI_SCALE must be a number"):
        config_setup.get_vision_dpi_scale()

    monkeypatch.setenv("VISION_DPI_SCALE", "0")
    with pytest.raises(ValueError, match="VISION_DPI_SCALE must be > 0.0"):
        config_setup.get_vision_dpi_scale()


def test_get_xlsx_config(monkeypatch):
    """Read XLSX-specific config with defaults."""
    assert config_setup.get_xlsx_vision_max_retries() == 3
    assert config_setup.get_xlsx_vision_retry_delay() == 2.0

    monkeypatch.setenv("XLSX_VISION_MAX_RETRIES", "5")
    assert config_setup.get_xlsx_vision_max_retries() == 5


def test_get_processor_retry_config(monkeypatch):
    """Read PDF, DOCX, and PPTX retry defaults."""
    assert config_setup.get_pdf_vision_max_retries() == 3
    assert config_setup.get_pdf_vision_retry_delay() == 2.0

    assert config_setup.get_docx_vision_max_retries() == 3
    assert config_setup.get_docx_vision_retry_delay() == 2.0

    assert config_setup.get_pptx_vision_max_retries() == 3
    assert config_setup.get_pptx_vision_retry_delay() == 2.0

    assert config_setup.get_content_extraction_max_retries() == 3
    assert config_setup.get_content_extraction_retry_delay() == 2.0

    assert config_setup.get_doc_metadata_max_retries() == 3
    assert config_setup.get_doc_metadata_retry_delay() == 2.0

    assert config_setup.get_section_detection_max_retries() == 3
    assert config_setup.get_section_detection_retry_delay() == 2.0

    assert config_setup.get_section_summary_max_retries() == 3
    assert config_setup.get_section_summary_retry_delay() == 2.0

    assert config_setup.get_doc_summary_max_retries() == 3
    assert config_setup.get_doc_summary_retry_delay() == 2.0

    monkeypatch.setenv("PDF_VISION_MAX_RETRIES", "5")
    assert config_setup.get_pdf_vision_max_retries() == 5

    monkeypatch.setenv("PDF_VISION_MAX_RETRIES", "0")
    with pytest.raises(
        ValueError,
        match="PDF_VISION_MAX_RETRIES must be >= 1",
    ):
        config_setup.get_pdf_vision_max_retries()

    monkeypatch.setenv("PDF_VISION_RETRY_DELAY_SECONDS", "-1")
    with pytest.raises(
        ValueError,
        match="PDF_VISION_RETRY_DELAY_SECONDS must be >= 0.0",
    ):
        config_setup.get_pdf_vision_retry_delay()

    monkeypatch.setenv("CONTENT_EXTRACTION_MAX_RETRIES", "5")
    assert config_setup.get_content_extraction_max_retries() == 5
    monkeypatch.setenv("CONTENT_EXTRACTION_RETRY_DELAY_SECONDS", "0.5")
    assert config_setup.get_content_extraction_retry_delay() == 0.5


def test_get_chunking_embedding_token_limit_default_and_override(
    monkeypatch,
):
    """Read chunking embedding token limit with default and override."""
    monkeypatch.delenv("CHUNKING_EMBEDDING_TOKEN_LIMIT", raising=False)
    assert config_setup.get_chunking_embedding_token_limit() == 8192

    monkeypatch.setenv("CHUNKING_EMBEDDING_TOKEN_LIMIT", "4096")
    assert config_setup.get_chunking_embedding_token_limit() == 4096


def test_get_chunking_truncation_token_limit_default_and_override(
    monkeypatch,
):
    """Read chunking truncation token limit with default and override."""
    monkeypatch.delenv("CHUNKING_TRUNCATION_TOKEN_LIMIT", raising=False)
    assert config_setup.get_chunking_truncation_token_limit() == 80000

    monkeypatch.setenv("CHUNKING_TRUNCATION_TOKEN_LIMIT", "50000")
    assert config_setup.get_chunking_truncation_token_limit() == 50000


def test_get_chunking_max_retries_default_and_override(monkeypatch):
    """Read chunking max retries with default and override."""
    monkeypatch.delenv("CHUNKING_MAX_RETRIES", raising=False)
    assert config_setup.get_chunking_max_retries() == 2

    monkeypatch.setenv("CHUNKING_MAX_RETRIES", "5")
    assert config_setup.get_chunking_max_retries() == 5

    monkeypatch.setenv("CHUNKING_MAX_RETRIES", "-1")
    with pytest.raises(ValueError, match="CHUNKING_MAX_RETRIES must be >= 0"):
        config_setup.get_chunking_max_retries()


def test_get_chunking_md_batch_size_default_and_override(monkeypatch):
    """Read markdown chunking batch size with default and override."""
    monkeypatch.delenv("CHUNKING_MD_BATCH_SIZE", raising=False)
    assert config_setup.get_chunking_md_batch_size() == 100

    monkeypatch.setenv("CHUNKING_MD_BATCH_SIZE", "200")
    assert config_setup.get_chunking_md_batch_size() == 200


def test_get_chunking_xlsx_batch_size_default_and_override(monkeypatch):
    """Read XLSX chunking batch size with default and override."""
    monkeypatch.delenv("CHUNKING_XLSX_BATCH_SIZE", raising=False)
    assert config_setup.get_chunking_xlsx_batch_size() == 50

    monkeypatch.setenv("CHUNKING_XLSX_BATCH_SIZE", "75")
    assert config_setup.get_chunking_xlsx_batch_size() == 75

    monkeypatch.setenv("CHUNKING_XLSX_BATCH_SIZE", "0")
    with pytest.raises(
        ValueError,
        match="CHUNKING_XLSX_BATCH_SIZE must be >= 1",
    ):
        config_setup.get_chunking_xlsx_batch_size()


def test_get_chunking_xlsx_header_rows_default_and_override(monkeypatch):
    """Read XLSX chunking header rows with default and override."""
    monkeypatch.delenv("CHUNKING_XLSX_HEADER_ROWS", raising=False)
    assert config_setup.get_chunking_xlsx_header_rows() == 5

    monkeypatch.setenv("CHUNKING_XLSX_HEADER_ROWS", "10")
    assert config_setup.get_chunking_xlsx_header_rows() == 10


def test_get_chunking_xlsx_overlap_rows_default_and_override(monkeypatch):
    """Read XLSX chunking overlap rows with default and override."""
    monkeypatch.delenv("CHUNKING_XLSX_OVERLAP_ROWS", raising=False)
    assert config_setup.get_chunking_xlsx_overlap_rows() == 3

    monkeypatch.setenv("CHUNKING_XLSX_OVERLAP_ROWS", "6")
    assert config_setup.get_chunking_xlsx_overlap_rows() == 6


def test_enrichment_config_defaults_and_overrides(monkeypatch):
    """Read enrichment config with defaults and overrides."""
    monkeypatch.delenv("DOC_METADATA_CONTEXT_BUDGET", raising=False)
    assert config_setup.get_doc_metadata_context_budget() == 30000
    monkeypatch.setenv("DOC_METADATA_CONTEXT_BUDGET", "50000")
    assert config_setup.get_doc_metadata_context_budget() == 50000

    monkeypatch.delenv("SECTION_DETECTION_BATCH_BUDGET", raising=False)
    assert config_setup.get_section_detection_batch_budget() == 80000

    monkeypatch.delenv("SUBSECTION_TOKEN_THRESHOLD", raising=False)
    assert config_setup.get_subsection_token_threshold() == 15000

    monkeypatch.delenv("CONTENT_EXTRACTION_BATCH_BUDGET", raising=False)
    assert config_setup.get_content_extraction_batch_budget() == 30000

    monkeypatch.delenv("SECTION_SUMMARY_BATCH_BUDGET", raising=False)
    assert config_setup.get_section_summary_batch_budget() == 80000

    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert config_setup.get_embedding_model() == ("text-embedding-3-large")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    assert config_setup.get_embedding_model() == ("text-embedding-3-small")

    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    assert config_setup.get_embedding_dimensions() == 3072

    monkeypatch.delenv("EMBEDDING_BATCH_SIZE", raising=False)
    assert config_setup.get_embedding_batch_size() == 20


def test_env_example_matches_live_config_surface():
    """Keep .env.example aligned with live config variables."""
    env_example = config_setup.PROJECT_ROOT / ".env.example"
    keys = {
        line.split("=", 1)[0].strip()
        for line in env_example.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }

    expected_present = {
        "AUTH_MODE",
        "OPENAI_API_KEY",
        "OAUTH_TOKEN_ENDPOINT",
        "OAUTH_CLIENT_ID",
        "OAUTH_CLIENT_SECRET",
        "OAUTH_SCOPE",
        "LLM_ENDPOINT",
        "STARTUP_MODEL",
        "STARTUP_MAX_TOKENS",
        "STARTUP_TEMPERATURE",
        "STARTUP_REASONING_EFFORT",
        "STARTUP_VERBOSITY",
        "TOKENIZER_MODEL",
        "EXTRACTION_MODEL",
        "EXTRACTION_MAX_TOKENS",
        "EXTRACTION_TEMPERATURE",
        "EXTRACTION_REASONING_EFFORT",
        "EXTRACTION_VERBOSITY",
        "CHUNKING_MODEL",
        "CHUNKING_MAX_TOKENS",
        "CHUNKING_TEMPERATURE",
        "CHUNKING_REASONING_EFFORT",
        "CHUNKING_VERBOSITY",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "DB_SCHEMA",
        "DATA_SOURCE_PATH",
        "RETENTION_COUNT",
        "ACCEPTED_FILETYPES",
        "DOCUMENT_CACHE_ROOT",
        "NON_CURRENT_VERSION_RETENTION_COUNT",
        "MAX_WORKERS",
        "EXTRACTION_PAGE_WORKERS",
        "VISION_DPI_SCALE",
        "PDF_VISION_MAX_RETRIES",
        "PDF_VISION_RETRY_DELAY_SECONDS",
        "DOCX_VISION_MAX_RETRIES",
        "DOCX_VISION_RETRY_DELAY_SECONDS",
        "PPTX_VISION_MAX_RETRIES",
        "PPTX_VISION_RETRY_DELAY_SECONDS",
        "XLSX_VISION_MAX_RETRIES",
        "XLSX_VISION_RETRY_DELAY_SECONDS",
        "CHUNKING_EMBEDDING_TOKEN_LIMIT",
        "CHUNKING_TRUNCATION_TOKEN_LIMIT",
        "CHUNKING_MD_BATCH_SIZE",
        "CHUNKING_XLSX_BATCH_SIZE",
        "CHUNKING_XLSX_HEADER_ROWS",
        "CHUNKING_XLSX_OVERLAP_ROWS",
        "DOC_METADATA_MODEL",
        "DOC_METADATA_MAX_TOKENS",
        "DOC_METADATA_TEMPERATURE",
        "DOC_METADATA_REASONING_EFFORT",
        "DOC_METADATA_VERBOSITY",
        "DOC_METADATA_CONTEXT_BUDGET",
        "SECTION_DETECTION_MODEL",
        "SECTION_DETECTION_MAX_TOKENS",
        "SECTION_DETECTION_TEMPERATURE",
        "SECTION_DETECTION_REASONING_EFFORT",
        "SECTION_DETECTION_VERBOSITY",
        "SECTION_DETECTION_BATCH_BUDGET",
        "SUBSECTION_TOKEN_THRESHOLD",
        "CONTENT_EXTRACTION_MODEL",
        "CONTENT_EXTRACTION_MAX_TOKENS",
        "CONTENT_EXTRACTION_TEMPERATURE",
        "CONTENT_EXTRACTION_REASONING_EFFORT",
        "CONTENT_EXTRACTION_VERBOSITY",
        "CONTENT_EXTRACTION_BATCH_BUDGET",
        "SECTION_SUMMARY_MODEL",
        "SECTION_SUMMARY_MAX_TOKENS",
        "SECTION_SUMMARY_TEMPERATURE",
        "SECTION_SUMMARY_REASONING_EFFORT",
        "SECTION_SUMMARY_VERBOSITY",
        "SECTION_SUMMARY_BATCH_BUDGET",
        "DOC_SUMMARY_MODEL",
        "DOC_SUMMARY_MAX_TOKENS",
        "DOC_SUMMARY_TEMPERATURE",
        "DOC_SUMMARY_REASONING_EFFORT",
        "DOC_SUMMARY_VERBOSITY",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_BATCH_SIZE",
    }
    expected_absent = {
        "CLASSIFICATION_MODEL",
        "CLASSIFICATION_MAX_TOKENS",
        "CLASSIFICATION_TEMPERATURE",
        "CLASSIFICATION_REASONING_EFFORT",
        "CLASSIFICATION_VERBOSITY",
        "PAGE_CLASSIFICATION_MODEL",
        "PAGE_CLASSIFICATION_MAX_TOKENS",
        "PPTX_CLASSIFICATION_MODEL",
        "PPTX_CLASSIFICATION_MAX_TOKENS",
        "XLSX_CLASSIFICATION_MODEL",
        "XLSX_CLASSIFICATION_MAX_TOKENS",
        "DENSE_TABLE_DESCRIPTION_MODEL",
        "DENSE_TABLE_DESCRIPTION_MAX_TOKENS",
        "CONTENT_CHUNKING_MODEL",
        "CONTENT_CHUNKING_MAX_TOKENS",
    }

    assert expected_present <= keys
    assert expected_absent.isdisjoint(keys)
