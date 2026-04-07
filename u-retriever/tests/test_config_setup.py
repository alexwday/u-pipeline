"""Tests for configuration helpers."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from retriever.utils import config_setup
from retriever.utils.config_setup import _require_env as require_env


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

    assert config_setup.get_stage_model_config("startup") == {
        "model": "gpt-5-mini",
        "max_tokens": 100,
        "temperature": 0.5,
        "reasoning_effort": "low",
    }

    monkeypatch.setenv("STARTUP_TEMPERATURE", "")
    monkeypatch.setenv("STARTUP_REASONING_EFFORT", "")
    assert config_setup.get_stage_model_config("startup") == {
        "model": "gpt-5-mini",
        "max_tokens": 100,
        "temperature": None,
        "reasoning_effort": None,
    }


def test_database_config_and_schema(monkeypatch):
    """Read database connection parameters and schema."""
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "catalog")
    monkeypatch.setenv("DB_USER", "tester")
    monkeypatch.setenv("DB_PASSWORD", "secret")
    monkeypatch.setenv("DB_SCHEMA", "u_pipeline")

    assert config_setup.get_database_config() == {
        "host": "localhost",
        "port": "5432",
        "dbname": "catalog",
        "user": "tester",
        "password": "secret",
    }
    assert config_setup.get_database_schema() == "u_pipeline"

    monkeypatch.delenv("DB_SCHEMA")
    with pytest.raises(ValueError, match="DB_SCHEMA is required"):
        config_setup.get_database_schema()

    assert isinstance(config_setup.PROJECT_ROOT, Path)


def test_get_trace_root(monkeypatch, tmp_path):
    """Read trace root with override and project default."""
    default_path = config_setup.PROJECT_ROOT / "traces"
    monkeypatch.setenv("RETRIEVAL_TRACE_ROOT", str(default_path))
    assert config_setup.get_trace_root() == default_path

    override = tmp_path / "custom-traces"
    monkeypatch.setenv("RETRIEVAL_TRACE_ROOT", str(override))
    assert config_setup.get_trace_root() == override


def test_embedding_config_defaults_and_overrides(
    monkeypatch,
):
    """Read embedding model and dimensions with overrides."""
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    assert config_setup.get_embedding_model() == "text-embedding-3-large"
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    assert config_setup.get_embedding_model() == "text-embedding-3-small"

    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
    assert config_setup.get_embedding_dimensions() == 3072
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
    assert config_setup.get_embedding_dimensions() == 1536


def test_search_parameter_defaults(monkeypatch):
    """Read search limits with expected values."""
    monkeypatch.setenv("SEARCH_TOP_K", "20")
    assert config_setup.get_search_top_k() == 20

    monkeypatch.setenv("SEARCH_BM25_TOP_K", "20")
    assert config_setup.get_bm25_top_k() == 20

    monkeypatch.setenv("SEARCH_KEYWORD_MATCH_LIMIT", "50")
    assert config_setup.get_keyword_match_limit() == 50

    monkeypatch.setenv("SEARCH_ENTITY_MATCH_LIMIT", "10")
    assert config_setup.get_entity_match_limit() == 10

    monkeypatch.setenv("SEARCH_BM25_TERM_CAP", "6")
    assert config_setup.get_bm25_term_cap() == 6

    monkeypatch.setenv("ORCHESTRATOR_MAX_WORKERS", "4")
    assert config_setup.get_orchestrator_max_workers() == 4


def test_search_parameter_overrides(monkeypatch):
    """Override search limits via env vars."""
    monkeypatch.setenv("SEARCH_TOP_K", "30")
    assert config_setup.get_search_top_k() == 30

    monkeypatch.setenv("SEARCH_BM25_TOP_K", "15")
    assert config_setup.get_bm25_top_k() == 15

    monkeypatch.setenv("SEARCH_BM25_TERM_CAP", "10")
    assert config_setup.get_bm25_term_cap() == 10

    monkeypatch.setenv("SEARCH_KEYWORD_MATCH_LIMIT", "100")
    assert config_setup.get_keyword_match_limit() == 100

    monkeypatch.setenv("SEARCH_ENTITY_MATCH_LIMIT", "75")
    assert config_setup.get_entity_match_limit() == 75

    monkeypatch.setenv("ORCHESTRATOR_MAX_WORKERS", "6")
    assert config_setup.get_orchestrator_max_workers() == 6


def test_threshold_defaults(monkeypatch):
    """Read threshold values with expected values."""
    monkeypatch.setenv("SMALL_DOC_TOKEN_THRESHOLD", "4000")
    assert config_setup.get_small_doc_token_threshold() == 4000

    monkeypatch.setenv("RERANK_PREVIEW_MAX_TOKENS", "200")
    assert config_setup.get_rerank_preview_max_tokens() == 200

    monkeypatch.setenv("RERANK_CANDIDATE_LIMIT", "30")
    assert config_setup.get_rerank_candidate_limit() == 30

    monkeypatch.setenv("RERANK_MIN_KEEP", "10")
    assert config_setup.get_rerank_min_keep() == 10

    monkeypatch.setenv("EXPAND_SECTION_TOKEN_THRESHOLD", "6000")
    assert config_setup.get_expand_section_threshold() == 6000

    monkeypatch.setenv("EXPAND_SUBSECTION_TOKEN_THRESHOLD", "3000")
    assert config_setup.get_expand_subsection_threshold() == 3000

    monkeypatch.setenv("EXPAND_NEIGHBOR_COUNT", "4")
    assert config_setup.get_expand_neighbor_count() == 4


def test_threshold_overrides(monkeypatch):
    """Override threshold values via env vars."""
    monkeypatch.setenv("SMALL_DOC_TOKEN_THRESHOLD", "8000")
    assert config_setup.get_small_doc_token_threshold() == 8000

    monkeypatch.setenv("RERANK_PREVIEW_MAX_TOKENS", "500")
    assert config_setup.get_rerank_preview_max_tokens() == 500

    monkeypatch.setenv("RERANK_CANDIDATE_LIMIT", "12")
    assert config_setup.get_rerank_candidate_limit() == 12

    monkeypatch.setenv("RERANK_MIN_KEEP", "5")
    assert config_setup.get_rerank_min_keep() == 5

    monkeypatch.setenv("EXPAND_SECTION_TOKEN_THRESHOLD", "5000")
    assert config_setup.get_expand_section_threshold() == 5000

    monkeypatch.setenv("EXPAND_SUBSECTION_TOKEN_THRESHOLD", "2500")
    assert config_setup.get_expand_subsection_threshold() == 2500

    monkeypatch.setenv("EXPAND_NEIGHBOR_COUNT", "4")
    assert config_setup.get_expand_neighbor_count() == 4


def test_research_config_defaults(monkeypatch):
    """Read research loop parameters with expected values."""
    monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "3")
    assert config_setup.get_research_max_iterations() == 3

    monkeypatch.setenv("RESEARCH_ADDITIONAL_SEARCH_TOP_K", "10")
    assert config_setup.get_research_additional_top_k() == 10


def test_research_config_overrides(monkeypatch):
    """Override research parameters via env vars."""
    monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "5")
    assert config_setup.get_research_max_iterations() == 5

    monkeypatch.setenv("RESEARCH_ADDITIONAL_SEARCH_TOP_K", "25")
    assert config_setup.get_research_additional_top_k() == 25


def test_score_weights_defaults(monkeypatch):
    """Read score weights with expected values."""
    monkeypatch.setenv("WEIGHT_CONTENT_VECTOR", "0.25")
    monkeypatch.setenv("WEIGHT_HYDE_VECTOR", "0.20")
    monkeypatch.setenv("WEIGHT_SUBQUERY_VECTOR", "0.15")
    monkeypatch.setenv("WEIGHT_KEYWORD_VECTOR", "0.10")
    monkeypatch.setenv("WEIGHT_SECTION_SUMMARY", "0.10")
    monkeypatch.setenv("WEIGHT_BM25", "0.12")
    monkeypatch.setenv("WEIGHT_KEYWORD_ARRAY", "0.07")
    monkeypatch.setenv("WEIGHT_ENTITY_ARRAY", "0.01")

    weights = config_setup.get_score_weights()
    assert weights["content_vector"] == 0.25
    assert weights["hyde_vector"] == 0.20
    assert weights["subquery_vector"] == 0.15
    assert weights["keyword_vector"] == 0.10
    assert weights["section_summary"] == 0.10
    assert weights["bm25"] == 0.12
    assert weights["keyword_array"] == 0.07
    assert weights["entity_array"] == 0.01


def test_score_weights_overrides(monkeypatch):
    """Override score weights via env vars."""
    monkeypatch.setenv("WEIGHT_CONTENT_VECTOR", "0.30")
    monkeypatch.setenv("WEIGHT_HYDE_VECTOR", "0.20")
    monkeypatch.setenv("WEIGHT_SUBQUERY_VECTOR", "0.15")
    monkeypatch.setenv("WEIGHT_KEYWORD_VECTOR", "0.10")
    monkeypatch.setenv("WEIGHT_SECTION_SUMMARY", "0.10")
    monkeypatch.setenv("WEIGHT_BM25", "0.05")
    monkeypatch.setenv("WEIGHT_KEYWORD_ARRAY", "0.07")
    monkeypatch.setenv("WEIGHT_ENTITY_ARRAY", "0.01")

    weights = config_setup.get_score_weights()
    assert weights["content_vector"] == 0.30
    assert weights["bm25"] == 0.05


def test_env_example_matches_live_config_surface():
    """Keep .env.example aligned with config variables."""
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
        "QUERY_PREP_MODEL",
        "QUERY_PREP_MAX_TOKENS",
        "QUERY_PREP_TEMPERATURE",
        "QUERY_PREP_REASONING_EFFORT",
        "RERANK_MODEL",
        "RERANK_MAX_TOKENS",
        "RERANK_TEMPERATURE",
        "RERANK_REASONING_EFFORT",
        "RESEARCH_MODEL",
        "RESEARCH_MAX_TOKENS",
        "RESEARCH_TEMPERATURE",
        "RESEARCH_REASONING_EFFORT",
        "CONSOLIDATION_MODEL",
        "CONSOLIDATION_MAX_TOKENS",
        "CONSOLIDATION_TEMPERATURE",
        "CONSOLIDATION_REASONING_EFFORT",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "DB_SCHEMA",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "SEARCH_TOP_K",
        "SEARCH_BM25_TOP_K",
        "SEARCH_KEYWORD_MATCH_LIMIT",
        "SEARCH_ENTITY_MATCH_LIMIT",
        "ORCHESTRATOR_MAX_WORKERS",
        "WEIGHT_CONTENT_VECTOR",
        "WEIGHT_HYDE_VECTOR",
        "WEIGHT_SUBQUERY_VECTOR",
        "WEIGHT_KEYWORD_VECTOR",
        "WEIGHT_SECTION_SUMMARY",
        "WEIGHT_BM25",
        "WEIGHT_KEYWORD_ARRAY",
        "WEIGHT_ENTITY_ARRAY",
        "SMALL_DOC_TOKEN_THRESHOLD",
        "RERANK_PREVIEW_MAX_TOKENS",
        "RERANK_CANDIDATE_LIMIT",
        "RERANK_MIN_KEEP",
        "EXPAND_SECTION_TOKEN_THRESHOLD",
        "EXPAND_SUBSECTION_TOKEN_THRESHOLD",
        "EXPAND_NEIGHBOR_COUNT",
        "RESEARCH_MAX_ITERATIONS",
        "RESEARCH_ADDITIONAL_SEARCH_TOP_K",
    }

    assert expected_present <= keys
