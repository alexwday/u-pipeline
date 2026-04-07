"""Tests for LLM connector streaming and content extraction."""

from unittest.mock import MagicMock, patch

import pytest

from retriever.utils.llm_connector import LLMClient, extract_content_text

_MOD = "retriever.utils.llm_connector"


@pytest.fixture(autouse=True)
def _set_stage_env(monkeypatch):
    """Set required env vars for stage model config."""
    monkeypatch.setenv("CONSOLIDATION_MODEL", "gpt-5-mini")
    monkeypatch.setenv("CONSOLIDATION_MAX_TOKENS", "3500")
    monkeypatch.setenv("CONSOLIDATION_TEMPERATURE", "")
    monkeypatch.setenv("CONSOLIDATION_REASONING_EFFORT", "")


def _make_stream_chunk(content=None, usage=None):
    """Build a mock streaming chunk object."""
    chunk = MagicMock()
    if content is not None:
        delta = MagicMock()
        delta.content = content
        choice = MagicMock()
        choice.delta = delta
        chunk.choices = [choice]
    else:
        chunk.choices = []
    if usage is not None:
        usage_obj = MagicMock()
        usage_obj.prompt_tokens = usage.get("prompt_tokens", 0)
        usage_obj.completion_tokens = usage.get("completion_tokens", 0)
        usage_obj.total_tokens = usage.get("total_tokens", 0)
        chunk.usage = usage_obj
    else:
        chunk.usage = None
    return chunk


def _build_mock_llm_client(chunks):
    """Build an LLMClient with mocked OpenAI streaming."""
    with (
        patch(f"{_MOD}.get_auth_mode", return_value="api_key"),
        patch(f"{_MOD}.get_api_key", return_value="sk-test"),
        patch(f"{_MOD}.get_llm_endpoint", return_value="http://test"),
    ):
        with patch(f"{_MOD}.OpenAI") as mock_openai_cls:
            mock_openai = MagicMock()
            mock_openai_cls.return_value = mock_openai
            mock_openai.chat.completions.create.return_value = iter(chunks)

            client = LLMClient()
            return client, mock_openai


def test_stream_yields_text_chunks():
    """Stream method yields content strings from response chunks."""
    chunks = [
        _make_stream_chunk(content="Hello "),
        _make_stream_chunk(content="world"),
        _make_stream_chunk(
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        ),
    ]
    client, _ = _build_mock_llm_client(chunks)

    gen = client.stream(
        messages=[{"role": "user", "content": "test"}],
        stage="consolidation",
    )
    collected = []
    usage = None
    try:
        while True:
            collected.append(next(gen))
    except StopIteration as stop:
        usage = stop.value

    assert collected == ["Hello ", "world"]
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 15


def test_stream_skips_empty_deltas():
    """Stream skips chunks with no content in delta."""
    chunks = [
        _make_stream_chunk(content=None),
        _make_stream_chunk(content="data"),
        _make_stream_chunk(content=""),
        _make_stream_chunk(
            usage={
                "prompt_tokens": 8,
                "completion_tokens": 2,
                "total_tokens": 10,
            },
        ),
    ]
    client, _ = _build_mock_llm_client(chunks)

    gen = client.stream(
        messages=[{"role": "user", "content": "test"}],
        stage="consolidation",
    )
    collected = list(gen)
    assert collected == ["data"]


def test_stream_empty_response():
    """Stream with no content chunks returns empty and zero usage."""
    chunks = [
        _make_stream_chunk(
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 0,
                "total_tokens": 5,
            },
        ),
    ]
    client, _ = _build_mock_llm_client(chunks)

    gen = client.stream(
        messages=[{"role": "user", "content": "test"}],
        stage="consolidation",
    )
    collected = []
    usage = None
    try:
        while True:
            collected.append(next(gen))
    except StopIteration as stop:
        usage = stop.value

    assert not collected
    assert usage["completion_tokens"] == 0


def test_stream_passes_correct_kwargs():
    """Stream passes stream=True and stream_options to the API."""
    chunks = [
        _make_stream_chunk(
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        ),
    ]
    client, mock_openai = _build_mock_llm_client(chunks)

    gen = client.stream(
        messages=[{"role": "user", "content": "hi"}],
        stage="consolidation",
        context="test_context",
    )
    list(gen)

    call_kwargs = mock_openai.chat.completions.create.call_args[1]
    assert call_kwargs["stream"] is True
    assert call_kwargs["stream_options"] == {"include_usage": True}
    assert call_kwargs["model"] == "gpt-5-mini"
    assert "tools" not in call_kwargs
    assert "tool_choice" not in call_kwargs


def test_extract_content_text_from_response():
    """Extract plain text from a standard LLM response dict."""
    response = {
        "choices": [
            {
                "message": {
                    "content": "The answer is 42.",
                },
            },
        ],
    }
    assert extract_content_text(response) == "The answer is 42."


def test_extract_content_text_empty_choices():
    """Return empty string when choices is empty."""
    assert extract_content_text({"choices": []}) == ""
    assert extract_content_text({}) == ""


def test_extract_content_text_none_content():
    """Return empty string when content is None."""
    response = {
        "choices": [
            {
                "message": {
                    "content": None,
                },
            },
        ],
    }
    assert extract_content_text(response) == ""
