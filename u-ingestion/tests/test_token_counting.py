"""Tests for prompt token estimation helpers."""

from types import SimpleNamespace

from ingestion.utils import token_counting as mod


def test_count_text_tokens_uses_configured_tokenizer(monkeypatch):
    """Token counting uses TOKENIZER_MODEL with a cached encoder."""
    calls = []

    def fake_get_encoding(name):
        calls.append(name)
        return SimpleNamespace(encode=lambda text: text.split())

    monkeypatch.setattr(mod, "get_tokenizer_model", lambda: "gpt-5-mini")
    monkeypatch.setattr(
        mod,
        "tiktoken",
        SimpleNamespace(
            encoding_name_for_model=lambda model: "fake-enc",
            get_encoding=fake_get_encoding,
        ),
    )
    mod.get_encoder.cache_clear()

    assert mod.count_text_tokens("one two three") == 3
    assert calls == ["fake-enc"]


def test_count_text_tokens_falls_back_to_o200k(monkeypatch):
    """Unknown tokenizer models fall back to o200k_base."""
    monkeypatch.setattr(mod, "get_tokenizer_model", lambda: "unknown-model")
    monkeypatch.setattr(
        mod,
        "tiktoken",
        SimpleNamespace(
            encoding_name_for_model=lambda model: (_ for _ in ()).throw(
                KeyError(model)
            ),
            get_encoding=lambda name: SimpleNamespace(
                encode=lambda text: text.split()
            ),
        ),
    )
    mod.get_encoder.cache_clear()

    assert mod.count_text_tokens("one two") == 2


def test_count_message_tokens_renders_roles(monkeypatch):
    """Message counting wraps string content with role tags."""
    captured = {}

    def fake_count_text_tokens(text):
        captured["text"] = text
        return 7

    monkeypatch.setattr(mod, "count_text_tokens", fake_count_text_tokens)

    result = mod.count_message_tokens(
        [
            {"role": "system", "content": "System text"},
            {"role": "user", "content": "User text"},
            {"role": "tool", "content": {"skip": True}},
        ]
    )

    assert result == 7
    assert "<system>" in captured["text"]
    assert "System text" in captured["text"]
    assert "<user>" in captured["text"]
    assert "User text" in captured["text"]
    assert "skip" not in captured["text"]


def test_count_message_tokens_includes_tool_schema(monkeypatch):
    """Tool schemas are serialized and counted with the messages."""
    captured = {}

    def fake_count_text_tokens(text):
        captured["text"] = text
        return 42

    monkeypatch.setattr(mod, "count_text_tokens", fake_count_text_tokens)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_metadata",
                "description": "ExtractMetadata",
            },
        }
    ]
    result = mod.count_message_tokens(
        [{"role": "user", "content": "hi"}],
        tools=tools,
    )

    assert result == 42
    assert "extract_metadata" in captured["text"]
    assert "ExtractMetadata" in captured["text"]


def test_count_message_tokens_ignores_empty_tools(monkeypatch):
    """Empty tools list (or None) adds nothing to the rendered text."""
    captured = {}

    def fake_count_text_tokens(text):
        captured["text"] = text
        return 3

    monkeypatch.setattr(mod, "count_text_tokens", fake_count_text_tokens)

    result = mod.count_message_tokens(
        [{"role": "user", "content": "hello"}],
        tools=None,
    )

    assert result == 3
    assert "function" not in captured["text"]

    mod.count_message_tokens(
        [{"role": "user", "content": "hello"}],
        tools=[],
    )
    assert "function" not in captured["text"]
