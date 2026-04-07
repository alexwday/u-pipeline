"""Helpers for estimating prompt token usage."""

from functools import lru_cache

import tiktoken

from .config_setup import get_tokenizer_model


@lru_cache(maxsize=None)
def _get_encoder(model: str) -> tiktoken.Encoding:
    """Get a tokenizer encoder for token estimation.

    Params:
        model: Model or encoding name

    Returns:
        tiktoken.Encoding -- cached encoder instance
    """
    try:
        encoding_name = tiktoken.encoding_name_for_model(model)
    except KeyError:
        encoding_name = "o200k_base"
    return tiktoken.get_encoding(encoding_name)


def count_text_tokens(text: str) -> int:
    """Count tokens for plain text using the configured tokenizer model.

    Params:
        text: Text to tokenize

    Returns:
        int -- estimated token count
    """
    encoder = _get_encoder(get_tokenizer_model())
    return len(encoder.encode(text))


def count_message_tokens(messages: list[dict]) -> int:
    """Count tokens for a list of chat messages.

    Params:
        messages: Chat message dicts with role/content

    Returns:
        int -- estimated token count across rendered messages
    """
    rendered_parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        rendered_parts.append(f"<{role}>\n{content}\n</{role}>")
    return count_text_tokens("\n\n".join(rendered_parts))


# ------------------------------------------------------------------
# Public aliases for testing
# ------------------------------------------------------------------
get_encoder = _get_encoder
