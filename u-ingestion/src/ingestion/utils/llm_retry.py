"""Shared retry wrapper for LLM tool-calling requests.

Used by every enrichment stage that makes an LLM call with a
tool-schema response. Catches transport errors (rate limit, timeout,
connection, server) and structural errors (malformed tool calls,
missing fields, duplicate ids) and retries with linear backoff up
to the caller's configured attempt limit. Non-retryable exceptions
propagate immediately.
"""

import logging
import time
from typing import Any, Callable, TypeVar

import openai

from .llm_connector import LLMClient

logger = logging.getLogger(__name__)

RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    ValueError,
)

T = TypeVar("T")


def call_with_retry(
    llm: LLMClient,
    messages: list,
    prompt: dict[str, Any],
    parser: Callable[[dict], T],
    *,
    stage: str,
    context: str,
    max_retries: int,
    retry_delay: float,
    validator: Callable[[T], None] | None = None,
) -> T:
    """Call the LLM, parse the response, and retry on transient errors.

    Wraps llm.call + parser + optional validator in a bounded retry
    loop. Retries on transport errors (rate limit, timeout, connection,
    server) and structural errors raised by the parser or validator
    (ValueError). Retrying at temperature=0 may not recover from truly
    deterministic failures; operators can tune max_retries per stage
    via the stage-specific env vars.

    Params:
        llm: LLMClient instance
        messages: Message list for the API call
        prompt: Loaded prompt dict with tools and tool_choice
        parser: Callable that parses the raw response dict into type T
        stage: Pipeline stage name for model config lookup
        context: Log label for the request (":attempt_N" is appended)
        max_retries: Maximum number of attempts (1 means no retry)
        retry_delay: Base backoff delay in seconds (scales linearly)
        validator: Optional callable that validates the parsed result
            and raises ValueError when the batch is inconsistent

    Returns:
        T -- the parsed (and optionally validated) response

    Example:
        >>> metadata = call_with_retry(
        ...     llm,
        ...     messages,
        ...     prompt,
        ...     parser=_parse_metadata_response,
        ...     stage="doc_metadata",
        ...     context="doc_metadata:doc.pdf",
        ...     max_retries=3,
        ...     retry_delay=2.0,
        ... )
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.call(
                messages=messages,
                stage=stage,
                tools=prompt.get("tools"),
                tool_choice=prompt.get("tool_choice"),
                context=f"{context}:attempt_{attempt}",
            )
            parsed = parser(response)
            if validator is not None:
                validator(parsed)
            return parsed
        except RETRYABLE_ERRORS as exc:
            if attempt == max_retries:
                logger.error(
                    "%s failed after %d retries: %s",
                    context,
                    max_retries,
                    exc,
                )
                raise
            wait = retry_delay * attempt
            logger.warning(
                "%s retry %d/%d after %.1fs: %s",
                context,
                attempt,
                max_retries,
                wait,
                exc,
            )
            time.sleep(wait)
    raise RuntimeError(f"{context} exited retry loop without a response")
