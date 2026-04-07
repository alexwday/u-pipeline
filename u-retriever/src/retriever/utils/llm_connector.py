"""LLM connector with swappable auth (OAuth or API key)."""

import json
import logging
from collections.abc import Generator
from copy import deepcopy
from typing import Any

from openai import OpenAI

from .config_setup import (
    get_api_key,
    get_auth_mode,
    get_llm_endpoint,
    get_oauth_config,
    get_stage_model_config,
)
from .oauth_connector import OAuthClient

logger = logging.getLogger(__name__)

_HEALTH_CHECK_PROMPT: dict[str, Any] = {
    "stage": "startup",
    "system_prompt": (
        "You are a startup health check agent. Confirm "
        "connectivity by calling the provided tool."
    ),
    "user_prompt": "Respond with status ok.",
    "tool_choice": {
        "type": "function",
        "function": {"name": "ping"},
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "ping",
                "description": (
                    "Use this tool when responding to the startup "
                    "health check. It returns the status field needed "
                    "to confirm connectivity."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                    },
                    "required": ["status"],
                    "additionalProperties": False,
                },
            },
        }
    ],
}


class LLMClient:
    """OpenAI-compatible LLM client with pluggable auth.

    Supports OAuth (token auto-refreshes before each call)
    or static API key, controlled by AUTH_MODE env var.

    Params:
        None — configuration loaded from environment

    Example:
        >>> client = LLMClient()
        >>> result = client.call(
        ...     messages=[{"role": "user", "content": "hi"}],
        ...     stage="classification",
        ... )
    """

    def __init__(self):
        self.auth_mode = get_auth_mode()
        self.endpoint = get_llm_endpoint()
        self.oauth_client = None
        self.static_client = None

        if self.auth_mode == "oauth":
            oauth_cfg = get_oauth_config()
            self.oauth_client = OAuthClient(config=oauth_cfg)
            logger.info("LLM client configured with OAuth")
        else:
            api_key = get_api_key()
            self.static_client = OpenAI(
                api_key=api_key,
                base_url=self.endpoint,
            )
            logger.info("LLM client configured with API key")

    def get_client(self) -> OpenAI:
        """Build or return an OpenAI client with current auth.

        Params:
            None

        Returns:
            OpenAI — configured client instance

        Example:
            >>> client = LLMClient()
            >>> openai_client = client.get_client()
        """
        if self.static_client:
            return self.static_client
        token = self.oauth_client.get_token()
        return OpenAI(
            api_key=token,
            base_url=self.endpoint,
        )

    def call(
        self,
        messages: list,
        stage: str = "startup",
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        context: str = "",
    ) -> dict:
        """Make an LLM tool-calling request.

        Model and max_tokens are read from env vars based
        on the stage name ({STAGE}_MODEL, {STAGE}_MAX_TOKENS).

        Params:
            messages: List of message dicts
                (e.g. [{"role": "user", "content": "..."}])
            stage: Pipeline stage name for model config
                (e.g. "startup", "classification")
            tools: Optional list of tool definitions
            tool_choice: Optional tool choice constraint
                (e.g. "required", "auto", "none", or a
                function-mapping selector)
            context: Optional short log label for the request

        Returns:
            dict — the full API response as a dict

        Example:
            >>> resp = client.call(
            ...     messages=[{"role": "user", "content": "hi"}],
            ...     stage="classification",
            ... )
        """
        client = self.get_client()
        model_config = get_stage_model_config(stage)
        strict_tools = _apply_tool_strictness(tools)

        kwargs = {
            "model": model_config["model"],
            "messages": messages,
            "max_completion_tokens": model_config["max_tokens"],
        }
        if model_config["temperature"] is not None:
            kwargs["temperature"] = model_config["temperature"]
        if model_config.get("reasoning_effort") is not None:
            kwargs["reasoning_effort"] = model_config["reasoning_effort"]
        if strict_tools:
            kwargs["tools"] = strict_tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        log_parts = []
        if context:
            log_parts.append(context)
        log_parts.append(f"model={model_config['model']}")
        log_parts.append(f"max_tokens={model_config['max_tokens']}")
        if model_config["temperature"] is not None:
            log_parts.append(f"temp={model_config['temperature']}")
        if model_config["reasoning_effort"] is not None:
            log_parts.append(f"reasoning={model_config['reasoning_effort']}")
        log_parts.append(f"messages={len(messages)}")
        if strict_tools:
            log_parts.append(f"tools={len(strict_tools)}")

        logger.debug(
            "LLM call: %s",
            ", ".join(log_parts),
            extra={"stage": stage},
        )

        response = client.chat.completions.create(**kwargs)
        return response.model_dump()

    def stream(
        self,
        messages: list,
        stage: str = "startup",
        context: str = "",
    ) -> Generator[str, None, dict[str, int]]:
        """Stream an LLM response, yielding text chunks.

        No tools or tool_choice — streaming is for plain-text
        responses only (e.g. consolidation).

        Yields str chunks as they arrive. After exhaustion,
        the return value (via StopIteration.value) contains
        usage metrics.

        Params:
            messages: List of message dicts
            stage: Pipeline stage name for model config
            context: Optional short log label

        Returns:
            Generator yielding str chunks; return value is
            dict with prompt_tokens, completion_tokens,
            total_tokens

        Example:
            >>> gen = client.stream(messages, stage="consolidation")
            >>> for chunk in gen:
            ...     print(chunk, end="")
        """
        client = self.get_client()
        model_config = get_stage_model_config(stage)

        kwargs: dict[str, Any] = {
            "model": model_config["model"],
            "messages": messages,
            "max_completion_tokens": model_config["max_tokens"],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if model_config["temperature"] is not None:
            kwargs["temperature"] = model_config["temperature"]
        if model_config.get("reasoning_effort") is not None:
            kwargs["reasoning_effort"] = model_config["reasoning_effort"]

        log_parts = []
        if context:
            log_parts.append(context)
        log_parts.append(f"model={model_config['model']}")
        log_parts.append(f"max_tokens={model_config['max_tokens']}")
        log_parts.append("stream=True")
        log_parts.append(f"messages={len(messages)}")

        logger.debug(
            "LLM stream: %s",
            ", ".join(log_parts),
            extra={"stage": stage},
        )

        response_stream = client.chat.completions.create(**kwargs)
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        for chunk in response_stream:
            if chunk.usage is not None:
                usage["prompt_tokens"] = chunk.usage.prompt_tokens or 0
                usage["completion_tokens"] = chunk.usage.completion_tokens or 0
                usage["total_tokens"] = chunk.usage.total_tokens or 0
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is not None and delta.content:
                yield delta.content
        return usage

    def embed(
        self,
        texts: list[str],
        model: str = "",
        dimensions: int = 0,
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Uses the embedding API endpoint. Auth is handled
        by the same client (OAuth or API key).

        Params:
            texts: List of strings to embed
            model: Embedding model name
                (e.g. text-embedding-3-large)
            dimensions: Output vector dimensions
                (e.g. 3072)

        Returns:
            list[list[float]] -- one vector per input text,
                in the same order as the input

        Example:
            >>> vecs = client.embed(
            ...     ["hello", "world"],
            ...     model="text-embedding-3-large",
            ...     dimensions=3072,
            ... )
            >>> len(vecs)
            2
        """
        if not texts:
            return []
        client = self.get_client()
        kwargs: dict = {
            "model": model,
            "input": texts,
        }
        if dimensions > 0:
            kwargs["dimensions"] = dimensions
        response = client.embeddings.create(**kwargs)
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [item.embedding for item in sorted_data]

    def test_connection(self) -> bool:
        """Validate LLM connectivity with a tool-calling request.

        Sends a minimal health-check prompt and verifies
        the model returns an actual tool call.

        Params:
            None

        Returns:
            bool — True if the call succeeds

        Example:
            >>> client.test_connection()
            True
        """
        prompt = _HEALTH_CHECK_PROMPT
        messages = []
        if prompt.get("system_prompt"):
            messages.append(
                {"role": "system", "content": prompt["system_prompt"]}
            )
        messages.append({"role": "user", "content": prompt["user_prompt"]})
        try:
            response = self.call(
                messages=messages,
                stage=prompt["stage"],
                tools=prompt.get("tools"),
                tool_choice=prompt.get("tool_choice"),
            )
            choices = response.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            if not message:
                raise RuntimeError("LLM returned no usable response")
            if not has_tool_call(response):
                logger.warning(
                    "LLM health check returned no tool call; "
                    "accepting response for connectivity validation"
                )
            logger.info("LLM connection test passed")
            return True
        except Exception:
            logger.error("LLM connection test failed")
            raise


def _apply_tool_strictness(tools: list | None) -> list | None:
    """Copy tools and enforce strict JSON mode on functions."""
    if tools is None:
        return None
    strict_tools = deepcopy(tools)
    for tool in strict_tools:
        function_data = tool.get("function")
        if isinstance(function_data, dict):
            function_data["strict"] = True
    return strict_tools


def has_tool_call(response: dict) -> bool:
    """Return whether the response contains a tool or function call."""
    choices = response.get("choices", [])
    if not choices:
        return False
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        return True
    return isinstance(message.get("function_call"), dict)


def extract_tool_arguments(response: dict) -> dict[str, Any]:
    """Extract parsed JSON arguments from a tool-calling response."""
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("LLM returned no choices")
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        function_data = tool_calls[0].get("function", {})
    else:
        function_data = message.get("function_call", {})
    if not function_data:
        raise ValueError("LLM did not return a tool call")
    arguments_raw = function_data.get("arguments", "{}")
    return json.loads(arguments_raw)


def get_usage_metrics(response: dict) -> dict[str, int]:
    """Extract prompt/completion token counts from an LLM response."""
    usage = response.get("usage", {})
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


def extract_content_text(response: dict) -> str:
    """Extract plain text content from a non-tool-call response.

    Params: response (dict). Returns: str.
    """
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return message.get("content", "") or ""
