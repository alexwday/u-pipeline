"""Tests for prompt loading and validation."""

from pathlib import Path
from textwrap import dedent

import pytest

from ingestion.utils import prompt_loader
from ingestion.utils.prompt_loader import (
    _validate_optional_str as validate_optional_str,
    _validate_prompt as validate_prompt,
    _validate_required_str as validate_required_str,
    _validate_tool_choice as validate_tool_choice,
    _validate_tools as validate_tools,
)


def _valid_tool():
    """Build a standard-compliant tool definition."""
    return {
        "type": "function",
        "function": {
            "name": "extract",
            "description": (
                "Call this tool when the input has been fully reviewed "
                "and you are ready to return the final structured result. "
                "It converts the observed content into the schema below. "
                "Returns the completed extraction payload."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Structured content to return.",
                    }
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        },
    }


def test_validate_required_and_optional_strings():
    """Validate required and optional prompt string fields."""
    data = {"field": "value"}

    assert validate_required_str(data, "field", "prompt") == "value"
    assert validate_optional_str(data, "field", "prompt") == "value"
    assert validate_optional_str({}, "field", "prompt") is None

    with pytest.raises(ValueError, match="requires non-empty string"):
        validate_required_str({}, "field", "prompt")
    with pytest.raises(ValueError, match="must be a string"):
        validate_optional_str({"field": 1}, "field", "prompt")


@pytest.mark.parametrize(
    ("tools", "message"),
    [
        ("bad", "must be a list"),
        ([], "cannot be empty"),
        ([1], "must be a mapping"),
        ([{"type": "bad"}], "must have type 'function'"),
        ([{"type": "function"}], "requires a 'function' mapping"),
        (
            [{"type": "function", "function": {"name": ""}}],
            "requires non-empty function.name",
        ),
        (
            [{"type": "function", "function": {"name": "extract"}}],
            "requires non-empty function.description",
        ),
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "extract",
                        "description": "Call this tool after reviewing.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "text",
                                }
                            },
                            "required": ["content"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "description must start with a WHEN clause",
        ),
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "extract",
                        "description": (
                            "Call this tool when ready. "
                            "It extracts content. "
                            "Returns the result."
                        ),
                    },
                }
            ],
            "requires a 'parameters' mapping",
        ),
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "extract",
                        "description": (
                            "Call this tool when ready. "
                            "It extracts content. "
                            "Returns the result."
                        ),
                        "parameters": {"type": "array"},
                    },
                }
            ],
            "parameters must have type 'object'",
        ),
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "extract",
                        "description": (
                            "Call this tool when ready. "
                            "It extracts content. "
                            "Returns the result."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "require a non-empty 'properties' mapping",
        ),
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "extract",
                        "description": (
                            "Call this tool when ready. "
                            "It extracts content. "
                            "Returns the result."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "text",
                                }
                            },
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "require a non-empty 'required' list",
        ),
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "extract",
                        "description": (
                            "Call this tool when ready. "
                            "It extracts content. "
                            "Returns the result."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "text",
                                }
                            },
                            "required": ["other"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "must list every property in required",
        ),
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "extract",
                        "description": (
                            "Call this tool when ready. "
                            "It extracts content. "
                            "Returns the result."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "text",
                                }
                            },
                            "required": ["content"],
                            "additionalProperties": True,
                        },
                    },
                }
            ],
            "must set additionalProperties to false",
        ),
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "extract",
                        "description": (
                            "Call this tool when ready. "
                            "It extracts content. "
                            "Returns the result."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "text",
                                }
                            },
                            "required": [1],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "required entries must be strings",
        ),
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "extract",
                        "description": (
                            "Call this tool when ready. "
                            "It extracts content. "
                            "Returns the result."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {"content": "bad"},
                            "required": ["content"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "property 'content' must be a mapping",
        ),
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "extract",
                        "description": (
                            "Call this tool when ready. "
                            "It extracts content. "
                            "Returns the result."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "description": "text",
                                }
                            },
                            "required": ["content"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "property 'content' requires field 'type'",
        ),
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "extract",
                        "description": (
                            "Call this tool when ready. "
                            "It extracts content. "
                            "Returns the result."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {"content": {"type": "string"}},
                            "required": ["content"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "requires non-empty description",
        ),
    ],
)
def test_validate_tools_errors(tools, message):
    """Reject malformed tool definitions."""
    with pytest.raises(ValueError, match=message):
        validate_tools("prompt", tools)


def test_validate_tools_success():
    """Accept well-formed tool definitions."""
    tools = [_valid_tool()]

    assert validate_tools("prompt", None) is None
    assert validate_tools("prompt", tools) == tools


@pytest.mark.parametrize(
    ("tool_choice", "message"),
    [
        ("bad", "must be one of"),
        (1, "must be a string or mapping"),
        ({"type": "bad"}, "must have type 'function'"),
        ({"type": "function"}, "requires a 'function' mapping"),
        (
            {"type": "function", "function": {"name": ""}},
            "requires non-empty function.name",
        ),
    ],
)
def test_validate_tool_choice_errors(tool_choice, message):
    """Reject malformed tool_choice values."""
    with pytest.raises(ValueError, match=message):
        validate_tool_choice("prompt", tool_choice)


def test_validate_tool_choice_success():
    """Accept supported tool_choice shapes."""
    mapping = {"type": "function", "function": {"name": "extract"}}

    assert validate_tool_choice("prompt", None) is None
    assert validate_tool_choice("prompt", "required") == "required"
    assert validate_tool_choice("prompt", mapping) == mapping


def test_validate_prompt_errors_and_success():
    """Normalize prompt files and reject invalid definitions."""
    valid = {
        "stage": "discovery",
        "version": 1,
        "description": "desc",
        "system_prompt": "You route files. Always use the provided tool.",
        "user_prompt": "Do the thing",
        "tools": [_valid_tool()],
        "tool_choice": "required",
    }
    valid["user_prompt"] = (
        "Do the thing.\n\nRules:\n"
        "1. Start with the input.\n"
        "2. Return the result."
    )

    prompt = validate_prompt("prompt", valid)
    assert prompt == {
        "stage": "discovery",
        "version": "1",
        "description": "desc",
        "system_prompt": "You route files. Always use the provided tool.",
        "user_prompt": valid["user_prompt"],
        "tools": valid["tools"],
        "tool_choice": "required",
    }

    with pytest.raises(ValueError, match="top-level mapping"):
        validate_prompt("prompt", [])
    with pytest.raises(ValueError, match="must be a string or number"):
        validate_prompt(
            "prompt",
            {
                "stage": "x",
                "description": "desc",
                "system_prompt": "Always use the provided tool.",
                "user_prompt": "Rules:\n1. x",
                "version": [],
            },
        )
    with pytest.raises(ValueError, match="requires field 'version'"):
        validate_prompt(
            "prompt",
            {
                "stage": "x",
                "description": "desc",
                "system_prompt": "Always use the provided tool.",
                "user_prompt": "Rules:\n1. x",
            },
        )
    with pytest.raises(ValueError, match="must include a Rules section"):
        validate_prompt(
            "prompt",
            {
                "stage": "x",
                "version": "1.0",
                "description": "desc",
                "system_prompt": "Always use the provided tool.",
                "user_prompt": "No rules here",
            },
        )
    with pytest.raises(ValueError, match="must include numbered rules"):
        validate_prompt(
            "prompt",
            {
                "stage": "x",
                "version": "1.0",
                "description": "desc",
                "system_prompt": "Always use the provided tool.",
                "user_prompt": "Rules:\n- bullet only",
            },
        )
    with pytest.raises(
        ValueError, match="cannot set tool_choice without tools"
    ):
        validate_prompt(
            "prompt",
            {
                "stage": "x",
                "version": "1.0",
                "description": "desc",
                "system_prompt": "Always use the provided tool.",
                "user_prompt": "Rules:\n1. x",
                "tool_choice": "required",
            },
        )
    with pytest.raises(
        ValueError,
        match="system_prompt must include 'Always use the provided tool.'",
    ):
        validate_prompt(
            "prompt",
            {
                "stage": "x",
                "version": "1.0",
                "description": "desc",
                "system_prompt": "You route files.",
                "user_prompt": "Rules:\n1. x",
                "tools": [_valid_tool()],
                "tool_choice": "required",
            },
        )
    with pytest.raises(
        ValueError, match="must reference a declared tool name"
    ):
        validate_prompt(
            "prompt",
            {
                "stage": "x",
                "version": "1.0",
                "description": "desc",
                "system_prompt": "Always use the provided tool.",
                "user_prompt": "Rules:\n1. x",
                "tools": [_valid_tool()],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "other"},
                },
            },
        )


def test_load_prompt(tmp_path):
    """Load prompt YAML, missing files, and invalid YAML."""
    with pytest.raises(ValueError, match="prompts_dir is required"):
        prompt_loader.load_prompt("prompt")

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Prompt file not found"):
        prompt_loader.load_prompt("missing", prompts_dir)

    invalid_file = prompts_dir / "invalid.yaml"
    invalid_file.write_text("stage: [bad", encoding="utf-8")
    with pytest.raises(ValueError, match="contains invalid YAML"):
        prompt_loader.load_prompt("invalid", prompts_dir)

    valid_file = prompts_dir / "valid.yaml"
    valid_file.write_text(
        dedent("""
        stage: discovery
        version: "1.0"
        description: Test prompt
        system_prompt: >
          You route files. Always use the provided tool.
        user_prompt: >
          Route the file.

          Rules:
          1. Read the input.
          2. Return the schema.
        tool_choice: required
        tools:
          - type: function
            function:
              name: extract
              description: >
                Call this tool when the input has been fully reviewed.
                It returns the structured result.
                Returns the extraction payload.
              parameters:
                type: object
                properties:
                  content:
                    type: string
                    description: Structured result text.
                required:
                  - content
                additionalProperties: false
        """).strip(),
        encoding="utf-8",
    )
    prompt = prompt_loader.load_prompt("valid", prompts_dir)

    assert prompt["stage"] == "discovery"
    assert prompt["version"] == "1.0"
    assert prompt["description"] == "Test prompt"
    assert "Always use the provided tool." in prompt["system_prompt"]
    assert "Rules:" in prompt["user_prompt"]
    assert prompt["tool_choice"] == "required"
    assert prompt["tools"][0]["function"]["name"] == "extract"
    assert prompt["tools"][0]["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Structured result text.",
            }
        },
        "required": ["content"],
        "additionalProperties": False,
    }


def test_repo_prompt_files_follow_project_standard():
    """Load every checked-in prompt and assert key standard markers."""
    repo_root = Path(__file__).resolve().parent.parent
    prompt_paths = sorted(
        repo_root.glob("src/ingestion/processors/*/prompts/*.yaml")
    ) + sorted(repo_root.glob("src/ingestion/stages/chunkers/prompts/*.yaml"))

    assert prompt_paths

    for path in prompt_paths:
        prompt = prompt_loader.load_prompt(path.stem, path.parent)
        assert "Rules:" in prompt["user_prompt"]
        assert "Examples:" in prompt["user_prompt"]
        assert "Always use the provided tool." in prompt["system_prompt"]
        for tool in prompt["tools"]:
            description = tool["function"]["description"]
            assert description.startswith("Call this tool when")
