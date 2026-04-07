"""Validation tests for real enrichment prompt YAML files."""

import re
from pathlib import Path

import pytest

from ingestion.utils.prompt_loader import load_prompt

PROMPTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ingestion"
    / "stages"
    / "enrichment"
    / "prompts"
)

PROMPT_NAMES = (
    "doc_metadata",
    "section_detection",
    "subsection_detection",
    "content_extraction",
    "section_summary",
    "doc_summary",
)


def _assert_object_schemas_closed(schema):
    """Assert additionalProperties is false for object schemas."""
    if not isinstance(schema, dict):
        return

    if schema.get("type") == "object":
        assert schema.get("additionalProperties") is False
        for subschema in schema.get("properties", {}).values():
            _assert_object_schemas_closed(subschema)

    if schema.get("type") == "array":
        _assert_object_schemas_closed(schema.get("items"))


@pytest.mark.parametrize("prompt_name", PROMPT_NAMES)
def test_real_enrichment_prompts_load_and_match_basics(prompt_name):
    """Prompt files load from disk with the expected base structure."""
    prompt = load_prompt(prompt_name, prompts_dir=PROMPTS_DIR)

    assert prompt["stage"]
    assert prompt["version"]
    assert prompt["description"]
    assert prompt["system_prompt"]
    assert prompt["user_prompt"]
    assert prompt["tool_choice"] == "required"
    assert prompt["tools"]


@pytest.mark.parametrize("prompt_name", PROMPT_NAMES)
def test_real_enrichment_prompts_follow_reviewed_tool_standards(prompt_name):
    """Tool descriptions include a when-clause and closed schemas."""
    prompt = load_prompt(prompt_name, prompts_dir=PROMPTS_DIR)

    for tool in prompt["tools"]:
        function = tool["function"]
        assert "when" in function["description"].lower()
        _assert_object_schemas_closed(function["parameters"])


@pytest.mark.parametrize("prompt_name", PROMPT_NAMES)
def test_real_enrichment_prompts_use_numbered_rules(prompt_name):
    """User prompts expose numbered rules and avoid aggressive language."""
    prompt = load_prompt(prompt_name, prompts_dir=PROMPTS_DIR)
    user_prompt = prompt["user_prompt"]
    system_prompt = prompt["system_prompt"]

    assert "Rules:" in user_prompt
    assert re.search(r"(^|\n)\s*1\.", user_prompt)
    assert "NEVER" not in system_prompt
    assert "NEVER" not in user_prompt
