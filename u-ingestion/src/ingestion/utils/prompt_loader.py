"""Load prompt definitions from YAML resources."""

from pathlib import Path
import re
from typing import Any

import yaml

ALLOWED_TOOL_CHOICES = frozenset({"required", "auto", "none"})
WHEN_TOOL_DESCRIPTION_RE = re.compile(r"^\s*(Call|Use) this tool when\b")
NUMBERED_RULE_RE = re.compile(r"Rules:\s*(?:\n|\r\n|\s)+1\.\s")


def _validate_required_str(
    data: dict[str, Any],
    field_name: str,
    prompt_name: str,
) -> str:
    """Get a required string field or raise.

    Params: data, field_name, prompt_name. Returns: str.
    """
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Prompt '{prompt_name}' requires non-empty string "
            f"field '{field_name}'"
        )
    return value


def _validate_optional_str(
    data: dict[str, Any],
    field_name: str,
    prompt_name: str,
) -> str | None:
    """Get an optional string field or raise.

    Params: data, field_name, prompt_name. Returns: str | None.
    """
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"Prompt '{prompt_name}' field '{field_name}' " f"must be a string"
        )
    return value


def _validate_required_version(
    data: dict[str, Any],
    prompt_name: str,
) -> str:
    """Get a required version field or raise.

    Params: data, prompt_name. Returns: str.
    """
    version = data.get("version")
    if version is None:
        raise ValueError(f"Prompt '{prompt_name}' requires field 'version'")
    if not isinstance(version, (str, int, float)):
        raise ValueError(
            f"Prompt '{prompt_name}' field 'version' must be "
            f"a string or number"
        )
    return str(version)


def _validate_user_prompt(prompt_name: str, user_prompt: str) -> str:
    """Validate the required user prompt structure.

    Params: prompt_name, user_prompt. Returns: str.
    """
    if "Rules:" not in user_prompt:
        raise ValueError(
            f"Prompt '{prompt_name}' user_prompt must include a Rules section"
        )
    if NUMBERED_RULE_RE.search(user_prompt) is None:
        raise ValueError(
            f"Prompt '{prompt_name}' user_prompt must include numbered rules"
        )
    return user_prompt


def _validate_tool_parameters(
    prompt_name: str,
    index: int,
    parameters: Any,
) -> dict[str, Any]:
    """Validate one tool parameter schema.

    Params: prompt_name, index, parameters. Returns: dict[str, Any].
    """
    if not isinstance(parameters, dict):
        raise ValueError(
            f"Prompt '{prompt_name}' tool #{index} requires a "
            f"'parameters' mapping"
        )
    if parameters.get("type") != "object":
        raise ValueError(
            f"Prompt '{prompt_name}' tool #{index} parameters must have "
            f"type 'object'"
        )

    properties = parameters.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise ValueError(
            f"Prompt '{prompt_name}' tool #{index} parameters require "
            f"a non-empty 'properties' mapping"
        )

    required = parameters.get("required")
    if not isinstance(required, list) or not required:
        raise ValueError(
            f"Prompt '{prompt_name}' tool #{index} parameters require "
            f"a non-empty 'required' list"
        )
    if not all(isinstance(field_name, str) for field_name in required):
        raise ValueError(
            f"Prompt '{prompt_name}' tool #{index} required entries "
            f"must be strings"
        )

    if parameters.get("additionalProperties") is not False:
        raise ValueError(
            f"Prompt '{prompt_name}' tool #{index} parameters must set "
            f"additionalProperties to false"
        )

    property_names = set(properties)
    required_names = set(required)
    if property_names != required_names:
        raise ValueError(
            f"Prompt '{prompt_name}' tool #{index} parameters must list "
            f"every property in required"
        )

    for property_name, schema in properties.items():
        if not isinstance(schema, dict):
            raise ValueError(
                f"Prompt '{prompt_name}' tool #{index} property "
                f"'{property_name}' must be a mapping"
            )
        if "type" not in schema:
            raise ValueError(
                f"Prompt '{prompt_name}' tool #{index} property "
                f"'{property_name}' requires field 'type'"
            )
        description = schema.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"Prompt '{prompt_name}' tool #{index} property "
                f"'{property_name}' requires non-empty description"
            )
    return parameters


def _validate_tools(prompt_name: str, tools: Any) -> list | None:
    """Validate tool definitions.

    Params: prompt_name, tools. Returns: list | None.
    """
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ValueError(
            f"Prompt '{prompt_name}' field 'tools' must be a list"
        )
    if not tools:
        raise ValueError(
            f"Prompt '{prompt_name}' field 'tools' cannot be empty"
        )

    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(
                f"Prompt '{prompt_name}' tool #{index} must be a mapping"
            )
        if tool.get("type") != "function":
            raise ValueError(
                f"Prompt '{prompt_name}' tool #{index} must have "
                f"type 'function'"
            )
        function_data = tool.get("function")
        if not isinstance(function_data, dict):
            raise ValueError(
                f"Prompt '{prompt_name}' tool #{index} requires a "
                f"'function' mapping"
            )
        function_name = function_data.get("name")
        if not isinstance(function_name, str) or not function_name.strip():
            raise ValueError(
                f"Prompt '{prompt_name}' tool #{index} requires "
                f"non-empty function.name"
            )
        description = function_data.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"Prompt '{prompt_name}' tool #{index} requires "
                f"non-empty function.description"
            )
        if WHEN_TOOL_DESCRIPTION_RE.search(description) is None:
            raise ValueError(
                f"Prompt '{prompt_name}' tool #{index} description must "
                f"start with a WHEN clause"
            )
        _validate_tool_parameters(
            prompt_name,
            index,
            function_data.get("parameters"),
        )
    return tools


def _validate_tool_choice(
    prompt_name: str, tool_choice: Any
) -> str | dict | None:
    """Validate tool_choice.

    Params: prompt_name, tool_choice. Returns: str | dict | None.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice not in ALLOWED_TOOL_CHOICES:
            raise ValueError(
                f"Prompt '{prompt_name}' field 'tool_choice' must be one "
                f"of {sorted(ALLOWED_TOOL_CHOICES)}"
            )
        return tool_choice
    if not isinstance(tool_choice, dict):
        raise ValueError(
            f"Prompt '{prompt_name}' field 'tool_choice' must be "
            f"a string or mapping"
        )

    if tool_choice.get("type") != "function":
        raise ValueError(
            f"Prompt '{prompt_name}' mapping tool_choice must have "
            f"type 'function'"
        )
    function_data = tool_choice.get("function")
    if not isinstance(function_data, dict):
        raise ValueError(
            f"Prompt '{prompt_name}' mapping tool_choice requires "
            f"a 'function' mapping"
        )
    function_name = function_data.get("name")
    if not isinstance(function_name, str) or not function_name.strip():
        raise ValueError(
            f"Prompt '{prompt_name}' mapping tool_choice requires "
            f"non-empty function.name"
        )
    return tool_choice


def _validate_prompt(name: str, data: Any) -> dict[str, Any]:
    """Validate and normalize prompt data.

    Params: name, data. Returns: dict.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Prompt '{name}' must contain a top-level mapping")

    prompt: dict[str, Any] = {
        "stage": _validate_required_str(data, "stage", name),
        "version": _validate_required_version(data, name),
        "description": _validate_required_str(data, "description", name),
        "system_prompt": _validate_required_str(data, "system_prompt", name),
        "user_prompt": _validate_user_prompt(
            name,
            _validate_required_str(data, "user_prompt", name),
        ),
    }

    tools = _validate_tools(name, data.get("tools"))
    tool_choice = _validate_tool_choice(name, data.get("tool_choice"))
    if tools is None and tool_choice not in (None, "none"):
        raise ValueError(
            f"Prompt '{name}' cannot set tool_choice without tools"
        )
    if isinstance(tool_choice, dict) and tools is not None:
        tool_names = {tool["function"]["name"] for tool in tools}
        chosen_name = tool_choice["function"]["name"]
        if chosen_name not in tool_names:
            raise ValueError(
                f"Prompt '{name}' mapping tool_choice must reference "
                f"a declared tool name"
            )
    if tools is not None and tool_choice != "none":
        if "Always use the provided tool." not in prompt["system_prompt"]:
            raise ValueError(
                f"Prompt '{name}' system_prompt must include "
                f"'Always use the provided tool.'"
            )
    if tools is not None:
        prompt["tools"] = tools
    if tool_choice is not None:
        prompt["tool_choice"] = tool_choice

    return prompt


def load_prompt(
    name: str,
    prompts_dir: Path | None = None,
) -> dict[str, Any]:
    """Load a prompt definition by filename stem.

    Params:
        name: Prompt filename stem (e.g. "page_extraction")
        prompts_dir: Optional directory containing prompt YAML files

    Returns:
        dict with validated keys such as stage, version,
        description, system_prompt, user_prompt, tools,
        and tool_choice

    Example:
        >>> prompt = load_prompt("page_extraction", prompts_dir)
        >>> prompt["stage"]
        "extraction"
    """
    if prompts_dir is None:
        raise ValueError(
            "prompts_dir is required — all prompts live in "
            "processor-specific directories"
        )
    directory = prompts_dir
    path = directory / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {name}.yaml")

    with path.open("r", encoding="utf-8") as file_handle:
        try:
            data = yaml.safe_load(file_handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Prompt '{name}' contains invalid YAML") from exc
    return _validate_prompt(name, data)
