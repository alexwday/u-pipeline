"""Stage 10: Document-level summary and keyword refinement.

Generates an executive summary and refined keywords/entities
from section summaries built in the section_summary stage.
One LLM call per document.
"""

import json
import logging
from pathlib import Path
from typing import Any

from ...utils.file_types import ExtractionResult
from ...utils.llm_connector import LLMClient
from ...utils.logging_setup import get_stage_logger
from ...utils.prompt_loader import load_prompt

STAGE = "10-DOC_SUMMARY"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

logger = logging.getLogger(__name__)


def _get_primary_sections(
    sections: list[dict],
) -> list[dict]:
    """Filter to primary sections sorted by sequence.

    Params: sections (list[dict]). Returns: list[dict].
    """
    primaries = [s for s in sections if s.get("level") == "section"]
    return sorted(primaries, key=lambda s: s.get("sequence", 0))


def _format_toc(sections: list[dict]) -> str:
    """Format sections as a table of contents block.

    Each entry shows title, summary, keywords, entities,
    and page range for a section.

    Params:
        sections: Primary section dicts with summaries

    Returns:
        str -- formatted TOC block
    """
    lines: list[str] = []
    for section in sections:
        sid = section.get("section_id", "")
        title = section.get("title", "")
        page_start = section.get("page_start", 0)
        page_end = section.get("page_end", 0)
        summary = section.get("summary", "")
        keywords = section.get("keywords", [])
        entities = section.get("entities", [])

        page_range = str(page_start)
        if page_end and page_end != page_start:
            page_range = f"{page_start}-{page_end}"

        kw_str = json.dumps(keywords)
        ent_str = json.dumps(entities)

        lines.append(f"[{sid}] {title} (p.{page_range})" f' -- "{summary}"')
        lines.append(f"    keywords: {kw_str}" f"  entities: {ent_str}")

    return "\n".join(lines)


def _format_doc_metadata(metadata: dict) -> str:
    """Format document metadata as key-value lines.

    Params:
        metadata: Document metadata dict

    Returns:
        str -- formatted metadata lines
    """
    lines: list[str] = []
    field_map = [
        ("title", "title"),
        ("authors", "authors"),
        ("publication_date", "publication_date"),
        ("data_source", "data_source"),
        ("filter_1", "filter_1"),
        ("filter_2", "filter_2"),
    ]
    for key, label in field_map:
        value = metadata.get(key, "")
        if value:
            lines.append(f'{label}: "{value}"')
    return "\n".join(lines)


def _format_user_input(
    doc_metadata_str: str,
    toc_str: str,
    prompt: dict[str, Any],
) -> str:
    """Assemble the full user message with XML tags.

    Params:
        doc_metadata_str: Formatted metadata block
        toc_str: Formatted TOC block
        prompt: Loaded prompt dict with user_prompt

    Returns:
        str -- complete user message
    """
    parts: list[str] = []

    if doc_metadata_str:
        parts.append(
            f"<document_metadata>\n"
            f"{doc_metadata_str}\n"
            f"</document_metadata>"
        )

    parts.append(
        f"<table_of_contents>\n" f"{toc_str}\n" f"</table_of_contents>"
    )

    user_input = "\n\n".join(parts)
    return prompt["user_prompt"].format(user_input=user_input)


def _parse_doc_summary_response(
    response: dict,
) -> dict[str, Any]:
    """Extract summary fields from LLM tool-call response.

    Params:
        response: Raw LLM response dict

    Returns:
        dict with executive_summary, keywords, entities

    Raises:
        ValueError: When response has no valid tool call
    """
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("LLM response contains no choices")

    finish_reason = choices[0].get("finish_reason", "")
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        content_preview = str(message.get("content", ""))[:200]
        raise ValueError(
            f"LLM response contains no tool calls "
            f"(finish_reason={finish_reason}, "
            f"content={content_preview!r})"
        )

    args_raw = tool_calls[0].get("function", {}).get("arguments", "")
    try:
        arguments = json.loads(args_raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Failed to parse tool arguments: {exc}") from exc

    executive_summary = arguments.get("executive_summary")
    if not isinstance(executive_summary, str):
        raise ValueError("Response missing executive_summary field")

    keywords = arguments.get("keywords")
    if not isinstance(keywords, list):
        raise ValueError("Response missing or invalid keywords field")

    entities = arguments.get("entities")
    if not isinstance(entities, list):
        raise ValueError("Response missing or invalid entities field")

    return {
        "executive_summary": executive_summary,
        "keywords": keywords,
        "entities": entities,
    }


def _update_metadata(
    result: ExtractionResult,
    summary_data: dict[str, Any],
) -> None:
    """Apply summary data to document_metadata.

    Params:
        result: ExtractionResult to update
        summary_data: Dict with executive_summary,
            keywords, entities

    Returns:
        None
    """
    if result.document_metadata is None:
        result.document_metadata = {}
    result.document_metadata["executive_summary"] = summary_data[
        "executive_summary"
    ]
    result.document_metadata["keywords"] = list(summary_data["keywords"])
    result.document_metadata["entities"] = list(summary_data["entities"])


def summarize_document(
    result: ExtractionResult,
    llm: LLMClient,
) -> ExtractionResult:
    """Generate executive summary and refine metadata.

    Loads the doc_summary prompt, builds input from section
    summaries and document metadata, makes one LLM call, and
    updates the document metadata with executive_summary,
    keywords, and entities.

    Params:
        result: ExtractionResult from upstream stage
        llm: Initialized LLM client

    Returns:
        ExtractionResult with document-level summary and
            refined keywords/entities in document_metadata
    """
    stage_log = get_stage_logger(__name__, STAGE)

    prompt = load_prompt("doc_summary", prompts_dir=_PROMPTS_DIR)

    metadata = result.document_metadata or {}
    primaries = _get_primary_sections(result.sections)

    doc_metadata_str = _format_doc_metadata(metadata)
    toc_str = _format_toc(primaries)

    user_message = _format_user_input(doc_metadata_str, toc_str, prompt)

    messages = [
        {
            "role": "system",
            "content": prompt["system_prompt"],
        },
        {"role": "user", "content": user_message},
    ]

    response = llm.call(
        messages=messages,
        stage="doc_summary",
        tools=prompt.get("tools"),
        tool_choice=prompt.get("tool_choice"),
        context=(f"doc_summary:{Path(result.file_path).name}"),
    )

    summary_data = _parse_doc_summary_response(response)
    _update_metadata(result, summary_data)

    kw_count = len(summary_data["keywords"])
    ent_count = len(summary_data["entities"])
    stage_log.info(
        "Document summary complete -- %d keywords, "
        "%d entities, %d sections synthesized",
        kw_count,
        ent_count,
        len(primaries),
    )

    return result


# ------------------------------------------------------------------
# Public aliases for testing
# ------------------------------------------------------------------
get_primary_sections = _get_primary_sections
format_toc = _format_toc
format_doc_metadata = _format_doc_metadata
format_user_input = _format_user_input
parse_doc_summary_response = _parse_doc_summary_response
update_metadata = _update_metadata
