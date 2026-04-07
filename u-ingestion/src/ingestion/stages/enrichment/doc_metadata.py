"""Stage 6: Document metadata enrichment.

Extracts document-level metadata, detects table of contents,
and classifies structure type in a single LLM call.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from ...utils.config_setup import get_doc_metadata_context_budget
from ...utils.file_types import ExtractionResult
from ...utils.llm_connector import LLMClient
from ...utils.logging_setup import get_stage_logger
from ...utils.prompt_loader import load_prompt
from ...utils.source_context import get_result_source_context
from ...utils.token_counting import count_message_tokens

STAGE = "6-DOC_METADATA"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_SHEET_NAME_RE = re.compile(r"^#\s+Sheet:\s+(.+)$", re.MULTILINE)
_HEADING_RE = re.compile(r"^#+\s+(.+)$", re.MULTILINE)

_VALID_STRUCTURE_TYPES = frozenset(
    {
        "sheet_based",
        "chapters",
        "sections",
        "topic_based",
        "semantic",
    }
)

logger = logging.getLogger(__name__)


def _extract_sheet_names(pages: list) -> list[str]:
    """Parse sheet names from XLSX page raw_content.

    Params:
        pages: List of PageResult objects

    Returns:
        list[str] -- sheet names in page order
    """
    names: list[str] = []
    for page in sorted(pages, key=lambda p: p.page_number):
        match = _SHEET_NAME_RE.search(page.raw_content)
        if match:
            names.append(match.group(1).strip())
    return names


def _extract_page_headings(pages: list) -> list[str]:
    """Find the first markdown heading from each page.

    Params:
        pages: List of PageResult objects

    Returns:
        list[str] -- first heading per page, empty string
            when no heading is found
    """
    headings: list[str] = []
    for page in sorted(pages, key=lambda p: p.page_number):
        match = _HEADING_RE.search(page.raw_content)
        if match:
            headings.append(match.group(1).strip())
        else:
            headings.append("")
    return headings


def _build_content_within_budget(
    pages: list,
    budget: int,
    prompt: dict[str, Any] | None = None,
    file_metadata: str = "",
    page_names: str = "",
    layout_summary: str = "",
) -> str:
    """Walk pages in order and collect content within budget.

    Params:
        pages: List of PageResult objects
        budget: Maximum total raw_token_count to include
        prompt: Optional loaded prompt for full-message budgeting
        file_metadata: Optional file metadata block
        page_names: Optional page names block
        layout_summary: Optional layout summary block

    Returns:
        str -- concatenated page content within budget
    """
    ordered_parts: list[tuple[str, int]] = []
    sorted_pages = sorted(pages, key=lambda p: p.page_number)
    for page in sorted_pages:
        if page.chunk_id:
            continue
        label = f"[Page {page.page_number}]"
        ordered_parts.append(
            (f"{label}\n{page.raw_content}", page.raw_token_count or 0)
        )

    chunked = [p for p in sorted_pages if p.chunk_id]
    chunked.sort(key=lambda p: (p.page_number, p.chunk_id))
    for chunk in chunked:
        label = f"[Page {chunk.page_number} Chunk {chunk.chunk_id}]"
        ordered_parts.append(
            (f"{label}\n{chunk.raw_content}", chunk.raw_token_count or 0)
        )

    parts: list[str] = []
    used = 0
    for content_part, raw_cost in ordered_parts:
        candidate_parts = parts + [content_part]
        if prompt is not None:
            user_message = _format_user_input(
                file_metadata,
                page_names,
                layout_summary,
                "\n\n".join(candidate_parts),
                prompt,
            )
            messages = [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": user_message},
            ]
            if parts and count_message_tokens(messages) > budget:
                break
        elif used + raw_cost > budget and parts:
            break
        parts.append(content_part)
        used += raw_cost

    return "\n\n".join(parts)


def _build_file_metadata(result: ExtractionResult) -> str:
    """Build file metadata string from the extraction result.

    Params:
        result: ExtractionResult with file_path and filetype

    Returns:
        str -- formatted file metadata block
    """
    path = Path(result.file_path)
    source_context = get_result_source_context(result)

    lines = [
        f"filename: {path.name}",
        f"filetype: {result.filetype}",
        f"total_pages: {result.total_pages}",
    ]
    if source_context["data_source"]:
        lines.append(f"data_source: {source_context['data_source']}")
    for idx, filter_key in enumerate(
        ("filter_1", "filter_2", "filter_3"),
        start=1,
    ):
        fval = source_context[filter_key]
        if fval:
            lines.append(f"filter_{idx}: {fval}")
    return "\n".join(lines)


def _build_page_names(
    result: ExtractionResult,
    pages: list,
) -> str:
    """Build page/sheet name listing for the prompt.

    Params:
        result: ExtractionResult with filetype
        pages: List of PageResult objects

    Returns:
        str -- formatted page names block
    """
    if result.filetype == "xlsx":
        sheet_names = _extract_sheet_names(pages)
        lines = []
        for idx, name in enumerate(sheet_names, 1):
            lines.append(f"Sheet {idx}: {name}")
        return "\n".join(lines)

    headings = _extract_page_headings(pages)
    lines = []
    sorted_pages = sorted(pages, key=lambda p: p.page_number)
    seen_pages: set[int] = set()
    for page in sorted_pages:
        if page.page_number in seen_pages:
            continue
        seen_pages.add(page.page_number)
        idx = page.page_number - 1
        heading = ""
        if 0 <= idx < len(headings):
            heading = headings[idx]
        if heading:
            lines.append(f"Page {page.page_number}: {heading}")
        else:
            lines.append(f"Page {page.page_number}")
    return "\n".join(lines)


def _build_layout_summary(pages: list) -> str:
    """Count layout_type occurrences across pages.

    Params:
        pages: List of PageResult objects

    Returns:
        str -- formatted layout type counts
    """
    counts: dict[str, int] = {}
    for page in pages:
        layout = page.layout_type or "unknown"
        counts[layout] = counts.get(layout, 0) + 1
    lines = []
    for layout_type, count in sorted(counts.items()):
        lines.append(f"{layout_type}: {count}")
    return "\n".join(lines)


def _format_user_input(
    file_metadata: str,
    page_names: str,
    layout_summary: str,
    content: str,
    prompt: dict[str, Any],
) -> str:
    """Assemble the full user message from components.

    Params:
        file_metadata: Formatted file metadata block
        page_names: Formatted page/sheet names block
        layout_summary: Formatted layout type counts
        content: Page content within budget
        prompt: Loaded prompt dict with user_prompt template

    Returns:
        str -- complete user message
    """
    user_input_parts = [
        f"<file_metadata>\n{file_metadata}\n</file_metadata>",
        f"<page_names>\n{page_names}\n</page_names>",
        f"<layout_summary>\n{layout_summary}\n</layout_summary>",
        f"<content>\n{content}\n</content>",
    ]
    user_input = "\n\n".join(user_input_parts)
    return prompt["user_prompt"].format(user_input=user_input)


def _parse_metadata_response(response: dict) -> dict:
    """Extract metadata from LLM tool call response.

    Params:
        response: Raw LLM response dict

    Returns:
        dict -- parsed metadata fields

    Raises:
        ValueError: When response has no valid tool call
    """
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("LLM response contains no choices")

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        raise ValueError("LLM response contains no tool calls")

    arguments_raw = tool_calls[0].get("function", {}).get("arguments", "")
    try:
        arguments = json.loads(arguments_raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Failed to parse tool arguments: {exc}") from exc

    required_fields = {
        "title",
        "authors",
        "publication_date",
        "language",
        "structure_type",
        "has_toc",
        "toc_entries",
        "rationale",
    }
    missing = required_fields - set(arguments.keys())
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    if arguments["structure_type"] not in _VALID_STRUCTURE_TYPES:
        raise ValueError(
            f"Invalid structure_type: {arguments['structure_type']}"
        )

    return arguments


def enrich_doc_metadata(
    result: ExtractionResult,
    llm: LLMClient,
) -> ExtractionResult:
    """Extract document-level metadata via LLM.

    Loads the doc_metadata prompt, builds context from
    the extraction result pages within the configured
    token budget, calls the LLM, and sets
    result.document_metadata.

    Params:
        result: ExtractionResult from upstream stage
        llm: Initialized LLM client

    Returns:
        ExtractionResult with document_metadata populated
    """
    stage_log = get_stage_logger(__name__, STAGE)

    prompt = load_prompt("doc_metadata", prompts_dir=_PROMPTS_DIR)
    budget = get_doc_metadata_context_budget()

    unique_pages = _deduplicate_pages(result.pages)

    file_metadata = _build_file_metadata(result)
    page_names = _build_page_names(result, unique_pages)
    layout_summary = _build_layout_summary(result.pages)
    content = _build_content_within_budget(
        result.pages,
        budget,
        prompt=prompt,
        file_metadata=file_metadata,
        page_names=page_names,
        layout_summary=layout_summary,
    )

    user_message = _format_user_input(
        file_metadata,
        page_names,
        layout_summary,
        content,
        prompt,
    )

    messages = [
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": user_message},
    ]

    response = llm.call(
        messages=messages,
        stage="doc_metadata",
        tools=prompt.get("tools"),
        tool_choice=prompt.get("tool_choice"),
        context=f"doc_metadata:{Path(result.file_path).name}",
    )

    metadata = _parse_metadata_response(response)
    source_context = get_result_source_context(result)

    if result.filetype == "xlsx":
        if metadata["structure_type"] != "sheet_based":
            stage_log.debug(
                "Overriding structure_type from '%s' to "
                "'sheet_based' for XLSX file",
                metadata["structure_type"],
            )
            metadata["structure_type"] = "sheet_based"

    result.document_metadata = {
        "title": metadata["title"],
        "authors": metadata["authors"],
        "publication_date": metadata["publication_date"],
        "language": metadata["language"],
        "structure_type": metadata["structure_type"],
        "data_source": source_context["data_source"],
        "filter_1": source_context["filter_1"],
        "filter_2": source_context["filter_2"],
        "filter_3": source_context["filter_3"],
        "has_toc": metadata["has_toc"],
        "toc_entries": metadata["toc_entries"],
        "source_toc_entries": metadata["toc_entries"],
        "generated_toc_entries": [],
        "rationale": metadata["rationale"],
    }

    stage_log.info(
        "Metadata extracted — title='%s', structure=%s, "
        "has_toc=%s, toc_entries=%d",
        metadata["title"][:50],
        metadata["structure_type"],
        metadata["has_toc"],
        len(metadata["toc_entries"]),
    )

    return result


def _deduplicate_pages(pages: list) -> list:
    """Return one representative page per page_number.

    Params:
        pages: List of PageResult objects (may include chunks)

    Returns:
        list -- one page per unique page_number, preferring
            unchunked pages
    """
    by_number: dict[int, Any] = {}
    for page in pages:
        existing = by_number.get(page.page_number)
        if existing is None:
            by_number[page.page_number] = page
        elif page.chunk_id == "" and existing.chunk_id != "":
            by_number[page.page_number] = page
    return sorted(by_number.values(), key=lambda p: p.page_number)


# ------------------------------------------------------------------
# Public aliases for testing
# ------------------------------------------------------------------
extract_sheet_names = _extract_sheet_names
extract_page_headings = _extract_page_headings
build_content_within_budget = _build_content_within_budget
build_file_metadata = _build_file_metadata
build_page_names = _build_page_names
build_layout_summary = _build_layout_summary
format_user_input = _format_user_input
parse_metadata_response = _parse_metadata_response
deduplicate_pages = _deduplicate_pages
