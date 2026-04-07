"""Stage 7: Section and subsection detection.

Detects primary section boundaries and subsections within
large sections. XLSX files use automatic sheet-based
detection, semantic documents get one section per page,
and all other structure types use LLM-based detection.
"""

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ...utils.config_setup import (
    get_section_detection_batch_budget,
    get_subsection_token_threshold,
)
from ...utils.file_types import (
    ExtractionResult,
    SectionResult,
    get_content_unit_id,
)
from ...utils.llm_connector import LLMClient
from ...utils.logging_setup import get_stage_logger
from ...utils.prompt_loader import load_prompt
from ...utils.token_counting import count_message_tokens

STAGE = "7-SECTION_DETECTION"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_SHEET_NAME_RE = re.compile(r"^#\s+Sheet:\s+(.+)$", re.MULTILINE)

logger = logging.getLogger(__name__)


def _deduplicate_pages(pages: list) -> list:
    """Return one representative page per page_number.

    Params:
        pages: List of PageResult objects (may include
            chunks)

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


def _content_unit_id(page: Any) -> str:
    """Get the identifier for a content unit.

    Params:
        page: PageResult object

    Returns:
        str -- chunk_id if set, else str(page_number)
    """
    return get_content_unit_id(page)


def _parse_sheet_name(raw_content: str) -> str:
    """Extract sheet name from XLSX raw_content header.

    Params:
        raw_content: Raw content string with sheet header

    Returns:
        str -- sheet name or empty string
    """
    match = _SHEET_NAME_RE.search(raw_content)
    if match:
        return match.group(1).strip()
    return ""


def _chunk_id_sort_key(chunk_id: str) -> tuple[int, tuple]:
    """Build a numeric-aware sort key for chunk ids.

    Params:
        chunk_id: Chunk identifier like "24.10"

    Returns:
        tuple[int, tuple] -- stable sort key
    """
    if not chunk_id:
        return (0, ())

    key_parts: list[tuple[int, int | str]] = []
    for part in chunk_id.split("."):
        if part.isdigit():
            key_parts.append((0, int(part)))
        else:
            key_parts.append((1, part))
    return (1, tuple(key_parts))


def _detect_sheet_based_sections(
    result: ExtractionResult,
) -> list[SectionResult]:
    """Build sections from XLSX sheets automatically.

    Each unique page_number becomes one section. Sheet
    name is parsed from the raw_content header. Chunks
    within the same sheet are collected as chunk_ids.

    Params:
        result: ExtractionResult with XLSX pages

    Returns:
        list[SectionResult] -- one section per sheet
    """
    page_groups: dict[int, list] = {}
    for page in result.pages:
        pn = page.page_number
        if pn not in page_groups:
            page_groups[pn] = []
        page_groups[pn].append(page)

    sections: list[SectionResult] = []
    for seq, pn in enumerate(sorted(page_groups), start=1):
        group = sorted(
            page_groups[pn],
            key=lambda page: _chunk_id_sort_key(page.chunk_id),
        )
        title = ""
        for page in group:
            title = _parse_sheet_name(page.raw_content)
            if title:
                break
        if not title:
            title = f"Sheet {pn}"

        chunk_ids = [_content_unit_id(p) for p in group]
        total_tokens = sum(p.raw_token_count or 0 for p in group)

        sections.append(
            SectionResult(
                section_id=str(seq),
                parent_section_id="",
                level="section",
                title=title,
                sequence=seq,
                page_start=pn,
                page_end=pn,
                chunk_ids=chunk_ids,
                token_count=total_tokens,
            )
        )
    return sections


def _detect_semantic_sections(
    result: ExtractionResult,
) -> list[SectionResult]:
    """Build flat sections for semantic documents.

    Each page becomes one standalone section with no
    hierarchy.

    Params:
        result: ExtractionResult with pages

    Returns:
        list[SectionResult] -- one section per page
    """
    unique = _deduplicate_pages(result.pages)
    sections: list[SectionResult] = []
    for seq, page in enumerate(unique, start=1):
        page_units = [
            p for p in result.pages if p.page_number == page.page_number
        ]
        chunk_ids = [_content_unit_id(p) for p in page_units]
        total_tokens = sum(p.raw_token_count or 0 for p in page_units)
        sections.append(
            SectionResult(
                section_id=str(seq),
                parent_section_id="",
                level="section",
                title=f"Page {page.page_number}",
                sequence=seq,
                page_start=page.page_number,
                page_end=page.page_number,
                chunk_ids=chunk_ids,
                token_count=total_tokens,
            )
        )
    return sections


def _parse_section_response(response: dict) -> list[dict]:
    """Extract sections list from LLM tool call response.

    Params:
        response: Raw LLM response dict

    Returns:
        list[dict] -- parsed section entries

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

    args_raw = tool_calls[0].get("function", {}).get("arguments", "")
    try:
        arguments = json.loads(args_raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Failed to parse tool arguments: {exc}") from exc

    sections = arguments.get("sections")
    if sections is None:
        sections = arguments.get("subsections")
    if sections is None:
        raise ValueError("Response missing sections/subsections field")
    if not isinstance(sections, list):
        raise ValueError("sections field must be a list")

    for entry in sections:
        if "title" not in entry or "start_page" not in entry:
            raise ValueError("Each section must have title and start_page")
    return sections


def _format_section_batch(
    toc_entries: list,
    previous_sections: list[dict],
    structure_type: str,
    batch_pages: list,
    prompt: dict[str, Any],
) -> str:
    """Build the full user message for a section batch.

    Params:
        toc_entries: TOC entries from document metadata
        previous_sections: Already-detected sections
        structure_type: Document structure classification
        batch_pages: Pages in the current batch
        prompt: Loaded prompt dict with user_prompt

    Returns:
        str -- complete user message
    """
    parts: list[str] = []

    if toc_entries:
        toc_lines = []
        for entry in toc_entries:
            name = entry.get("name", "")
            page = entry.get("page_number", "")
            toc_lines.append(f"- {name} (p.{page})")
        toc_block = "\n".join(toc_lines)
        parts.append(f"<toc_hint>\n{toc_block}\n</toc_hint>")

    if previous_sections:
        prev_lines = []
        for sec in previous_sections:
            prev_lines.append(
                f"- {sec['title']} (starts p." f"{sec['start_page']})"
            )
        prev_block = "\n".join(prev_lines)
        parts.append(
            f"<previous_sections>\n" f"{prev_block}\n" f"</previous_sections>"
        )

    parts.append(f"<structure_type>" f"{structure_type}" f"</structure_type>")

    content_lines = []
    for page in batch_pages:
        label = f"[Page {page.page_number}]"
        content_lines.append(f"{label}\n{page.raw_content}")
    content_block = "\n\n".join(content_lines)
    parts.append(f"<content>\n{content_block}\n</content>")

    user_input = "\n\n".join(parts)
    return prompt["user_prompt"].format(user_input=user_input)


def _detect_llm_sections(
    result: ExtractionResult,
    llm: LLMClient,
    metadata: dict,
) -> list[SectionResult]:
    """Detect sections via LLM for structured documents.

    Processes deduplicated pages in batches within the
    configured token budget. Accumulates section
    boundaries across batches and computes end pages.

    Params:
        result: ExtractionResult with pages
        llm: Initialized LLM client
        metadata: Document metadata dict

    Returns:
        list[SectionResult] -- detected primary sections
    """
    stage_log = get_stage_logger(__name__, STAGE)
    prompt = load_prompt("section_detection", prompts_dir=_PROMPTS_DIR)
    budget = get_section_detection_batch_budget()
    toc_entries = metadata.get(
        "source_toc_entries",
        metadata.get("toc_entries", []),
    )
    structure_type = metadata.get("structure_type", "")

    unique_pages = _deduplicate_pages(result.pages)
    if not unique_pages:
        return []

    raw_sections: list[dict] = []
    remaining_pages = list(unique_pages)

    while remaining_pages:
        batch: list = []
        for page in remaining_pages:
            candidate_batch = batch + [page]
            user_message = _format_section_batch(
                toc_entries,
                raw_sections,
                structure_type,
                candidate_batch,
                prompt,
            )
            messages = [
                {
                    "role": "system",
                    "content": prompt["system_prompt"],
                },
                {"role": "user", "content": user_message},
            ]
            if batch and count_message_tokens(messages) > budget:
                break
            batch = candidate_batch

        raw_sections.extend(
            _call_section_llm(
                llm,
                prompt,
                toc_entries,
                raw_sections,
                structure_type,
                batch,
                result,
            )
        )
        remaining_pages = remaining_pages[len(batch) :]

    last_page = unique_pages[-1].page_number
    sections = _build_section_results(raw_sections, result, last_page)

    stage_log.info(
        "Detected %d sections via LLM across %d pages",
        len(sections),
        len(unique_pages),
    )
    return sections


def _call_section_llm(
    llm: LLMClient,
    prompt: dict[str, Any],
    toc_entries: list,
    previous_raw: list[dict],
    structure_type: str,
    batch_pages: list,
    result: ExtractionResult,
) -> list[dict]:
    """Call LLM for one batch and return raw sections.

    Params:
        llm: LLM client
        prompt: Loaded prompt dict
        toc_entries: TOC entries from metadata
        previous_raw: Previously detected raw sections
        structure_type: Document structure type
        batch_pages: Pages in this batch
        result: Full ExtractionResult for context

    Returns:
        list[dict] -- raw section dicts from LLM
    """
    user_message = _format_section_batch(
        toc_entries,
        previous_raw,
        structure_type,
        batch_pages,
        prompt,
    )
    messages = [
        {
            "role": "system",
            "content": prompt["system_prompt"],
        },
        {"role": "user", "content": user_message},
    ]
    response = llm.call(
        messages=messages,
        stage="section_detection",
        tools=prompt.get("tools"),
        tool_choice=prompt.get("tool_choice"),
        context=(f"section_detection:" f"{Path(result.file_path).name}"),
    )
    return _parse_section_response(response)


def _build_section_results(
    raw_sections: list[dict],
    result: ExtractionResult,
    last_page: int,
) -> list[SectionResult]:
    """Convert raw section dicts to SectionResult objects.

    Computes end_page for each section and assigns
    chunk_ids from the content units that fall within
    each section's page range.

    Params:
        raw_sections: Parsed section dicts with title
            and start_page
        result: ExtractionResult with pages
        last_page: Last page number in the document

    Returns:
        list[SectionResult] -- ordered primary sections
    """
    if not raw_sections:
        return []

    sorted_secs = sorted(raw_sections, key=lambda s: s["start_page"])

    sections: list[SectionResult] = []
    for idx, raw in enumerate(sorted_secs):
        seq = idx + 1
        start = raw["start_page"]
        if idx + 1 < len(sorted_secs):
            end = sorted_secs[idx + 1]["start_page"] - 1
        else:
            end = last_page

        pages_in_range = [
            p for p in result.pages if start <= p.page_number <= end
        ]
        chunk_ids = [_content_unit_id(p) for p in pages_in_range]
        total_tokens = sum(p.raw_token_count or 0 for p in pages_in_range)

        sections.append(
            SectionResult(
                section_id=str(seq),
                parent_section_id="",
                level="section",
                title=raw["title"],
                sequence=seq,
                page_start=start,
                page_end=end,
                chunk_ids=chunk_ids,
                token_count=total_tokens,
            )
        )
    return sections


def _format_subsection_batch(
    section_title: str,
    page_start: int,
    page_end: int,
    section_pages: list,
    prompt: dict[str, Any],
) -> str:
    """Build the user message for subsection detection.

    Params:
        section_title: Title of the parent section
        page_start: First page of the section
        page_end: Last page of the section
        section_pages: Pages within the section
        prompt: Loaded prompt dict

    Returns:
        str -- complete user message
    """
    parts: list[str] = []

    parts.append(
        f"<section_info>\n"
        f"Title: {section_title}\n"
        f"Pages: {page_start}-{page_end}\n"
        f"</section_info>"
    )

    content_lines = []
    for page in section_pages:
        label = f"[Page {page.page_number}]"
        content_lines.append(f"{label}\n{page.raw_content}")
    content_block = "\n\n".join(content_lines)
    parts.append(f"<content>\n{content_block}\n</content>")

    user_input = "\n\n".join(parts)
    return prompt["user_prompt"].format(user_input=user_input)


def _xlsx_chunks_as_subsections(
    section: SectionResult,
    pages: list,
) -> list[SectionResult]:
    """Create subsections from XLSX chunks automatically.

    Each chunk within the section becomes a subsection.
    Subsection title uses the chunk_context field.

    Params:
        section: Parent section
        pages: All pages in the ExtractionResult

    Returns:
        list[SectionResult] -- subsections for this section
    """
    chunks = [
        p for p in pages if p.page_number == section.page_start and p.chunk_id
    ]
    chunks.sort(key=lambda page: _chunk_id_sort_key(page.chunk_id))

    subsections: list[SectionResult] = []
    for idx, chunk in enumerate(chunks, start=1):
        sub_id = f"{section.section_id}.{idx}"
        chunk_label = chunk.chunk_context or f"Chunk {idx}"
        title = (
            f"{section.title} \u2014 {chunk_label}"
            if section.title
            else chunk_label
        )
        subsections.append(
            SectionResult(
                section_id=sub_id,
                parent_section_id=section.section_id,
                level="subsection",
                title=title,
                sequence=idx,
                page_start=chunk.page_number,
                page_end=chunk.page_number,
                chunk_ids=[_content_unit_id(chunk)],
                token_count=chunk.raw_token_count or 0,
            )
        )
    return subsections


def _build_subsection_results(
    sorted_subs: list[dict],
    section: SectionResult,
    result: ExtractionResult,
) -> list[SectionResult]:
    """Convert sorted raw subsection dicts to SectionResults.

    Params:
        sorted_subs: Subsection dicts sorted by start_page
        section: Parent primary section
        result: ExtractionResult with pages

    Returns:
        list[SectionResult] -- ordered subsections
    """
    subsections: list[SectionResult] = []
    for idx, raw in enumerate(sorted_subs):
        seq = idx + 1
        sub_id = f"{section.section_id}.{seq}"
        start = raw["start_page"]
        if idx + 1 < len(sorted_subs):
            end = sorted_subs[idx + 1]["start_page"] - 1
        else:
            end = section.page_end

        pages_in_range = [
            p for p in result.pages if start <= p.page_number <= end
        ]
        chunk_ids = [_content_unit_id(p) for p in pages_in_range]
        total_tokens = sum(p.raw_token_count or 0 for p in pages_in_range)

        subsections.append(
            SectionResult(
                section_id=sub_id,
                parent_section_id=section.section_id,
                level="subsection",
                title=raw["title"],
                sequence=seq,
                page_start=start,
                page_end=end,
                chunk_ids=chunk_ids,
                token_count=total_tokens,
            )
        )
    return subsections


def _detect_llm_subsections(
    section: SectionResult,
    result: ExtractionResult,
    llm: LLMClient,
) -> list[SectionResult]:
    """Detect subsections via LLM for one large section.

    Params:
        section: Parent primary section
        result: ExtractionResult with pages
        llm: Initialized LLM client

    Returns:
        list[SectionResult] -- subsections within section
    """
    prompt = load_prompt("subsection_detection", prompts_dir=_PROMPTS_DIR)

    unique_pages = _deduplicate_pages(result.pages)
    section_pages = [
        p
        for p in unique_pages
        if section.page_start <= p.page_number <= section.page_end
    ]
    if not section_pages:
        return []

    user_message = _format_subsection_batch(
        section.title,
        section.page_start,
        section.page_end,
        section_pages,
        prompt,
    )
    messages = [
        {
            "role": "system",
            "content": prompt["system_prompt"],
        },
        {"role": "user", "content": user_message},
    ]
    response = llm.call(
        messages=messages,
        stage="section_detection",
        tools=prompt.get("tools"),
        tool_choice=prompt.get("tool_choice"),
        context=(
            f"subsection_detection:"
            f"{Path(result.file_path).name}:"
            f"{section.title}"
        ),
    )
    raw_subs = _parse_section_response(response)
    if not raw_subs:
        return []

    sorted_subs = sorted(raw_subs, key=lambda s: s["start_page"])
    return _build_subsection_results(sorted_subs, section, result)


def _detect_subsections(
    result: ExtractionResult,
    llm: LLMClient,
    sections: list[SectionResult],
) -> list[SectionResult]:
    """Add subsections to sections exceeding threshold.

    For XLSX files with chunks, subsections are created
    automatically from the existing chunks. For other
    file types, LLM detection is used.

    Params:
        result: ExtractionResult with pages
        llm: Initialized LLM client
        sections: Primary sections list

    Returns:
        list[SectionResult] -- primaries + subsections
    """
    stage_log = get_stage_logger(__name__, STAGE)
    threshold = get_subsection_token_threshold()
    combined: list[SectionResult] = []
    sub_count = 0
    sections_with_subs = 0

    for section in sections:
        combined.append(section)
        xlsx_chunk_count = sum(
            1
            for page in result.pages
            if page.page_number == section.page_start and page.chunk_id
        )
        if result.filetype == "xlsx" and xlsx_chunk_count > 1:
            subs = _xlsx_chunks_as_subsections(section, result.pages)
        else:
            if section.token_count <= threshold:
                continue
            subs = _detect_llm_subsections(section, result, llm)

        if not subs:
            continue

        sub_count += len(subs)
        sections_with_subs += 1
        combined.extend(subs)

    if sub_count:
        stage_log.info(
            "Detected %d subsections across %d sections",
            sub_count,
            sections_with_subs,
        )
    return combined


def _apply_section_ids(pages: list, sections: list[SectionResult]) -> None:
    """Assign the deepest owning section_id to each page or chunk.

    Params:
        pages: PageResult objects to update in place
        sections: Ordered primary sections and subsections

    Returns:
        None
    """
    unit_to_section: dict[str, str] = {}
    for section in sections:
        for unit_id in section.chunk_ids:
            unit_to_section[unit_id] = section.section_id

    for page in pages:
        page.section_id = unit_to_section.get(get_content_unit_id(page), "")


def detect_sections(
    result: ExtractionResult,
    llm: LLMClient,
) -> ExtractionResult:
    """Detect document sections and subsections.

    Routes to automatic detection for sheet_based and
    semantic structure types, or LLM-based detection
    for chapters, sections, and topic_based documents.
    Large sections are further divided into subsections.

    Params:
        result: ExtractionResult from upstream stage
        llm: Initialized LLM client

    Returns:
        ExtractionResult with sections populated
    """
    stage_log = get_stage_logger(__name__, STAGE)
    metadata = result.document_metadata
    structure_type = metadata.get("structure_type", "")

    stage_log.info(
        "Section detection starting — structure=%s, pages=%d",
        structure_type,
        len(result.pages),
    )

    if structure_type == "sheet_based":
        sections = _detect_sheet_based_sections(result)
    elif structure_type == "semantic":
        sections = _detect_semantic_sections(result)
    else:
        sections = _detect_llm_sections(result, llm, metadata)

    sections = _detect_subsections(result, llm, sections)
    phantom_count = len(sections)
    sections = [
        s
        for s in sections
        if not (s.token_count == 0 and s.page_start > s.page_end)
    ]
    phantom_count -= len(sections)
    if phantom_count:
        stage_log.info(
            "Dropped %d phantom sections (empty, inverted range)",
            phantom_count,
        )
    _apply_section_ids(result.pages, sections)

    result.sections = [asdict(s) for s in sections]

    primary = [s for s in sections if s.level == "section"]
    subs = [s for s in sections if s.level == "subsection"]
    stage_log.info(
        "Section detection complete — %d sections, %d subsections",
        len(primary),
        len(subs),
    )

    return result


# ------------------------------------------------------------------
# Public aliases for testing
# ------------------------------------------------------------------
deduplicate_pages = _deduplicate_pages
content_unit_id = _content_unit_id
parse_sheet_name = _parse_sheet_name
chunk_id_sort_key = _chunk_id_sort_key
detect_sheet_based_sections = _detect_sheet_based_sections
detect_semantic_sections = _detect_semantic_sections
parse_section_response = _parse_section_response
format_section_batch = _format_section_batch
detect_llm_sections = _detect_llm_sections
build_section_results = _build_section_results
build_subsection_results = _build_subsection_results
format_subsection_batch = _format_subsection_batch
xlsx_chunks_as_subsections = _xlsx_chunks_as_subsections
detect_llm_subsections = _detect_llm_subsections
detect_subsections_fn = _detect_subsections
apply_section_ids = _apply_section_ids
