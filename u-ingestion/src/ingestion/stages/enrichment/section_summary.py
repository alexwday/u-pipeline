"""Stage 9: Section summarization with progressive TOC.

Generates summaries for each primary section and builds a
progressive table of contents. Sections are processed
sequentially in document order so each batch sees the TOC
built from prior sections. Small sections can be grouped
into one LLM call within the token budget.
"""

import json
import logging
from pathlib import Path
from typing import Any

from ...utils.config_setup import (
    get_section_summary_batch_budget,
    get_section_summary_max_retries,
    get_section_summary_retry_delay,
)
from ...utils.file_types import ExtractionResult, get_content_unit_id
from ...utils.llm_connector import LLMClient
from ...utils.llm_retry import call_with_retry
from ...utils.logging_setup import get_stage_logger
from ...utils.prompt_loader import load_prompt
from ...utils.source_context import get_result_source_context
from ...utils.token_counting import count_message_tokens

STAGE = "9-SECTION_SUMMARY"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

logger = logging.getLogger(__name__)


def _get_primary_sections(
    sections: list[dict],
) -> list[dict]:
    """Filter sections to primary level only.

    Params:
        sections: All section dicts from ExtractionResult

    Returns:
        list[dict] -- sections where level is "section",
            sorted by sequence
    """
    primaries = [s for s in sections if s.get("level") == "section"]
    return sorted(primaries, key=lambda s: s.get("sequence", 0))


def _find_page_for_unit(
    unit_id: str,
    pages: list,
) -> Any:
    """Find the page matching a content unit identifier.

    Params:
        unit_id: chunk_id or str(page_number) to match
        pages: All PageResult objects

    Returns:
        PageResult or None
    """
    for page in pages:
        if get_content_unit_id(page) == unit_id:
            return page
    return None


def _gather_section_content(
    section: dict,
    pages: list,
) -> str:
    """Build content string for one unchunked section.

    Collects raw_content from pages matching the section's
    chunk_ids. Includes per-unit keywords and entities from
    the content extraction stage.

    Params:
        section: Section dict with chunk_ids and metadata
        pages: All PageResult objects

    Returns:
        str -- formatted section content for the LLM
    """
    section_id = section.get("section_id", "")
    title = section.get("title", "")
    page_start = section.get("page_start", 0)
    page_end = section.get("page_end", 0)
    page_range = str(page_start)
    if page_end and page_end != page_start:
        page_range = f"{page_start}-{page_end}"

    header = (
        f'<section id="{section_id}" title="{title}" ' f'pages="{page_range}">'
    )

    unit_lines: list[str] = []
    chunk_ids = section.get("chunk_ids", [])
    for uid in chunk_ids:
        page = _find_page_for_unit(uid, pages)
        if page is None:
            continue
        kw_line = ""
        if page.keywords:
            kw_str = ", ".join(f'"{k}"' for k in page.keywords)
            kw_line = f"  keywords: [{kw_str}]\n"
        ent_line = ""
        if page.entities:
            ent_str = ", ".join(f'"{e}"' for e in page.entities)
            ent_line = f"  entities: [{ent_str}]\n"
        unit_lines.append(f"{kw_line}{ent_line}" f"  {page.raw_content}")

    body = "\n\n".join(unit_lines)
    return f"{header}\n{body}\n</section>"


def _gather_xlsx_section_content(
    section: dict,
    pages: list,
) -> str:
    """Build content for a chunked XLSX section.

    Includes chunk_header and sheet_passthrough_content once
    at the top, then raw_content for each chunk in order.
    Also includes per-unit keywords and entities.

    Params:
        section: Section dict with chunk_ids and metadata
        pages: All PageResult objects

    Returns:
        str -- formatted XLSX section content for the LLM
    """
    section_id = section.get("section_id", "")
    title = section.get("title", "")
    page_start = section.get("page_start", 0)

    header = (
        f'<section id="{section_id}" title="{title}" ' f'pages="{page_start}">'
    )

    chunk_ids = section.get("chunk_ids", [])
    chunk_pages = []
    for uid in chunk_ids:
        page = _find_page_for_unit(uid, pages)
        if page is not None:
            chunk_pages.append((uid, page))

    parts = [
        block
        for block in [
            _format_xlsx_context_block(
                "sheet_header",
                _first_chunk_page_value(chunk_pages, "chunk_header"),
            ),
            _format_xlsx_context_block(
                "sheet_context",
                _first_chunk_page_value(
                    chunk_pages,
                    "sheet_passthrough_content",
                ),
            ),
            _format_xlsx_context_block(
                "section_context",
                _first_chunk_page_value(
                    chunk_pages,
                    "section_passthrough_content",
                ),
            ),
        ]
        if block
    ]

    unit_lines = [
        _format_xlsx_unit_line(uid, page) for uid, page in chunk_pages
    ]

    if unit_lines:
        units_block = "\n\n".join(unit_lines)
        parts.append(f"  <units>\n{units_block}\n  </units>")

    body = "\n\n".join(parts)
    return f"{header}\n{body}\n</section>"


def _first_chunk_page_value(
    chunk_pages: list[tuple[str, Any]],
    attribute: str,
) -> str:
    """Return the first non-empty page attribute.

    Params: chunk_pages (list[tuple[str, Any]]), attribute (str).
    Returns: str.
    """
    for _, page in chunk_pages:
        value = getattr(page, attribute, "")
        if value:
            return value
    return ""


def _format_xlsx_context_block(tag: str, value: str) -> str:
    """Format an optional XLSX context block.

    Params: tag (str), value (str). Returns: str.
    """
    if not value:
        return ""
    return f"  <{tag}>\n  {value}\n  </{tag}>"


def _format_xlsx_unit_line(uid: str, page: Any) -> str:
    """Format one XLSX chunk with keyword/entity metadata.

    Params: uid (str), page (Any). Returns: str.
    """
    context_label = page.chunk_context or uid
    suffix_parts: list[str] = []
    if page.keywords:
        suffix_parts.append(
            " keywords: ["
            + ", ".join(f'"{keyword}"' for keyword in page.keywords)
            + "]"
        )
    if page.entities:
        suffix_parts.append(
            " entities: ["
            + ", ".join(f'"{entity}"' for entity in page.entities)
            + "]"
        )
    suffix = "".join(suffix_parts)
    return (
        f"  [Chunk {uid} - {context_label}]"
        f"{suffix}\n"
        f"  {page.raw_content}"
    )


def _section_is_xlsx_chunked(
    section: dict,
    pages: list,
) -> bool:
    """Check if a section contains XLSX chunks.

    Params:
        section: Section dict with chunk_ids
        pages: All PageResult objects

    Returns:
        bool -- True if any matching page has a chunk_id
    """
    for uid in section.get("chunk_ids", []):
        page = _find_page_for_unit(uid, pages)
        if page is not None and page.chunk_id:
            return True
    return False


def _build_progressive_toc(
    completed: list[dict],
) -> str:
    """Format completed sections as a progressive TOC.

    Params:
        completed: List of dicts with section_id, title,
            page_start, and summary

    Returns:
        str -- formatted TOC block or empty string
    """
    if not completed:
        return ""

    lines: list[str] = []
    for entry in completed:
        sid = entry.get("section_id", "")
        title = entry.get("title", "")
        page = entry.get("page_start", "")
        summary = entry.get("summary", "")
        snippet = summary[:80] if summary else ""
        if snippet and len(summary) > 80:
            snippet += "..."
        lines.append(f"  [{sid}] {title} (p.{page})" f' -- "{snippet}"')

    toc_block = "\n".join(lines)
    return (
        f"<table_of_contents_so_far>\n"
        f"{toc_block}\n"
        f"</table_of_contents_so_far>"
    )


def _estimate_section_tokens(
    section: dict,
    pages: list,
) -> int:
    """Estimate total tokens for a section's content.

    Params:
        section: Section dict with chunk_ids
        pages: All PageResult objects

    Returns:
        int -- sum of raw_token_count for matching pages
    """
    total = 0
    for uid in section.get("chunk_ids", []):
        page = _find_page_for_unit(uid, pages)
        if page is not None:
            total += page.raw_token_count or 0
    return total


def _batch_sections(
    sections: list[dict],
    pages: list,
    budget: int,
    doc_context: str = "",
    toc_so_far: str = "",
    prompt: dict[str, Any] | None = None,
) -> list[list[dict]]:
    """Group sections into batches within token budget.

    When prompt data is provided, batches are sized against
    the fully formatted request. Otherwise raw section token
    estimates are used as a fallback.

    Params:
        sections: Primary sections sorted by sequence
        pages: All PageResult objects
        budget: Maximum total tokens per batch
        doc_context: Optional formatted document context
        toc_so_far: Optional progressive TOC block
        prompt: Optional loaded prompt for full-message sizing

    Returns:
        list[list[dict]] -- batches of section dicts
    """
    if not sections:
        return []

    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_tokens = 0

    for section in sections:
        candidate_batch = current_batch + [section]
        if prompt is not None:
            user_message = _format_summary_batch(
                candidate_batch,
                pages,
                doc_context,
                toc_so_far,
                prompt,
            )
            messages = [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": user_message},
            ]
            exceeds_budget = (
                bool(current_batch)
                and count_message_tokens(messages, prompt.get("tools"))
                > budget
            )
        else:
            cost = _estimate_section_tokens(section, pages)
            exceeds_budget = (
                bool(current_batch) and current_tokens + cost > budget
            )

        if exceeds_budget:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
            candidate_batch = [section]

        current_batch = candidate_batch
        current_tokens += _estimate_section_tokens(section, pages)

    if current_batch:
        batches.append(current_batch)

    return batches


def _build_doc_context(result: ExtractionResult) -> str:
    """Build document context string for the LLM.

    Params:
        result: ExtractionResult with metadata

    Returns:
        str -- formatted document context block
    """
    metadata = result.document_metadata or {}
    source_context = get_result_source_context(result)

    lines: list[str] = []
    title = metadata.get("title", "")
    if title:
        lines.append(f'title: "{title}"')
    if source_context["data_source"]:
        lines.append(f'data_source: "{source_context["data_source"]}"')
    for filter_key in ("filter_1", "filter_2", "filter_3"):
        if source_context[filter_key]:
            lines.append(f'{filter_key}: "{source_context[filter_key]}"')
    if metadata.get("structure_type"):
        lines.append(f'structure_type: "{metadata["structure_type"]}"')
    if not lines:
        return ""
    ctx = "\n".join(lines)
    return f"<document_context>\n{ctx}\n</document_context>"


def _format_summary_batch(
    batch: list[dict],
    pages: list,
    doc_context: str,
    toc_so_far: str,
    prompt: dict[str, Any],
) -> str:
    """Build the full user message for one summary batch.

    Params:
        batch: List of section dicts in this batch
        pages: All PageResult objects
        doc_context: Formatted document context block
        toc_so_far: Progressive TOC block
        prompt: Loaded prompt dict with user_prompt

    Returns:
        str -- complete user message
    """
    parts: list[str] = []

    if doc_context:
        parts.append(doc_context)

    if toc_so_far:
        parts.append(toc_so_far)

    section_blocks: list[str] = []
    for section in batch:
        if _section_is_xlsx_chunked(section, pages):
            block = _gather_xlsx_section_content(section, pages)
        else:
            block = _gather_section_content(section, pages)
        section_blocks.append(block)

    sections_block = "\n\n".join(section_blocks)
    parts.append(f"<sections>\n{sections_block}\n</sections>")

    user_input = "\n\n".join(parts)
    return prompt["user_prompt"].format(user_input=user_input)


def _parse_summary_response(
    response: dict,
) -> dict[str, dict]:
    """Parse LLM response into section_id -> summary map.

    Params:
        response: Raw LLM response dict

    Returns:
        dict[str, dict] -- mapping of section_id to
            {summary, keywords, entities}

    Raises:
        ValueError: When response has no valid tool call
            or items field
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

    items = arguments.get("items")
    if not isinstance(items, list):
        raise ValueError("Response missing or invalid items field")

    result: dict[str, dict] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Response item {idx} must be an object, got {type(item)}"
            )
        sid = item.get("section_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"Response item {idx} missing valid section_id")
        if sid in result:
            raise ValueError(f"Response contains duplicate section_id: {sid}")
        summary = item.get("summary")
        keywords = item.get("keywords")
        entities = item.get("entities")
        if not isinstance(summary, str):
            raise ValueError(f"Response item {sid} has invalid summary field")
        if not isinstance(keywords, list) or any(
            not isinstance(value, str) for value in keywords
        ):
            raise ValueError(f"Response item {sid} has invalid keywords field")
        if not isinstance(entities, list) or any(
            not isinstance(value, str) for value in entities
        ):
            raise ValueError(f"Response item {sid} has invalid entities field")
        result[sid] = {
            "summary": summary,
            "keywords": keywords,
            "entities": entities,
        }
    return result


def _validate_batch_results(
    batch: list[dict],
    batch_results: dict[str, dict],
) -> None:
    """Ensure a batch returned exactly the requested section ids.

    Params:
        batch: Requested section batch
        batch_results: Parsed batch results keyed by section_id

    Returns:
        None

    Raises:
        ValueError: When returned ids differ from the batch
    """
    expected_ids = {section["section_id"] for section in batch}
    actual_ids = set(batch_results)
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    if missing_ids or extra_ids:
        problems: list[str] = []
        if missing_ids:
            problems.append(f"missing section_ids={missing_ids}")
        if extra_ids:
            problems.append(f"unexpected section_ids={extra_ids}")
        raise ValueError(
            "Batch response ids do not match request: " + ", ".join(problems)
        )


def _update_sections(
    result: ExtractionResult,
    all_summaries: dict[str, dict],
) -> None:
    """Apply summaries, keywords, and entities to sections.

    Params:
        result: ExtractionResult whose sections are updated
        all_summaries: section_id -> {summary, keywords,
            entities}

    Returns:
        None
    """
    for section in result.sections:
        sid = section.get("section_id", "")
        data = all_summaries.get(sid)
        if data:
            section["summary"] = data["summary"]
            section["keywords"] = list(data["keywords"])
            section["entities"] = list(data["entities"])


def _build_toc_entries(
    result: ExtractionResult,
) -> list[dict]:
    """Build final TOC entries from summarized sections.

    Params:
        result: ExtractionResult with sections

    Returns:
        list[dict] -- TOC entries with section_id, title,
            page_start, and summary
    """
    entries: list[dict] = []
    primaries = _get_primary_sections(result.sections)
    for section in primaries:
        entries.append(
            {
                "section_id": section.get("section_id", ""),
                "title": section.get("title", ""),
                "page_start": section.get("page_start", 0),
                "summary": section.get("summary", ""),
            }
        )
    return entries


def _process_batches(
    sections: list[dict],
    result: ExtractionResult,
    llm: LLMClient,
    doc_context: str,
    prompt: dict[str, Any],
    budget: int,
) -> tuple[dict[str, dict], int]:
    """Process section batches sequentially with TOC.

    Calls the LLM for each batch, building a progressive
    TOC from completed sections to feed into subsequent
    calls.

    Params:
        sections: Ordered primary section dicts
        result: ExtractionResult with pages
        llm: Initialized LLM client
        doc_context: Formatted document context
        prompt: Loaded prompt dict
        budget: Maximum request token budget per batch

    Returns:
        tuple[dict[str, dict], int] -- all section summaries
            keyed by section_id, plus batch count
    """
    all_summaries: dict[str, dict] = {}
    completed_toc: list[dict] = []
    batch_count = 0
    remaining_sections = list(sections)

    while remaining_sections:
        toc_so_far = _build_progressive_toc(completed_toc)
        batch = _batch_sections(
            remaining_sections,
            result.pages,
            budget,
            doc_context=doc_context,
            toc_so_far=toc_so_far,
            prompt=prompt,
        )[0]
        user_message = _format_summary_batch(
            batch,
            result.pages,
            doc_context,
            toc_so_far,
            prompt,
        )
        messages = [
            {
                "role": "system",
                "content": prompt["system_prompt"],
            },
            {"role": "user", "content": user_message},
        ]
        batch_results = call_with_retry(
            llm,
            messages,
            prompt,
            parser=_parse_summary_response,
            stage="section_summary",
            context=(
                f"section_summary:"
                f"{Path(result.file_path).name}:"
                f"batch_{batch_count + 1}"
            ),
            max_retries=get_section_summary_max_retries(),
            retry_delay=get_section_summary_retry_delay(),
            validator=lambda parsed, _batch=batch: _validate_batch_results(
                _batch, parsed
            ),
        )
        all_summaries.update(batch_results)
        batch_count += 1

        for section in batch:
            sid = section.get("section_id", "")
            data = batch_results.get(sid, {})
            completed_toc.append(
                {
                    "section_id": sid,
                    "title": section.get("title", ""),
                    "page_start": section.get("page_start", 0),
                    "summary": data.get("summary", ""),
                }
            )
        remaining_sections = remaining_sections[len(batch) :]

    return all_summaries, batch_count


def summarize_sections(
    result: ExtractionResult,
    llm: LLMClient,
) -> ExtractionResult:
    """Summarize each primary section with progressive TOC.

    Loads the section_summary prompt, batches primary
    sections by token budget, processes batches sequentially
    to build a progressive table of contents, and updates
    section summaries, keywords, and entities.

    Params:
        result: ExtractionResult from upstream stage
        llm: Initialized LLM client

    Returns:
        ExtractionResult with section summaries populated
            and generated_toc_entries updated in
            document_metadata
    """
    stage_log = get_stage_logger(__name__, STAGE)

    primaries = _get_primary_sections(result.sections)
    if not primaries:
        stage_log.info("No primary sections — skipping")
        return result

    prompt = load_prompt("section_summary", prompts_dir=_PROMPTS_DIR)
    budget = get_section_summary_batch_budget()

    doc_context = _build_doc_context(result)
    all_summaries, batch_count = _process_batches(
        primaries,
        result,
        llm,
        doc_context,
        prompt,
        budget,
    )

    _update_sections(result, all_summaries)

    xlsx_subs = [
        s
        for s in result.sections
        if s.get("level") == "subsection"
        and s.get("parent_section_id")
        and not s.get("summary")
    ]
    if xlsx_subs:
        sub_summaries, sub_batches = _process_batches(
            xlsx_subs,
            result,
            llm,
            doc_context,
            prompt,
            budget,
        )
        _update_sections(result, sub_summaries)
        batch_count += sub_batches

    toc_entries = _build_toc_entries(result)
    if result.document_metadata is None:
        result.document_metadata = {}
    result.document_metadata["generated_toc_entries"] = toc_entries

    total_summaries = sum(1 for s in result.sections if s.get("summary"))
    stage_log.info(
        "Section summary complete — %d sections "
        "summarized, %d batches, %d TOC entries",
        total_summaries,
        batch_count,
        len(toc_entries),
    )

    return result


# ------------------------------------------------------------------
# Public aliases for testing
# ------------------------------------------------------------------
get_primary_sections = _get_primary_sections
find_page_for_unit = _find_page_for_unit
gather_section_content = _gather_section_content
gather_xlsx_section_content = _gather_xlsx_section_content
section_is_xlsx_chunked = _section_is_xlsx_chunked
build_progressive_toc = _build_progressive_toc
estimate_section_tokens = _estimate_section_tokens
batch_sections = _batch_sections
build_doc_context = _build_doc_context
format_summary_batch = _format_summary_batch
parse_summary_response = _parse_summary_response
first_chunk_page_value = _first_chunk_page_value
format_xlsx_context_block = _format_xlsx_context_block
format_xlsx_unit_line = _format_xlsx_unit_line
validate_batch_results = _validate_batch_results
update_sections = _update_sections
build_toc_entries = _build_toc_entries
process_batches = _process_batches
