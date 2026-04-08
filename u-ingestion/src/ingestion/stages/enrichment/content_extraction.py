"""Stage 8: Keyword and entity extraction per content unit.

Extracts keywords and named entities from every content
unit (page or chunk) for sparse retrieval. Units are
batched by token budget to minimize LLM calls.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import openai

from ...utils.config_setup import (
    get_content_extraction_batch_budget,
    get_content_extraction_max_retries,
    get_content_extraction_retry_delay,
)
from ...utils.file_types import ExtractionResult, get_content_unit_id
from ...utils.llm_connector import LLMClient
from ...utils.logging_setup import get_stage_logger
from ...utils.prompt_loader import load_prompt
from ...utils.source_context import get_result_source_context
from ...utils.token_counting import count_message_tokens

STAGE = "8-CONTENT_EXTRACTION"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_SHEET_NAME_RE = re.compile(r"^#\s+Sheet:\s+(.+)$", re.MULTILINE)
_HEADING_RE = re.compile(r"^#+\s+(.+)$", re.MULTILINE)

_RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    ValueError,
)

logger = logging.getLogger(__name__)


def _get_unit_name(page: Any) -> str:
    """Extract a display name from page content.

    For XLSX pages the sheet name is parsed from the
    header line. For other types the first markdown
    heading is used.

    Params:
        page: PageResult object

    Returns:
        str -- sheet name, heading, or empty string
    """
    match = _SHEET_NAME_RE.search(page.raw_content)
    if match:
        return match.group(1).strip()
    match = _HEADING_RE.search(page.raw_content)
    if match:
        return match.group(1).strip()
    return ""


def _build_unit_list(result: ExtractionResult) -> list[dict]:
    """Build a list of content units with identifiers.

    Each unit includes its id, page number, display name,
    raw content, token count, and optional chunk context.

    Params:
        result: ExtractionResult with pages

    Returns:
        list[dict] -- unit descriptors sorted by page
            number then chunk_id
    """
    units: list[dict] = []
    sorted_pages = sorted(
        result.pages,
        key=lambda p: (p.page_number, p.chunk_id),
    )
    for page in sorted_pages:
        units.append(
            {
                "unit_id": get_content_unit_id(page),
                "page_number": page.page_number,
                "name": _get_unit_name(page),
                "raw_content": page.raw_content,
                "raw_token_count": page.raw_token_count or 0,
                "context": page.chunk_context or "",
                "section_id": page.section_id or "",
            }
        )
    return units


def _batch_units(
    units: list[dict],
    budget: int,
    doc_context: dict | None = None,
    prompt: dict[str, Any] | None = None,
) -> list[list[dict]]:
    """Group content units into batches within token budget.

    When prompt and document context are provided, batches
    are sized against the fully formatted request. Without
    them, raw_token_count is used as a fallback estimate.

    Params:
        units: List of unit descriptors with raw_token_count
        budget: Maximum total raw_token_count per batch
        doc_context: Optional document context for formatting
        prompt: Optional loaded prompt for full-message sizing

    Returns:
        list[list[dict]] -- batches of unit descriptors
    """
    if not units:
        return []

    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_tokens = 0

    for unit in units:
        candidate_batch = current_batch + [unit]
        if prompt is not None and doc_context is not None:
            user_message = _format_batch(
                candidate_batch,
                doc_context,
                prompt,
            )
            messages = [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": user_message},
            ]
            exceeds_budget = (
                bool(current_batch) and count_message_tokens(messages) > budget
            )
        else:
            cost = unit["raw_token_count"]
            exceeds_budget = (
                bool(current_batch) and current_tokens + cost > budget
            )

        if exceeds_budget:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
            candidate_batch = [unit]

        current_batch = candidate_batch
        current_tokens += unit["raw_token_count"]

    if current_batch:
        batches.append(current_batch)

    return batches


def _format_batch(
    batch: list[dict],
    doc_context: dict,
    prompt: dict[str, Any],
) -> str:
    """Build the full user message for one batch.

    Formats document context and content units with XML
    tags, then injects into the prompt template.

    Params:
        batch: List of unit descriptors
        doc_context: Dict with title and data_source
        prompt: Loaded prompt dict with user_prompt

    Returns:
        str -- complete user message
    """
    parts: list[str] = []

    ctx_lines = []
    if doc_context.get("title"):
        ctx_lines.append(f"title: \"{doc_context['title']}\"")
    if doc_context.get("data_source"):
        ctx_lines.append(f"data_source: \"{doc_context['data_source']}\"")
    for filter_key in ("filter_1", "filter_2", "filter_3"):
        if doc_context.get(filter_key):
            ctx_lines.append(f'{filter_key}: "{doc_context[filter_key]}"')
    if ctx_lines:
        ctx_block = "\n".join(ctx_lines)
        parts.append(
            f"<document_context>\n" f"{ctx_block}\n" f"</document_context>"
        )

    unit_lines: list[str] = []
    for unit in batch:
        attrs = [f'id="{unit["unit_id"]}"']
        attrs.append(f'page="{unit["page_number"]}"')
        if unit["name"]:
            attrs.append(f'name="{unit["name"]}"')
        if unit["context"]:
            attrs.append(f'context="{unit["context"]}"')
        attr_str = " ".join(attrs)
        unit_lines.append(
            f"<unit {attr_str}>\n" f"{unit['raw_content']}\n" f"</unit>"
        )

    units_block = "\n\n".join(unit_lines)
    parts.append(f"<content_units>\n{units_block}\n</content_units>")

    user_input = "\n\n".join(parts)
    return prompt["user_prompt"].format(user_input=user_input)


def _parse_extraction_response(
    response: dict,
) -> dict[str, dict]:
    """Parse LLM response into unit_id -> extraction map.

    Params:
        response: Raw LLM response dict

    Returns:
        dict[str, dict] -- mapping of unit_id to
            {keywords, entities}

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
        uid = item.get("unit_id")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"Response item {idx} missing valid unit_id")
        if uid in result:
            raise ValueError(f"Response contains duplicate unit_id: {uid}")
        keywords = item.get("keywords")
        entities = item.get("entities")
        if not isinstance(keywords, list) or any(
            not isinstance(value, str) for value in keywords
        ):
            raise ValueError(f"Response item {uid} has invalid keywords field")
        if not isinstance(entities, list) or any(
            not isinstance(value, str) for value in entities
        ):
            raise ValueError(f"Response item {uid} has invalid entities field")
        result[uid] = {
            "keywords": keywords,
            "entities": entities,
        }
    return result


def _validate_batch_results(
    batch: list[dict],
    batch_results: dict[str, dict],
) -> None:
    """Ensure a batch returned exactly the requested unit ids.

    Params:
        batch: Requested content-unit batch
        batch_results: Parsed batch results keyed by unit_id

    Returns:
        None

    Raises:
        ValueError: When returned ids differ from the batch
    """
    expected_ids = {unit["unit_id"] for unit in batch}
    actual_ids = set(batch_results)
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    if missing_ids or extra_ids:
        problems: list[str] = []
        if missing_ids:
            problems.append(f"missing unit_ids={missing_ids}")
        if extra_ids:
            problems.append(f"unexpected unit_ids={extra_ids}")
        raise ValueError(
            "Batch response ids do not match request: " + ", ".join(problems)
        )


def _apply_to_pages(
    result: ExtractionResult,
    all_extractions: dict[str, dict],
) -> None:
    """Set keywords and entities on PageResult objects.

    Params:
        result: ExtractionResult whose pages are updated
        all_extractions: unit_id -> {keywords, entities}

    Returns:
        None
    """
    for page in result.pages:
        extraction = all_extractions.get(get_content_unit_id(page))
        if extraction:
            page.keywords = list(extraction["keywords"])
            page.entities = list(extraction["entities"])


def _build_doc_context(result: ExtractionResult) -> dict:
    """Build document context dict from result metadata.

    Params:
        result: ExtractionResult with document_metadata

    Returns:
        dict -- with title and data_source keys
    """
    metadata = result.document_metadata or {}
    source_context = get_result_source_context(result)
    return {
        "title": metadata.get("title", ""),
        "data_source": source_context["data_source"],
        "filter_1": source_context["filter_1"],
        "filter_2": source_context["filter_2"],
        "filter_3": source_context["filter_3"],
    }


def _build_content_units(result: ExtractionResult) -> list[dict]:
    """Materialize downstream content-unit records from pages.

    Params:
        result: ExtractionResult with enriched pages

    Returns:
        list[dict] -- normalized content-unit records
    """
    content_units: list[dict] = []
    for page in result.pages:
        content_units.append(
            {
                "content_unit_id": get_content_unit_id(page),
                "chunk_id": page.chunk_id,
                "section_id": page.section_id,
                "page_number": page.page_number,
                "parent_page_number": page.parent_page_number,
                "raw_content": page.raw_content,
                "chunk_context": page.chunk_context,
                "chunk_header": page.chunk_header,
                "sheet_passthrough_content": (page.sheet_passthrough_content),
                "section_passthrough_content": (
                    page.section_passthrough_content
                ),
                "keywords": list(page.keywords),
                "entities": list(page.entities),
                "raw_token_count": page.raw_token_count,
                "embedding_token_count": page.embedding_token_count,
                "token_count": page.token_count,
            }
        )
    return content_units


def _call_with_retry(
    llm: LLMClient,
    messages: list,
    prompt: dict[str, Any],
    batch: list[dict],
    context: str,
) -> dict[str, dict]:
    """Call LLM for one batch with retry on transient and structural errors.

    Wraps llm.call + _parse_extraction_response + _validate_batch_results
    so that missing tool_calls, duplicate unit_ids, and batch-id
    mismatches — all known nondeterministic LLM failure modes — survive
    bounded retries instead of killing the whole file. Note: retrying at
    temperature=0 is unlikely to recover from deterministic failures;
    operators can bump CONTENT_EXTRACTION_MAX_RETRIES if observation
    shows retries aren't helping.

    Params:
        llm: LLMClient instance
        messages: Message list for the API call
        prompt: Loaded content_extraction prompt
        batch: Content-unit batch for validation
        context: Log label for the request

    Returns:
        dict[str, dict] -- parsed unit_id -> {keywords, entities}
    """
    max_retries = get_content_extraction_max_retries()
    retry_delay = get_content_extraction_retry_delay()

    for attempt in range(1, max_retries + 1):
        try:
            response = llm.call(
                messages=messages,
                stage="content_extraction",
                tools=prompt.get("tools"),
                tool_choice=prompt.get("tool_choice"),
                context=f"{context}:attempt_{attempt}",
            )
            batch_results = _parse_extraction_response(response)
            _validate_batch_results(batch, batch_results)
            return batch_results
        except _RETRYABLE_ERRORS as exc:
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


def extract_content(
    result: ExtractionResult,
    llm: LLMClient,
) -> ExtractionResult:
    """Extract keywords and entities per content unit.

    Loads the content_extraction prompt, batches all
    content units by token budget, calls the LLM for
    each batch, and sets keywords/entities on every
    PageResult.

    Params:
        result: ExtractionResult from upstream stage
        llm: Initialized LLM client

    Returns:
        ExtractionResult with keywords and entities
            populated on each page
    """
    stage_log = get_stage_logger(__name__, STAGE)

    if not result.pages:
        stage_log.info("No pages to extract — skipping")
        return result

    prompt = load_prompt("content_extraction", prompts_dir=_PROMPTS_DIR)
    budget = get_content_extraction_batch_budget()

    units = _build_unit_list(result)
    doc_context = _build_doc_context(result)
    batches = _batch_units(
        units,
        budget,
        doc_context=doc_context,
        prompt=prompt,
    )

    all_extractions: dict[str, dict] = {}

    for batch_idx, batch in enumerate(batches):
        user_message = _format_batch(batch, doc_context, prompt)
        messages = [
            {
                "role": "system",
                "content": prompt["system_prompt"],
            },
            {"role": "user", "content": user_message},
        ]
        batch_context = (
            f"content_extraction:"
            f"{Path(result.file_path).name}:"
            f"batch_{batch_idx + 1}"
        )
        batch_results = _call_with_retry(
            llm,
            messages,
            prompt,
            batch,
            batch_context,
        )
        all_extractions.update(batch_results)

    _apply_to_pages(result, all_extractions)
    result.content_units = _build_content_units(result)

    total_keywords = sum(len(page.keywords) for page in result.pages)
    total_entities = sum(len(page.entities) for page in result.pages)

    stage_log.info(
        "Content extraction complete — %d units, "
        "%d keywords, %d entities, %d batches",
        len(result.pages),
        total_keywords,
        total_entities,
        len(batches),
    )

    return result


# ------------------------------------------------------------------
# Public aliases for testing
# ------------------------------------------------------------------
get_unit_name = _get_unit_name
build_unit_list = _build_unit_list
batch_units = _batch_units
format_batch = _format_batch
parse_extraction_response = _parse_extraction_response
validate_batch_results = _validate_batch_results
apply_to_pages = _apply_to_pages
build_doc_context = _build_doc_context
build_content_units = _build_content_units
call_with_retry = _call_with_retry
