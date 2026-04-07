"""Markdown chunker — split oversized markdown by LLM breakpoints."""

import json
import logging
from pathlib import Path
from typing import Any

from ...utils.config_setup import get_chunking_md_batch_size
from ...utils.file_types import PageResult
from ...utils.llm_connector import LLMClient
from ...utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _chunk_id_prefix(page: PageResult) -> str:
    """Build the prefix for child chunk IDs. Params: page. Returns: str."""
    if page.chunk_id:
        return page.chunk_id
    return str(page.parent_page_number or page.page_number)


def _root_page_number(page: PageResult) -> int:
    """Resolve the original source page number. Params: page. Returns: int."""
    return page.parent_page_number or page.page_number


def _index_lines(content: str) -> list[tuple[int, str]]:
    """Build 1-indexed (line_number, text) pairs.

    Params:
        content: Raw markdown string

    Returns:
        list of (int, str) tuples starting at line 1

    Example:
        >>> _index_lines("a\\nb")
        [(1, 'a'), (2, 'b')]
    """
    return [(i + 1, line) for i, line in enumerate(content.split("\n"))]


def _format_batch(
    indexed_lines: list[tuple[int, str]],
) -> str:
    """Format indexed lines as numbered text for LLM input.

    Params:
        indexed_lines: Subset of 1-indexed (line_number, text) pairs

    Returns:
        str — formatted as ``[1] text\\n[2] text\\n...``

    Example:
        >>> _format_batch([(1, "hello"), (2, "world")])
        '[1] hello\\n[2] world'
    """
    return "\n".join(f"[{num}] {text}" for num, text in indexed_lines)


def _parse_breakpoints_response(response: dict) -> list[int]:
    """Extract breakpoint line numbers from an LLM tool-call response.

    Params:
        response: Full API response dict from LLMClient.call()

    Returns:
        list[int] — sorted breakpoint line numbers

    Raises:
        ValueError: When the response structure is malformed

    Example:
        >>> _parse_breakpoints_response(resp)
        [5, 10]
    """
    choices = response.get("choices")
    if not choices:
        raise ValueError("Response has no choices")

    message = choices[0].get("message")
    if message is None:
        raise ValueError("First choice has no message")

    tool_calls = message.get("tool_calls")
    if not tool_calls:
        raise ValueError("Message has no tool_calls")

    arguments_raw = tool_calls[0].get("function", {}).get("arguments")
    if not arguments_raw:
        raise ValueError("Tool call has no arguments")

    try:
        parsed = json.loads(arguments_raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Cannot parse tool arguments: {exc}") from exc

    breakpoints = parsed.get("breakpoints")
    if not isinstance(breakpoints, list):
        raise ValueError("Arguments missing 'breakpoints' list")

    return sorted(int(bp) for bp in breakpoints)


def _collect_breakpoints(
    indexed_lines: list[tuple[int, str]],
    llm: LLMClient,
    prompt: dict[str, Any],
    batch_size: int,
    embedding_limit: int,
    context: str,
) -> list[int]:
    """Iterate lines in batches, calling LLM for breakpoints.

    Params:
        indexed_lines: Full document as 1-indexed pairs
        llm: Configured LLM client
        prompt: Loaded prompt dict with system_prompt, user_prompt,
            stage, tools, and tool_choice
        batch_size: Number of lines per LLM call
        embedding_limit: Target token limit per chunk
        context: Log label for LLM calls

    Returns:
        list[int] — sorted, deduplicated breakpoint line numbers

    Example:
        >>> _collect_breakpoints(lines, llm, prompt, 100, 8192, "")
        [10, 25, 50]
    """
    all_breakpoints: set[int] = set()

    for start in range(0, len(indexed_lines), batch_size):
        batch = indexed_lines[start : start + batch_size]
        batch_content = _format_batch(batch)

        user_text = prompt["user_prompt"].format(
            batch_content=batch_content,
            embedding_limit=embedding_limit,
        )

        messages: list[dict[str, str]] = []
        system_prompt = prompt.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})

        response = llm.call(
            messages=messages,
            stage=prompt["stage"],
            tools=prompt["tools"],
            tool_choice=prompt.get("tool_choice", "required"),
            context=context,
        )

        try:
            batch_bps = _parse_breakpoints_response(response)
            all_breakpoints.update(batch_bps)
        except ValueError:
            logger.warning(
                "Breakpoint parse failed at line %d",
                batch[0][0],
            )

    return sorted(all_breakpoints)


def _assemble_chunks(
    content: str,
    breakpoints: list[int],
    page: PageResult,
) -> list[PageResult]:
    """Split content at breakpoint line numbers into PageResults.

    Params:
        content: Full markdown content string
        breakpoints: Sorted 1-indexed line numbers where new chunks
            begin
        page: Original PageResult supplying metadata

    Returns:
        list[PageResult] — one per chunk with metadata set

    Example:
        >>> chunks = _assemble_chunks(text, [5, 10], page)
        >>> chunks[0].chunk_id
        '1.1'
    """
    lines = content.split("\n")
    total_lines = len(lines)

    edges = [1] + [bp for bp in breakpoints if 1 < bp <= total_lines]
    if edges[-1] != total_lines + 1:
        edges.append(total_lines + 1)

    chunk_prefix = _chunk_id_prefix(page)
    root_page_number = _root_page_number(page)
    chunks: list[PageResult] = []
    for i in range(len(edges) - 1):
        start_line = edges[i]
        end_line = edges[i + 1] - 1
        chunk_lines = lines[start_line - 1 : end_line]
        chunk_content = "\n".join(chunk_lines)

        chunks.append(
            PageResult(
                page_number=page.page_number,
                raw_content=chunk_content,
                token_count=0,
                token_tier="",
                chunk_id=f"{chunk_prefix}.{i + 1}",
                parent_page_number=root_page_number,
                layout_type=page.layout_type,
                chunk_context=(f"Lines {start_line}-{end_line}"),
            )
        )

    return chunks


def chunk_markdown_page(
    page: PageResult,
    llm: LLMClient,
    embedding_limit: int,
) -> list[PageResult]:
    """Split an oversized markdown page into smaller chunks.

    Loads the markdown chunking prompt, indexes lines, collects
    LLM-identified breakpoints, and assembles chunk PageResults.
    When the LLM returns no breakpoints the full page is returned
    as a single chunk.

    Params:
        page: PageResult with oversized markdown content
        llm: Configured LLM client
        embedding_limit: Target token limit per chunk

    Returns:
        list[PageResult] — one or more chunks with metadata

    Example:
        >>> chunks = chunk_markdown_page(page, llm, 8192)
        >>> len(chunks) >= 1
        True
    """
    prompt = load_prompt("markdown_chunking", prompts_dir=_PROMPTS_DIR)
    batch_size = get_chunking_md_batch_size()

    indexed = _index_lines(page.raw_content)
    context = f"page {page.page_number} markdown chunking"

    breakpoints = _collect_breakpoints(
        indexed_lines=indexed,
        llm=llm,
        prompt=prompt,
        batch_size=batch_size,
        embedding_limit=embedding_limit,
        context=context,
    )

    if not breakpoints:
        chunk_prefix = _chunk_id_prefix(page)
        root_page_number = _root_page_number(page)
        return [
            PageResult(
                page_number=page.page_number,
                raw_content=page.raw_content,
                token_count=0,
                token_tier="",
                chunk_id=f"{chunk_prefix}.1",
                parent_page_number=root_page_number,
                layout_type=page.layout_type,
                chunk_context=("Lines 1-" + str(len(indexed))),
            )
        ]

    return _assemble_chunks(page.raw_content, breakpoints, page)


# ---- Expose internals for testing ----
index_lines = _index_lines
format_batch = _format_batch
parse_breakpoints_response = _parse_breakpoints_response
collect_breakpoints = _collect_breakpoints
assemble_chunks = _assemble_chunks
