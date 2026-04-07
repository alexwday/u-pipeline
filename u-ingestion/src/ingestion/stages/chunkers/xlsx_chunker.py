"""XLSX chunker — split oversized sheet content by LLM breakpoints."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from ...utils.config_setup import (
    get_chunking_xlsx_batch_size,
    get_chunking_xlsx_header_rows,
    get_chunking_xlsx_overlap_rows,
)
from ...utils.file_types import PageResult
from ...utils.llm_connector import LLMClient
from ...utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_ROW_NUMBER_RE = re.compile(r"^\|\s*(\d+)\s*\|")


def _chunk_id_prefix(page: PageResult) -> str:
    """Build the prefix for child chunk IDs. Params: page. Returns: str."""
    if page.chunk_id:
        return page.chunk_id
    return str(page.parent_page_number or page.page_number)


def _root_page_number(page: PageResult) -> int:
    """Resolve the original source page number. Params: page. Returns: int."""
    return page.parent_page_number or page.page_number


def _parse_sheet_content(content: str) -> dict[str, Any]:
    """Parse XLSX markdown into structural components.

    Params:
        content: Markdown string produced by the extraction stage

    Returns:
        dict with keys heading, table_header, separator,
        data_rows (list of (row_number, line) tuples),
        and visual_blocks (list of blockquote strings)

    Example:
        >>> parsed = _parse_sheet_content("# Sheet: S1\\n...")
        >>> parsed["heading"]
        '# Sheet: S1\\n'
    """
    lines = content.split("\n")
    heading = ""
    table_header = ""
    separator = ""
    data_rows: list[tuple[int, str]] = []
    visual_blocks: list[str] = []
    past_table = False

    for line in lines:
        if not heading and line.startswith("#"):
            heading = line + "\n"
            continue

        if not table_header and line.startswith("| Row |"):
            table_header = line
            continue

        if not separator and line.startswith("| ---"):
            separator = line
            continue

        if past_table:
            if line.startswith(">"):
                visual_blocks.append(line)
            continue

        match = _ROW_NUMBER_RE.match(line)
        if match:
            row_num = int(match.group(1))
            data_rows.append((row_num, line))
        elif line.startswith(">"):
            past_table = True
            visual_blocks.append(line)

    return {
        "heading": heading,
        "table_header": table_header,
        "separator": separator,
        "data_rows": data_rows,
        "visual_blocks": visual_blocks,
    }


def _format_xlsx_batch(
    header_rows_context: list[str],
    sheet_passthrough_rows: list[str],
    section_context_rows: list[str],
    overlap_rows: list[str],
    batch_rows: list[tuple[int, str]],
) -> str:
    """Build XML-tagged batch content for the LLM prompt.

    Params:
        header_rows_context: Lines from the top of the sheet
        sheet_passthrough_rows: Permanent passthrough row lines
        section_context_rows: Section passthrough from prior batch
        overlap_rows: Trailing rows from the previous batch
        batch_rows: Current batch as (row_number, line) tuples

    Returns:
        str — formatted batch with XML section tags

    Example:
        >>> _format_xlsx_batch(["| 2 | T |"], [], [], [], [])
        '<sheet_context>\\n| 2 | T |\\n</sheet_context>'
    """
    sections: list[str] = []

    if header_rows_context:
        body = "\n".join(header_rows_context)
        sections.append(f"<sheet_context>\n{body}\n</sheet_context>")

    if sheet_passthrough_rows:
        body = "\n".join(sheet_passthrough_rows)
        sections.append(
            "<sheet_passthrough>\n" f"{body}\n" "</sheet_passthrough>"
        )

    if section_context_rows:
        body = "\n".join(section_context_rows)
        sections.append("<section_context>\n" f"{body}\n" "</section_context>")

    if overlap_rows:
        body = "\n".join(overlap_rows)
        sections.append(
            "<prior_chunk_overlap>\n" f"{body}\n" "</prior_chunk_overlap>"
        )

    if batch_rows:
        row_lines = [f"[{row_num}] {line}" for row_num, line in batch_rows]
        body = "\n".join(row_lines)
        sections.append(f"<current_batch>\n{body}\n</current_batch>")

    return "\n\n".join(sections)


def _parse_xlsx_response(
    response: dict,
) -> tuple[list[int], list[int], list[int]]:
    """Extract breakpoints and passthrough tiers from LLM response.

    Params:
        response: Raw LLM API response dict

    Returns:
        tuple of (breakpoints, sheet_passthrough_rows,
        section_passthrough_rows) integer lists

    Example:
        >>> _parse_xlsx_response({"choices": [...]})
        ([15], [2], [10])
    """
    try:
        args_str = response["choices"][0]["message"]["tool_calls"][0][
            "function"
        ]["arguments"]
        parsed = json.loads(args_str)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "Malformed LLM response: missing expected fields"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Malformed LLM response: invalid JSON in arguments"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("Malformed LLM response: arguments must be an object")

    breakpoints = parsed.get("breakpoints", [])
    sheet_pt = parsed.get("sheet_passthrough_rows", [])
    section_pt = parsed.get("section_passthrough_rows", [])

    if (
        not isinstance(breakpoints, list)
        or not isinstance(sheet_pt, list)
        or not isinstance(section_pt, list)
    ):
        raise ValueError(
            "Malformed LLM response: breakpoints, "
            "sheet_passthrough_rows, and "
            "section_passthrough_rows must be arrays"
        )

    return (
        [int(b) for b in breakpoints],
        [int(p) for p in sheet_pt],
        [int(s) for s in section_pt],
    )


def _build_batch_messages(
    prompt: dict[str, Any],
    batch_content: str,
    embedding_limit: int,
) -> list[dict[str, str]]:
    """Build LLM messages for one chunking batch.

    Params: prompt, batch_content, embedding_limit.
    Returns: list of message dicts.
    """
    user_text = prompt["user_prompt"]
    user_text = user_text.replace("{batch_content}", batch_content)
    user_text = user_text.replace("{embedding_limit}", str(embedding_limit))

    messages: list[dict[str, str]] = []
    if prompt.get("system_prompt"):
        messages.append(
            {
                "role": "system",
                "content": prompt["system_prompt"],
            }
        )
    messages.append({"role": "user", "content": user_text})
    return messages


def _row_content(line: str) -> str:
    """Extract cell content from a row line, ignoring the row number.

    Params: line (str). Returns: str — pipe-delimited cells after
        the row number column.
    """
    parts = line.split("|")
    if len(parts) > 2:
        return "|".join(parts[2:]).strip()
    return line


def _update_sheet_passthrough_lines(
    passthrough_lines: list[str],
    batch_pt: list[int],
    batch: list[tuple[int, str]],
) -> None:
    """Add newly flagged sheet passthrough lines in place.

    Deduplicates by cell content so repeated rows with
    different row numbers are not added twice.

    Params: passthrough_lines, batch_pt, batch. Returns: None.
    """
    existing_content = {_row_content(ln) for ln in passthrough_lines}
    row_lookup = dict(batch)
    for pt_num in batch_pt:
        if pt_num in row_lookup:
            pt_line = row_lookup[pt_num]
            content = _row_content(pt_line)
            if content not in existing_content:
                passthrough_lines.append(pt_line)
                existing_content.add(content)


def _build_section_context_lines(
    section_pt: list[int],
    row_lookup: dict[int, str],
) -> list[str]:
    """Build section passthrough lines from a row lookup.

    Unlike sheet passthrough, section lines do not accumulate.
    Each batch replaces the previous section context entirely.

    Params: section_pt, row_lookup. Returns: list of line strs.
    """
    lines: list[str] = []
    for pt_num in section_pt:
        if pt_num in row_lookup:
            lines.append(row_lookup[pt_num])
    return lines


def _call_llm_for_batch(
    llm: LLMClient,
    prompt: dict[str, Any],
    batch_content: str,
    embedding_limit: int,
    context: str,
) -> tuple[list[int], list[int], list[int]]:
    """Send one batch to the LLM and parse the response.

    Params: llm, prompt, batch_content, embedding_limit, context.
    Returns: tuple of (breakpoints, sheet_pt, section_pt).
    """
    messages = _build_batch_messages(
        prompt,
        batch_content,
        embedding_limit,
    )
    response = llm.call(
        messages=messages,
        stage=prompt["stage"],
        tools=prompt.get("tools"),
        tool_choice=prompt.get("tool_choice"),
        context=context,
    )
    return _parse_xlsx_response(response)


def _process_batch_result(
    batch_bp: list[int],
    batch_sheet_pt: list[int],
    batch_section_pt: list[int],
    batch: list[tuple[int, str]],
    state: dict[str, Any],
) -> None:
    """Apply one batch's LLM results to accumulator state.

    Params: batch_bp, batch_sheet_pt, batch_section_pt,
        batch, state. Returns: None.
    """
    state["all_bp"].extend(batch_bp)
    state["all_sheet_pt"].update(batch_sheet_pt)

    _update_sheet_passthrough_lines(
        state["sheet_pt_lines"], batch_sheet_pt, batch
    )

    batch_snap = sorted(set(batch_section_pt))
    for _ in batch_bp:
        state["section_snapshots"].append(batch_snap)

    state["section_pt_lines"] = _build_section_context_lines(
        batch_section_pt, state["all_row_lookup"]
    )

    overlap_count = state["overlap_count"]
    state["overlap"] = [line for _num, line in batch[-overlap_count:]]


def _collect_xlsx_breakpoints(
    parsed: dict[str, Any],
    llm: LLMClient,
    prompt: dict[str, Any],
    batch_size: int,
    row_counts: tuple[int, int],
    embedding_limit: int,
    context: str,
) -> tuple[list[int], list[int], list[list[int]]]:
    """Process data rows in batches to collect breakpoints.

    Sends batches of data rows to the LLM to identify where
    chunks should split, which rows are permanent sheet-level
    passthrough, and which are temporary section-level
    passthrough.

    Params:
        parsed: Output of _parse_sheet_content
        llm: LLM client instance
        prompt: Loaded prompt definition dict
        batch_size: Number of data rows per LLM call
        row_counts: Tuple of (header_row_count,
            overlap_row_count) controlling context sizes
        embedding_limit: Target token ceiling per chunk
        context: Log label for LLM calls

    Returns:
        tuple of (sorted deduplicated breakpoints,
        sorted deduplicated sheet passthrough row indices,
        section_snapshots list aligned 1:1 with breakpoints)
    """
    header_row_count, overlap_row_count = row_counts
    data_rows = parsed["data_rows"]
    header_ctx = [line for _num, line in data_rows[:header_row_count]]

    state: dict[str, Any] = {
        "all_bp": [],
        "all_sheet_pt": set(),
        "sheet_pt_lines": [],
        "section_pt_lines": [],
        "section_snapshots": [],
        "overlap": [],
        "all_row_lookup": dict(data_rows),
        "overlap_count": overlap_row_count,
    }

    for start in range(0, len(data_rows), batch_size):
        batch = data_rows[start : start + batch_size]

        batch_content = _format_xlsx_batch(
            header_ctx,
            state["sheet_pt_lines"],
            state["section_pt_lines"],
            state["overlap"],
            batch,
        )

        result = _call_llm_for_batch(
            llm,
            prompt,
            batch_content,
            embedding_limit,
            context,
        )
        _process_batch_result(result[0], result[1], result[2], batch, state)

    return (
        sorted(set(state["all_bp"])),
        sorted(state["all_sheet_pt"]),
        state["section_snapshots"],
    )


def _row_num_from_line(line: str) -> int:
    """Extract Excel row number from a table line.

    Params: line (str). Returns: int or -1 if not found.
    """
    match = _ROW_NUMBER_RE.match(line)
    if match:
        return int(match.group(1))
    return -1


def _build_chunk_prefix(parsed: dict[str, Any]) -> str:
    """Build the heading + table_header + separator prefix.

    Params: parsed (dict). Returns: str.
    """
    parts = [
        p
        for p in (
            parsed["heading"].rstrip("\n"),
            parsed["table_header"],
            parsed["separator"],
        )
        if p
    ]
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def _split_into_segments(
    data_rows: list[tuple[int, str]],
    breakpoints: list[int],
) -> list[list[tuple[int, str]]]:
    """Split data rows into segments at breakpoint boundaries.

    Params: data_rows, breakpoints. Returns: list of segments.
    """
    bp_set = set(breakpoints)
    segments: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []

    for row_num, line in data_rows:
        if row_num in bp_set and current:
            segments.append(current)
            current = []
        current.append((row_num, line))

    if current:
        segments.append(current)

    return segments


def _collect_sheet_pt_lines(
    data_rows: list[tuple[int, str]],
    sheet_pt_indices: list[int],
) -> list[str]:
    """Collect deduplicated sheet passthrough row lines.

    Params: data_rows, sheet_pt_indices. Returns: list[str].
    """
    sheet_pt_set = set(sheet_pt_indices)
    seen: set[str] = set()
    lines: list[str] = []
    for row_num, line in data_rows:
        if row_num in sheet_pt_set:
            cell_content = _row_content(line)
            if cell_content not in seen:
                lines.append(line)
                seen.add(cell_content)
    return lines


def _segment_raw_text(
    segment: list[tuple[int, str]],
    visuals: list[str],
    is_last: bool,
) -> str:
    """Assemble raw content text for one chunk segment.

    Params: segment, visuals, is_last. Returns: str.
    """
    raw_parts = [line for _rn, line in segment]
    if is_last and visuals:
        raw_parts.append("")
        raw_parts.extend(visuals)
    return "\n".join(raw_parts)


def _segment_section_text(
    snap: list[int],
    row_lookup: dict[int, str],
    segment: list[tuple[int, str]],
) -> str:
    """Build section passthrough text for one chunk.

    Params: snap, row_lookup, segment. Returns: str.
    """
    seg_lines = {ln for _, ln in segment}
    sec_lines = [
        row_lookup[rn]
        for rn in snap
        if rn in row_lookup and row_lookup[rn] not in seg_lines
    ]
    return "\n".join(sec_lines)


def _assemble_xlsx_chunks(
    parsed: dict[str, Any],
    breakpoints: list[int],
    sheet_pt_indices: list[int],
    section_snapshots: list[list[int]],
    page: PageResult,
) -> list[PageResult]:
    """Build PageResult chunks from breakpoints.

    Params:
        parsed: Output of _parse_sheet_content
        breakpoints: Sorted row numbers where chunks begin
        sheet_pt_indices: Row numbers included in every chunk
            as permanent sheet-level context
        section_snapshots: List aligned 1:1 with breakpoints;
            each entry is the section passthrough row indices
            active at that breakpoint
        page: Original PageResult being chunked

    Returns:
        list of PageResult instances, one per chunk

    Example:
        >>> chunks = _assemble_xlsx_chunks(
        ...     parsed, [15], [2], [[10]], p
        ... )
        >>> chunks[0].chunk_id
        '1.1'
    """
    data_rows = parsed["data_rows"]
    sheet_pt_lines = _collect_sheet_pt_lines(data_rows, sheet_pt_indices)
    row_lookup = dict(data_rows)

    prefix = _build_chunk_prefix(parsed)
    segments = _split_into_segments(data_rows, breakpoints)
    chunk_prefix = _chunk_id_prefix(page)
    table_header = parsed.get("table_header", "")
    separator = parsed.get("separator", "")
    header_parts = [p for p in (table_header, separator) if p]
    header_prefix = "\n".join(header_parts)
    pt_lines_with_header = (
        [header_prefix] + sheet_pt_lines
        if header_prefix and sheet_pt_lines
        else sheet_pt_lines
    )
    sheet_pt_text = "\n".join(pt_lines_with_header)

    chunks: list[PageResult] = []
    for idx, segment in enumerate(segments):
        snap = (
            section_snapshots[idx - 1]
            if idx > 0 and idx - 1 < len(section_snapshots)
            else []
        )
        row_nums = [rn for rn, _ in segment]
        chunks.append(
            PageResult(
                page_number=page.page_number,
                raw_content=_segment_raw_text(
                    segment,
                    parsed["visual_blocks"],
                    idx == len(segments) - 1,
                ),
                chunk_id=f"{chunk_prefix}.{idx + 1}",
                parent_page_number=_root_page_number(page),
                layout_type=page.layout_type,
                chunk_context=(f"Rows {row_nums[0]}-{row_nums[-1]}"),
                chunk_header=prefix,
                sheet_passthrough_content=sheet_pt_text,
                section_passthrough_content=(
                    _segment_section_text(snap, row_lookup, segment)
                ),
            )
        )

    return chunks


def chunk_xlsx_page(
    page: PageResult,
    llm: LLMClient,
    embedding_limit: int,
) -> list[PageResult]:
    """Chunk an oversized XLSX page by LLM-identified breakpoints.

    Loads the chunking prompt, parses the sheet markdown, collects
    breakpoints across batches, and assembles final chunks. Returns
    a single-element list when no splitting is needed.

    Params:
        page: PageResult whose content is XLSX sheet markdown
        llm: LLM client for breakpoint identification
        embedding_limit: Target token ceiling per chunk

    Returns:
        list of PageResult instances (one per chunk)

    Example:
        >>> chunks = chunk_xlsx_page(page, llm, 8192)
        >>> len(chunks) >= 1
        True
    """
    prompt = load_prompt("xlsx_chunking", prompts_dir=_PROMPTS_DIR)

    parsed = _parse_sheet_content(page.raw_content)

    if not parsed["data_rows"]:
        return [page]

    batch_size = get_chunking_xlsx_batch_size()
    header_row_count = get_chunking_xlsx_header_rows()
    overlap_row_count = get_chunking_xlsx_overlap_rows()

    context = f"xlsx_chunking page={page.page_number}"

    breakpoints, sheet_pt, section_snapshots = _collect_xlsx_breakpoints(
        parsed,
        llm,
        prompt,
        batch_size,
        (header_row_count, overlap_row_count),
        embedding_limit,
        context,
    )

    if not breakpoints:
        return [page]

    return _assemble_xlsx_chunks(
        parsed, breakpoints, sheet_pt, section_snapshots, page
    )


parse_sheet_content = _parse_sheet_content
format_xlsx_batch = _format_xlsx_batch
parse_xlsx_response = _parse_xlsx_response
collect_xlsx_breakpoints = _collect_xlsx_breakpoints
assemble_xlsx_chunks = _assemble_xlsx_chunks
row_num_from_line = _row_num_from_line
build_batch_messages = _build_batch_messages
row_content = _row_content
update_sheet_passthrough_lines = _update_sheet_passthrough_lines
build_section_context_lines = _build_section_context_lines
process_batch_result = _process_batch_result
build_chunk_prefix = _build_chunk_prefix
split_into_segments = _split_into_segments
call_llm_for_batch = _call_llm_for_batch
