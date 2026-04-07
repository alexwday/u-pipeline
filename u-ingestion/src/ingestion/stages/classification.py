"""Stage 4: Classify pages into stable layout types."""

from ..utils.file_types import ExtractionResult
from ..utils.logging_setup import get_stage_logger

STAGE = "4-CLASSIFICATION"

_EMPTY_LAYOUT = "empty"
_TABLE_LAYOUT = "table"
_DENSE_TABLE_LAYOUT = "dense_table"
_LONGFORM_TEXT_LAYOUT = "longform_text"
_NARRATIVE_TEXT_LAYOUT = "narrative_text"
_PRESENTATION_LAYOUT = "presentation_slide"
_SPREADSHEET_LAYOUT = "spreadsheet"
_VISUAL_LAYOUT = "visual_heavy"
_BULLET_PREFIXES = ("- ", "* ", "• ", "1. ", "2. ", "3. ")
_VISUAL_MARKERS = (
    "![",
    "chart",
    "diagram",
    "figure",
    "image",
    "photo",
    "screenshot",
)


def _content_lines(raw_content: str) -> list[str]:
    """Return normalized non-empty content lines.

    Params: raw_content (str). Returns: list[str].
    """
    return [line.strip() for line in raw_content.splitlines() if line.strip()]


def _count_table_lines(lines: list[str]) -> int:
    """Count markdown-table lines. Params: lines (list[str]). Returns: int."""
    return sum(
        1
        for line in lines
        if line.startswith("|") and line.endswith("|") and line.count("|") >= 2
    )


def _count_bullet_lines(lines: list[str]) -> int:
    """Count bullet-style lines. Params: lines (list[str]). Returns: int."""
    return sum(
        1
        for line in lines
        if any(line.startswith(prefix) for prefix in _BULLET_PREFIXES)
    )


def _has_visual_marker(lines: list[str]) -> bool:
    """Detect visual-oriented content markers.

    Params: lines (list[str]). Returns: bool.
    """
    lowered = "\n".join(lines).lower()
    return any(marker in lowered for marker in _VISUAL_MARKERS)


def _lookup_layout_type(
    filetype: str,
    raw_content: str,
    token_tier: str,
) -> str:
    """Map filetype, content, and token tier to a layout label.

    Params:
        filetype: Source file extension
        raw_content: Page content to classify
        token_tier: Token-size tier from tokenization

    Returns:
        str — stable layout label for downstream consumers
    """
    if filetype == "xlsx":
        return _SPREADSHEET_LAYOUT
    if filetype == "pptx":
        return _PRESENTATION_LAYOUT

    lines = _content_lines(raw_content)
    if not lines:
        return _EMPTY_LAYOUT

    table_lines = _count_table_lines(lines)
    if table_lines >= 4:
        return _DENSE_TABLE_LAYOUT if token_tier == "high" else _TABLE_LAYOUT

    if _has_visual_marker(lines) or _count_bullet_lines(lines) >= 4:
        return _VISUAL_LAYOUT

    if token_tier == "high":
        return _LONGFORM_TEXT_LAYOUT

    return _NARRATIVE_TEXT_LAYOUT


lookup_layout_type = _lookup_layout_type


def classify_result(result: ExtractionResult) -> ExtractionResult:
    """Assign a meaningful layout type to each page.

    Uses deterministic heuristics so layout_type is stable
    across retries and resumes without introducing another
    LLM dependency.

    Params:
        result: ExtractionResult from the tokenization stage

    Returns:
        ExtractionResult with layout_type set on each page

    Example:
        >>> classify_result(result).pages[0].layout_type
        'narrative_text'
    """
    logger = get_stage_logger(__name__, STAGE)

    counts: dict[str, int] = {}
    for page in result.pages:
        layout = _lookup_layout_type(
            result.filetype,
            page.raw_content,
            page.token_tier,
        )
        page.layout_type = layout
        counts[layout] = counts.get(layout, 0) + 1

    summary_parts = [f"{label}: {count}" for label, count in counts.items()]
    logger.info(
        "%s — %d pages classified (%s)",
        result.file_path,
        len(result.pages),
        ", ".join(summary_parts) if summary_parts else "none",
    )

    return result
