"""PDF processor — renders pages to PNG, extracts via LLM vision."""

import base64
import json
import logging
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import openai
import fitz
from PIL import Image

from ...utils.llm_connector import LLMClient
from ...utils.config_setup import (
    get_extraction_page_workers,
    get_extraction_region_workers,
    get_pdf_vision_max_retries,
    get_pdf_vision_retry_delay,
    get_vision_dpi_scale,
)
from ...utils.fitz_rendering import (
    FITZ_ERRORS,
    open_fitz_document,
    render_fitz_page_to_png,
)
from ...utils.file_types import ExtractionResult, PageResult
from ...utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_BBOX_PADDING = 0.025
_SKIPPED_REGION_TYPES = frozenset({"legend"})

RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

_PARSE_RETRYABLE_ERRORS = RETRYABLE_ERRORS + (ValueError,)


@dataclass
class RenderedPage:
    """A single rendered PDF page ready for LLM extraction.

    Params:
        page_number: 1-indexed page number
        img_bytes: PNG image bytes
    """

    page_number: int
    img_bytes: bytes


@dataclass
class RenderedPdf:
    """Open PDF handle configured for page-by-page rendering.

    Params:
        pdf_path: Path to the source PDF
        document: Open fitz document handle
        matrix: Rendering matrix derived from the DPI scale
        total_pages: Total number of pages in the document
    """

    pdf_path: Path
    document: Any
    matrix: Any
    total_pages: int


@dataclass
class VisualRegion:
    """A crop-worthy page region detected in the layout pass.

    Params:
        region_id: Unique page-local identifier
        region_type: Region type such as line_chart or table
        label: Visible region title or label
        bbox: Normalized [x1, y1, x2, y2] coordinates
        context: Nearby text or legend context
        requires_region_extraction: Whether crop extraction should run
        confidence: Model confidence in the region boundary
    """

    region_id: str
    region_type: str
    label: str
    bbox: list[float]
    context: str
    requires_region_extraction: bool
    confidence: float


@dataclass
class PageLayout:
    """Text-only page OCR plus detected visual regions.

    Params:
        page_text_markdown: Text OCR excluding dense visual values
        visual_regions: Regions to crop and extract separately
        rationale: Brief layout rationale
    """

    page_text_markdown: str
    visual_regions: list[VisualRegion]
    rationale: str


@contextmanager
def open_rendered_pdf(
    pdf_path: Path, dpi_scale: float
) -> Iterator[RenderedPdf]:
    """Open PDF once for streaming page rendering.

    Params:
        pdf_path: Path to the PDF file
        dpi_scale: DPI multiplier (e.g. 2.0 for 144 DPI)

    Returns:
        Iterator yielding a RenderedPdf handle

    Example:
        >>> with open_rendered_pdf(Path("doc.pdf"), 2.0) as rendered:
        ...     rendered.total_pages
        10
    """
    try:
        document = open_fitz_document(fitz, pdf_path)
    except FITZ_ERRORS as exc:
        raise RuntimeError(
            f"Failed to open PDF '{pdf_path.name}': {exc}"
        ) from exc

    rendered = RenderedPdf(
        pdf_path=pdf_path,
        document=document,
        matrix=fitz.Matrix(dpi_scale, dpi_scale),
        total_pages=document.page_count,
    )
    try:
        yield rendered
    finally:
        document.close()


def render_page(rendered_pdf: RenderedPdf, page_number: int) -> bytes:
    """Render a single 1-indexed PDF page to PNG bytes.

    Params:
        rendered_pdf: Open RenderedPdf handle
        page_number: 1-indexed page number to render

    Returns:
        bytes — rendered PNG payload for the page

    Example:
        >>> with open_rendered_pdf(Path("doc.pdf"), 2.0) as rendered:
        ...     page = render_page(rendered, 1)
        >>> isinstance(page, bytes)
        True
    """
    if page_number < 1 or page_number > rendered_pdf.total_pages:
        raise ValueError(
            f"Page {page_number} is out of range for "
            f"'{rendered_pdf.pdf_path.name}'"
        )

    page_index = page_number - 1
    try:
        return render_fitz_page_to_png(
            fitz,
            rendered_pdf.document,
            page_index,
            rendered_pdf.matrix,
        )
    except FITZ_ERRORS as exc:
        raise RuntimeError(
            f"Failed to render page {page_number}"
            f" of '{rendered_pdf.pdf_path.name}': {exc}"
        ) from exc


def render_all_pages(rendered_pdf: RenderedPdf) -> list[RenderedPage]:
    """Render all PDF pages to PNG sequentially under the fitz lock.

    Params:
        rendered_pdf: Open RenderedPdf handle

    Returns:
        list[RenderedPage] — all pages rendered as PNG bytes
    """
    pages = []
    for page_num in range(1, rendered_pdf.total_pages + 1):
        img_bytes = render_page(rendered_pdf, page_num)
        pages.append(RenderedPage(page_number=page_num, img_bytes=img_bytes))
    return pages


def parse_tool_arguments(response: dict) -> dict[str, Any]:
    """Extract tool arguments from an LLM tool-call response.

    Params: response (dict). Returns: dict[str, Any].
    """
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices")

    finish_reason = choices[0].get("finish_reason", "")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response missing message payload")

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        content_preview = str(message.get("content", ""))[:200]
        raise ValueError(
            f"LLM response missing tool calls "
            f"(finish_reason={finish_reason}, "
            f"content={content_preview!r})"
        )

    function_data = tool_calls[0].get("function")
    if not isinstance(function_data, dict):
        raise ValueError("LLM response missing function payload")

    arguments = function_data.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("LLM response missing function arguments")

    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise ValueError("LLM tool arguments must decode to an object")
    return parsed


def parse_extraction_response(response: dict) -> str:
    """Extract content string from an LLM tool-call response.

    Params: response (dict). Returns: str — markdown content.
    """
    parsed = parse_tool_arguments(response)

    content = parsed.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM returned empty or missing content")
    return content


def _parse_bbox(value: Any, region_id: str) -> list[float]:
    """Validate a normalized bbox. Params: value, region_id. Returns: list."""
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"Region {region_id!r} has invalid bbox")
    bbox = [float(coord) for coord in value]
    x1, y1, x2, y2 = bbox
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"Region {region_id!r} bbox out of range")
    return bbox


def _coerce_visual_region(data: Any, index: int) -> VisualRegion:
    """Build a VisualRegion from decoded JSON. Params: data, index."""
    if not isinstance(data, dict):
        raise ValueError(f"Visual region #{index} must be an object")
    region_id = str(data.get("region_id", "")).strip()
    if not region_id:
        region_id = f"region_{index + 1}"
    region_type = str(data.get("type", "")).strip() or "other_visual"
    label = str(data.get("label", "")).strip()
    context = str(data.get("context", "")).strip()
    requires = bool(data.get("requires_region_extraction", True))
    confidence = float(data.get("confidence", 0.0) or 0.0)
    return VisualRegion(
        region_id=region_id,
        region_type=region_type,
        label=label,
        bbox=_parse_bbox(data.get("bbox"), region_id),
        context=context,
        requires_region_extraction=requires,
        confidence=confidence,
    )


def parse_page_layout_response(response: dict) -> PageLayout:
    """Parse the layout/OCR tool response.

    Params: response. Returns: PageLayout.
    """
    parsed = parse_tool_arguments(response)
    page_text = parsed.get("page_text_markdown")
    regions_json = parsed.get("visual_regions_json")
    rationale = parsed.get("rationale")
    if not isinstance(page_text, str) or not page_text.strip():
        raise ValueError("LLM returned empty or missing page_text_markdown")
    if not isinstance(regions_json, str):
        raise ValueError("LLM returned missing visual_regions_json")
    if not isinstance(rationale, str):
        raise ValueError("LLM returned missing layout rationale")

    decoded = json.loads(regions_json or "[]")
    if not isinstance(decoded, list):
        raise ValueError("visual_regions_json must decode to a list")
    return PageLayout(
        page_text_markdown=page_text,
        visual_regions=[
            _coerce_visual_region(region, index)
            for index, region in enumerate(decoded)
        ],
        rationale=rationale,
    )


def parse_visual_region_response(response: dict) -> str:
    """Parse a visual-region extraction response.

    Params: response. Returns: str.
    """
    parsed = parse_tool_arguments(response)
    content = parsed.get("content")
    confidence_notes = parsed.get("confidence_notes")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM returned empty or missing region content")
    if isinstance(confidence_notes, str) and confidence_notes.strip():
        notes = confidence_notes.strip()
        return f"{content.strip()}\n\nConfidence notes: {notes}"
    return content.strip()


def _image_content(img_bytes: bytes) -> dict[str, Any]:
    """Build an OpenAI image content part. Params: img_bytes. Returns: dict."""
    b64 = base64.b64encode(img_bytes).decode()
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{b64}",
            "detail": "high",
        },
    }


def _build_messages(prompt: dict, user_content: list[dict]) -> list[dict]:
    """Build chat messages for a prompt. Params: prompt, user_content."""
    return [
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": user_content},
    ]


def _call_tool_with_retries(
    llm: LLMClient,
    messages: list[dict],
    prompt: dict,
    context: str = "",
    parser: Callable[[dict], Any] = parse_tool_arguments,
) -> Any:
    """Call the LLM and return parsed tool args with retry handling.

    Retries on transient API errors up to the configured limit.
    Raises on any non-transient error or after retries are exhausted.

    Params:
        llm: LLMClient instance
        messages: OpenAI chat messages
        prompt: Loaded prompt dict with system_prompt,
            user_prompt, stage, tools, tool_choice
        context: Short log label for the request
        parser: Response parser to run inside the retry loop

    Returns:
        Any — parsed response from parser
    """
    max_retries = get_pdf_vision_max_retries()
    retry_delay = get_pdf_vision_retry_delay()

    for attempt in range(1, max_retries + 1):
        try:
            response = llm.call(
                messages=messages,
                stage=prompt["stage"],
                tools=prompt["tools"],
                tool_choice=prompt.get("tool_choice", "required"),
                context=context,
            )
            return parser(response)

        except _PARSE_RETRYABLE_ERRORS as exc:
            if attempt == max_retries:
                logger.error(
                    "%s failed after %d retries: %s",
                    context or "Vision",
                    max_retries,
                    exc,
                )
                raise
            wait = retry_delay * attempt
            logger.warning(
                "%s retry %d/%d after %.1fs: %s",
                context or "Vision",
                attempt,
                max_retries,
                wait,
                exc,
            )
            time.sleep(wait)

    raise RuntimeError("Vision call exited retry loop without a response")


def extract_page(
    llm: LLMClient,
    img_bytes: bytes,
    prompt: dict,
    context: str = "",
) -> str:
    """Send a page image to the LLM and extract markdown content.

    Params:
        llm: LLMClient instance
        img_bytes: PNG image bytes
        prompt: Loaded extraction prompt
        context: Short log label for the request

    Returns:
        str — extracted markdown content

    Example:
        >>> content = extract_page(llm, img, prompt, "doc.pdf 1/10")
    """
    messages = _build_messages(
        prompt,
        [
            _image_content(img_bytes),
            {"type": "text", "text": prompt["user_prompt"]},
        ],
    )
    return _call_tool_with_retries(
        llm,
        messages,
        prompt,
        context,
        parse_extraction_response,
    )


def _crop_region(img_bytes: bytes, bbox: list[float]) -> bytes:
    """Crop a normalized bbox from PNG bytes. Params: img_bytes, bbox."""
    with Image.open(BytesIO(img_bytes)) as image:
        width, height = image.size
        x1, y1, x2, y2 = bbox
        crop_box = (
            round(max(0.0, x1 - _BBOX_PADDING) * width),
            round(max(0.0, y1 - _BBOX_PADDING) * height),
            round(min(1.0, x2 + _BBOX_PADDING) * width),
            round(min(1.0, y2 + _BBOX_PADDING) * height),
        )
        cropped = image.crop(crop_box)
        output = BytesIO()
        cropped.save(output, format="PNG")
        return output.getvalue()


def _extract_page_layout(
    llm: LLMClient,
    img_bytes: bytes,
    prompt: dict,
    context: str,
) -> PageLayout:
    """Run full-page text OCR and visual-region detection."""
    messages = _build_messages(
        prompt,
        [
            _image_content(img_bytes),
            {"type": "text", "text": prompt["user_prompt"]},
        ],
    )
    return _call_tool_with_retries(
        llm,
        messages,
        prompt,
        context,
        parse_page_layout_response,
    )


def _region_context(
    layout: PageLayout,
    region: VisualRegion,
) -> str:
    """Build text context for a visual crop. Params: layout, region."""
    region_data = {
        "region_id": region.region_id,
        "type": region.region_type,
        "label": region.label,
        "bbox": region.bbox,
        "context": region.context,
    }
    return (
        f"{layout.page_text_markdown.strip()}\n\n"
        "<region_metadata>\n"
        f"{json.dumps(region_data, indent=2)}\n"
        "</region_metadata>"
    )


def _extract_visual_region(
    llm: LLMClient,
    img_bytes: bytes,
    prompt: dict,
    layout: PageLayout,
    region: VisualRegion,
    context: str,
) -> str:
    """Extract one cropped visual region into markdown."""
    user_prompt = (
        f"{prompt['user_prompt']}\n\n"
        "<full_page_text_context>\n"
        f"{_region_context(layout, region)}\n"
        "</full_page_text_context>"
    )
    messages = _build_messages(
        prompt,
        [
            _image_content(_crop_region(img_bytes, region.bbox)),
            {"type": "text", "text": user_prompt},
        ],
    )
    return _call_tool_with_retries(
        llm,
        messages,
        prompt,
        context,
        parse_visual_region_response,
    )


def _should_extract_region(region: VisualRegion) -> bool:
    """Return whether a region should get crop extraction."""
    if not region.requires_region_extraction:
        return False
    return region.region_type.lower() not in _SKIPPED_REGION_TYPES


def _combine_page_content(
    layout: PageLayout,
    region_contents: list[tuple[VisualRegion, str]],
) -> str:
    """Combine text OCR and region extractions into one page chunk."""
    parts = [layout.page_text_markdown.strip()]
    if region_contents:
        parts.append("## Extracted Visual Regions")
    for region, content in region_contents:
        header = (
            f"### {region.label or region.region_id}"
            f" ({region.region_type}; region_id={region.region_id};"
            f" bbox={region.bbox})"
        )
        parts.append(f"{header}\n\n{content.strip()}")
    return "\n\n".join(part for part in parts if part.strip())


def _extract_visual_regions(
    llm: LLMClient,
    img_bytes: bytes,
    region_prompt: dict,
    layout: PageLayout,
    context: str,
) -> list[tuple[VisualRegion, str]]:
    """Extract crop-worthy visual regions with bounded parallelism."""
    regions = [
        region
        for region in layout.visual_regions
        if _should_extract_region(region)
    ]
    if not regions:
        return []

    region_workers = min(get_extraction_region_workers(), len(regions))
    if region_workers == 1:
        return [
            (
                region,
                _extract_visual_region(
                    llm,
                    img_bytes,
                    region_prompt,
                    layout,
                    region,
                    f"{context} {region.region_id}",
                ),
            )
            for region in regions
        ]

    results: list[tuple[VisualRegion, str] | None] = [None] * len(regions)
    with ThreadPoolExecutor(max_workers=region_workers) as pool:
        futures = {
            pool.submit(
                _extract_visual_region,
                llm,
                img_bytes,
                region_prompt,
                layout,
                region,
                f"{context} {region.region_id}",
            ): index
            for index, region in enumerate(regions)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = (regions[index], future.result())

    return [result for result in results if result is not None]


def extract_region_aware_page(
    llm: LLMClient,
    img_bytes: bytes,
    layout_prompt: dict,
    region_prompt: dict,
    context: str = "",
) -> str:
    """Extract a page via text-only layout plus visual crop passes."""
    layout = _extract_page_layout(
        llm,
        img_bytes,
        layout_prompt,
        f"{context} layout",
    )
    region_contents = _extract_visual_regions(
        llm,
        img_bytes,
        region_prompt,
        layout,
        context,
    )
    return _combine_page_content(layout, region_contents)


def _extract_single_page(
    llm: LLMClient,
    rendered_page: RenderedPage,
    total_pages: int,
    layout_prompt: dict,
    region_prompt: dict,
    file_label: str,
) -> PageResult:
    """Extract a single rendered page and return a PageResult.

    Params:
        llm: LLMClient instance
        rendered_page: RenderedPage with page_number and img_bytes
        total_pages: Total pages in the document
        layout_prompt: Loaded layout/OCR prompt
        region_prompt: Loaded visual-region extraction prompt
        file_label: Filename for log context

    Returns:
        PageResult with page_number and extracted content
    """
    context = f"{file_label} page {rendered_page.page_number}/{total_pages}"
    content = extract_region_aware_page(
        llm,
        rendered_page.img_bytes,
        layout_prompt,
        region_prompt,
        context,
    )
    logger.info("%s extracted (%d chars)", context, len(content))
    return PageResult(
        page_number=rendered_page.page_number,
        raw_content=content,
    )


def _extract_rendered_pages(
    llm: LLMClient,
    rendered_pdf: RenderedPdf,
    layout_prompt: dict[str, Any],
    region_prompt: dict[str, Any],
    file_label: str,
) -> list[PageResult]:
    """Render and extract pages with bounded in-flight page images.

    Params:
        llm: LLMClient instance
        rendered_pdf: Open rendered PDF handle
        layout_prompt: Loaded layout/OCR prompt
        region_prompt: Loaded visual-region extraction prompt
        file_label: Source filename for logs

    Returns:
        list[PageResult] — extracted pages in document order
    """
    total_pages = rendered_pdf.total_pages
    if total_pages == 0:
        return []

    page_workers = get_extraction_page_workers()
    results: list[PageResult | None] = [None] * total_pages
    in_flight: dict[Any, int] = {}

    with ThreadPoolExecutor(max_workers=page_workers) as pool:
        for page_number in range(1, total_pages + 1):
            rendered_page = RenderedPage(
                page_number=page_number,
                img_bytes=render_page(rendered_pdf, page_number),
            )
            future = pool.submit(
                _extract_single_page,
                llm,
                rendered_page,
                total_pages,
                layout_prompt,
                region_prompt,
                file_label,
            )
            in_flight[future] = page_number - 1
            if len(in_flight) >= page_workers:
                completed = next(as_completed(in_flight))
                results[in_flight.pop(completed)] = completed.result()

        for future in as_completed(in_flight):
            results[in_flight[future]] = future.result()

    return [page for page in results if page is not None]


def process_pdf(file_path: str, llm: LLMClient) -> ExtractionResult:
    """Extract content from a PDF file via vision processing.

    Renders each page to PNG sequentially (fitz is not thread-safe)
    while extracting rendered pages and visual regions with bounded
    parallelism. Any page failure fails the entire file.

    Params:
        file_path: Absolute path to the PDF file
        llm: LLMClient instance

    Returns:
        ExtractionResult with per-page extraction details

    Example:
        >>> result = process_pdf("/data/report.pdf", llm)
        >>> result.total_pages
        10
    """
    pdf_path = Path(file_path)
    layout_prompt = load_prompt("page_layout", _PROMPTS_DIR)
    region_prompt = load_prompt("visual_region_extraction", _PROMPTS_DIR)

    with open_rendered_pdf(pdf_path, get_vision_dpi_scale()) as rendered:
        pages = _extract_rendered_pages(
            llm,
            rendered,
            layout_prompt,
            region_prompt,
            pdf_path.name,
        )

    if not pages:
        logger.warning("PDF '%s' contains no pages", pdf_path.name)
        return ExtractionResult(
            file_path=file_path,
            filetype="pdf",
            pages=[],
            total_pages=0,
        )

    return ExtractionResult(
        file_path=file_path,
        filetype="pdf",
        pages=pages,
        total_pages=len(pages),
    )
