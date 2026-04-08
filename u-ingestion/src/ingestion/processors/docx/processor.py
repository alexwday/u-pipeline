"""DOCX processor — converts to PDF, renders to PNG, extracts via vision."""

import base64
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
import fitz

from ...utils.llm_connector import LLMClient
from ...utils.config_setup import (
    get_docx_vision_max_retries,
    get_docx_vision_retry_delay,
    get_extraction_page_workers,
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

RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

_PARSE_RETRYABLE_ERRORS = RETRYABLE_ERRORS + (ValueError,)

_CONVERSION_LOCK = threading.Lock()

_SOFFICE_PATHS = [
    "soffice",
    "libreoffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/usr/bin/libreoffice",
    "/usr/bin/soffice",
]

_CONVERSION_TIMEOUT_S = 120


# -----------------------------------------------------------------
# DOCX to PDF conversion
# -----------------------------------------------------------------


def _find_soffice() -> str:
    """Locate the LibreOffice/soffice binary.

    Checks common paths and returns the first one found.

    Returns:
        str — command name or absolute path to soffice binary

    Example:
        >>> _find_soffice()
        "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    """
    for candidate in _SOFFICE_PATHS:
        if shutil.which(candidate):
            return candidate
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError(
        "LibreOffice not found. Install LibreOffice or add "
        "soffice to PATH for DOCX processing."
    )


def _build_user_installation_arg(profile_dir: Path) -> str:
    """Build an isolated LibreOffice profile argument.

    Params: profile_dir (Path). Returns: str.
    """
    return f"-env:UserInstallation={profile_dir.resolve().as_uri()}"


def _format_conversion_error(
    result: subprocess.CompletedProcess,
) -> str:
    """Format LibreOffice subprocess diagnostics.

    Params: result (CompletedProcess). Returns: str.
    """
    details = [f"return code {result.returncode}"]
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        details.append(f"stdout: {stdout}")
    if stderr:
        details.append(f"stderr: {stderr}")
    return "; ".join(details)


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the subprocess's entire process group via SIGKILL.

    Params: proc (subprocess.Popen). Returns: None.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def _run_soffice_conversion(
    cmd: list[str],
    source_name: str,
) -> subprocess.CompletedProcess:
    """Run LibreOffice in its own process group and kill the group on timeout.

    Using start_new_session=True places soffice in a new process group so
    SIGKILL via os.killpg reaches the whole tree, including soffice's
    child workers that would otherwise orphan on the direct-child kill.

    Params:
        cmd: soffice command and arguments
        source_name: Filename for diagnostics

    Returns:
        subprocess.CompletedProcess with stdout/stderr/returncode
    """
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=_CONVERSION_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            _kill_process_group(proc)
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"LibreOffice conversion timed out after "
                f"{_CONVERSION_TIMEOUT_S}s for '{source_name}'"
            ) from exc
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )


def convert_to_pdf(docx_path: Path, output_dir: Path) -> Path:
    """Convert a DOCX file to PDF using LibreOffice headless.

    Params:
        docx_path: Path to the source DOCX file
        output_dir: Directory to write the converted PDF

    Returns:
        Path to the generated PDF file

    Example:
        >>> pdf = convert_to_pdf(Path("doc.docx"), Path("/tmp"))
        >>> pdf.suffix
        ".pdf"
    """
    soffice = _find_soffice()
    profile_dir = output_dir / "soffice-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        soffice,
        _build_user_installation_arg(profile_dir),
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(docx_path),
    ]

    logger.info("Converting '%s' to PDF via LibreOffice", docx_path.name)

    with _CONVERSION_LOCK:
        result = _run_soffice_conversion(cmd, docx_path.name)

    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice conversion failed for "
            f"'{docx_path.name}': "
            f"{_format_conversion_error(result)}"
        )

    pdf_path = output_dir / (docx_path.stem + ".pdf")
    if not pdf_path.is_file():
        raise RuntimeError(
            f"LibreOffice conversion produced no output for "
            f"'{docx_path.name}'"
        )

    logger.info(
        "Converted '%s' to PDF (%d bytes)",
        docx_path.name,
        pdf_path.stat().st_size,
    )
    return pdf_path


# -----------------------------------------------------------------
# PDF rendering (from converted DOCX)
# -----------------------------------------------------------------


@dataclass
class RenderedPage:
    """A single rendered page ready for LLM extraction.

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


# -----------------------------------------------------------------
# LLM vision extraction
# -----------------------------------------------------------------


def parse_extraction_response(response: dict) -> str:
    """Extract content string from an LLM tool-call response.

    Params: response (dict). Returns: str — markdown content.
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

    content = parsed.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM returned empty or missing content")
    return content


def extract_page(
    llm: LLMClient,
    img_bytes: bytes,
    prompt: dict,
    context: str = "",
) -> str:
    """Send a page image to the LLM and extract markdown content.

    Retries on transient API errors up to the configured limit.
    Raises on any non-transient error or after retries are exhausted.

    Params:
        llm: LLMClient instance
        img_bytes: PNG image bytes
        prompt: Loaded prompt dict with system_prompt,
            user_prompt, stage, tools, tool_choice
        context: Short log label for the request

    Returns:
        str — extracted markdown content

    Example:
        >>> content = extract_page(llm, img, prompt, "doc.docx 1/10")
    """
    max_retries = get_docx_vision_max_retries()
    retry_delay = get_docx_vision_retry_delay()
    b64 = base64.b64encode(img_bytes).decode()

    messages = [
        {"role": "system", "content": prompt["system_prompt"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": prompt["user_prompt"],
                },
            ],
        },
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = llm.call(
                messages=messages,
                stage=prompt["stage"],
                tools=prompt["tools"],
                tool_choice=prompt.get("tool_choice", "required"),
                context=context,
            )
            return parse_extraction_response(response)

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


def _extract_single_page(
    llm: LLMClient,
    rendered_page: RenderedPage,
    total_pages: int,
    prompt: dict,
    file_label: str,
) -> PageResult:
    """Extract a single rendered page and return a PageResult.

    Params:
        llm: LLMClient instance
        rendered_page: RenderedPage with page_number and img_bytes
        total_pages: Total pages in the document
        prompt: Loaded extraction prompt
        file_label: Filename for log context

    Returns:
        PageResult with page_number and extracted content
    """
    context = f"{file_label} page {rendered_page.page_number}/{total_pages}"
    content = extract_page(llm, rendered_page.img_bytes, prompt, context)
    logger.info("%s extracted (%d chars)", context, len(content))
    return PageResult(
        page_number=rendered_page.page_number,
        raw_content=content,
    )


def _extract_rendered_pages(
    llm: LLMClient,
    rendered_pdf: RenderedPdf,
    prompt: dict[str, Any],
    file_label: str,
) -> list[PageResult]:
    """Render and extract pages with bounded in-flight page images.

    Params:
        llm: LLMClient instance
        rendered_pdf: Open rendered PDF handle
        prompt: Loaded extraction prompt
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
                prompt,
                file_label,
            )
            in_flight[future] = page_number - 1
            if len(in_flight) >= page_workers:
                completed = next(as_completed(in_flight))
                results[in_flight.pop(completed)] = completed.result()

        for future in as_completed(in_flight):
            results[in_flight[future]] = future.result()

    return [page for page in results if page is not None]


# -----------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------


def process_docx(file_path: str, llm: LLMClient) -> ExtractionResult:
    """Extract content from a DOCX file via vision processing.

    Converts DOCX to PDF using LibreOffice headless, renders all
    pages to PNG sequentially (fitz is not thread-safe), then
    extracts pages in parallel using a thread pool. Any page
    failure fails the entire file.

    Params:
        file_path: Absolute path to the DOCX file
        llm: LLMClient instance

    Returns:
        ExtractionResult with per-page extraction details

    Example:
        >>> result = process_docx("/data/report.docx", llm)
        >>> result.total_pages
        5
    """
    docx_path = Path(file_path)
    prompt = load_prompt("page_extraction", _PROMPTS_DIR)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = convert_to_pdf(docx_path, Path(tmp_dir))
        with open_rendered_pdf(pdf_path, get_vision_dpi_scale()) as rendered:
            pages = _extract_rendered_pages(
                llm,
                rendered,
                prompt,
                docx_path.name,
            )

    if not pages:
        logger.warning(
            "DOCX '%s' converted to a zero-page PDF", docx_path.name
        )
        return ExtractionResult(
            file_path=file_path,
            filetype="docx",
            pages=[],
            total_pages=0,
        )

    return ExtractionResult(
        file_path=file_path,
        filetype="docx",
        pages=pages,
        total_pages=len(pages),
    )
