"""Shared MuPDF rendering helpers protected by one process-wide lock."""

import threading
from pathlib import Path
from typing import Any

FITZ_ERRORS = (RuntimeError, ValueError, OSError)
FITZ_RENDER_LOCK = threading.Lock()


def open_fitz_document(fitz_module: Any, pdf_path: Path) -> Any:
    """Open a fitz document under the shared render lock.

    Params: fitz_module, pdf_path. Returns: Any.
    """
    with FITZ_RENDER_LOCK:
        fitz_module.TOOLS.mupdf_display_errors(False)
        try:
            document = fitz_module.open(str(pdf_path))
            fitz_module.TOOLS.mupdf_warnings()
            return document
        finally:
            fitz_module.TOOLS.mupdf_display_errors(True)


def render_fitz_page_to_png(
    fitz_module: Any,
    document: Any,
    page_index: int,
    matrix: Any,
) -> bytes:
    """Render a fitz page to PNG bytes under the shared render lock.

    Params: fitz_module, document, page_index, matrix. Returns: bytes.
    """
    with FITZ_RENDER_LOCK:
        fitz_module.TOOLS.mupdf_display_errors(False)
        try:
            try:
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                return pixmap.tobytes("png")
            finally:
                fitz_module.TOOLS.mupdf_warnings()
        finally:
            fitz_module.TOOLS.mupdf_display_errors(True)
