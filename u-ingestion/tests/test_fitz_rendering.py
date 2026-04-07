"""Tests for shared MuPDF rendering helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ingestion.utils import fitz_rendering


class _FakeTools:
    """Minimal fitz tools stub."""

    def __init__(self):
        self.display_calls = []
        self.warning_calls = 0

    def mupdf_display_errors(self, enabled):
        """Record display-error toggles."""
        self.display_calls.append(enabled)

    def mupdf_warnings(self):
        """Record warning flushes."""
        self.warning_calls += 1


class _FakePixmap:
    """Minimal pixmap stub."""

    def tobytes(self, fmt):
        """Return PNG bytes."""
        assert fmt == "png"
        return b"png-bytes"

    def render(self, fmt):
        """Mirror tobytes for lint-friendly test doubles."""
        return self.tobytes(fmt)


class _FakePage:
    """Minimal page stub."""

    def get_pixmap(self, matrix, alpha):
        """Return a rendered pixmap."""
        assert matrix == ("matrix", 2.0, 2.0)
        assert alpha is False
        return _FakePixmap()

    def render(self, matrix, alpha):
        """Mirror get_pixmap for lint-friendly test doubles."""
        return self.get_pixmap(matrix, alpha)


class _FakeDocument:
    """Minimal document stub."""

    def load_page(self, index):
        """Return a fake page by index."""
        assert index == 0
        return _FakePage()

    def page(self, index):
        """Mirror load_page for lint-friendly test doubles."""
        return self.load_page(index)


def test_open_fitz_document_returns_document_and_flushes_warnings():
    """Open documents under the shared lock and flush warnings on success."""
    tools = _FakeTools()
    document = object()
    fake_fitz = SimpleNamespace(
        TOOLS=tools,
        open=lambda _path: document,
    )

    loaded = fitz_rendering.open_fitz_document(
        fake_fitz,
        Path("/tmp/doc.pdf"),
    )

    assert loaded is document
    assert tools.display_calls == [False, True]
    assert tools.warning_calls == 1


def test_open_fitz_document_restores_display_errors_on_failure():
    """Re-enable MuPDF display errors even when opening fails."""
    tools = _FakeTools()
    fake_fitz = SimpleNamespace(
        TOOLS=tools,
        open=lambda _path: (_ for _ in ()).throw(OSError("bad pdf")),
    )

    with pytest.raises(OSError, match="bad pdf"):
        fitz_rendering.open_fitz_document(fake_fitz, Path("/tmp/doc.pdf"))

    assert tools.display_calls == [False, True]
    assert tools.warning_calls == 0


def test_render_fitz_page_to_png_returns_png_bytes():
    """Render one page to PNG bytes under the shared lock."""
    tools = _FakeTools()
    fake_fitz = SimpleNamespace(TOOLS=tools)

    rendered = fitz_rendering.render_fitz_page_to_png(
        fake_fitz,
        _FakeDocument(),
        0,
        ("matrix", 2.0, 2.0),
    )

    assert rendered == b"png-bytes"
    assert tools.display_calls == [False, True]
    assert tools.warning_calls == 1


def test_render_fitz_page_to_png_flushes_warnings_on_failure():
    """Flush MuPDF warnings and restore display errors when rendering fails."""
    tools = _FakeTools()
    fake_fitz = SimpleNamespace(TOOLS=tools)
    document = SimpleNamespace(
        load_page=lambda _index: (_ for _ in ()).throw(OSError("boom"))
    )

    with pytest.raises(OSError, match="boom"):
        fitz_rendering.render_fitz_page_to_png(
            fake_fitz,
            document,
            0,
            ("matrix", 2.0, 2.0),
        )

    assert tools.display_calls == [False, True]
    assert tools.warning_calls == 1
