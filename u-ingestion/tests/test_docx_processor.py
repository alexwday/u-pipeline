"""Tests for the DOCX processor."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import subprocess

import pytest

from ingestion.processors.docx import processor as docx_module
from ingestion.utils.file_types import PageResult


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

    def __init__(self, page_count=2):
        self.page_count = page_count
        self.closed = False

    def close(self):
        """Track close calls."""
        self.closed = True

    def load_page(self, index):
        """Return a fake page by index."""
        assert index in (0, 1)
        return _FakePage()


def _mock_prompt():
    """Build a minimal extraction prompt."""
    return {
        "system_prompt": "system",
        "user_prompt": "user",
        "stage": "extraction",
        "tools": [{"type": "function"}],
        "tool_choice": "required",
    }


def test_find_soffice_prefers_path_lookup(monkeypatch):
    """Return the first binary found via PATH."""
    monkeypatch.setattr(
        docx_module.shutil,
        "which",
        lambda candidate: (
            "/usr/bin/soffice" if candidate == "soffice" else None
        ),
    )

    assert getattr(docx_module, "_find_soffice")() == "soffice"


def test_find_soffice_accepts_existing_binary_path(monkeypatch):
    """Return an explicit binary path when it exists on disk."""
    target = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    monkeypatch.setattr(docx_module.shutil, "which", lambda _candidate: None)
    monkeypatch.setattr(
        docx_module.Path,
        "is_file",
        lambda self: str(self) == target,
    )

    assert getattr(docx_module, "_find_soffice")() == target


def test_find_soffice_raises_when_missing(monkeypatch):
    """Raise when no LibreOffice binary is available."""
    monkeypatch.setattr(docx_module.shutil, "which", lambda _candidate: None)
    monkeypatch.setattr(docx_module.Path, "is_file", lambda self: False)

    with pytest.raises(RuntimeError, match="LibreOffice not found"):
        getattr(docx_module, "_find_soffice")()


def test_build_user_installation_arg(tmp_path):
    """Build an isolated LibreOffice profile URI."""
    result = getattr(docx_module, "_build_user_installation_arg")(
        tmp_path / "profile"
    )

    assert result.startswith("-env:UserInstallation=file:")


def test_format_conversion_error_includes_streams():
    """Include stdout and stderr in conversion diagnostics."""
    result = SimpleNamespace(returncode=1, stdout="out", stderr="err")

    assert getattr(docx_module, "_format_conversion_error")(result) == (
        "return code 1; stdout: out; stderr: err"
    )


def test_convert_to_pdf_returns_generated_file(monkeypatch, tmp_path):
    """Return the converted PDF path when LibreOffice succeeds."""
    docx_path = tmp_path / "doc.docx"
    output_dir = tmp_path / "out"
    pdf_path = output_dir / "doc.pdf"
    docx_path.write_text("docx")
    output_dir.mkdir()

    def fake_run(*_args, **_kwargs):
        pdf_path.write_bytes(b"%PDF")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(docx_module, "_find_soffice", lambda: "soffice")
    monkeypatch.setattr(docx_module.subprocess, "run", fake_run)

    assert docx_module.convert_to_pdf(docx_path, output_dir) == pdf_path


def test_convert_to_pdf_wraps_timeout(monkeypatch, tmp_path):
    """Wrap LibreOffice timeouts in RuntimeError."""
    docx_path = tmp_path / "doc.docx"
    output_dir = tmp_path / "out"
    docx_path.write_text("docx")
    output_dir.mkdir()
    monkeypatch.setattr(docx_module, "_find_soffice", lambda: "soffice")
    monkeypatch.setattr(
        docx_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd="soffice", timeout=120)
        ),
    )

    with pytest.raises(RuntimeError, match="conversion timed out"):
        docx_module.convert_to_pdf(docx_path, output_dir)


def test_convert_to_pdf_rejects_nonzero_exit(monkeypatch, tmp_path):
    """Raise when LibreOffice reports failure."""
    docx_path = tmp_path / "doc.docx"
    output_dir = tmp_path / "out"
    docx_path.write_text("docx")
    output_dir.mkdir()
    monkeypatch.setattr(docx_module, "_find_soffice", lambda: "soffice")
    monkeypatch.setattr(
        docx_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="bad",
            stderr="worse",
        ),
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        docx_module.convert_to_pdf(docx_path, output_dir)


def test_convert_to_pdf_requires_output_file(monkeypatch, tmp_path):
    """Raise when no PDF is produced."""
    docx_path = tmp_path / "doc.docx"
    output_dir = tmp_path / "out"
    docx_path.write_text("docx")
    output_dir.mkdir()
    monkeypatch.setattr(docx_module, "_find_soffice", lambda: "soffice")
    monkeypatch.setattr(
        docx_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="produced no output"):
        docx_module.convert_to_pdf(docx_path, output_dir)


def test_open_rendered_pdf_yields_handle_and_closes(monkeypatch):
    """Open a converted PDF handle and close it on exit."""
    document = _FakeDocument(page_count=3)
    tools = _FakeTools()
    fake_fitz = SimpleNamespace(
        TOOLS=tools,
        open=lambda _path: document,
        Matrix=lambda x, y: ("matrix", x, y),
    )
    monkeypatch.setattr(docx_module, "fitz", fake_fitz)

    with docx_module.open_rendered_pdf(Path("/tmp/doc.pdf"), 2.0) as rendered:
        assert rendered.pdf_path == Path("/tmp/doc.pdf")
        assert rendered.matrix == ("matrix", 2.0, 2.0)
        assert rendered.total_pages == 3

    assert document.closed is True


def test_open_rendered_pdf_wraps_open_errors(monkeypatch):
    """Wrap fitz open failures in RuntimeError."""
    tools = _FakeTools()
    fake_fitz = SimpleNamespace(
        TOOLS=tools,
        open=lambda _path: (_ for _ in ()).throw(ValueError("bad pdf")),
        Matrix=lambda x, y: ("matrix", x, y),
    )
    monkeypatch.setattr(docx_module, "fitz", fake_fitz)

    with pytest.raises(RuntimeError, match="Failed to open PDF 'doc.pdf'"):
        with docx_module.open_rendered_pdf(Path("/tmp/doc.pdf"), 2.0):
            pass

    assert tools.display_calls == [False, True]


def test_render_page_returns_png_bytes(monkeypatch):
    """Render a valid converted page to PNG bytes."""
    tools = _FakeTools()
    monkeypatch.setattr(
        docx_module,
        "fitz",
        SimpleNamespace(TOOLS=tools),
    )
    rendered = docx_module.RenderedPdf(
        pdf_path=Path("/tmp/doc.pdf"),
        document=_FakeDocument(page_count=1),
        matrix=("matrix", 2.0, 2.0),
        total_pages=1,
    )

    assert docx_module.render_page(rendered, 1) == b"png-bytes"


def test_render_page_validates_bounds_and_wraps_render_errors(monkeypatch):
    """Reject invalid page numbers and wrap fitz render failures."""
    tools = _FakeTools()
    monkeypatch.setattr(
        docx_module,
        "fitz",
        SimpleNamespace(TOOLS=tools),
    )
    rendered = docx_module.RenderedPdf(
        pdf_path=Path("/tmp/doc.pdf"),
        document=SimpleNamespace(
            load_page=lambda _index: (_ for _ in ()).throw(ValueError("bad"))
        ),
        matrix=("matrix", 2.0, 2.0),
        total_pages=1,
    )

    with pytest.raises(ValueError, match="Page 2 is out of range"):
        docx_module.render_page(rendered, 2)

    with pytest.raises(RuntimeError, match="Failed to render page 1"):
        docx_module.render_page(rendered, 1)


def test_render_all_pages_uses_render_page(monkeypatch):
    """Render every converted page in order."""
    rendered = docx_module.RenderedPdf(
        pdf_path=Path("/tmp/doc.pdf"),
        document=object(),
        matrix=object(),
        total_pages=2,
    )
    monkeypatch.setattr(
        docx_module,
        "render_page",
        Mock(side_effect=[b"page-1", b"page-2"]),
    )

    pages = docx_module.render_all_pages(rendered)

    assert pages == [
        docx_module.RenderedPage(page_number=1, img_bytes=b"page-1"),
        docx_module.RenderedPage(page_number=2, img_bytes=b"page-2"),
    ]


def test_parse_extraction_response_reads_tool_arguments():
    """Parse extracted content from a tool-call response."""
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"arguments": '{"content": "# Page"}'}}
                    ]
                }
            }
        ]
    }

    assert docx_module.parse_extraction_response(response) == "# Page"


def test_parse_extraction_response_rejects_missing_choices():
    """Reject DOCX extraction payloads without choices."""
    with pytest.raises(ValueError, match="LLM response missing choices"):
        docx_module.parse_extraction_response({})


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"choices": [{"message": None}]}, "missing message payload"),
        (
            {"choices": [{"message": {"tool_calls": None}}]},
            "missing tool calls",
        ),
        (
            {"choices": [{"message": {"tool_calls": [{"function": None}]}}]},
            "missing function payload",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [{"function": {"arguments": None}}]
                        }
                    }
                ]
            },
            "missing function arguments",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": '["not-an-object"]'
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            "must decode to an object",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": '{"content": "   "}'
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            "empty or missing content",
        ),
    ],
)
def test_parse_extraction_response_rejects_payload_variants(response, message):
    """Reject malformed DOCX extraction payload shapes."""
    with pytest.raises(ValueError, match=message):
        docx_module.parse_extraction_response(response)


def test_extract_page_retries_then_succeeds(monkeypatch):
    """Retry transient errors and return parsed content."""

    class DummyRetryable(Exception):
        """Retryable stand-in."""

    llm = Mock()
    llm.call.side_effect = [
        DummyRetryable("retry"),
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"arguments": '{"content": "ok"}'}}
                        ]
                    }
                }
            ]
        },
    ]
    monkeypatch.setattr(docx_module, "RETRYABLE_ERRORS", (DummyRetryable,))
    monkeypatch.setattr(docx_module, "get_docx_vision_max_retries", lambda: 2)
    monkeypatch.setattr(
        docx_module,
        "get_docx_vision_retry_delay",
        lambda: 0.0,
    )
    monkeypatch.setattr(docx_module.time, "sleep", lambda _seconds: None)

    result = docx_module.extract_page(
        llm=llm,
        img_bytes=b"img",
        prompt=_mock_prompt(),
        context="doc.docx page 1/1",
    )

    assert result == "ok"
    assert getattr(llm.call, "call_count") == 2


def test_extract_page_raises_after_final_retry(monkeypatch):
    """Raise the last retryable error after exhausting retries."""

    class DummyRetryable(Exception):
        """Retryable stand-in."""

    llm = Mock()
    llm.call.side_effect = DummyRetryable("retry")
    monkeypatch.setattr(docx_module, "RETRYABLE_ERRORS", (DummyRetryable,))
    monkeypatch.setattr(docx_module, "get_docx_vision_max_retries", lambda: 1)
    monkeypatch.setattr(
        docx_module,
        "get_docx_vision_retry_delay",
        lambda: 0.0,
    )

    with pytest.raises(DummyRetryable, match="retry"):
        docx_module.extract_page(
            llm=llm,
            img_bytes=b"img",
            prompt=_mock_prompt(),
            context="doc.docx page 1/1",
        )


def test_extract_page_raises_when_retry_loop_never_runs(monkeypatch):
    """Guard against a zero-retry DOCX vision configuration."""
    monkeypatch.setattr(docx_module, "get_docx_vision_max_retries", lambda: 0)
    monkeypatch.setattr(
        docx_module,
        "get_docx_vision_retry_delay",
        lambda: 0.0,
    )

    with pytest.raises(
        RuntimeError, match="exited retry loop without a response"
    ):
        docx_module.extract_page(
            llm=Mock(),
            img_bytes=b"img",
            prompt=_mock_prompt(),
            context="doc.docx page 1/1",
        )


def test_extract_single_page_wraps_content():
    """Return a PageResult for one rendered page."""
    llm = Mock()
    prompt = _mock_prompt()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            docx_module,
            "extract_page",
            Mock(return_value="content"),
        )
        result = getattr(docx_module, "_extract_single_page")(
            llm=llm,
            rendered_page=docx_module.RenderedPage(2, b"img"),
            total_pages=4,
            prompt=prompt,
            file_label="doc.docx",
        )

    assert result == PageResult(page_number=2, raw_content="content")


def test_process_docx_returns_empty_for_zero_page_document(monkeypatch):
    """Return an empty result when conversion yields no pages."""

    @contextmanager
    def fake_open_rendered_pdf(_path, _dpi):
        yield SimpleNamespace(total_pages=0)

    monkeypatch.setattr(
        docx_module, "load_prompt", lambda *_args: _mock_prompt()
    )
    monkeypatch.setattr(
        docx_module, "convert_to_pdf", lambda *_args: Path("/tmp/doc.pdf")
    )
    monkeypatch.setattr(docx_module, "get_vision_dpi_scale", lambda: 2.0)
    monkeypatch.setattr(
        docx_module,
        "open_rendered_pdf",
        fake_open_rendered_pdf,
    )

    result = docx_module.process_docx("/tmp/doc.docx", Mock())

    assert result.file_path == "/tmp/doc.docx"
    assert not result.pages
    assert result.total_pages == 0


def test_process_docx_extracts_pages_in_document_order(monkeypatch):
    """Process pages sequentially in render order."""
    calls = []

    @contextmanager
    def fake_open_rendered_pdf(_path, _dpi):
        yield SimpleNamespace(total_pages=2)

    def fake_extract(_llm, rendered_page, total_pages, prompt, file_label):
        calls.append((rendered_page.page_number, total_pages, file_label))
        assert prompt == _mock_prompt()
        return PageResult(
            page_number=rendered_page.page_number,
            raw_content=f"page-{rendered_page.page_number}",
        )

    monkeypatch.setattr(
        docx_module, "load_prompt", lambda *_args: _mock_prompt()
    )
    monkeypatch.setattr(
        docx_module, "convert_to_pdf", lambda *_args: Path("/tmp/doc.pdf")
    )
    monkeypatch.setattr(docx_module, "get_vision_dpi_scale", lambda: 2.0)
    monkeypatch.setattr(
        docx_module,
        "open_rendered_pdf",
        fake_open_rendered_pdf,
    )
    monkeypatch.setattr(
        docx_module,
        "render_page",
        lambda _rendered, page_number: f"page-{page_number}".encode(),
    )
    monkeypatch.setattr(
        docx_module,
        "get_extraction_page_workers",
        lambda: 2,
    )
    monkeypatch.setattr(docx_module, "_extract_single_page", fake_extract)

    result = docx_module.process_docx("/tmp/doc.docx", Mock())

    assert calls == [(1, 2, "doc.docx"), (2, 2, "doc.docx")]
    assert [page.page_number for page in result.pages] == [1, 2]
    assert result.total_pages == 2
