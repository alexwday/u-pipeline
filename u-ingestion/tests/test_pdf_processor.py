"""Tests for the PDF processor."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ingestion.processors.pdf import processor as pdf_module
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


def test_open_rendered_pdf_yields_handle_and_closes(monkeypatch):
    """Open a PDF handle and close it on exit."""
    document = _FakeDocument(page_count=3)
    tools = _FakeTools()
    fake_fitz = SimpleNamespace(
        TOOLS=tools,
        open=lambda _path: document,
        Matrix=lambda x, y: ("matrix", x, y),
    )
    monkeypatch.setattr(pdf_module, "fitz", fake_fitz)

    with pdf_module.open_rendered_pdf(Path("/tmp/doc.pdf"), 2.0) as rendered:
        assert rendered.pdf_path == Path("/tmp/doc.pdf")
        assert rendered.matrix == ("matrix", 2.0, 2.0)
        assert rendered.total_pages == 3

    assert document.closed is True
    assert tools.display_calls == [False, True]
    assert tools.warning_calls == 1


def test_open_rendered_pdf_wraps_open_errors(monkeypatch):
    """Wrap fitz open failures in RuntimeError."""
    tools = _FakeTools()
    fake_fitz = SimpleNamespace(
        TOOLS=tools,
        open=lambda _path: (_ for _ in ()).throw(ValueError("bad pdf")),
        Matrix=lambda x, y: ("matrix", x, y),
    )
    monkeypatch.setattr(pdf_module, "fitz", fake_fitz)

    with pytest.raises(RuntimeError, match="Failed to open PDF 'doc.pdf'"):
        with pdf_module.open_rendered_pdf(Path("/tmp/doc.pdf"), 2.0):
            pass

    assert tools.display_calls == [False, True]


def test_render_page_returns_png_bytes(monkeypatch):
    """Render a valid page to PNG bytes."""
    tools = _FakeTools()
    monkeypatch.setattr(
        pdf_module,
        "fitz",
        SimpleNamespace(TOOLS=tools),
    )
    rendered = pdf_module.RenderedPdf(
        pdf_path=Path("/tmp/doc.pdf"),
        document=_FakeDocument(page_count=1),
        matrix=("matrix", 2.0, 2.0),
        total_pages=1,
    )

    assert pdf_module.render_page(rendered, 1) == b"png-bytes"
    assert tools.display_calls == [False, True]
    assert tools.warning_calls == 1


def test_render_page_validates_bounds_and_wraps_render_errors(monkeypatch):
    """Reject invalid page numbers and wrap fitz failures."""
    tools = _FakeTools()
    monkeypatch.setattr(
        pdf_module,
        "fitz",
        SimpleNamespace(TOOLS=tools),
    )
    rendered = pdf_module.RenderedPdf(
        pdf_path=Path("/tmp/doc.pdf"),
        document=SimpleNamespace(
            load_page=lambda _index: (_ for _ in ()).throw(OSError("boom"))
        ),
        matrix=("matrix", 2.0, 2.0),
        total_pages=1,
    )

    with pytest.raises(ValueError, match="Page 2 is out of range"):
        pdf_module.render_page(rendered, 2)

    with pytest.raises(RuntimeError, match="Failed to render page 1"):
        pdf_module.render_page(rendered, 1)


def test_render_all_pages_uses_render_page(monkeypatch):
    """Render every page in order."""
    rendered = pdf_module.RenderedPdf(
        pdf_path=Path("/tmp/doc.pdf"),
        document=object(),
        matrix=object(),
        total_pages=2,
    )
    monkeypatch.setattr(
        pdf_module,
        "render_page",
        Mock(side_effect=[b"page-1", b"page-2"]),
    )

    pages = pdf_module.render_all_pages(rendered)

    assert pages == [
        pdf_module.RenderedPage(page_number=1, img_bytes=b"page-1"),
        pdf_module.RenderedPage(page_number=2, img_bytes=b"page-2"),
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

    assert pdf_module.parse_extraction_response(response) == "# Page"


def test_parse_extraction_response_rejects_invalid_payload():
    """Reject malformed tool payloads."""
    with pytest.raises(ValueError, match="LLM response missing choices"):
        pdf_module.parse_extraction_response({})


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
    """Reject malformed payload shapes beyond the top-level choices list."""
    with pytest.raises(ValueError, match=message):
        pdf_module.parse_extraction_response(response)


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
    monkeypatch.setattr(
        pdf_module, "_PARSE_RETRYABLE_ERRORS", (DummyRetryable,)
    )
    monkeypatch.setattr(pdf_module, "get_pdf_vision_max_retries", lambda: 2)
    monkeypatch.setattr(pdf_module, "get_pdf_vision_retry_delay", lambda: 0.0)
    monkeypatch.setattr(pdf_module.time, "sleep", lambda _seconds: None)

    result = pdf_module.extract_page(
        llm=llm,
        img_bytes=b"img",
        prompt=_mock_prompt(),
        context="doc.pdf page 1/1",
    )

    assert result == "ok"
    assert getattr(llm.call, "call_count") == 2


def test_extract_page_retries_on_missing_tool_calls(monkeypatch):
    """Retry when the model returns an empty tool_calls payload."""
    bad_response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"tool_calls": None, "content": "oops"},
            }
        ]
    }
    good_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"arguments": '{"content": "ok"}'}}
                    ]
                }
            }
        ]
    }

    llm = Mock()
    llm.call.side_effect = [bad_response, good_response]
    monkeypatch.setattr(pdf_module, "get_pdf_vision_max_retries", lambda: 2)
    monkeypatch.setattr(pdf_module, "get_pdf_vision_retry_delay", lambda: 0.0)
    monkeypatch.setattr(pdf_module.time, "sleep", lambda _seconds: None)

    result = pdf_module.extract_page(
        llm=llm,
        img_bytes=b"img",
        prompt=_mock_prompt(),
        context="doc.pdf page 1/1",
    )

    assert result == "ok"
    assert getattr(llm.call, "call_count") == 2


def test_extract_page_raises_value_error_after_exhausted_retries(monkeypatch):
    """Raise ValueError when every retry returns a missing-tool-calls body."""
    bad_response = {
        "choices": [{"message": {"tool_calls": None, "content": "still bad"}}]
    }

    llm = Mock()
    llm.call.side_effect = [bad_response, bad_response]
    monkeypatch.setattr(pdf_module, "get_pdf_vision_max_retries", lambda: 2)
    monkeypatch.setattr(pdf_module, "get_pdf_vision_retry_delay", lambda: 0.0)
    monkeypatch.setattr(pdf_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(ValueError, match="missing tool calls"):
        pdf_module.extract_page(
            llm=llm,
            img_bytes=b"img",
            prompt=_mock_prompt(),
            context="doc.pdf page 1/1",
        )
    assert getattr(llm.call, "call_count") == 2


def test_parse_extraction_response_missing_tool_calls_includes_diagnostics():
    """Enriched missing-tool-calls error includes finish_reason and preview."""
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "tool_calls": None,
                    "content": "truncated thinking...",
                },
            }
        ]
    }

    with pytest.raises(ValueError) as exc_info:
        pdf_module.parse_extraction_response(response)

    message = str(exc_info.value)
    assert "missing tool calls" in message
    assert "finish_reason=length" in message
    assert "truncated thinking" in message


def test_extract_page_raises_after_final_retry(monkeypatch):
    """Raise the last retryable error after exhausting retries."""

    class DummyRetryable(Exception):
        """Retryable stand-in."""

    llm = Mock()
    llm.call.side_effect = DummyRetryable("retry")
    monkeypatch.setattr(
        pdf_module, "_PARSE_RETRYABLE_ERRORS", (DummyRetryable,)
    )
    monkeypatch.setattr(pdf_module, "get_pdf_vision_max_retries", lambda: 1)
    monkeypatch.setattr(pdf_module, "get_pdf_vision_retry_delay", lambda: 0.0)

    with pytest.raises(DummyRetryable, match="retry"):
        pdf_module.extract_page(
            llm=llm,
            img_bytes=b"img",
            prompt=_mock_prompt(),
            context="doc.pdf page 1/1",
        )


def test_extract_page_raises_when_retry_loop_never_runs(monkeypatch):
    """Guard against a zero-retry configuration."""
    monkeypatch.setattr(pdf_module, "get_pdf_vision_max_retries", lambda: 0)
    monkeypatch.setattr(pdf_module, "get_pdf_vision_retry_delay", lambda: 0.0)

    with pytest.raises(
        RuntimeError, match="exited retry loop without a response"
    ):
        pdf_module.extract_page(
            llm=Mock(),
            img_bytes=b"img",
            prompt=_mock_prompt(),
            context="doc.pdf page 1/1",
        )


def test_extract_single_page_wraps_content():
    """Return a PageResult for one rendered page."""
    llm = Mock()
    prompt = _mock_prompt()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            pdf_module,
            "extract_page",
            Mock(return_value="content"),
        )
        result = getattr(pdf_module, "_extract_single_page")(
            llm=llm,
            rendered_page=pdf_module.RenderedPage(2, b"img"),
            total_pages=4,
            prompt=prompt,
            file_label="doc.pdf",
        )

    assert result == PageResult(page_number=2, raw_content="content")


def test_process_pdf_returns_empty_for_zero_page_document(monkeypatch):
    """Return an empty result instead of building a zero-worker pool."""

    @contextmanager
    def fake_open_rendered_pdf(_path, _dpi):
        yield SimpleNamespace(total_pages=0)

    monkeypatch.setattr(
        pdf_module, "load_prompt", lambda *_args: _mock_prompt()
    )
    monkeypatch.setattr(pdf_module, "get_vision_dpi_scale", lambda: 2.0)
    monkeypatch.setattr(
        pdf_module,
        "open_rendered_pdf",
        fake_open_rendered_pdf,
    )

    result = pdf_module.process_pdf("/tmp/doc.pdf", Mock())

    assert result.file_path == "/tmp/doc.pdf"
    assert not result.pages
    assert result.total_pages == 0


def test_process_pdf_extracts_pages_in_document_order(monkeypatch):
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
        pdf_module, "load_prompt", lambda *_args: _mock_prompt()
    )
    monkeypatch.setattr(pdf_module, "get_vision_dpi_scale", lambda: 2.0)
    monkeypatch.setattr(
        pdf_module,
        "open_rendered_pdf",
        fake_open_rendered_pdf,
    )
    monkeypatch.setattr(
        pdf_module,
        "render_page",
        lambda _rendered, page_number: f"page-{page_number}".encode(),
    )
    monkeypatch.setattr(
        pdf_module,
        "get_extraction_page_workers",
        lambda: 2,
    )
    monkeypatch.setattr(pdf_module, "_extract_single_page", fake_extract)

    result = pdf_module.process_pdf("/tmp/doc.pdf", Mock())

    assert calls == [(1, 2, "doc.pdf"), (2, 2, "doc.pdf")]
    assert [page.page_number for page in result.pages] == [1, 2]
    assert result.total_pages == 2
