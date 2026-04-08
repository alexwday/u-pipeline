"""Tests for the PPTX processor."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import subprocess

import pytest

from ingestion.processors.pptx import processor as pptx_module
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


def test_find_soffice_raises_when_missing(monkeypatch):
    """Raise when no LibreOffice binary is available."""
    monkeypatch.setattr(pptx_module.shutil, "which", lambda _candidate: None)
    monkeypatch.setattr(pptx_module.Path, "is_file", lambda self: False)

    with pytest.raises(RuntimeError, match="LibreOffice not found"):
        getattr(pptx_module, "_find_soffice")()


def test_find_soffice_prefers_path_lookup(monkeypatch):
    """Return the first binary found via PATH."""
    monkeypatch.setattr(
        pptx_module.shutil,
        "which",
        lambda candidate: (
            "/usr/bin/soffice" if candidate == "soffice" else None
        ),
    )

    assert getattr(pptx_module, "_find_soffice")() == "soffice"


def test_find_soffice_accepts_existing_binary_path(monkeypatch):
    """Return an explicit binary path when it exists on disk."""
    target = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    monkeypatch.setattr(pptx_module.shutil, "which", lambda _candidate: None)
    monkeypatch.setattr(
        pptx_module.Path,
        "is_file",
        lambda self: str(self) == target,
    )

    assert getattr(pptx_module, "_find_soffice")() == target


def test_format_conversion_error_includes_streams():
    """Include stdout and stderr in conversion diagnostics."""
    result = SimpleNamespace(returncode=1, stdout="out", stderr="err")

    assert getattr(pptx_module, "_format_conversion_error")(result) == (
        "return code 1; stdout: out; stderr: err"
    )


def test_convert_to_pdf_returns_generated_file(monkeypatch, tmp_path):
    """Return the converted PDF path when LibreOffice succeeds."""
    pptx_path = tmp_path / "deck.pptx"
    output_dir = tmp_path / "out"
    pdf_path = output_dir / "deck.pdf"
    pptx_path.write_text("pptx")
    output_dir.mkdir()

    def fake_run_soffice(_cmd, _source_name):
        pdf_path.write_bytes(b"%PDF")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pptx_module, "_find_soffice", lambda: "soffice")
    monkeypatch.setattr(
        pptx_module, "_run_soffice_conversion", fake_run_soffice
    )

    assert pptx_module.convert_to_pdf(pptx_path, output_dir) == pdf_path


def test_convert_to_pdf_propagates_timeout_runtime_error(
    monkeypatch, tmp_path
):
    """Propagate the RuntimeError raised on conversion timeout."""
    pptx_path = tmp_path / "deck.pptx"
    output_dir = tmp_path / "out"
    pptx_path.write_text("pptx")
    output_dir.mkdir()
    monkeypatch.setattr(pptx_module, "_find_soffice", lambda: "soffice")

    def fake_run_soffice(_cmd, source_name):
        raise RuntimeError(
            f"LibreOffice conversion timed out after 120s for '{source_name}'"
        )

    monkeypatch.setattr(
        pptx_module, "_run_soffice_conversion", fake_run_soffice
    )

    with pytest.raises(RuntimeError, match="conversion timed out"):
        pptx_module.convert_to_pdf(pptx_path, output_dir)


def test_convert_to_pdf_rejects_nonzero_exit(monkeypatch, tmp_path):
    """Raise when LibreOffice reports failure."""
    pptx_path = tmp_path / "deck.pptx"
    output_dir = tmp_path / "out"
    pptx_path.write_text("pptx")
    output_dir.mkdir()
    monkeypatch.setattr(pptx_module, "_find_soffice", lambda: "soffice")
    monkeypatch.setattr(
        pptx_module,
        "_run_soffice_conversion",
        lambda *_args: SimpleNamespace(
            returncode=1,
            stdout="bad",
            stderr="worse",
        ),
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        pptx_module.convert_to_pdf(pptx_path, output_dir)


def test_convert_to_pdf_requires_output_file(monkeypatch, tmp_path):
    """Raise when no PDF is produced."""
    pptx_path = tmp_path / "deck.pptx"
    output_dir = tmp_path / "out"
    pptx_path.write_text("pptx")
    output_dir.mkdir()
    monkeypatch.setattr(pptx_module, "_find_soffice", lambda: "soffice")
    monkeypatch.setattr(
        pptx_module,
        "_run_soffice_conversion",
        lambda *_args: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="produced no output"):
        pptx_module.convert_to_pdf(pptx_path, output_dir)


class _FakePopen:
    """Minimal Popen stand-in for _run_soffice_conversion tests."""

    def __init__(self, timeout_first=False, returncode=0, pid=12345):
        self.pid = pid
        self.returncode = returncode
        self._timeout_first = timeout_first
        self._calls = 0

    def communicate(self, timeout=None):
        """Mimic Popen.communicate: optionally raise TimeoutExpired once."""
        self._calls += 1
        if self._timeout_first and self._calls == 1:
            raise subprocess.TimeoutExpired(cmd="soffice", timeout=timeout)
        return ("stdout-content", "stderr-content")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_run_soffice_conversion_returns_completed_process(monkeypatch):
    """Happy path: Popen communicates successfully and returns a result."""
    fake = _FakePopen()
    monkeypatch.setattr(
        pptx_module.subprocess, "Popen", lambda *_a, **_kw: fake
    )

    result = getattr(pptx_module, "_run_soffice_conversion")(
        ["soffice", "--headless"], "deck.pptx"
    )

    assert result.returncode == 0
    assert result.stdout == "stdout-content"
    assert result.stderr == "stderr-content"


def test_run_soffice_conversion_kills_process_group_on_timeout(monkeypatch):
    """On timeout, the helper kills the whole process group and raises."""
    fake = _FakePopen(timeout_first=True, pid=98765)
    killpg_calls: list[tuple] = []
    monkeypatch.setattr(
        pptx_module.subprocess, "Popen", lambda *_a, **_kw: fake
    )
    monkeypatch.setattr(pptx_module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        pptx_module.os,
        "killpg",
        lambda pgid, sig: killpg_calls.append((pgid, sig)),
    )

    with pytest.raises(RuntimeError, match="conversion timed out"):
        getattr(pptx_module, "_run_soffice_conversion")(
            ["soffice", "--headless"], "deck.pptx"
        )

    assert killpg_calls == [(98765, pptx_module.signal.SIGKILL)]


def test_kill_process_group_falls_back_when_process_missing(monkeypatch):
    """If os.killpg raises ProcessLookupError, call proc.kill() as fallback."""
    kill_calls: list[int] = []
    fake = SimpleNamespace(pid=42, kill=lambda: kill_calls.append(42))
    monkeypatch.setattr(pptx_module.os, "getpgid", lambda pid: pid)

    def raise_lookup(_pgid, _sig):
        raise ProcessLookupError("gone")

    monkeypatch.setattr(pptx_module.os, "killpg", raise_lookup)

    getattr(pptx_module, "_kill_process_group")(fake)

    assert kill_calls == [42]


def test_extract_all_speaker_notes_reads_non_empty_notes(monkeypatch):
    """Return notes for slides that contain text."""
    slides = [
        SimpleNamespace(
            has_notes_slide=True,
            notes_slide=SimpleNamespace(
                notes_text_frame=SimpleNamespace(text=" intro ")
            ),
        ),
        SimpleNamespace(
            has_notes_slide=True,
            notes_slide=SimpleNamespace(
                notes_text_frame=SimpleNamespace(text="   ")
            ),
        ),
        SimpleNamespace(has_notes_slide=False),
    ]
    monkeypatch.setattr(
        pptx_module,
        "Presentation",
        lambda _path: SimpleNamespace(slides=slides),
    )

    assert pptx_module.extract_all_speaker_notes("/tmp/deck.pptx") == {
        1: "intro"
    }


def test_extract_all_speaker_notes_handles_pptx_errors(monkeypatch):
    """Return an empty mapping when python-pptx parsing fails."""
    monkeypatch.setattr(
        pptx_module,
        "Presentation",
        lambda _path: (_ for _ in ()).throw(ValueError("bad deck")),
    )

    assert not pptx_module.extract_all_speaker_notes("/tmp/deck.pptx")


def test_open_rendered_pdf_yields_handle_and_closes(monkeypatch):
    """Open a converted PDF handle and close it on exit."""
    document = _FakeDocument(page_count=3)
    tools = _FakeTools()
    fake_fitz = SimpleNamespace(
        TOOLS=tools,
        open=lambda _path: document,
        Matrix=lambda x, y: ("matrix", x, y),
    )
    monkeypatch.setattr(pptx_module, "fitz", fake_fitz)

    with pptx_module.open_rendered_pdf(Path("/tmp/deck.pdf"), 2.0) as rendered:
        assert rendered.pdf_path == Path("/tmp/deck.pdf")
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
    monkeypatch.setattr(pptx_module, "fitz", fake_fitz)

    with pytest.raises(RuntimeError, match="Failed to open PDF 'deck.pdf'"):
        with pptx_module.open_rendered_pdf(Path("/tmp/deck.pdf"), 2.0):
            pass

    assert tools.display_calls == [False, True]


def test_render_page_returns_png_bytes(monkeypatch):
    """Render a valid converted slide to PNG bytes."""
    tools = _FakeTools()
    monkeypatch.setattr(
        pptx_module,
        "fitz",
        SimpleNamespace(TOOLS=tools),
    )
    rendered = pptx_module.RenderedPdf(
        pdf_path=Path("/tmp/deck.pdf"),
        document=_FakeDocument(page_count=1),
        matrix=("matrix", 2.0, 2.0),
        total_pages=1,
    )

    assert pptx_module.render_page(rendered, 1) == b"png-bytes"


def test_render_page_validates_bounds_and_wraps_render_errors(monkeypatch):
    """Reject invalid slide numbers and wrap fitz failures."""
    tools = _FakeTools()
    monkeypatch.setattr(
        pptx_module,
        "fitz",
        SimpleNamespace(TOOLS=tools),
    )
    rendered = pptx_module.RenderedPdf(
        pdf_path=Path("/tmp/deck.pdf"),
        document=SimpleNamespace(
            load_page=lambda _index: (_ for _ in ()).throw(OSError("boom"))
        ),
        matrix=("matrix", 2.0, 2.0),
        total_pages=1,
    )

    with pytest.raises(ValueError, match="Page 2 is out of range"):
        pptx_module.render_page(rendered, 2)

    with pytest.raises(RuntimeError, match="Failed to render page 1"):
        pptx_module.render_page(rendered, 1)


def test_render_all_pages_uses_render_page(monkeypatch):
    """Render every slide in order."""
    rendered = pptx_module.RenderedPdf(
        pdf_path=Path("/tmp/deck.pdf"),
        document=object(),
        matrix=object(),
        total_pages=2,
    )
    monkeypatch.setattr(
        pptx_module,
        "render_page",
        Mock(side_effect=[b"page-1", b"page-2"]),
    )

    pages = pptx_module.render_all_pages(rendered)

    assert pages == [
        pptx_module.RenderedPage(page_number=1, img_bytes=b"page-1"),
        pptx_module.RenderedPage(page_number=2, img_bytes=b"page-2"),
    ]


def test_parse_extraction_response_reads_tool_arguments():
    """Parse extracted content from a tool-call response."""
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"arguments": '{"content": "# Slide"}'}}
                    ]
                }
            }
        ]
    }

    assert pptx_module.parse_extraction_response(response) == "# Slide"


def test_parse_extraction_response_rejects_missing_choices():
    """Reject PPTX extraction payloads without choices."""
    with pytest.raises(ValueError, match="LLM response missing choices"):
        pptx_module.parse_extraction_response({})


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
    """Reject malformed PPTX extraction payload shapes."""
    with pytest.raises(ValueError, match=message):
        pptx_module.parse_extraction_response(response)


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
        pptx_module, "_PARSE_RETRYABLE_ERRORS", (DummyRetryable,)
    )
    monkeypatch.setattr(pptx_module, "get_pptx_vision_max_retries", lambda: 2)
    monkeypatch.setattr(
        pptx_module,
        "get_pptx_vision_retry_delay",
        lambda: 0.0,
    )
    monkeypatch.setattr(pptx_module.time, "sleep", lambda _seconds: None)

    result = pptx_module.extract_page(
        llm=llm,
        img_bytes=b"img",
        prompt=_mock_prompt(),
        context="deck.pptx slide 1/1",
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
    monkeypatch.setattr(pptx_module, "get_pptx_vision_max_retries", lambda: 2)
    monkeypatch.setattr(
        pptx_module,
        "get_pptx_vision_retry_delay",
        lambda: 0.0,
    )
    monkeypatch.setattr(pptx_module.time, "sleep", lambda _seconds: None)

    result = pptx_module.extract_page(
        llm=llm,
        img_bytes=b"img",
        prompt=_mock_prompt(),
        context="deck.pptx slide 1/1",
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
    monkeypatch.setattr(pptx_module, "get_pptx_vision_max_retries", lambda: 2)
    monkeypatch.setattr(
        pptx_module,
        "get_pptx_vision_retry_delay",
        lambda: 0.0,
    )
    monkeypatch.setattr(pptx_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(ValueError, match="missing tool calls"):
        pptx_module.extract_page(
            llm=llm,
            img_bytes=b"img",
            prompt=_mock_prompt(),
            context="deck.pptx slide 1/1",
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
        pptx_module.parse_extraction_response(response)

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
        pptx_module, "_PARSE_RETRYABLE_ERRORS", (DummyRetryable,)
    )
    monkeypatch.setattr(pptx_module, "get_pptx_vision_max_retries", lambda: 1)
    monkeypatch.setattr(
        pptx_module,
        "get_pptx_vision_retry_delay",
        lambda: 0.0,
    )

    with pytest.raises(DummyRetryable, match="retry"):
        pptx_module.extract_page(
            llm=llm,
            img_bytes=b"img",
            prompt=_mock_prompt(),
            context="deck.pptx slide 1/1",
        )


def test_extract_page_raises_when_retry_loop_never_runs(monkeypatch):
    """Guard against a zero-retry PPTX vision configuration."""
    monkeypatch.setattr(pptx_module, "get_pptx_vision_max_retries", lambda: 0)
    monkeypatch.setattr(
        pptx_module,
        "get_pptx_vision_retry_delay",
        lambda: 0.0,
    )

    with pytest.raises(
        RuntimeError, match="exited retry loop without a response"
    ):
        pptx_module.extract_page(
            llm=Mock(),
            img_bytes=b"img",
            prompt=_mock_prompt(),
            context="deck.pptx slide 1/1",
        )


def test_extract_single_page_appends_speaker_notes():
    """Append notes to extracted slide content when present."""
    llm = Mock()
    prompt = _mock_prompt()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            pptx_module,
            "extract_page",
            Mock(return_value="content"),
        )
        result = getattr(pptx_module, "_extract_single_page")(
            llm=llm,
            rendered_page=pptx_module.RenderedPage(2, b"img"),
            total_pages=4,
            prompt=prompt,
            file_label="deck.pptx",
            speaker_notes={2: "notes"},
        )

    assert result == PageResult(
        page_number=2,
        raw_content="content\n\n## Speaker Notes\n\nnotes",
    )


def test_process_pptx_returns_empty_for_zero_page_document(monkeypatch):
    """Return an empty result when conversion yields no slides."""

    @contextmanager
    def fake_open_rendered_pdf(_path, _dpi):
        yield SimpleNamespace(total_pages=0)

    monkeypatch.setattr(
        pptx_module, "load_prompt", lambda *_args: _mock_prompt()
    )
    monkeypatch.setattr(
        pptx_module, "extract_all_speaker_notes", lambda *_args: {}
    )
    monkeypatch.setattr(
        pptx_module, "convert_to_pdf", lambda *_args: Path("/tmp/deck.pdf")
    )
    monkeypatch.setattr(pptx_module, "get_vision_dpi_scale", lambda: 2.0)
    monkeypatch.setattr(
        pptx_module,
        "open_rendered_pdf",
        fake_open_rendered_pdf,
    )

    result = pptx_module.process_pptx("/tmp/deck.pptx", Mock())

    assert result.file_path == "/tmp/deck.pptx"
    assert not result.pages
    assert result.total_pages == 0


def test_process_pptx_extracts_pages_in_document_order(monkeypatch):
    """Process slides sequentially in render order."""
    calls = []

    @contextmanager
    def fake_open_rendered_pdf(_path, _dpi):
        yield SimpleNamespace(total_pages=2)

    def fake_extract(
        _llm,
        rendered_page,
        total_pages,
        prompt,
        file_label,
        speaker_notes,
    ):
        calls.append(
            (
                rendered_page.page_number,
                total_pages,
                file_label,
                speaker_notes,
            )
        )
        assert prompt == _mock_prompt()
        return PageResult(
            page_number=rendered_page.page_number,
            raw_content=f"slide-{rendered_page.page_number}",
        )

    monkeypatch.setattr(
        pptx_module, "load_prompt", lambda *_args: _mock_prompt()
    )
    monkeypatch.setattr(
        pptx_module,
        "extract_all_speaker_notes",
        lambda *_args: {2: "notes"},
    )
    monkeypatch.setattr(
        pptx_module, "convert_to_pdf", lambda *_args: Path("/tmp/deck.pdf")
    )
    monkeypatch.setattr(pptx_module, "get_vision_dpi_scale", lambda: 2.0)
    monkeypatch.setattr(
        pptx_module,
        "open_rendered_pdf",
        fake_open_rendered_pdf,
    )
    monkeypatch.setattr(
        pptx_module,
        "render_page",
        lambda _rendered, page_number: f"page-{page_number}".encode(),
    )
    monkeypatch.setattr(
        pptx_module,
        "get_extraction_page_workers",
        lambda: 2,
    )
    monkeypatch.setattr(pptx_module, "_extract_single_page", fake_extract)

    result = pptx_module.process_pptx("/tmp/deck.pptx", Mock())

    assert calls == [
        (1, 2, "deck.pptx", {2: "notes"}),
        (2, 2, "deck.pptx", {2: "notes"}),
    ]
    assert [page.page_number for page in result.pages] == [1, 2]
    assert result.total_pages == 2
