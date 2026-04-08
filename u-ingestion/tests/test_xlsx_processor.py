"""Tests for the XLSX processor."""

import threading
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ingestion.processors.xlsx import processor as xlsx_module
from ingestion.utils.file_types import PageResult


class _FakeCell:
    """Minimal worksheet cell stub."""

    def __init__(self, row, column, value, number_format="General"):
        self.row = row
        self.column = column
        self.value = value
        self.number_format = number_format

    def as_tuple(self):
        """Expose coordinates and value for lint-friendly stubs."""
        return self.row, self.column, self.value

    def is_empty(self):
        """Return True when the fake cell has no value."""
        return self.value is None


class _FakeSheet:
    """Minimal worksheet stub."""

    def __init__(self, title, rows, charts=None, images=None):
        self.title = title
        self._rows = rows
        self._charts = charts or []
        self._images = images or []

    def iter_rows(self):
        """Yield worksheet rows."""
        return iter(self._rows)

    def cell(self, row, column):
        """Return a cell by coordinates."""
        for row_cells in self._rows:
            for cell in row_cells:
                if cell.row == row and cell.column == column:
                    return cell
        return _FakeCell(row, column, None)


class _FakeWorkbook:
    """Minimal workbook stub."""

    def __init__(self, sheets):
        self._sheets = {sheet.title: sheet for sheet in sheets}
        self.sheetnames = list(self._sheets)
        self.closed = False

    def __getitem__(self, key):
        """Return a sheet by name."""
        return self._sheets[key]

    def close(self):
        """Track close calls."""
        self.closed = True


def _make_visual_response(content):
    """Build a visual extraction response."""
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": f'{{"content": "{content}"}}'
                            }
                        }
                    ]
                }
            }
        ]
    }


def test_normalize_cell_value_handles_supported_types():
    """Normalize common worksheet value types."""
    normalize = getattr(xlsx_module, "_normalize_cell_value")

    assert normalize(None) == ""
    assert normalize(True) == "TRUE"
    assert normalize(False) == "FALSE"
    assert normalize(date(2026, 3, 27)) == "2026-03-27"
    assert normalize(datetime(2026, 3, 27, 0, 0, 0)) == "2026-03-27"
    assert normalize(datetime(2026, 3, 27, 12, 30, 0)) == (
        "2026-03-27 12:30:00"
    )
    assert normalize("a\r\nb\rc") == "a\nb\nc"


def test_escape_table_text_escapes_pipes_and_newlines():
    """Escape markdown table delimiters."""
    assert getattr(xlsx_module, "_escape_table_text")("a|b\nc") == (
        "a\\|b<br>c"
    )


def test_open_workbooks_returns_formula_and_cached_versions(monkeypatch):
    """Open both workbook views from the same file."""
    formula_workbook = SimpleNamespace(close=Mock())
    cached_workbook = SimpleNamespace(close=Mock())
    calls = []

    def fake_load_workbook(filename, data_only):
        calls.append((filename, data_only))
        return cached_workbook if data_only else formula_workbook

    monkeypatch.setattr(xlsx_module, "load_workbook", fake_load_workbook)

    assert getattr(xlsx_module, "_open_workbooks")("/tmp/data.xlsx") == (
        formula_workbook,
        cached_workbook,
    )
    assert calls == [
        ("/tmp/data.xlsx", False),
        ("/tmp/data.xlsx", True),
    ]


def test_open_workbooks_wraps_primary_open_failure(monkeypatch):
    """Raise when the initial workbook open fails."""

    def fake_load_workbook(filename, data_only):
        assert filename == "/tmp/data.xlsx"
        assert data_only is False
        raise OSError("bad xlsx")

    monkeypatch.setattr(xlsx_module, "load_workbook", fake_load_workbook)

    with pytest.raises(RuntimeError, match="Failed to open XLSX 'data.xlsx'"):
        getattr(xlsx_module, "_open_workbooks")("/tmp/data.xlsx")


def test_open_workbooks_closes_formula_book_when_cached_open_fails(
    monkeypatch,
):
    """Close the primary workbook when the cached load fails."""
    formula_workbook = SimpleNamespace(close=Mock())

    def fake_load_workbook(filename, data_only):
        assert filename == "/tmp/data.xlsx"
        if not data_only:
            return formula_workbook
        raise OSError("bad xlsx")

    monkeypatch.setattr(xlsx_module, "load_workbook", fake_load_workbook)

    with pytest.raises(RuntimeError, match="Failed to open XLSX 'data.xlsx'"):
        getattr(xlsx_module, "_open_workbooks")("/tmp/data.xlsx")

    formula_workbook.close.assert_called_once_with()


def test_open_workbooks_closes_formula_book_on_unexpected_exception(
    monkeypatch,
):
    """Close the primary workbook when the cached load raises an unexpected
    exception outside the known (OSError, ValueError) catch tuple."""
    formula_workbook = SimpleNamespace(close=Mock())

    def fake_load_workbook(filename, data_only):
        assert filename == "/tmp/data.xlsx"
        if not data_only:
            return formula_workbook
        raise KeyError("corrupt zip entry")

    monkeypatch.setattr(xlsx_module, "load_workbook", fake_load_workbook)

    with pytest.raises(KeyError, match="corrupt zip entry"):
        getattr(xlsx_module, "_open_workbooks")("/tmp/data.xlsx")

    formula_workbook.close.assert_called_once_with()


def test_build_cached_values_collects_non_empty_cells():
    """Capture cached display values by coordinates."""
    sheet = _FakeSheet(
        "Data",
        rows=[
            [_FakeCell(1, 1, "A"), _FakeCell(1, 2, None)],
            [_FakeCell(2, 1, 10)],
        ],
    )

    assert getattr(xlsx_module, "_build_cached_values")(sheet) == {
        (1, 1): "A",
        (2, 1): 10,
    }


def test_is_chartsheet_checks_runtime_type_name():
    """Detect dedicated chartsheets by class name."""
    chartsheet_type = type("Chartsheet", (), {})

    is_chartsheet = getattr(xlsx_module, "_is_chartsheet")

    assert is_chartsheet(chartsheet_type()) is True
    assert is_chartsheet(SimpleNamespace()) is False


def test_collect_populated_cells_and_serialize_sheet():
    """Collect populated grid cells and serialize them to markdown."""
    sheet = _FakeSheet(
        "Data",
        rows=[
            [_FakeCell(1, 1, "Name"), _FakeCell(1, 2, "Value")],
            [_FakeCell(2, 1, "Revenue"), _FakeCell(2, 2, 42)],
        ],
    )
    cached_values = {(2, 2): 84}

    cell_data = getattr(xlsx_module, "_collect_populated_cells")(
        sheet, cached_values
    )

    assert cell_data == {
        "rows": {1: {1: "Name", 2: "Value"}, 2: {1: "Revenue", 2: "84"}},
        "columns": [1, 2],
    }
    assert getattr(xlsx_module, "_serialize_sheet")(
        "Data", cell_data
    ) == "\n".join(
        [
            "# Sheet: Data",
            "",
            "| Row | A | B |",
            "| --- | --- | --- |",
            "| 1 | Name | Value |",
            "| 2 | Revenue | 84 |",
        ]
    )


def test_collect_populated_cells_returns_none_for_empty_sheet():
    """Return None when no cells are populated."""
    sheet = _FakeSheet("Empty", rows=[[_FakeCell(1, 1, None)]])

    assert getattr(xlsx_module, "_collect_populated_cells")(sheet, {}) is None


def test_extract_title_text_and_anchor_helpers():
    """Extract chart text and anchor metadata."""
    rich_title = SimpleNamespace(
        tx=SimpleNamespace(
            rich=SimpleNamespace(
                p=[
                    SimpleNamespace(
                        r=[SimpleNamespace(t="Quarterly")],
                        fld=[SimpleNamespace(t=" Results")],
                    )
                ]
            )
        )
    )
    marker_anchor = SimpleNamespace(_from=SimpleNamespace(row=2, col=1))

    extract_title = getattr(xlsx_module, "_extract_title_text")
    extract_anchor = getattr(xlsx_module, "_extract_anchor_cell")

    assert extract_title(" Revenue ") == "Revenue"
    assert extract_title(rich_title) == "Quarterly Results"
    assert extract_title(None) == ""
    assert extract_anchor(marker_anchor) == "B3"
    assert extract_anchor("C5:D7") == "C5"
    assert extract_anchor(object()) == "unknown"


def test_extract_anchor_cell_supports_from_attribute():
    """Read anchors exposed through a from_ attribute."""
    anchor = SimpleNamespace(from_=SimpleNamespace(row=4, col=2))

    assert getattr(xlsx_module, "_extract_anchor_cell")(anchor) == "C5"


def test_load_reference_values_and_series_helpers():
    """Resolve chart references and derive series metadata."""
    formula_sheet = _FakeSheet(
        "Sheet1",
        rows=[
            [_FakeCell(1, 1, "Label"), _FakeCell(1, 2, "Series A")],
            [_FakeCell(2, 1, "Q1"), _FakeCell(2, 2, "=1+1")],
            [_FakeCell(3, 1, "Q2"), _FakeCell(3, 2, "=2+2")],
        ],
    )
    cached_sheet = _FakeSheet(
        "Sheet1",
        rows=[
            [_FakeCell(1, 1, "Label"), _FakeCell(1, 2, "Series A")],
            [_FakeCell(2, 1, "Q1"), _FakeCell(2, 2, 2)],
            [_FakeCell(3, 1, "Q2"), _FakeCell(3, 2, 4)],
        ],
    )
    workbook = _FakeWorkbook([formula_sheet])
    cached_workbook = _FakeWorkbook([cached_sheet])
    series = SimpleNamespace(
        tx=SimpleNamespace(strRef=SimpleNamespace(f="'Sheet1'!$B$1")),
        cat=SimpleNamespace(strRef=SimpleNamespace(f="'Sheet1'!$A$2:$A$3")),
        val=SimpleNamespace(numRef=SimpleNamespace(f="'Sheet1'!$B$2:$B$3")),
    )

    load_reference_values = getattr(xlsx_module, "_load_reference_values")
    extract_series_name = getattr(xlsx_module, "_extract_series_name")
    extract_series_points = getattr(xlsx_module, "_extract_series_points")

    assert load_reference_values(
        workbook,
        cached_workbook,
        "'Sheet1'!$A$2:$A$3",
    ) == ["Q1", "Q2"]
    assert not load_reference_values(workbook, cached_workbook, "")
    assert (
        extract_series_name(series, workbook, cached_workbook, 1) == "Series A"
    )
    assert extract_series_points(
        series,
        workbook,
        cached_workbook,
    ) == ["Q1: 2", "Q2: 4"]


def test_load_reference_values_returns_empty_for_bad_references():
    """Return no values for malformed or missing-sheet references."""
    workbook = _FakeWorkbook([])
    cached_workbook = _FakeWorkbook([])
    load_reference_values = getattr(xlsx_module, "_load_reference_values")

    assert not load_reference_values(workbook, cached_workbook, "bad-ref")
    assert not load_reference_values(
        workbook,
        cached_workbook,
        "'Missing'!$A$1:$A$2",
    )


def test_extract_series_name_prefers_literal_and_falls_back(monkeypatch):
    """Use literal series names first, then a generated default."""
    extract_series_name = getattr(xlsx_module, "_extract_series_name")
    monkeypatch.setattr(
        xlsx_module,
        "_load_reference_values",
        lambda *_args: [],
    )
    literal_series = SimpleNamespace(tx=SimpleNamespace(v=" Revenue "))
    fallback_series = SimpleNamespace(tx=SimpleNamespace(strRef=None, v=None))

    assert (
        extract_series_name(literal_series, object(), object(), 1) == "Revenue"
    )
    assert (
        extract_series_name(fallback_series, object(), object(), 3)
        == "Series 3"
    )


def test_extract_series_points_uses_numref_fallback_and_limit(monkeypatch):
    """Skip blank pairs, use numeric category refs, and cap point output."""
    extract_series_points = getattr(xlsx_module, "_extract_series_points")
    monkeypatch.setattr(
        xlsx_module,
        "_load_reference_values",
        lambda _workbook, _cached, formula: {
            "cats": ["", "Q2"] + [f"Q{idx}" for idx in range(3, 20)],
            "vals": ["", "20"] + [str(idx) for idx in range(3, 20)],
        }.get(formula, []),
    )
    series = SimpleNamespace(
        cat=SimpleNamespace(numRef=SimpleNamespace(f="cats"), strRef=None),
        val=SimpleNamespace(numRef=SimpleNamespace(f="vals")),
    )

    points = extract_series_points(series, object(), object())

    assert points[0] == "Q2: 20"
    assert len(points) == getattr(xlsx_module, "_VISUAL_SERIES_POINT_LIMIT")
    assert "Point 1" not in points[0]


def test_format_chart_markdown_renders_context_and_series():
    """Format chart metadata into markdown."""
    workbook = _FakeWorkbook([])
    cached_workbook = _FakeWorkbook([])
    chart = SimpleNamespace(
        anchor="A1:B5",
        title="Revenue",
        x_axis=SimpleNamespace(title="Quarter"),
        y_axis=SimpleNamespace(title="USD"),
        ser=[],
    )

    content = getattr(xlsx_module, "_format_chart_markdown")(
        "Data",
        chart,
        workbook,
        cached_workbook,
    )

    assert '> [Chart]: SimpleNamespace — "Revenue"' in content
    assert "X-axis: Quarter" in content
    assert "Y-axis: USD" in content
    assert "Sheet: Data, Cell: A1" in content


def test_format_chart_markdown_renders_series_variants(monkeypatch):
    """Render both populated and empty chart series summaries."""
    monkeypatch.setattr(
        xlsx_module,
        "_extract_series_name",
        lambda _series, _workbook, _cached, series_index: (
            f"Series {series_index}"
        ),
    )
    monkeypatch.setattr(
        xlsx_module,
        "_extract_series_points",
        lambda series, _workbook, _cached: (
            ["Q1: 10"] if getattr(series, "has_points", False) else []
        ),
    )
    chart = SimpleNamespace(
        anchor="A1:B5",
        title="Revenue",
        x_axis=SimpleNamespace(title=None),
        y_axis=SimpleNamespace(title=None),
        ser=[
            SimpleNamespace(has_points=True),
            SimpleNamespace(has_points=False),
        ],
    )

    content = getattr(xlsx_module, "_format_chart_markdown")(
        "Data",
        chart,
        object(),
        object(),
    )

    assert '> Series "Series 1": Q1: 10' in content
    assert '> Series "Series 2": no data points recovered' in content


def test_parse_visual_response_validates_payload():
    """Parse visual extraction content and reject malformed payloads."""
    parse_visual_response = getattr(xlsx_module, "_parse_visual_response")

    assert parse_visual_response(_make_visual_response("ok")) == "ok"

    with pytest.raises(ValueError, match="LLM response missing choices"):
        parse_visual_response({})


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
def test_parse_visual_response_rejects_payload_variants(response, message):
    """Reject malformed visual extraction payload shapes."""
    with pytest.raises(ValueError, match=message):
        getattr(xlsx_module, "_parse_visual_response")(response)


def test_llm_call_with_retry_retries_then_succeeds(monkeypatch):
    """Retry transient visual extraction errors."""

    class DummyRetryable(Exception):
        """Retryable stand-in."""

    llm = Mock()
    llm.call.side_effect = [
        DummyRetryable("retry"),
        _make_visual_response("ok"),
    ]
    monkeypatch.setattr(
        xlsx_module, "_PARSE_RETRYABLE_ERRORS", (DummyRetryable,)
    )
    monkeypatch.setattr(xlsx_module, "get_xlsx_vision_max_retries", lambda: 2)
    monkeypatch.setattr(
        xlsx_module,
        "get_xlsx_vision_retry_delay",
        lambda: 0.0,
    )
    monkeypatch.setattr(xlsx_module.time, "sleep", lambda _seconds: None)

    result = getattr(xlsx_module, "_llm_call_with_retry")(
        llm,
        [{"role": "user", "content": "hi"}],
        {"stage": "extraction", "tools": [], "tool_choice": "required"},
        "sheet image",
    )

    assert result == "ok"
    assert getattr(llm.call, "call_count") == 2


def test_llm_call_with_retry_retries_on_missing_tool_calls(monkeypatch):
    """Retry when the model returns an empty tool_calls payload."""
    bad_response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"tool_calls": None, "content": "oops"},
            }
        ]
    }

    llm = Mock()
    llm.call.side_effect = [bad_response, _make_visual_response("ok")]
    monkeypatch.setattr(xlsx_module, "get_xlsx_vision_max_retries", lambda: 2)
    monkeypatch.setattr(
        xlsx_module,
        "get_xlsx_vision_retry_delay",
        lambda: 0.0,
    )
    monkeypatch.setattr(xlsx_module.time, "sleep", lambda _seconds: None)

    result = getattr(xlsx_module, "_llm_call_with_retry")(
        llm,
        [{"role": "user", "content": "hi"}],
        {"stage": "extraction", "tools": [], "tool_choice": "required"},
        "sheet image",
    )

    assert result == "ok"
    assert getattr(llm.call, "call_count") == 2


def test_llm_call_with_retry_raises_value_error_after_exhausted_retries(
    monkeypatch,
):
    """Raise ValueError when every retry returns a missing-tool-calls body."""
    bad_response = {
        "choices": [{"message": {"tool_calls": None, "content": "still bad"}}]
    }

    llm = Mock()
    llm.call.side_effect = [bad_response, bad_response]
    monkeypatch.setattr(xlsx_module, "get_xlsx_vision_max_retries", lambda: 2)
    monkeypatch.setattr(
        xlsx_module,
        "get_xlsx_vision_retry_delay",
        lambda: 0.0,
    )
    monkeypatch.setattr(xlsx_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(ValueError, match="missing tool calls"):
        getattr(xlsx_module, "_llm_call_with_retry")(
            llm,
            [{"role": "user", "content": "hi"}],
            {"stage": "extraction", "tools": [], "tool_choice": "required"},
            "sheet image",
        )
    assert getattr(llm.call, "call_count") == 2


def test_parse_visual_response_missing_tool_calls_includes_diagnostics():
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
        getattr(xlsx_module, "_parse_visual_response")(response)

    message = str(exc_info.value)
    assert "missing tool calls" in message
    assert "finish_reason=length" in message
    assert "truncated thinking" in message


def test_llm_call_with_retry_raises_after_final_retry(monkeypatch):
    """Raise the last retryable error after exhausting retries."""

    class DummyRetryable(Exception):
        """Retryable stand-in."""

    llm = Mock()
    llm.call.side_effect = DummyRetryable("retry")
    monkeypatch.setattr(
        xlsx_module, "_PARSE_RETRYABLE_ERRORS", (DummyRetryable,)
    )
    monkeypatch.setattr(xlsx_module, "get_xlsx_vision_max_retries", lambda: 1)
    monkeypatch.setattr(
        xlsx_module,
        "get_xlsx_vision_retry_delay",
        lambda: 0.0,
    )

    with pytest.raises(DummyRetryable, match="retry"):
        getattr(xlsx_module, "_llm_call_with_retry")(
            llm,
            [{"role": "user", "content": "hi"}],
            {"stage": "extraction", "tools": [], "tool_choice": "required"},
            "sheet image",
        )


def test_llm_call_with_retry_raises_when_retry_loop_never_runs(monkeypatch):
    """Guard against a zero-retry XLSX vision configuration."""
    monkeypatch.setattr(xlsx_module, "get_xlsx_vision_max_retries", lambda: 0)
    monkeypatch.setattr(
        xlsx_module,
        "get_xlsx_vision_retry_delay",
        lambda: 0.0,
    )

    with pytest.raises(
        RuntimeError, match="sheet image exited retry loop without a response"
    ):
        getattr(xlsx_module, "_llm_call_with_retry")(
            Mock(),
            [{"role": "user", "content": "hi"}],
            {"stage": "extraction", "tools": [], "tool_choice": "required"},
            "sheet image",
        )


def test_describe_image_uses_correct_media_type(monkeypatch):
    """Preserve supported image types and fall back to PNG for others."""
    llm = Mock()
    prompt = {
        "system_prompt": "system",
        "user_prompt": "user",
        "stage": "extraction",
        "tools": [],
        "tool_choice": "required",
    }
    captured = []

    def fake_call_with_retry(_llm, messages, _prompt, _context):
        captured.append(messages[1]["content"][0]["image_url"]["url"])
        return "described"

    monkeypatch.setattr(
        xlsx_module, "_llm_call_with_retry", fake_call_with_retry
    )
    jpeg_image = SimpleNamespace(
        format="jpeg",
        anchor="B2:C4",
        _data=lambda: b"jpeg",
    )
    bmp_image = SimpleNamespace(
        format="bmp",
        anchor="D5:E8",
        _data=lambda: b"png",
    )

    cache: dict[str, str] = {}
    assert (
        getattr(xlsx_module, "_describe_image")(
            llm,
            prompt,
            "Sheet1",
            jpeg_image,
            "jpeg",
            cache,
        )
        == "described"
    )
    assert (
        getattr(xlsx_module, "_describe_image")(
            llm,
            prompt,
            "Sheet1",
            bmp_image,
            "bmp",
            cache,
        )
        == "described"
    )
    assert captured[0].startswith("data:image/jpeg;base64,")
    assert captured[1].startswith("data:image/png;base64,")


def test_describe_image_deduplicates_by_hash(monkeypatch):
    """Return cached description for identical image bytes without LLM call."""
    llm = Mock()
    prompt = {
        "system_prompt": "system",
        "user_prompt": "user",
        "stage": "extraction",
        "tools": [],
        "tool_choice": "required",
    }
    call_count = []

    def fake_call_with_retry(_llm, _messages, _prompt, _context):
        call_count.append(1)
        return "first description"

    monkeypatch.setattr(
        xlsx_module, "_llm_call_with_retry", fake_call_with_retry
    )
    same_bytes = b"identical-image-bytes"
    image_a = SimpleNamespace(
        format="png", anchor="B2", _data=lambda: same_bytes
    )
    image_b = SimpleNamespace(
        format="png", anchor="C3", _data=lambda: same_bytes
    )
    cache: dict[str, str] = {}

    first = getattr(xlsx_module, "_describe_image")(
        llm, prompt, "Sheet1", image_a, "img1", cache
    )
    second = getattr(xlsx_module, "_describe_image")(
        llm, prompt, "Sheet2", image_b, "img2", cache
    )

    assert first == "first description"
    assert second == "first description"
    assert len(call_count) == 1
    assert len(cache) == 1


def test_describe_image_requires_embedded_bytes():
    """Raise when an embedded image object cannot expose raw bytes."""
    with pytest.raises(RuntimeError, match="bytes unavailable"):
        getattr(xlsx_module, "_describe_image")(
            Mock(),
            {
                "system_prompt": "system",
                "user_prompt": "user",
                "stage": "extraction",
                "tools": [],
                "tool_choice": "required",
            },
            "Sheet1",
            SimpleNamespace(anchor="A1"),
            "image",
            {},
        )


def test_describe_all_images_and_format_all_charts(monkeypatch):
    """Describe every image and format every chart on a sheet."""
    sheet = SimpleNamespace(
        title="Sheet1",
        _images=[SimpleNamespace(), SimpleNamespace()],
        _charts=[SimpleNamespace(), SimpleNamespace()],
    )
    monkeypatch.setattr(
        xlsx_module,
        "_describe_image",
        lambda _llm, _prompt, sheet_name, _image, context, _cache: (
            f"{sheet_name}:{context}"
        ),
    )
    monkeypatch.setattr(
        xlsx_module,
        "_format_chart_markdown",
        lambda sheet_name, _chart, _workbook, _cached: f"chart:{sheet_name}",
    )

    assert getattr(xlsx_module, "_describe_all_images")(
        Mock(), {}, sheet, "book.xlsx", {}
    ) == [
        "Sheet1:book.xlsx 'Sheet1' image 1",
        "Sheet1:book.xlsx 'Sheet1' image 2",
    ]
    assert getattr(xlsx_module, "_format_all_charts")(
        sheet, object(), object()
    ) == [
        "chart:Sheet1",
        "chart:Sheet1",
    ]


def test_preread_image_data_extracts_bytes_and_metadata():
    """Pre-read image bytes, format, and anchor from an openpyxl image."""
    image = SimpleNamespace(
        format="jpeg",
        anchor="B2:C4",
        _data=lambda: b"jpeg-bytes",
    )
    result = getattr(xlsx_module, "_preread_image_data")(image)

    assert result == {
        "bytes": b"jpeg-bytes",
        "format": "jpeg",
        "anchor": "B2:C4",
    }


def test_preread_image_data_returns_none_without_data_attr():
    """Return None when the image object cannot expose raw bytes."""
    image = SimpleNamespace(format="png", anchor="A1")

    assert getattr(xlsx_module, "_preread_image_data")(image) is None


def test_describe_preloaded_image_calls_llm_and_caches(monkeypatch):
    """Describe a pre-loaded image via LLM and store in the cache."""

    captured = []
    cache_state = getattr(xlsx_module, "_ImageDescriptionState")()

    def fake_call_with_retry(_llm, messages, _prompt, _context):
        captured.append(messages[1]["content"][0]["image_url"]["url"])
        return "described"

    monkeypatch.setattr(
        xlsx_module,
        "_llm_call_with_retry",
        fake_call_with_retry,
    )

    image_data = {
        "bytes": b"test-image-bytes",
        "format": "jpeg",
        "anchor": "B2:C4",
    }

    result = getattr(xlsx_module, "_describe_preloaded_image")(
        Mock(),
        {
            "system_prompt": "system",
            "user_prompt": "user",
            "stage": "extraction",
            "tools": [],
            "tool_choice": "required",
        },
        "Sheet1",
        image_data,
        "context",
        cache_state,
    )

    assert result == "described"
    assert len(cache_state.descriptions) == 1
    assert captured[0].startswith("data:image/jpeg;base64,")


def test_describe_preloaded_image_deduplicates_by_hash(monkeypatch):
    """Return cached description for identical image bytes."""

    call_count = []
    cache_state = getattr(xlsx_module, "_ImageDescriptionState")()

    def fake_call_with_retry(_llm, _messages, _prompt, _context):
        call_count.append(1)
        return "first description"

    monkeypatch.setattr(
        xlsx_module,
        "_llm_call_with_retry",
        fake_call_with_retry,
    )

    same_bytes = b"identical-image-bytes"
    image_a = {"bytes": same_bytes, "format": "png", "anchor": "B2"}
    image_b = {"bytes": same_bytes, "format": "png", "anchor": "C3"}
    prompt = {
        "system_prompt": "system",
        "user_prompt": "user",
        "stage": "extraction",
        "tools": [],
        "tool_choice": "required",
    }
    describe = getattr(xlsx_module, "_describe_preloaded_image")

    first = describe(
        Mock(),
        prompt,
        "Sheet1",
        image_a,
        "img1",
        cache_state,
    )
    second = describe(
        Mock(),
        prompt,
        "Sheet2",
        image_b,
        "img2",
        cache_state,
    )

    assert first == "first description"
    assert second == "first description"
    assert len(call_count) == 1
    assert len(cache_state.descriptions) == 1


def test_describe_preloaded_image_deduplicates_inflight_requests(
    monkeypatch,
):
    """Concurrent callers should share one in-flight image description."""
    call_count = []
    cache_state = getattr(xlsx_module, "_ImageDescriptionState")()
    started = threading.Event()
    release = threading.Event()

    def fake_call_with_retry(_llm, _messages, _prompt, _context):
        call_count.append(1)
        started.set()
        release.wait(timeout=1)
        return "shared description"

    monkeypatch.setattr(
        xlsx_module,
        "_llm_call_with_retry",
        fake_call_with_retry,
    )

    image = {
        "bytes": b"same-image-bytes",
        "format": "png",
        "anchor": "B2",
    }
    prompt = {
        "system_prompt": "system",
        "user_prompt": "user",
        "stage": "extraction",
        "tools": [],
        "tool_choice": "required",
    }
    results = []

    def runner(context):
        results.append(
            getattr(xlsx_module, "_describe_preloaded_image")(
                Mock(),
                prompt,
                "Sheet1",
                image,
                context,
                cache_state,
            )
        )

    first = threading.Thread(target=runner, args=("img1",))
    second = threading.Thread(target=runner, args=("img2",))

    first.start()
    started.wait(timeout=1)
    second.start()
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert results == ["shared description", "shared description"]
    assert len(call_count) == 1
    assert len(cache_state.descriptions) == 1


def test_describe_preloaded_image_waiter_sees_owner_retryable_failure(
    monkeypatch,
):
    """Waiting callers should surface owner failures with shared context."""
    cache_state = getattr(xlsx_module, "_ImageDescriptionState")()
    started = threading.Event()
    release = threading.Event()
    prompt = {
        "system_prompt": "system",
        "user_prompt": "user",
        "stage": "extraction",
        "tools": [],
        "tool_choice": "required",
    }
    image = {
        "bytes": b"same-image-bytes",
        "format": "png",
        "anchor": "B2",
    }
    errors = []

    def fake_call_with_retry(_llm, _messages, _prompt, _context):
        started.set()
        release.wait(timeout=1)
        raise RuntimeError("vision failed")

    monkeypatch.setattr(
        xlsx_module,
        "_llm_call_with_retry",
        fake_call_with_retry,
    )

    def runner(context):
        try:
            getattr(xlsx_module, "_describe_preloaded_image")(
                Mock(),
                prompt,
                "Sheet1",
                image,
                context,
                cache_state,
            )
        except RuntimeError as exc:
            errors.append((context, exc))

    owner = threading.Thread(target=runner, args=("img1",))
    waiter = threading.Thread(target=runner, args=("img2",))

    owner.start()
    started.wait(timeout=1)
    waiter.start()
    release.set()
    owner.join(timeout=1)
    waiter.join(timeout=1)

    assert len(errors) == 2
    owner_error = dict(errors)["img1"]
    waiter_error = dict(errors)["img2"]
    assert str(owner_error) == "vision failed"
    assert "failed while waiting for a deduplicated image description" in str(
        waiter_error
    )
    assert "vision failed" in str(waiter_error)
    assert waiter_error.__cause__ is owner_error
    assert not cache_state.descriptions


def test_describe_preloaded_image_waiter_sees_owner_unexpected_failure(
    monkeypatch,
):
    """Waiting callers should get a synthesized error.

    The owner may fail unexpectedly before caching a shared result.
    """
    cache_state = getattr(xlsx_module, "_ImageDescriptionState")()
    started = threading.Event()
    release = threading.Event()
    prompt = {
        "system_prompt": "system",
        "user_prompt": "user",
        "stage": "extraction",
        "tools": [],
        "tool_choice": "required",
    }
    image = {
        "bytes": b"same-image-bytes",
        "format": "png",
        "anchor": "B2",
    }
    errors = []

    def fake_call_with_retry(_llm, _messages, _prompt, _context):
        started.set()
        release.wait(timeout=1)
        raise TypeError("unexpected failure")

    monkeypatch.setattr(
        xlsx_module,
        "_llm_call_with_retry",
        fake_call_with_retry,
    )

    def runner(context):
        try:
            getattr(xlsx_module, "_describe_preloaded_image")(
                Mock(),
                prompt,
                "Sheet1",
                image,
                context,
                cache_state,
            )
        except (RuntimeError, TypeError) as exc:
            errors.append((context, exc))

    owner = threading.Thread(target=runner, args=("img1",))
    waiter = threading.Thread(target=runner, args=("img2",))

    owner.start()
    started.wait(timeout=1)
    waiter.start()
    release.set()
    owner.join(timeout=1)
    waiter.join(timeout=1)

    assert len(errors) == 2
    owner_error = dict(errors)["img1"]
    waiter_error = dict(errors)["img2"]
    assert isinstance(owner_error, TypeError)
    assert str(owner_error) == "unexpected failure"
    assert "failed while waiting for a deduplicated image description" in str(
        waiter_error
    )
    assert "failed before caching a deduplicated image description" in str(
        waiter_error
    )
    assert isinstance(waiter_error.__cause__, RuntimeError)
    assert not cache_state.descriptions


def test_preread_all_sheets_collects_sheet_data(monkeypatch):
    """Pre-read cell data, charts, and images from all sheets."""
    image = SimpleNamespace(format="png", anchor="A1", _data=lambda: b"img")
    sheet1 = _FakeSheet(
        "Data",
        rows=[
            [_FakeCell(1, 1, "Name"), _FakeCell(1, 2, "Value")],
        ],
        charts=[SimpleNamespace()],
        images=[image],
    )
    sheet2 = _FakeSheet("Empty", rows=[[_FakeCell(1, 1, None)]])
    workbook = _FakeWorkbook([sheet1, sheet2])
    cached_workbook = _FakeWorkbook(
        [
            _FakeSheet(
                "Data",
                rows=[
                    [_FakeCell(1, 1, "Name"), _FakeCell(1, 2, "Value")],
                ],
            ),
            _FakeSheet("Empty", rows=[[_FakeCell(1, 1, None)]]),
        ]
    )
    monkeypatch.setattr(
        xlsx_module,
        "_format_all_charts",
        lambda sheet, _workbook, _cached: [f"chart:{sheet.title}"],
    )

    preread = getattr(xlsx_module, "_preread_all_sheets")(
        workbook, cached_workbook
    )

    assert len(preread) == 2
    assert preread[0].title == "Data"
    assert preread[0].page_number == 1
    assert preread[0].is_chartsheet is False
    assert preread[0].cell_data is not None
    assert preread[0].chart_descriptions == ["chart:Data"]
    assert len(preread[0].image_data) == 1
    assert preread[0].image_data[0]["bytes"] == b"img"
    assert preread[1].title == "Empty"
    assert preread[1].page_number == 2
    assert preread[1].cell_data is None
    assert preread[1].image_data == []


def test_process_single_sheet_assembles_content(monkeypatch):
    """Assemble content from pre-read cell data and visuals."""

    monkeypatch.setattr(
        xlsx_module,
        "_describe_preloaded_image",
        lambda _l, _p, sn, _i, _c, _state=None: f"image:{sn}",
    )

    sheet_data = xlsx_module.SheetData(
        title="Data",
        page_number=1,
        is_chartsheet=False,
        cell_data={
            "rows": {1: {1: "Name", 2: "Value"}},
            "columns": [1, 2],
        },
        chart_descriptions=["chart:Data"],
        image_data=[
            {"bytes": b"img", "format": "png", "anchor": "A1"},
        ],
    )
    prompt = {
        "system_prompt": "system",
        "user_prompt": "user",
        "stage": "extraction",
        "tools": [],
        "tool_choice": "required",
    }
    cache_state = getattr(xlsx_module, "_ImageDescriptionState")()

    result = getattr(xlsx_module, "_process_single_sheet")(
        sheet_data,
        Mock(),
        prompt,
        "book.xlsx",
        1,
        cache_state,
    )

    assert result.page_number == 1
    assert "# Sheet: Data" in result.raw_content
    assert "chart:Data" in result.raw_content
    assert "image:Data" in result.raw_content


def test_process_xlsx_builds_sheet_results_and_closes_workbooks(monkeypatch):
    """Build one page per sheet and close workbook handles."""
    image = SimpleNamespace(format="png", anchor="A1", _data=lambda: b"img")
    sheet1 = _FakeSheet(
        "Data",
        rows=[
            [_FakeCell(1, 1, "Name"), _FakeCell(1, 2, "Value")],
            [_FakeCell(2, 1, "Revenue"), _FakeCell(2, 2, 42)],
        ],
        charts=[SimpleNamespace()],
        images=[image],
    )
    chartsheet_type = type("Chartsheet", (), {})
    sheet2 = chartsheet_type()
    sheet2.title = "Charts"
    setattr(sheet2, "_charts", [SimpleNamespace()])
    setattr(sheet2, "_images", [])
    workbook = _FakeWorkbook([sheet1, sheet2])
    cached_workbook = _FakeWorkbook(
        [
            _FakeSheet(
                "Data",
                rows=[
                    [_FakeCell(1, 1, "Name"), _FakeCell(1, 2, "Value")],
                    [_FakeCell(2, 1, "Revenue"), _FakeCell(2, 2, 84)],
                ],
            ),
            _FakeSheet("Charts", rows=[]),
        ]
    )
    monkeypatch.setattr(
        xlsx_module,
        "_open_workbooks",
        lambda _path: (workbook, cached_workbook),
    )
    monkeypatch.setattr(
        xlsx_module,
        "load_prompt",
        lambda *_args: {
            "system_prompt": "system",
            "user_prompt": "user",
            "stage": "extraction",
            "tools": [],
            "tool_choice": "required",
        },
    )
    monkeypatch.setattr(
        xlsx_module,
        "_format_all_charts",
        lambda sheet, _workbook, _cached: [f"chart:{sheet.title}"],
    )
    monkeypatch.setattr(
        xlsx_module,
        "_describe_preloaded_image",
        lambda _llm, _prompt, sheet_name, _img, _ctx, _state=None: (
            f"image:{sheet_name}"
        ),
    )
    monkeypatch.setattr(
        xlsx_module,
        "get_extraction_page_workers",
        lambda: 2,
    )

    result = xlsx_module.process_xlsx("/tmp/data.xlsx", Mock())

    assert [page.page_number for page in result.pages] == [1, 2]
    assert "# Sheet: Data" in result.pages[0].raw_content
    assert "chart:Data" in result.pages[0].raw_content
    assert "image:Data" in result.pages[0].raw_content
    assert result.pages[1] == PageResult(
        page_number=2,
        raw_content="# Sheet: Charts\n\nchart:Charts",
    )
    assert workbook.closed is True
    assert cached_workbook.closed is True


def test_process_xlsx_handles_empty_and_grid_only_sheets(monkeypatch):
    """Build empty-sheet and grid-only content without visual prompts."""
    empty_sheet = _FakeSheet("Empty", rows=[[_FakeCell(1, 1, None)]])
    data_sheet = _FakeSheet(
        "Data",
        rows=[
            [_FakeCell(1, 1, "Name"), _FakeCell(1, 2, "Value")],
            [_FakeCell(2, 1, "Revenue"), _FakeCell(2, 2, 42)],
        ],
    )
    workbook = _FakeWorkbook([empty_sheet, data_sheet])
    cached_workbook = _FakeWorkbook(
        [
            _FakeSheet("Empty", rows=[[_FakeCell(1, 1, None)]]),
            _FakeSheet(
                "Data",
                rows=[
                    [_FakeCell(1, 1, "Name"), _FakeCell(1, 2, "Value")],
                    [_FakeCell(2, 1, "Revenue"), _FakeCell(2, 2, 42)],
                ],
            ),
        ]
    )
    load_prompt = Mock()
    monkeypatch.setattr(
        xlsx_module,
        "_open_workbooks",
        lambda _path: (workbook, cached_workbook),
    )
    monkeypatch.setattr(xlsx_module, "load_prompt", load_prompt)
    monkeypatch.setattr(
        xlsx_module,
        "_format_all_charts",
        lambda _sheet, _workbook, _cached: [],
    )
    monkeypatch.setattr(
        xlsx_module,
        "get_extraction_page_workers",
        lambda: 2,
    )

    result = xlsx_module.process_xlsx("/tmp/data.xlsx", Mock())

    assert result.pages[0] == PageResult(
        page_number=1,
        raw_content=("# Sheet: Empty\n\nThis sheet contains no data."),
    )
    assert result.pages[1].raw_content == "\n".join(
        [
            "# Sheet: Data",
            "",
            "| Row | A | B |",
            "| --- | --- | --- |",
            "| 1 | Name | Value |",
            "| 2 | Revenue | 42 |",
        ]
    )
    load_prompt.assert_not_called()
    assert workbook.closed is True
    assert cached_workbook.closed is True
