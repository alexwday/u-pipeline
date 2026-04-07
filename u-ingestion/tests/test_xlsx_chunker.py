"""Tests for the XLSX chunker module."""

import json
from unittest.mock import Mock

import pytest

from ingestion.stages.chunkers import xlsx_chunker as xlsx_chunker_module
from ingestion.utils.file_types import PageResult


def _make_xlsx_content(sheet_name="TestSheet", rows=None, visuals=None):
    """Build fake XLSX markdown content.

    Params: sheet_name, rows, visuals. Returns: str.
    """
    lines = [f"# Sheet: {sheet_name}", ""]
    if rows is not None:
        lines.append("| Row | B | C | D |")
        lines.append("| --- | --- | --- | --- |")
        for row_num, cells in rows:
            cell_str = " | ".join(cells)
            lines.append(f"| {row_num} | {cell_str} |")
    if visuals:
        lines.append("")
        lines.extend(visuals)
    return "\n".join(lines)


def _make_page(
    content,
    page_number=1,
    layout_type="standard",
    **overrides,
):
    """Build a PageResult for testing.

    Params: content, page_number, layout_type.
    Returns: PageResult.
    """
    page = PageResult(
        page_number=page_number,
        raw_content=content,
        layout_type=layout_type,
    )
    for key, value in overrides.items():
        setattr(page, key, value)
    return page


def _make_llm_response(breakpoints, sheet_pt, section_pt):
    """Build a mock LLM tool-call response dict.

    Params: breakpoints, sheet_pt, section_pt. Returns: dict.
    """
    args = json.dumps(
        {
            "breakpoints": breakpoints,
            "sheet_passthrough_rows": sheet_pt,
            "section_passthrough_rows": section_pt,
        }
    )
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": args,
                            }
                        }
                    ]
                }
            }
        ],
    }


# ------------------------------------------------------------------
# _parse_sheet_content
# ------------------------------------------------------------------


def test_parse_sheet_content_extracts_components():
    """Heading, header, separator, data rows, visual blocks."""
    content = _make_xlsx_content(
        sheet_name="CR6_A",
        rows=[
            (2, ["Title", "", ""]),
            (4, ["Desc", "", ""]),
            (6, ["Data1", "10", "20"]),
        ],
        visuals=["> [Figure]: Chart description"],
    )

    parsed = xlsx_chunker_module.parse_sheet_content(content)

    assert parsed["heading"] == "# Sheet: CR6_A\n"
    assert parsed["table_header"] == "| Row | B | C | D |"
    assert parsed["separator"] == "| --- | --- | --- | --- |"
    assert len(parsed["data_rows"]) == 3
    assert parsed["data_rows"][0] == (
        2,
        "| 2 | Title |  |  |",
    )
    assert parsed["data_rows"][2][0] == 6
    assert len(parsed["visual_blocks"]) == 1
    assert parsed["visual_blocks"][0] == ("> [Figure]: Chart description")


def test_parse_sheet_content_handles_no_visuals():
    """Sheet without visual blocks."""
    content = _make_xlsx_content(rows=[(1, ["A", "B", "C"])])

    parsed = xlsx_chunker_module.parse_sheet_content(content)

    assert not parsed["visual_blocks"]
    assert len(parsed["data_rows"]) == 1


def test_parse_sheet_content_handles_visual_only():
    """Sheet with only blockquotes, no table."""
    content = "\n".join(
        [
            "# Sheet: Charts",
            "",
            "> [Chart]: Revenue trend",
            "> [Figure]: Bar chart",
        ]
    )

    parsed = xlsx_chunker_module.parse_sheet_content(content)

    assert parsed["heading"] == "# Sheet: Charts\n"
    assert parsed["table_header"] == ""
    assert parsed["separator"] == ""
    assert not parsed["data_rows"]
    assert len(parsed["visual_blocks"]) == 2


# ------------------------------------------------------------------
# _format_xlsx_batch
# ------------------------------------------------------------------


def test_format_xlsx_batch_includes_all_sections():
    """All 5 XML sections present when content exists."""
    result = xlsx_chunker_module.format_xlsx_batch(
        header_rows_context=["| 2 | Title |"],
        sheet_passthrough_rows=["| 4 | Header |"],
        section_context_rows=["| 6 | Section |"],
        overlap_rows=["| 10 | Overlap |"],
        batch_rows=[(12, "| 12 | Data |")],
    )

    assert "<sheet_context>" in result
    assert "| 2 | Title |" in result
    assert "</sheet_context>" in result
    assert "<sheet_passthrough>" in result
    assert "| 4 | Header |" in result
    assert "</sheet_passthrough>" in result
    assert "<section_context>" in result
    assert "| 6 | Section |" in result
    assert "</section_context>" in result
    assert "<prior_chunk_overlap>" in result
    assert "| 10 | Overlap |" in result
    assert "</prior_chunk_overlap>" in result
    assert "<current_batch>" in result
    assert "[12] | 12 | Data |" in result
    assert "</current_batch>" in result


def test_format_xlsx_batch_omits_empty_sections():
    """Missing passthrough/overlap/section sections omitted."""
    result = xlsx_chunker_module.format_xlsx_batch(
        header_rows_context=["| 2 | Title |"],
        sheet_passthrough_rows=[],
        section_context_rows=[],
        overlap_rows=[],
        batch_rows=[(5, "| 5 | Data |")],
    )

    assert "<sheet_context>" in result
    assert "<current_batch>" in result
    assert "<sheet_passthrough>" not in result
    assert "<section_context>" not in result
    assert "<prior_chunk_overlap>" not in result


# ------------------------------------------------------------------
# _parse_xlsx_response
# ------------------------------------------------------------------


def test_parse_xlsx_response_extracts_both_fields():
    """Valid response parsed into three lists."""
    response = _make_llm_response([15, 30], [2, 4], [10])

    breakpoints, sheet_pt, section_pt = (
        xlsx_chunker_module.parse_xlsx_response(response)
    )

    assert breakpoints == [15, 30]
    assert sheet_pt == [2, 4]
    assert section_pt == [10]


def test_parse_xlsx_response_rejects_malformed():
    """Raises ValueError for malformed responses."""
    with pytest.raises(ValueError, match="Malformed"):
        xlsx_chunker_module.parse_xlsx_response({})

    with pytest.raises(ValueError, match="Malformed"):
        xlsx_chunker_module.parse_xlsx_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"arguments": ("not-json{")}}
                            ]
                        }
                    }
                ]
            }
        )

    with pytest.raises(ValueError, match="Malformed"):
        xlsx_chunker_module.parse_xlsx_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"arguments": ('"string"')}}
                            ]
                        }
                    }
                ]
            }
        )

    bad_args = json.dumps(
        {
            "breakpoints": "bad",
            "sheet_passthrough_rows": [],
            "section_passthrough_rows": [],
        }
    )
    with pytest.raises(ValueError, match="must be arrays"):
        xlsx_chunker_module.parse_xlsx_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": bad_args,
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        )


# ------------------------------------------------------------------
# _row_num_from_line
# ------------------------------------------------------------------


def test_row_num_from_line_returns_negative_for_non_row():
    """Return -1 when line has no row number pattern."""
    result = xlsx_chunker_module.row_num_from_line("no row number here")
    assert result == -1


def test_row_num_from_line_extracts_number():
    """Return row number from a valid table line."""
    result = xlsx_chunker_module.row_num_from_line("| 42 | data |")
    assert result == 42


# ------------------------------------------------------------------
# _build_section_context_lines
# ------------------------------------------------------------------


def test_build_section_context_lines_picks_from_lookup():
    """Lines built from matching row numbers in lookup."""
    lookup = {5: "| 5 | sec |", 6: "| 6 | data |"}
    lines = xlsx_chunker_module.build_section_context_lines([5], lookup)
    assert lines == ["| 5 | sec |"]


def test_build_section_context_lines_skips_missing():
    """Row numbers not in lookup are skipped."""
    lookup = {5: "| 5 | sec |"}
    lines = xlsx_chunker_module.build_section_context_lines([99], lookup)
    assert not lines


# ------------------------------------------------------------------
# _collect_xlsx_breakpoints
# ------------------------------------------------------------------


def _build_parsed_with_rows(row_count):
    """Build a parsed dict with sequential data rows.

    Params: row_count (int). Returns: dict.
    """
    data_rows = []
    for idx in range(1, row_count + 1):
        data_rows.append((idx, f"| {idx} | val_{idx} |"))
    return {
        "heading": "# Sheet: Test\n",
        "table_header": "| Row | B |",
        "separator": "| --- | --- |",
        "data_rows": data_rows,
        "visual_blocks": [],
    }


def test_collect_xlsx_breakpoints_accumulates_passthrough():
    """Sheet passthrough rows accumulate across batches."""
    parsed = _build_parsed_with_rows(10)
    call_count = [0]

    def fake_call(messages, stage, tools, tool_choice, context):
        """Return different breakpoints per call.

        Params: messages, stage, tools, tool_choice, context.
        Returns: dict.
        """
        assert messages is not None
        assert stage == "chunking"
        assert tools is not None
        assert tool_choice is not None
        assert context is not None
        call_count[0] += 1
        if call_count[0] == 1:
            return _make_llm_response([5], [2], [3])
        return _make_llm_response([8], [3], [7])

    llm = Mock()
    llm.call.side_effect = fake_call

    prompt = {
        "stage": "chunking",
        "system_prompt": "system",
        "user_prompt": ("{batch_content} limit={embedding_limit}"),
        "tools": [],
        "tool_choice": "required",
    }

    breakpoints, sheet_pt, section_snaps = (
        xlsx_chunker_module.collect_xlsx_breakpoints(
            parsed=parsed,
            llm=llm,
            prompt=prompt,
            batch_size=5,
            row_counts=(2, 2),
            embedding_limit=8192,
            context="test",
        )
    )

    assert breakpoints == [5, 8]
    assert sheet_pt == [2, 3]
    assert call_count[0] == 2
    assert len(section_snaps) == 2
    assert section_snaps[0] == [3]
    assert section_snaps[1] == [7]


def test_collect_xlsx_breakpoints_includes_header_and_overlap():
    """Header rows and overlap present in batch."""
    parsed = _build_parsed_with_rows(10)
    captured_messages: list[str] = []

    def fake_call(messages, stage, tools, tool_choice, context):
        """Capture messages and return empty response.

        Params: messages, stage, tools, tool_choice, context.
        Returns: dict.
        """
        assert stage is not None
        assert tools is not None
        assert tool_choice is not None
        assert context is not None
        user_msg = messages[-1]["content"]
        captured_messages.append(user_msg)
        return _make_llm_response([], [], [])

    llm = Mock()
    llm.call.side_effect = fake_call

    prompt = {
        "stage": "chunking",
        "system_prompt": "system",
        "user_prompt": ("{batch_content} limit={embedding_limit}"),
        "tools": [],
        "tool_choice": "required",
    }

    xlsx_chunker_module.collect_xlsx_breakpoints(
        parsed=parsed,
        llm=llm,
        prompt=prompt,
        batch_size=5,
        row_counts=(3, 2),
        embedding_limit=8192,
        context="test",
    )

    first_call = captured_messages[0]
    assert "<sheet_context>" in first_call
    assert "| 1 | val_1 |" in first_call
    assert "| 2 | val_2 |" in first_call
    assert "| 3 | val_3 |" in first_call

    second_call = captured_messages[1]
    assert "<prior_chunk_overlap>" in second_call
    assert "| 4 | val_4 |" in second_call
    assert "| 5 | val_5 |" in second_call


# ------------------------------------------------------------------
# _assemble_xlsx_chunks
# ------------------------------------------------------------------


def test_assemble_xlsx_chunks_includes_structural_markdown():
    """Each chunk has heading + header + separator."""
    parsed = _build_parsed_with_rows(6)
    page = _make_page(content="", page_number=3)

    chunks = xlsx_chunker_module.assemble_xlsx_chunks(
        parsed=parsed,
        breakpoints=[4],
        sheet_pt_indices=[],
        section_snapshots=[[1]],
        page=page,
    )

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.chunk_header.startswith("# Sheet: Test\n")
        assert "| Row | B |" in chunk.chunk_header
        assert "| --- | --- |" in chunk.chunk_header


def test_assemble_xlsx_chunks_includes_passthrough_rows():
    """Sheet passthrough rows in every chunk."""
    parsed = _build_parsed_with_rows(6)
    page = _make_page(content="", page_number=1)

    chunks = xlsx_chunker_module.assemble_xlsx_chunks(
        parsed=parsed,
        breakpoints=[4],
        sheet_pt_indices=[1],
        section_snapshots=[[]],
        page=page,
    )

    assert len(chunks) == 2
    for chunk in chunks:
        assert "| 1 | val_1 |" in chunk.sheet_passthrough_content


def test_assemble_xlsx_chunks_appends_visuals_to_last():
    """Visual blocks only on last chunk."""
    parsed = _build_parsed_with_rows(6)
    parsed["visual_blocks"] = ["> [Chart]: Revenue trend"]
    page = _make_page(content="", page_number=1)

    chunks = xlsx_chunker_module.assemble_xlsx_chunks(
        parsed=parsed,
        breakpoints=[4],
        sheet_pt_indices=[],
        section_snapshots=[[]],
        page=page,
    )

    assert len(chunks) == 2
    assert "> [Chart]: Revenue trend" not in chunks[0].raw_content
    assert "> [Chart]: Revenue trend" in chunks[1].raw_content


def test_row_can_be_breakpoint_and_passthrough():
    """Row in both breakpoints and sheet passthrough."""
    parsed = _build_parsed_with_rows(6)
    page = _make_page(content="", page_number=1)

    chunks = xlsx_chunker_module.assemble_xlsx_chunks(
        parsed=parsed,
        breakpoints=[4],
        sheet_pt_indices=[4],
        section_snapshots=[[]],
        page=page,
    )

    assert len(chunks) == 2
    assert chunks[0].chunk_context == "Rows 1-3"
    assert chunks[1].chunk_context == "Rows 4-6"
    assert "| 4 | val_4 |" in chunks[0].sheet_passthrough_content
    assert "| 4 | val_4 |" in chunks[1].sheet_passthrough_content


# ------------------------------------------------------------------
# Two-tier passthrough behavior
# ------------------------------------------------------------------


def test_sheet_passthrough_in_all_chunks():
    """Sheet passthrough rows appear in every chunk."""
    parsed = _build_parsed_with_rows(9)
    page = _make_page(content="", page_number=1)

    chunks = xlsx_chunker_module.assemble_xlsx_chunks(
        parsed=parsed,
        breakpoints=[4, 7],
        sheet_pt_indices=[1],
        section_snapshots=[[], []],
        page=page,
    )

    assert len(chunks) == 3
    for chunk in chunks:
        assert "| 1 | val_1 |" in chunk.sheet_passthrough_content


def test_section_passthrough_only_in_adjacent_chunks():
    """Section passthrough rows only in the intended chunk."""
    parsed = _build_parsed_with_rows(9)
    page = _make_page(content="", page_number=1)

    chunks = xlsx_chunker_module.assemble_xlsx_chunks(
        parsed=parsed,
        breakpoints=[4, 7],
        sheet_pt_indices=[],
        section_snapshots=[[2], [5]],
        page=page,
    )

    assert len(chunks) == 3
    assert "| 2 | val_2 |" in chunks[1].section_passthrough_content
    assert "| 2 | val_2 |" not in chunks[2].section_passthrough_content
    assert "| 5 | val_5 |" in chunks[2].section_passthrough_content
    assert "| 5 | val_5 |" not in chunks[0].section_passthrough_content


def test_section_passthrough_propagation():
    """Section rows carry from one batch to the next."""
    parsed = _build_parsed_with_rows(15)
    call_count = [0]

    def fake_call(messages, stage, tools, tool_choice, context):
        """Return section passthrough that propagates.

        Params: messages, stage, tools, tool_choice, context.
        Returns: dict.
        """
        assert messages is not None
        assert stage is not None
        assert tools is not None
        assert tool_choice is not None
        assert context is not None
        call_count[0] += 1
        if call_count[0] == 1:
            return _make_llm_response([5], [], [3])
        if call_count[0] == 2:
            return _make_llm_response([10], [], [3])
        return _make_llm_response([13], [], [])

    llm = Mock()
    llm.call.side_effect = fake_call

    prompt = {
        "stage": "chunking",
        "system_prompt": "system",
        "user_prompt": ("{batch_content} limit={embedding_limit}"),
        "tools": [],
        "tool_choice": "required",
    }

    breakpoints, sheet_pt, section_snaps = (
        xlsx_chunker_module.collect_xlsx_breakpoints(
            parsed=parsed,
            llm=llm,
            prompt=prompt,
            batch_size=5,
            row_counts=(2, 2),
            embedding_limit=8192,
            context="test",
        )
    )

    assert breakpoints == [5, 10, 13]
    assert sheet_pt == []
    assert section_snaps[0] == [3]
    assert section_snaps[1] == [3]
    assert section_snaps[2] == []

    captured = []

    def capture_call(messages, stage, tools, tool_choice, context):
        """Capture user messages for inspection.

        Params: messages, stage, tools, tool_choice, context.
        Returns: dict.
        """
        assert stage is not None
        assert tools is not None
        assert tool_choice is not None
        assert context is not None
        user_msg = messages[-1]["content"]
        captured.append(user_msg)
        return _make_llm_response([], [], [3])

    llm2 = Mock()
    llm2.call.side_effect = capture_call

    xlsx_chunker_module.collect_xlsx_breakpoints(
        parsed=parsed,
        llm=llm2,
        prompt=prompt,
        batch_size=5,
        row_counts=(2, 0),
        embedding_limit=8192,
        context="test",
    )

    assert "<section_context>" not in captured[0]
    assert "<section_context>" in captured[1]
    assert "| 3 | val_3 |" in captured[1]
    assert "<section_context>" in captured[2]
    assert "| 3 | val_3 |" in captured[2]


def test_section_passthrough_drops_when_not_reflagged():
    """Section rows drop when LLM stops flagging them."""
    parsed = _build_parsed_with_rows(15)
    call_count = [0]

    def fake_call(messages, stage, tools, tool_choice, context):
        """Section passthrough in batch 1 only.

        Params: messages, stage, tools, tool_choice, context.
        Returns: dict.
        """
        assert messages is not None
        assert stage is not None
        assert tools is not None
        assert tool_choice is not None
        assert context is not None
        call_count[0] += 1
        if call_count[0] == 1:
            return _make_llm_response([], [], [3])
        return _make_llm_response([], [], [])

    llm = Mock()
    llm.call.side_effect = fake_call

    prompt = {
        "stage": "chunking",
        "system_prompt": "system",
        "user_prompt": ("{batch_content} limit={embedding_limit}"),
        "tools": [],
        "tool_choice": "required",
    }

    captured: list[str] = []

    def capture_call(messages, stage, tools, tool_choice, context):
        """Capture then delegate.

        Params: messages, stage, tools, tool_choice, context.
        Returns: dict.
        """
        assert stage is not None
        assert tools is not None
        assert tool_choice is not None
        assert context is not None
        captured.append(messages[-1]["content"])
        call_count[0] += 1
        if call_count[0] == 1:
            return _make_llm_response([], [], [3])
        return _make_llm_response([], [], [])

    call_count[0] = 0
    llm.call.side_effect = capture_call

    xlsx_chunker_module.collect_xlsx_breakpoints(
        parsed=parsed,
        llm=llm,
        prompt=prompt,
        batch_size=5,
        row_counts=(2, 0),
        embedding_limit=8192,
        context="test",
    )

    assert "<section_context>" not in captured[0]
    assert "<section_context>" in captured[1]
    assert "| 3 | val_3 |" in captured[1]
    assert "<section_context>" not in captured[2]


# ------------------------------------------------------------------
# chunk_xlsx_page (end-to-end)
# ------------------------------------------------------------------


def test_chunk_xlsx_page_end_to_end(monkeypatch):
    """Full flow with monkeypatched LLM."""
    content = _make_xlsx_content(
        sheet_name="Revenue",
        rows=[
            (2, ["Title", "", ""]),
            (4, ["Header", "", ""]),
            (6, ["Data1", "10", "20"]),
            (8, ["Data2", "30", "40"]),
            (10, ["Section", "", ""]),
            (12, ["Data3", "50", "60"]),
            (14, ["Data4", "70", "80"]),
        ],
    )
    page = _make_page(content, page_number=2)

    monkeypatch.setattr(
        xlsx_chunker_module,
        "get_chunking_xlsx_batch_size",
        _stub_50,
    )
    monkeypatch.setattr(
        xlsx_chunker_module,
        "get_chunking_xlsx_header_rows",
        _stub_2,
    )
    monkeypatch.setattr(
        xlsx_chunker_module,
        "get_chunking_xlsx_overlap_rows",
        _stub_2,
    )

    llm = Mock()
    llm.call.return_value = _make_llm_response([10], [2], [4])

    chunks = xlsx_chunker_module.chunk_xlsx_page(page, llm, 8192)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "2.1"
    assert chunks[1].chunk_id == "2.2"
    assert chunks[0].parent_page_number == 2
    assert chunks[1].parent_page_number == 2
    assert chunks[0].layout_type == "standard"
    assert "| 2 | Title |" in chunks[0].sheet_passthrough_content
    assert "| 2 | Title |" in chunks[1].sheet_passthrough_content
    assert "Rows 2-8" in chunks[0].chunk_context
    assert "Rows 10-14" in chunks[1].chunk_context


def _stub_50():
    """Return 50. Returns: int."""
    return 50


def _stub_2():
    """Return 2. Returns: int."""
    return 2


def test_chunk_xlsx_page_no_breakpoints_single_chunk(
    monkeypatch,
):
    """Returns single chunk when LLM finds no breakpoints."""
    content = _make_xlsx_content(
        sheet_name="Simple",
        rows=[
            (1, ["A", "B", "C"]),
            (2, ["D", "E", "F"]),
        ],
    )
    page = _make_page(content, page_number=1)

    monkeypatch.setattr(
        xlsx_chunker_module,
        "get_chunking_xlsx_batch_size",
        _stub_50,
    )
    monkeypatch.setattr(
        xlsx_chunker_module,
        "get_chunking_xlsx_header_rows",
        _stub_2,
    )
    monkeypatch.setattr(
        xlsx_chunker_module,
        "get_chunking_xlsx_overlap_rows",
        _stub_2,
    )

    llm = Mock()
    llm.call.return_value = _make_llm_response([], [], [])

    chunks = xlsx_chunker_module.chunk_xlsx_page(page, llm, 8192)

    assert len(chunks) == 1
    assert chunks[0] is page


def test_chunk_xlsx_page_nests_ids_for_rechunked_page(
    monkeypatch,
):
    """Nested chunk IDs stay unique when re-chunked."""
    content = _make_xlsx_content(
        sheet_name="Revenue",
        rows=[
            (2, ["Title", "", ""]),
            (4, ["Header", "", ""]),
            (6, ["Data1", "10", "20"]),
            (8, ["Data2", "30", "40"]),
            (10, ["Section", "", ""]),
            (12, ["Data3", "50", "60"]),
        ],
    )
    page = _make_page(
        content,
        page_number=2,
        chunk_id="2.2",
        parent_page_number=2,
    )

    monkeypatch.setattr(
        xlsx_chunker_module,
        "get_chunking_xlsx_batch_size",
        _stub_50,
    )
    monkeypatch.setattr(
        xlsx_chunker_module,
        "get_chunking_xlsx_header_rows",
        _stub_2,
    )
    monkeypatch.setattr(
        xlsx_chunker_module,
        "get_chunking_xlsx_overlap_rows",
        _stub_2,
    )

    llm = Mock()
    llm.call.return_value = _make_llm_response([10], [2], [])

    chunks = xlsx_chunker_module.chunk_xlsx_page(page, llm, 8192)

    assert [chunk.chunk_id for chunk in chunks] == [
        "2.2.1",
        "2.2.2",
    ]
    assert all(chunk.parent_page_number == 2 for chunk in chunks)


def test_row_content_extracts_cells_after_row_number():
    """Extract cell content ignoring the row number column."""
    assert xlsx_chunker_module.row_content("| 5 | A | B |") == "A | B |"
    assert xlsx_chunker_module.row_content("no pipes") == "no pipes"


def test_build_chunk_prefix_returns_empty_for_empty_parsed():
    """Return empty string when heading, header, separator all empty."""
    parsed = {
        "heading": "",
        "table_header": "",
        "separator": "",
        "data_rows": [],
        "visual_blocks": [],
    }
    assert xlsx_chunker_module.build_chunk_prefix(parsed) == ""


def test_chunk_xlsx_page_no_data_rows_returns_page():
    """Return original page when content has no data rows."""
    content = "\n".join(
        [
            "# Sheet: Charts",
            "",
            "> [Chart]: Revenue trend",
        ]
    )
    page = _make_page(content, page_number=5)

    chunks = xlsx_chunker_module.chunk_xlsx_page(page, Mock(), 8192)

    assert len(chunks) == 1
    assert chunks[0] is page
