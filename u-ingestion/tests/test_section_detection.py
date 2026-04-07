"""Tests for the section_detection enrichment module."""

import json

import pytest

from ingestion.stages.enrichment import (
    section_detection as mod,
)
from ingestion.utils.file_types import (
    ExtractionResult,
    PageResult,
    SectionResult,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_page(
    raw_content="# Heading\nSome text",
    page_number=1,
    raw_token_count=100,
    **overrides,
):
    """Build a PageResult for testing.

    Params: raw_content, page_number, raw_token_count.
    Returns: PageResult.
    """
    page = PageResult(
        page_number=page_number,
        raw_content=raw_content,
        raw_token_count=raw_token_count,
    )
    for key, value in overrides.items():
        setattr(page, key, value)
    return page


def _make_result(
    pages=None,
    filetype="pdf",
    file_path="/tmp/data/source/2026/doc.pdf",
    metadata=None,
):
    """Build an ExtractionResult for testing.

    Params: pages, filetype, file_path, metadata.
    Returns: ExtractionResult.
    """
    if pages is None:
        pages = [_make_page()]
    result = ExtractionResult(
        file_path=file_path,
        filetype=filetype,
        pages=pages,
        total_pages=len(pages),
    )
    if metadata is not None:
        result.document_metadata = metadata
    return result


def _make_llm_response(sections_data, key="sections"):
    """Build a mock LLM tool-call response dict.

    Params: sections_data, key. Returns: dict.
    """
    args = json.dumps({key: sections_data, "rationale": "test"})
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


def _stub_prompt():
    """Build a minimal prompt dict for testing.

    Returns: dict.
    """
    return {
        "stage": "section_detection",
        "version": "1.0",
        "description": "test",
        "system_prompt": "You are a test agent.",
        "user_prompt": ("Analyze:\n\n{user_input}\n\n" "Rules:\n1. Do it."),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "detect_sections",
                    "description": ("Call this tool when ready."),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "required",
    }


def _make_fake_llm(responses=None):
    """Build a fake LLM client with canned responses.

    Params: responses -- list of response dicts.
    Returns: tuple of (fake_llm, call_log).
    """
    if responses is None:
        responses = [_make_llm_response([])]
    call_log = []
    call_idx = [0]

    def fake_call(
        messages,
        stage="",
        tools=None,
        tool_choice=None,
        context="",
    ):
        """Record call args and return canned response."""
        call_log.append(
            {
                "messages": messages,
                "stage": stage,
                "tools": tools,
                "tool_choice": tool_choice,
                "context": context,
            }
        )
        idx = min(call_idx[0], len(responses) - 1)
        call_idx[0] += 1
        return responses[idx]

    fake_llm = type(
        "FakeLLM",
        (),
        {"call": staticmethod(fake_call), "__doc__": "F."},
    )()
    return fake_llm, call_log


def _patch_core(monkeypatch, budget=50000, threshold=500):
    """Patch load_prompt and config for all tests.

    Params: monkeypatch, budget, threshold.
    Returns: None.
    """
    monkeypatch.setattr(
        mod,
        "load_prompt",
        lambda name, prompts_dir=None: _stub_prompt(),
    )
    monkeypatch.setattr(
        mod,
        "get_section_detection_batch_budget",
        lambda: budget,
    )
    monkeypatch.setattr(
        mod,
        "get_subsection_token_threshold",
        lambda: threshold,
    )
    monkeypatch.setattr(
        mod,
        "count_message_tokens",
        lambda messages: len(messages[1]["content"]),
    )


# ------------------------------------------------------------------
# XLSX / sheet_based tests
# ------------------------------------------------------------------


def test_detect_sections_xlsx_sheet_based(monkeypatch):
    """XLSX produces one section per sheet."""
    _patch_core(monkeypatch)
    pages = [
        _make_page(
            "# Sheet: KM1\ndata row 1",
            page_number=1,
            raw_token_count=50,
        ),
        _make_page(
            "# Sheet: CR6\ndata row 2",
            page_number=2,
            raw_token_count=50,
        ),
        _make_page(
            "# Sheet: LR2\ndata row 3",
            page_number=3,
            raw_token_count=50,
        ),
    ]
    result = _make_result(
        pages=pages,
        filetype="xlsx",
        metadata={"structure_type": "sheet_based"},
    )
    fake_llm, _ = _make_fake_llm()

    enriched = mod.detect_sections(result, fake_llm)

    assert len(enriched.sections) == 3
    assert enriched.sections[0]["title"] == "KM1"
    assert enriched.sections[1]["title"] == "CR6"
    assert enriched.sections[2]["title"] == "LR2"
    assert enriched.sections[0]["section_id"] == "1"
    assert enriched.sections[1]["section_id"] == "2"
    assert enriched.sections[0]["level"] == "section"


def test_detect_sections_xlsx_with_chunks(monkeypatch):
    """Chunked sheet has numerically sorted chunk_ids."""
    _patch_core(monkeypatch)
    pages = [
        _make_page(
            "# Sheet: KM1\nchunk A",
            page_number=1,
            raw_token_count=50,
            chunk_id="1.1",
        ),
        _make_page(
            "# Sheet: KM1\nchunk C",
            page_number=1,
            raw_token_count=50,
            chunk_id="1.10",
        ),
        _make_page(
            "# Sheet: KM1\nchunk B",
            page_number=1,
            raw_token_count=50,
            chunk_id="1.2",
        ),
    ]
    result = _make_result(
        pages=pages,
        filetype="xlsx",
        metadata={"structure_type": "sheet_based"},
    )
    fake_llm, _ = _make_fake_llm()

    enriched = mod.detect_sections(result, fake_llm)

    primaries = [s for s in enriched.sections if s["level"] == "section"]
    assert len(primaries) == 1
    section = primaries[0]
    assert section["chunk_ids"] == ["1.1", "1.2", "1.10"]
    assert section["title"] == "KM1"


def test_detect_sections_xlsx_subsections_from_chunks(
    monkeypatch,
):
    """Chunked sheets produce subsections even below threshold."""
    _patch_core(monkeypatch, threshold=500)
    pages = [
        _make_page(
            "# Sheet: KM1\nchunk A",
            page_number=1,
            raw_token_count=40,
            chunk_id="1.1",
            chunk_context="Rows 1-9",
        ),
        _make_page(
            "# Sheet: KM1\nchunk B",
            page_number=1,
            raw_token_count=40,
            chunk_id="1.2",
            chunk_context="Rows 10-19",
        ),
    ]
    result = _make_result(
        pages=pages,
        filetype="xlsx",
        metadata={"structure_type": "sheet_based"},
    )
    fake_llm, _ = _make_fake_llm()

    enriched = mod.detect_sections(result, fake_llm)

    primaries = [s for s in enriched.sections if s["level"] == "section"]
    subs = [s for s in enriched.sections if s["level"] == "subsection"]
    assert len(primaries) == 1
    assert len(subs) == 2
    assert subs[0]["section_id"] == "1.1"
    assert subs[0]["parent_section_id"] == "1"
    assert subs[0]["title"] == "KM1 \u2014 Rows 1-9"
    assert subs[1]["section_id"] == "1.2"
    assert subs[1]["title"] == "KM1 \u2014 Rows 10-19"


# ------------------------------------------------------------------
# Semantic tests
# ------------------------------------------------------------------


def test_detect_sections_semantic_one_per_page(
    monkeypatch,
):
    """Semantic structure produces one section per page."""
    _patch_core(monkeypatch)
    pages = [
        _make_page("Page A content", page_number=1),
        _make_page("Page B content", page_number=2),
        _make_page("Page C content", page_number=3),
    ]
    result = _make_result(
        pages=pages,
        metadata={"structure_type": "semantic"},
    )
    fake_llm, _ = _make_fake_llm()

    enriched = mod.detect_sections(result, fake_llm)

    assert len(enriched.sections) == 3
    for idx, sec in enumerate(enriched.sections, start=1):
        assert sec["section_id"] == str(idx)
        assert sec["level"] == "section"
        assert sec["page_start"] == idx
        assert sec["page_end"] == idx


# ------------------------------------------------------------------
# LLM-based detection tests
# ------------------------------------------------------------------


def test_detect_sections_pdf_with_llm(monkeypatch):
    """PDF sections detected via LLM batches."""
    _patch_core(monkeypatch)
    pages = [
        _make_page("# Intro\ntext", page_number=1),
        _make_page("# Methods\ntext", page_number=2),
        _make_page("# Results\ntext", page_number=3),
    ]
    result = _make_result(
        pages=pages,
        metadata={
            "structure_type": "chapters",
            "toc_entries": [],
        },
    )
    llm_resp = _make_llm_response(
        [
            {"title": "Introduction", "start_page": 1},
            {"title": "Methods", "start_page": 2},
            {"title": "Results", "start_page": 3},
        ]
    )
    fake_llm, call_log = _make_fake_llm([llm_resp])

    enriched = mod.detect_sections(result, fake_llm)

    assert len(enriched.sections) == 3
    assert enriched.sections[0]["title"] == "Introduction"
    assert enriched.sections[0]["page_start"] == 1
    assert enriched.sections[0]["page_end"] == 1
    assert enriched.sections[1]["page_start"] == 2
    assert enriched.sections[1]["page_end"] == 2
    assert enriched.sections[2]["page_start"] == 3
    assert enriched.sections[2]["page_end"] == 3
    assert len(call_log) == 1


def test_detect_sections_uses_toc_as_hint(monkeypatch):
    """TOC entries passed to LLM in the prompt."""
    _patch_core(monkeypatch)
    toc = [
        {"name": "Overview", "page_number": 1},
        {"name": "Details", "page_number": 3},
    ]
    pages = [
        _make_page("# Overview\ntext", page_number=1),
        _make_page("More content", page_number=2),
        _make_page("# Details\ntext", page_number=3),
    ]
    result = _make_result(
        pages=pages,
        metadata={
            "structure_type": "sections",
            "toc_entries": toc,
        },
    )
    llm_resp = _make_llm_response(
        [
            {"title": "Overview", "start_page": 1},
            {"title": "Details", "start_page": 3},
        ]
    )
    fake_llm, call_log = _make_fake_llm([llm_resp])

    mod.detect_sections(result, fake_llm)

    user_msg = call_log[0]["messages"][1]["content"]
    assert "<toc_hint>" in user_msg
    assert "Overview" in user_msg
    assert "Details" in user_msg


def test_detect_sections_accumulates_across_batches(
    monkeypatch,
):
    """Multiple batches produce merged sections."""
    _patch_core(monkeypatch, budget=450)
    pages = [
        _make_page(
            "# Part A",
            page_number=1,
            raw_token_count=200,
        ),
        _make_page(
            "# Part B",
            page_number=2,
            raw_token_count=200,
        ),
        _make_page(
            "# Part C",
            page_number=3,
            raw_token_count=200,
        ),
        _make_page(
            "# Part D",
            page_number=4,
            raw_token_count=200,
        ),
    ]
    result = _make_result(
        pages=pages,
        metadata={
            "structure_type": "chapters",
            "toc_entries": [],
        },
    )
    resp1 = _make_llm_response([{"title": "Part A", "start_page": 1}])
    resp2 = _make_llm_response([{"title": "Part C", "start_page": 3}])
    fake_llm, call_log = _make_fake_llm([resp1, resp2])

    enriched = mod.detect_sections(result, fake_llm)

    assert len(call_log) == 2
    assert len(enriched.sections) == 2
    assert enriched.sections[0]["title"] == "Part A"
    assert enriched.sections[1]["title"] == "Part C"


def test_detect_sections_counts_formatted_request_budget(
    monkeypatch,
):
    """Rendered section-detection requests can force an extra batch."""
    _patch_core(monkeypatch, budget=150)
    monkeypatch.setattr(
        mod,
        "count_message_tokens",
        lambda messages: (
            200 if "[Page 2]" in messages[1]["content"] else 100
        ),
    )
    pages = [
        _make_page(
            "# Part A",
            page_number=1,
            raw_token_count=10,
        ),
        _make_page(
            "# Part B",
            page_number=2,
            raw_token_count=10,
        ),
    ]
    result = _make_result(
        pages=pages,
        metadata={
            "structure_type": "chapters",
            "toc_entries": [],
        },
    )
    resp1 = _make_llm_response([{"title": "Part A", "start_page": 1}])
    resp2 = _make_llm_response([{"title": "Part B", "start_page": 2}])
    fake_llm, call_log = _make_fake_llm([resp1, resp2])

    enriched = mod.detect_sections(result, fake_llm)

    assert len(call_log) == 2
    assert len(enriched.sections) == 2


def test_detect_sections_computes_end_pages(monkeypatch):
    """end_page inferred correctly from next start."""
    _patch_core(monkeypatch)
    pages = [
        _make_page("text", page_number=1),
        _make_page("text", page_number=2),
        _make_page("text", page_number=3),
        _make_page("text", page_number=4),
        _make_page("text", page_number=5),
    ]
    result = _make_result(
        pages=pages,
        metadata={
            "structure_type": "sections",
            "toc_entries": [],
        },
    )
    llm_resp = _make_llm_response(
        [
            {"title": "Sec A", "start_page": 1},
            {"title": "Sec B", "start_page": 3},
        ]
    )
    fake_llm, _ = _make_fake_llm([llm_resp])

    enriched = mod.detect_sections(result, fake_llm)

    assert enriched.sections[0]["page_start"] == 1
    assert enriched.sections[0]["page_end"] == 2
    assert enriched.sections[1]["page_start"] == 3
    assert enriched.sections[1]["page_end"] == 5


def test_detect_sections_assigns_chunk_ids(monkeypatch):
    """Content units mapped to sections by page range."""
    _patch_core(monkeypatch)
    pages = [
        _make_page("text", page_number=1),
        _make_page("text", page_number=2),
        _make_page("text", page_number=3),
    ]
    result = _make_result(
        pages=pages,
        metadata={
            "structure_type": "chapters",
            "toc_entries": [],
        },
    )
    llm_resp = _make_llm_response(
        [
            {"title": "A", "start_page": 1},
            {"title": "B", "start_page": 3},
        ]
    )
    fake_llm, _ = _make_fake_llm([llm_resp])

    enriched = mod.detect_sections(result, fake_llm)

    assert enriched.sections[0]["chunk_ids"] == ["1", "2"]
    assert enriched.sections[1]["chunk_ids"] == ["3"]


# ------------------------------------------------------------------
# Subsection detection
# ------------------------------------------------------------------


def test_subsection_detection_skips_small_sections(
    monkeypatch,
):
    """Sections under threshold not subdivided."""
    _patch_core(monkeypatch, threshold=5000)
    pages = [
        _make_page("text", page_number=1, raw_token_count=100),
        _make_page("text", page_number=2, raw_token_count=100),
    ]
    result = _make_result(
        pages=pages,
        metadata={
            "structure_type": "chapters",
            "toc_entries": [],
        },
    )
    llm_resp = _make_llm_response([{"title": "Small", "start_page": 1}])
    fake_llm, call_log = _make_fake_llm([llm_resp])

    enriched = mod.detect_sections(result, fake_llm)

    subs = [s for s in enriched.sections if s["level"] == "subsection"]
    assert len(subs) == 0
    assert len(call_log) == 1


def test_subsection_detection_large_pdf_section(
    monkeypatch,
):
    """Large PDF section gets LLM subsections."""
    _patch_core(monkeypatch, threshold=150)
    pages = [
        _make_page(
            "# Overview\ntext",
            page_number=1,
            raw_token_count=100,
        ),
        _make_page(
            "# Details\ntext",
            page_number=2,
            raw_token_count=100,
        ),
    ]
    result = _make_result(
        pages=pages,
        metadata={
            "structure_type": "chapters",
            "toc_entries": [],
        },
    )
    section_resp = _make_llm_response(
        [{"title": "Big Section", "start_page": 1}]
    )
    subsection_resp = _make_llm_response(
        [
            {"title": "Sub A", "start_page": 1},
            {"title": "Sub B", "start_page": 2},
        ],
        key="subsections",
    )
    fake_llm, call_log = _make_fake_llm([section_resp, subsection_resp])

    enriched = mod.detect_sections(result, fake_llm)

    primaries = [s for s in enriched.sections if s["level"] == "section"]
    subs = [s for s in enriched.sections if s["level"] == "subsection"]
    assert len(primaries) == 1
    assert len(subs) == 2
    assert subs[0]["section_id"] == "1.1"
    assert subs[0]["parent_section_id"] == "1"
    assert subs[0]["title"] == "Sub A"
    assert subs[1]["section_id"] == "1.2"
    assert len(call_log) == 2


def test_detect_sections_assigns_page_section_ids(monkeypatch):
    """Pages and chunks are stamped with their owning section ids."""
    _patch_core(monkeypatch, threshold=50)
    pages = [
        _make_page("Intro", page_number=1),
        _make_page(
            "# Sheet: KM1\nchunk A",
            page_number=2,
            raw_token_count=40,
            chunk_id="2.1",
            chunk_context="Rows 1-9",
        ),
        _make_page(
            "# Sheet: KM1\nchunk B",
            page_number=2,
            raw_token_count=40,
            chunk_id="2.2",
            chunk_context="Rows 10-19",
        ),
    ]
    result = _make_result(
        pages=pages,
        filetype="xlsx",
        metadata={"structure_type": "sheet_based"},
    )
    fake_llm, _ = _make_fake_llm()

    enriched = mod.detect_sections(result, fake_llm)

    assert enriched.pages[0].section_id == "1"
    assert enriched.pages[1].section_id == "2.1"
    assert enriched.pages[2].section_id == "2.2"


# ------------------------------------------------------------------
# parse_section_response
# ------------------------------------------------------------------


def test_parse_section_response_valid():
    """Valid response parsed correctly."""
    response = _make_llm_response(
        [
            {"title": "Intro", "start_page": 1},
            {"title": "Body", "start_page": 5},
        ]
    )

    result = mod.parse_section_response(response)

    assert len(result) == 2
    assert result[0]["title"] == "Intro"
    assert result[1]["start_page"] == 5


def test_parse_section_response_malformed():
    """Bad response raises ValueError."""
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": "not json{",
                            }
                        }
                    ]
                }
            }
        ],
    }

    with pytest.raises(ValueError, match="parse tool"):
        mod.parse_section_response(response)


def test_parse_section_response_no_choices():
    """Raises ValueError when response has no choices."""
    with pytest.raises(ValueError, match="no choices"):
        mod.parse_section_response({"choices": []})


def test_parse_section_response_no_tool_calls():
    """Raises ValueError when message has no tool calls."""
    response = {
        "choices": [{"message": {"tool_calls": None}}],
    }
    with pytest.raises(ValueError, match="no tool calls"):
        mod.parse_section_response(response)


def test_parse_section_response_missing_field():
    """Raises ValueError for entries missing fields."""
    response = _make_llm_response([{"title": "No start_page"}])

    with pytest.raises(ValueError, match="start_page"):
        mod.parse_section_response(response)


def test_parse_section_response_no_sections_key():
    """Raises ValueError when neither key exists."""
    args = json.dumps({"rationale": "none"})
    response = {
        "choices": [
            {"message": {"tool_calls": [{"function": {"arguments": args}}]}}
        ],
    }

    with pytest.raises(ValueError, match="missing"):
        mod.parse_section_response(response)


def test_parse_section_response_not_a_list():
    """Raises ValueError when sections is not a list."""
    args = json.dumps({"sections": "not a list", "rationale": "x"})
    response = {
        "choices": [
            {"message": {"tool_calls": [{"function": {"arguments": args}}]}}
        ],
    }

    with pytest.raises(ValueError, match="must be a list"):
        mod.parse_section_response(response)


# ------------------------------------------------------------------
# Metadata preservation
# ------------------------------------------------------------------


def test_detect_sections_preserves_metadata(monkeypatch):
    """document_metadata unchanged after detection."""
    _patch_core(monkeypatch)
    original_meta = {
        "structure_type": "semantic",
        "title": "Test Doc",
        "has_toc": False,
        "toc_entries": [],
    }
    result = _make_result(
        metadata=dict(original_meta),
    )
    fake_llm, _ = _make_fake_llm()

    enriched = mod.detect_sections(result, fake_llm)

    assert enriched.document_metadata["title"] == "Test Doc"
    assert enriched.document_metadata["structure_type"] == "semantic"


# ------------------------------------------------------------------
# Helper function unit tests
# ------------------------------------------------------------------


def test_content_unit_id_with_chunk():
    """Returns chunk_id when set."""
    page = _make_page(chunk_id="3.2")

    assert mod.content_unit_id(page) == "3.2"


def test_content_unit_id_without_chunk():
    """Returns str(page_number) when no chunk_id."""
    page = _make_page(page_number=7, chunk_id="")

    assert mod.content_unit_id(page) == "7"


def test_parse_sheet_name_found():
    """Parses sheet name from header."""
    raw = "# Sheet: Balance Sheet\nrow1\nrow2"

    assert mod.parse_sheet_name(raw) == "Balance Sheet"


def test_parse_sheet_name_missing():
    """Returns empty string when no header."""
    raw = "No sheet header here"

    assert mod.parse_sheet_name(raw) == ""


def test_deduplicate_pages_prefers_unchunked():
    """Deduplication prefers unchunked pages."""
    unchunked = _make_page("Full page", page_number=1, chunk_id="")
    chunked = _make_page("Chunk only", page_number=1, chunk_id="1.1")

    result = mod.deduplicate_pages([chunked, unchunked])

    assert len(result) == 1
    assert result[0].raw_content == "Full page"


def test_format_section_batch_includes_context():
    """Format includes TOC, previous sections, content."""
    prompt = _stub_prompt()
    toc = [{"name": "Intro", "page_number": 1}]
    prev = [{"title": "Prev Sec", "start_page": 1}]
    pages = [_make_page("content", page_number=5)]

    output = mod.format_section_batch(toc, prev, "chapters", pages, prompt)

    assert "<toc_hint>" in output
    assert "Intro" in output
    assert "<previous_sections>" in output
    assert "Prev Sec" in output
    assert "<structure_type>" in output
    assert "<content>" in output
    assert "[Page 5]" in output


def test_format_section_batch_no_toc_no_prev():
    """Format omits TOC and previous when empty."""
    prompt = _stub_prompt()

    output = mod.format_section_batch(
        [], [], "sections", [_make_page()], prompt
    )

    assert "<toc_hint>" not in output
    assert "<previous_sections>" not in output
    assert "<structure_type>" in output
    assert "<content>" in output


def test_format_subsection_batch_includes_section_info():
    """Subsection format includes section info."""
    prompt = _stub_prompt()
    pages = [
        _make_page("sub content", page_number=3),
    ]

    output = mod.format_subsection_batch("Big Section", 3, 5, pages, prompt)

    assert "<section_info>" in output
    assert "Big Section" in output
    assert "3-5" in output
    assert "<content>" in output


def test_xlsx_chunks_as_subsections():
    """XLSX chunks become subsections in numeric chunk order."""
    section = SectionResult(
        section_id="2",
        page_start=3,
        page_end=3,
        token_count=1000,
    )
    pages = [
        _make_page(
            "chunk A",
            page_number=3,
            chunk_id="3.1",
            chunk_context="Rows 1-10",
            raw_token_count=500,
        ),
        _make_page(
            "chunk B",
            page_number=3,
            chunk_id="3.10",
            chunk_context="Rows 21-30",
            raw_token_count=500,
        ),
        _make_page(
            "chunk C",
            page_number=3,
            chunk_id="3.2",
            chunk_context="Rows 11-20",
            raw_token_count=500,
        ),
        _make_page(
            "other sheet",
            page_number=4,
            chunk_id="4.1",
        ),
    ]

    result = mod.xlsx_chunks_as_subsections(section, pages)

    assert len(result) == 3
    assert result[0].section_id == "2.1"
    assert result[0].parent_section_id == "2"
    assert result[0].title == "Rows 1-10"
    assert result[0].chunk_ids == ["3.1"]
    assert result[1].section_id == "2.2"
    assert result[1].title == "Rows 11-20"
    assert result[1].chunk_ids == ["3.2"]
    assert result[2].section_id == "2.3"
    assert result[2].title == "Rows 21-30"
    assert result[2].chunk_ids == ["3.10"]


def test_chunk_id_sort_key_orders_multi_digit_chunks():
    """Sort key keeps 3.10 after 3.2."""
    chunk_ids = ["3.10", "3.2", "3.1"]

    ordered = sorted(chunk_ids, key=mod.chunk_id_sort_key)

    assert ordered == ["3.1", "3.2", "3.10"]


def test_chunk_id_sort_key_handles_non_numeric_parts():
    """Sort key preserves non-numeric chunk suffixes."""
    chunk_ids = ["3.b", "3.a"]

    ordered = sorted(chunk_ids, key=mod.chunk_id_sort_key)

    assert ordered == ["3.a", "3.b"]


def test_build_section_results_empty():
    """Empty raw sections produces empty result."""
    result = _make_result()

    sections = mod.build_section_results([], result, 5)

    assert not sections


def test_detect_sections_empty_pages(monkeypatch):
    """Empty pages list produces no sections."""
    _patch_core(monkeypatch)
    result = _make_result(
        pages=[],
        metadata={
            "structure_type": "chapters",
            "toc_entries": [],
        },
    )
    result.total_pages = 0
    fake_llm, _ = _make_fake_llm()

    enriched = mod.detect_sections(result, fake_llm)

    assert enriched.sections == []


def test_detect_sections_llm_returns_empty(monkeypatch):
    """LLM returning no sections produces empty list."""
    _patch_core(monkeypatch)
    pages = [
        _make_page("text", page_number=1),
    ]
    result = _make_result(
        pages=pages,
        metadata={
            "structure_type": "chapters",
            "toc_entries": [],
        },
    )
    llm_resp = _make_llm_response([])
    fake_llm, _ = _make_fake_llm([llm_resp])

    enriched = mod.detect_sections(result, fake_llm)

    assert enriched.sections == []


def test_xlsx_sheet_name_fallback(monkeypatch):
    """XLSX uses fallback name when header not found."""
    _patch_core(monkeypatch)
    pages = [
        _make_page(
            "No sheet header",
            page_number=1,
            raw_token_count=50,
        ),
    ]
    result = _make_result(
        pages=pages,
        filetype="xlsx",
        metadata={"structure_type": "sheet_based"},
    )
    fake_llm, _ = _make_fake_llm()

    enriched = mod.detect_sections(result, fake_llm)

    assert enriched.sections[0]["title"] == "Sheet 1"


def test_semantic_with_chunks(monkeypatch):
    """Semantic mode collects all chunks per page."""
    _patch_core(monkeypatch)
    pages = [
        _make_page(
            "Full page",
            page_number=1,
            chunk_id="",
            raw_token_count=50,
        ),
        _make_page(
            "Chunk A",
            page_number=1,
            chunk_id="1.1",
            raw_token_count=50,
        ),
    ]
    result = _make_result(
        pages=pages,
        metadata={"structure_type": "semantic"},
    )
    fake_llm, _ = _make_fake_llm()

    enriched = mod.detect_sections(result, fake_llm)

    assert len(enriched.sections) == 1
    assert "1" in enriched.sections[0]["chunk_ids"]
    assert "1.1" in enriched.sections[0]["chunk_ids"]


def test_section_sequence_is_one_indexed(monkeypatch):
    """Section sequence numbers start at 1."""
    _patch_core(monkeypatch)
    pages = [
        _make_page("A", page_number=1),
        _make_page("B", page_number=2),
    ]
    result = _make_result(
        pages=pages,
        metadata={
            "structure_type": "chapters",
            "toc_entries": [],
        },
    )
    llm_resp = _make_llm_response(
        [
            {"title": "First", "start_page": 1},
            {"title": "Second", "start_page": 2},
        ]
    )
    fake_llm, _ = _make_fake_llm([llm_resp])

    enriched = mod.detect_sections(result, fake_llm)

    assert enriched.sections[0]["sequence"] == 1
    assert enriched.sections[1]["sequence"] == 2


def test_llm_subsections_no_pages_returns_empty(
    monkeypatch,
):
    """Subsection detection returns empty for no pages."""
    _patch_core(monkeypatch)
    section = SectionResult(
        section_id="1",
        page_start=99,
        page_end=99,
        token_count=20000,
    )
    result = _make_result(
        pages=[_make_page("text", page_number=1)],
    )
    fake_llm, _ = _make_fake_llm()

    subs = mod.detect_llm_subsections(section, result, fake_llm)

    assert not subs


def test_llm_subsections_empty_response(monkeypatch):
    """LLM returning no subsections yields empty list."""
    _patch_core(monkeypatch)
    section = SectionResult(
        section_id="1",
        page_start=1,
        page_end=2,
        token_count=20000,
    )
    pages = [
        _make_page("text", page_number=1),
        _make_page("text", page_number=2),
    ]
    result = _make_result(pages=pages)
    empty_resp = _make_llm_response([], key="subsections")
    fake_llm, _ = _make_fake_llm([empty_resp])

    subs = mod.detect_llm_subsections(section, result, fake_llm)

    assert not subs
