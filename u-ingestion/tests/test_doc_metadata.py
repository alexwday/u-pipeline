"""Tests for the doc_metadata enrichment module."""

import json

import pytest

from ingestion.stages.enrichment import doc_metadata as mod
from ingestion.utils.file_types import ExtractionResult, PageResult

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
):
    """Build an ExtractionResult for testing.

    Params: pages, filetype, file_path. Returns: ExtractionResult.
    """
    if pages is None:
        pages = [_make_page()]
    return ExtractionResult(
        file_path=file_path,
        filetype=filetype,
        pages=pages,
        total_pages=len(pages),
    )


def _make_llm_response(metadata_dict):
    """Build a mock LLM tool-call response dict.

    Params: metadata_dict. Returns: dict.
    """
    args = json.dumps(metadata_dict)
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


def _default_metadata(**overrides):
    """Build a default metadata dict with overrides.

    Params: overrides. Returns: dict.
    """
    base = {
        "title": "Test Document",
        "authors": "Test Corp",
        "publication_date": "2026-01-15",
        "language": "en",
        "structure_type": "sections",
        "has_toc": False,
        "toc_entries": [],
        "rationale": "Numbered section headings found.",
    }
    base.update(overrides)
    return base


def _stub_prompt():
    """Build a minimal prompt dict for testing. Returns: dict."""
    return {
        "stage": "doc_metadata",
        "version": "1.0",
        "description": "test",
        "system_prompt": "You are a test agent.",
        "user_prompt": ("Analyze:\n\n{user_input}\n\nRules:\n1. Do it."),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "extract_document_metadata",
                    "description": "Call this tool when ready.",
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


def _patch_dependencies(monkeypatch, metadata_dict=None):
    """Patch load_prompt and config for all tests.

    Params: monkeypatch, metadata_dict. Returns: dict of mocks.
    """
    if metadata_dict is None:
        metadata_dict = _default_metadata()

    prompt = _stub_prompt()

    monkeypatch.setattr(
        mod,
        "load_prompt",
        lambda name, prompts_dir=None: prompt,
    )
    monkeypatch.setattr(
        mod,
        "get_doc_metadata_context_budget",
        lambda: 50000,
    )

    call_log = []

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
        return _make_llm_response(metadata_dict)

    fake_llm = type(
        "FakeLLM",
        (),
        {"call": staticmethod(fake_call), "__doc__": "Fake."},
    )()

    return {"llm": fake_llm, "call_log": call_log}


# ------------------------------------------------------------------
# extract_sheet_names
# ------------------------------------------------------------------


def test_extract_sheet_names():
    """Parses sheet names from XLSX raw_content headers."""
    pages = [
        _make_page("# Sheet: KM1\ndata", page_number=1),
        _make_page("# Sheet: CR6_A\ndata", page_number=2),
        _make_page("# Sheet: Overview\ndata", page_number=3),
    ]
    result = mod.extract_sheet_names(pages)

    assert result == ["KM1", "CR6_A", "Overview"]


def test_extract_sheet_names_no_match():
    """Returns empty list when no sheet headers found."""
    pages = [_make_page("Just text", page_number=1)]
    result = mod.extract_sheet_names(pages)

    assert not result


# ------------------------------------------------------------------
# extract_page_headings
# ------------------------------------------------------------------


def test_extract_page_headings():
    """Finds first markdown heading from each page."""
    pages = [
        _make_page("# Cover Page\ntext", page_number=1),
        _make_page("## Table of Contents\ntext", page_number=2),
        _make_page("No heading here", page_number=3),
    ]
    result = mod.extract_page_headings(pages)

    assert result == ["Cover Page", "Table of Contents", ""]


# ------------------------------------------------------------------
# build_content_within_budget
# ------------------------------------------------------------------


def test_build_content_within_budget():
    """Stops adding pages when budget is exceeded."""
    pages = [
        _make_page("Page one", page_number=1, raw_token_count=100),
        _make_page("Page two", page_number=2, raw_token_count=100),
        _make_page("Page three", page_number=3, raw_token_count=100),
    ]
    result = mod.build_content_within_budget(pages, 150)

    assert "[Page 1]" in result
    assert "Page one" in result
    assert "[Page 2]" not in result


def test_build_content_includes_first_even_if_over_budget():
    """First page is always included even if over budget."""
    pages = [
        _make_page("Big page", page_number=1, raw_token_count=200),
    ]
    result = mod.build_content_within_budget(pages, 50)

    assert "Big page" in result


def test_build_content_includes_chunks():
    """Chunks are included after unchunked pages."""
    pages = [
        _make_page(
            "Chunk data",
            page_number=1,
            raw_token_count=50,
            chunk_id="1.1",
        ),
    ]
    result = mod.build_content_within_budget(pages, 5000)

    assert "[Page 1 Chunk 1.1]" in result
    assert "Chunk data" in result


def test_build_content_chunks_stop_at_budget():
    """Chunks stop being added when budget is exceeded."""
    pages = [
        _make_page(
            "Chunk A",
            page_number=1,
            raw_token_count=100,
            chunk_id="1.1",
        ),
        _make_page(
            "Chunk B",
            page_number=1,
            raw_token_count=100,
            chunk_id="1.2",
        ),
    ]
    result = mod.build_content_within_budget(pages, 150)

    assert "Chunk A" in result
    assert "Chunk B" not in result


def test_build_content_within_budget_counts_formatted_prompt(
    monkeypatch,
):
    """Prompt-aware budgeting stops when the rendered request grows."""
    pages = [
        _make_page("Page one", page_number=1, raw_token_count=10),
        _make_page("Page two", page_number=2, raw_token_count=10),
    ]
    prompt = _stub_prompt()
    monkeypatch.setattr(
        mod,
        "count_message_tokens",
        lambda messages, tools=None: (
            200 if "Page two" in messages[1]["content"] else 100
        ),
    )

    result = mod.build_content_within_budget(
        pages,
        150,
        prompt=prompt,
        file_metadata="filename: test.pdf",
        page_names="Page 1\nPage 2",
        layout_summary="text: 2",
    )

    assert "Page one" in result
    assert "Page two" not in result


# ------------------------------------------------------------------
# parse_metadata_response
# ------------------------------------------------------------------


def test_parse_metadata_response_valid():
    """Valid tool call response is parsed correctly."""
    metadata = _default_metadata()
    response = _make_llm_response(metadata)

    result = mod.parse_metadata_response(response)

    assert result["title"] == "Test Document"
    assert result["structure_type"] == "sections"
    assert result["has_toc"] is False


def test_parse_metadata_response_no_choices():
    """Raises ValueError when response has no choices."""
    with pytest.raises(ValueError, match="no choices"):
        mod.parse_metadata_response({"choices": []})


def test_parse_metadata_response_no_tool_calls():
    """Raises ValueError when message has no tool calls."""
    response = {"choices": [{"message": {"tool_calls": None}}]}
    with pytest.raises(ValueError, match="no tool calls"):
        mod.parse_metadata_response(response)


def test_parse_metadata_response_malformed_json():
    """Raises ValueError for invalid JSON arguments."""
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
    with pytest.raises(ValueError, match="parse tool arguments"):
        mod.parse_metadata_response(response)


def test_parse_metadata_response_missing_fields():
    """Raises ValueError when required fields are missing."""
    response = _make_llm_response({"title": "Only title"})

    with pytest.raises(ValueError, match="Missing required"):
        mod.parse_metadata_response(response)


def test_parse_metadata_response_invalid_structure_type():
    """Raises ValueError for an unknown structure_type."""
    metadata = _default_metadata(structure_type="invalid")
    response = _make_llm_response(metadata)

    with pytest.raises(ValueError, match="Invalid structure_type"):
        mod.parse_metadata_response(response)


# ------------------------------------------------------------------
# enrich_doc_metadata — integration-style (mocked LLM)
# ------------------------------------------------------------------


def test_enrich_doc_metadata_pdf_classifies_structure(
    monkeypatch,
):
    """PDF file gets LLM-determined structure_type."""
    mocks = _patch_dependencies(
        monkeypatch,
        _default_metadata(structure_type="chapters"),
    )
    result = _make_result(filetype="pdf")

    enriched = mod.enrich_doc_metadata(result, mocks["llm"])

    assert enriched.document_metadata["structure_type"] == "chapters"


def test_enrich_doc_metadata_xlsx_sets_sheet_based(
    monkeypatch,
):
    """XLSX file gets structure_type overridden."""
    mocks = _patch_dependencies(
        monkeypatch,
        _default_metadata(structure_type="chapters"),
    )
    pages = [
        _make_page("# Sheet: KM1\ndata", page_number=1),
    ]
    result = _make_result(
        pages=pages,
        filetype="xlsx",
        file_path="/tmp/data/source/report.xlsx",
    )

    enriched = mod.enrich_doc_metadata(result, mocks["llm"])

    assert enriched.document_metadata["structure_type"] == "sheet_based"


def test_enrich_doc_metadata_extracts_title_and_date(
    monkeypatch,
):
    """Title and publication_date populated from response."""
    mocks = _patch_dependencies(
        monkeypatch,
        _default_metadata(
            title="Q4 2025 Results",
            publication_date="2025-12-05",
        ),
    )
    result = _make_result()

    enriched = mod.enrich_doc_metadata(result, mocks["llm"])

    assert enriched.document_metadata["title"] == "Q4 2025 Results"
    assert enriched.document_metadata["publication_date"] == "2025-12-05"


def test_enrich_doc_metadata_preserves_pages(monkeypatch):
    """Pages pass through unchanged after enrichment."""
    mocks = _patch_dependencies(monkeypatch)
    pages = [
        _make_page("Page one", page_number=1),
        _make_page("Page two", page_number=2),
    ]
    result = _make_result(pages=pages)

    enriched = mod.enrich_doc_metadata(result, mocks["llm"])

    assert len(enriched.pages) == 2
    assert enriched.pages[0].raw_content == "Page one"
    assert enriched.pages[1].raw_content == "Page two"


def test_xlsx_overrides_to_sheet_based(monkeypatch):
    """Non-sheet_based for XLSX is overridden."""
    wrong_types = [
        "chapters",
        "sections",
        "topic_based",
        "semantic",
    ]
    for wrong_type in wrong_types:
        mocks = _patch_dependencies(
            monkeypatch,
            _default_metadata(structure_type=wrong_type),
        )
        result = _make_result(
            filetype="xlsx",
            file_path="/tmp/data/source/report.xlsx",
        )

        enriched = mod.enrich_doc_metadata(result, mocks["llm"])

        assert enriched.document_metadata["structure_type"] == "sheet_based"


def test_enrich_doc_metadata_includes_toc(monkeypatch):
    """TOC entries populated when has_toc is true."""
    toc = [
        {"name": "Introduction", "page_number": 1},
        {"name": "Results", "page_number": 5},
    ]
    mocks = _patch_dependencies(
        monkeypatch,
        _default_metadata(has_toc=True, toc_entries=toc),
    )
    result = _make_result()

    enriched = mod.enrich_doc_metadata(result, mocks["llm"])

    assert enriched.document_metadata["has_toc"] is True
    entries = enriched.document_metadata["toc_entries"]
    assert len(entries) == 2
    assert entries[0]["name"] == "Introduction"
    assert entries[1]["page_number"] == 5
    assert enriched.document_metadata["source_toc_entries"] == toc
    assert enriched.document_metadata["generated_toc_entries"] == []


def test_enrich_doc_metadata_calls_llm_with_stage(
    monkeypatch,
):
    """LLM is called with stage='doc_metadata'."""
    mocks = _patch_dependencies(monkeypatch)
    result = _make_result()

    mod.enrich_doc_metadata(result, mocks["llm"])

    assert len(mocks["call_log"]) == 1
    assert mocks["call_log"][0]["stage"] == "doc_metadata"


def test_enrich_doc_metadata_rationale_stored(monkeypatch):
    """Rationale field is stored in document_metadata."""
    mocks = _patch_dependencies(
        monkeypatch,
        _default_metadata(rationale="Clear numbered sections."),
    )
    result = _make_result()

    enriched = mod.enrich_doc_metadata(result, mocks["llm"])

    assert (
        enriched.document_metadata["rationale"] == "Clear numbered sections."
    )


def test_enrich_doc_metadata_carries_source_context(monkeypatch):
    """Source and filter fields are copied into document metadata."""
    mocks = _patch_dependencies(
        monkeypatch,
        metadata_dict=_default_metadata(),
    )
    result = _make_result()
    result.data_source = "pillar3"
    result.filter_1 = "2026"
    result.filter_2 = "Q1"

    enriched = mod.enrich_doc_metadata(result, mocks["llm"])

    assert enriched.document_metadata["data_source"] == "pillar3"
    assert enriched.document_metadata["filter_1"] == "2026"
    assert enriched.document_metadata["filter_2"] == "Q1"


def test_enrich_doc_metadata_empty_pages(monkeypatch):
    """Empty documents still produce metadata without crashing."""
    mocks = _patch_dependencies(monkeypatch)
    result = _make_result(pages=[])
    result.total_pages = 0

    enriched = mod.enrich_doc_metadata(result, mocks["llm"])

    assert enriched.document_metadata["title"] == "Test Document"
    assert len(mocks["call_log"]) == 1


# ------------------------------------------------------------------
# build_file_metadata
# ------------------------------------------------------------------


def test_build_file_metadata_includes_filename():
    """File metadata includes the filename."""
    result = _make_result(
        file_path="/data/reports/2026/Q1/report.pdf",
    )
    result.data_source = "reports"
    result.filter_1 = "2026"
    result.filter_2 = "Q1"

    output = mod.build_file_metadata(result)

    assert "filename: report.pdf" in output
    assert "filetype: pdf" in output
    assert "data_source: reports" in output
    assert "filter_1: 2026" in output
    assert "filter_2: Q1" in output


# ------------------------------------------------------------------
# build_page_names
# ------------------------------------------------------------------


def test_build_page_names_xlsx():
    """XLSX page names use sheet names."""
    pages = [
        _make_page("# Sheet: Overview\ndata", page_number=1),
        _make_page("# Sheet: Details\ndata", page_number=2),
    ]
    result = _make_result(pages=pages, filetype="xlsx")

    output = mod.build_page_names(result, pages)

    assert "Sheet 1: Overview" in output
    assert "Sheet 2: Details" in output


def test_build_page_names_pdf():
    """PDF page names use page numbers and headings."""
    pages = [
        _make_page("# Cover\ntext", page_number=1),
        _make_page("Plain text only", page_number=2),
    ]
    result = _make_result(pages=pages, filetype="pdf")

    output = mod.build_page_names(result, pages)

    assert "Page 1: Cover" in output
    assert "Page 2" in output


def test_build_page_names_deduplicates():
    """Duplicate page numbers only appear once."""
    pages = [
        _make_page("# Intro\ntext", page_number=1),
        _make_page("# Intro\ntext", page_number=1),
        _make_page("# Next\ntext", page_number=2),
    ]
    result = _make_result(pages=pages, filetype="pdf")

    output = mod.build_page_names(result, pages)

    lines = [line for line in output.splitlines() if line.strip()]
    assert len(lines) == 2


# ------------------------------------------------------------------
# build_layout_summary
# ------------------------------------------------------------------


def test_build_layout_summary_counts():
    """Layout summary counts types correctly."""
    pages = [
        _make_page(layout_type="text", page_number=1),
        _make_page(layout_type="text", page_number=2),
        _make_page(layout_type="spreadsheet", page_number=3),
    ]

    output = mod.build_layout_summary(pages)

    assert "spreadsheet: 1" in output
    assert "text: 2" in output


# ------------------------------------------------------------------
# format_user_input
# ------------------------------------------------------------------


def test_format_user_input_assembles_sections():
    """User input contains all XML sections."""
    prompt = _stub_prompt()

    output = mod.format_user_input(
        "filename: doc.pdf",
        "Page 1: Cover",
        "text: 5",
        "[Page 1]\nContent here",
        prompt,
    )

    assert "<file_metadata>" in output
    assert "</file_metadata>" in output
    assert "<page_names>" in output
    assert "<content>" in output
    assert "<layout_summary>" in output


# ------------------------------------------------------------------
# deduplicate_pages
# ------------------------------------------------------------------


def test_deduplicate_pages_prefers_unchunked():
    """Deduplication prefers unchunked pages."""
    unchunked = _make_page("Full page", page_number=1, chunk_id="")
    chunked = _make_page("Chunk only", page_number=1, chunk_id="1.1")

    result = mod.deduplicate_pages([chunked, unchunked])

    assert len(result) == 1
    assert result[0].raw_content == "Full page"
