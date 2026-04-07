"""Tests for the doc_summary enrichment module."""

import json

import pytest

from ingestion.stages.enrichment import (
    doc_summary as mod,
)
from ingestion.utils.file_types import (
    ExtractionResult,
    PageResult,
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


def _make_section(
    section_id="1",
    title="Section One",
    sequence=1,
    page_start=1,
    page_end=1,
    chunk_ids=None,
    level="section",
    **overrides,
):
    """Build a section dict for testing.

    Params: section_id, title, sequence, page_start,
        page_end, chunk_ids, level, plus overrides for
        summary, keywords, entities.
    Returns: dict.
    """
    summary = overrides.pop("summary", "")
    keywords = overrides.pop("keywords", None)
    entities = overrides.pop("entities", None)
    section = {
        "section_id": section_id,
        "parent_section_id": "",
        "level": level,
        "title": title,
        "sequence": sequence,
        "page_start": page_start,
        "page_end": page_end,
        "chunk_ids": chunk_ids if chunk_ids else [],
        "summary": summary,
        "keywords": keywords if keywords else [],
        "entities": entities if entities else [],
        "token_count": 0,
    }
    section.update(overrides)
    return section


def _make_result(
    pages=None,
    sections=None,
    filetype="pdf",
    file_path="/tmp/data/source/2026/doc.pdf",
    metadata=None,
):
    """Build an ExtractionResult for testing.

    Params: pages, sections, filetype, file_path, metadata.
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
    if sections is not None:
        result.sections = sections
    if metadata is not None:
        result.document_metadata = metadata
    return result


def _make_llm_response(
    executive_summary="Test summary.",
    keywords=None,
    entities=None,
    rationale="test",
):
    """Build a mock LLM tool-call response dict.

    Params: executive_summary, keywords, entities,
        rationale.
    Returns: dict.
    """
    if keywords is None:
        keywords = ["kw1", "kw2"]
    if entities is None:
        entities = ["ent1"]
    args = json.dumps(
        {
            "executive_summary": executive_summary,
            "keywords": keywords,
            "entities": entities,
            "rationale": rationale,
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


def _stub_prompt():
    """Build a minimal prompt dict for testing.

    Returns: dict.
    """
    return {
        "stage": "doc_summary",
        "version": "1.0",
        "description": "test",
        "system_prompt": "You are a test agent.",
        "user_prompt": ("Summarize:\n\n{user_input}\n\nRules:\n1. Do it."),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "summarize_document",
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


def _patch_dependencies(
    monkeypatch,
    executive_summary="Executive summary text.",
    keywords=None,
    entities=None,
):
    """Patch load_prompt and build a fake LLM.

    Params: monkeypatch, executive_summary, keywords,
        entities.
    Returns: dict of mocks.
    """
    if keywords is None:
        keywords = ["CET1", "RWA", "capital"]
    if entities is None:
        entities = ["OSFI", "RBC"]

    prompt = _stub_prompt()

    monkeypatch.setattr(
        mod,
        "load_prompt",
        lambda name, prompts_dir=None: prompt,
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
        return _make_llm_response(
            executive_summary=executive_summary,
            keywords=keywords,
            entities=entities,
        )

    fake_llm = type(
        "FakeLLM",
        (),
        {"call": staticmethod(fake_call), "__doc__": "F."},
    )()

    return {"llm": fake_llm, "call_log": call_log}


# ------------------------------------------------------------------
# test_summarize_document_sets_executive_summary
# ------------------------------------------------------------------


def test_summarize_document_sets_executive_summary(
    monkeypatch,
):
    """Executive summary populated in document_metadata."""
    mocks = _patch_dependencies(
        monkeypatch,
        executive_summary="The document covers capital.",
    )
    sections = [
        _make_section(
            section_id="1",
            title="KM1",
            summary="Key metrics.",
            keywords=["CET1"],
            entities=["RBC"],
            chunk_ids=["1"],
        ),
    ]
    result = _make_result(
        sections=sections,
        metadata={"title": "Report"},
    )

    enriched = mod.summarize_document(result, mocks["llm"])

    assert (
        enriched.document_metadata["executive_summary"]
        == "The document covers capital."
    )


# ------------------------------------------------------------------
# test_summarize_document_sets_keywords
# ------------------------------------------------------------------


def test_summarize_document_sets_keywords(monkeypatch):
    """Document-level keywords populated."""
    mocks = _patch_dependencies(
        monkeypatch,
        keywords=["CET1", "leverage", "TLAC"],
    )
    sections = [
        _make_section(
            section_id="1",
            summary="Summary.",
            chunk_ids=["1"],
        ),
    ]
    result = _make_result(
        sections=sections,
        metadata={"title": "Report"},
    )

    enriched = mod.summarize_document(result, mocks["llm"])

    assert enriched.document_metadata["keywords"] == [
        "CET1",
        "leverage",
        "TLAC",
    ]


# ------------------------------------------------------------------
# test_summarize_document_sets_entities
# ------------------------------------------------------------------


def test_summarize_document_sets_entities(monkeypatch):
    """Document-level entities populated."""
    mocks = _patch_dependencies(
        monkeypatch,
        entities=["OSFI", "Royal Bank of Canada"],
    )
    sections = [
        _make_section(
            section_id="1",
            summary="Summary.",
            chunk_ids=["1"],
        ),
    ]
    result = _make_result(
        sections=sections,
        metadata={"title": "Report"},
    )

    enriched = mod.summarize_document(result, mocks["llm"])

    assert enriched.document_metadata["entities"] == [
        "OSFI",
        "Royal Bank of Canada",
    ]


# ------------------------------------------------------------------
# test_summarize_document_uses_section_summaries
# ------------------------------------------------------------------


def test_summarize_document_uses_section_summaries(
    monkeypatch,
):
    """LLM input contains section summaries."""
    mocks = _patch_dependencies(monkeypatch)
    sections = [
        _make_section(
            section_id="1",
            title="KM1",
            summary="Key capital metrics CET1 13.7%.",
            keywords=["CET1"],
            entities=["OSFI"],
            chunk_ids=["1"],
        ),
        _make_section(
            section_id="2",
            title="OV1",
            summary="Overview of RWA by risk type.",
            keywords=["RWA"],
            entities=["RBC"],
            chunk_ids=["2"],
        ),
    ]
    result = _make_result(
        sections=sections,
        metadata={"title": "Pillar 3"},
    )

    mod.summarize_document(result, mocks["llm"])

    user_msg = mocks["call_log"][0]["messages"][1]
    content = user_msg["content"]
    assert "Key capital metrics CET1 13.7%." in content
    assert "Overview of RWA by risk type." in content
    assert "KM1" in content
    assert "OV1" in content


# ------------------------------------------------------------------
# test_summarize_document_includes_metadata
# ------------------------------------------------------------------


def test_summarize_document_includes_metadata(
    monkeypatch,
):
    """LLM input contains document title and authors."""
    mocks = _patch_dependencies(monkeypatch)
    sections = [
        _make_section(
            section_id="1",
            summary="Sum.",
            chunk_ids=["1"],
        ),
    ]
    result = _make_result(
        sections=sections,
        metadata={
            "title": "Pillar 3 Disclosures",
            "authors": "Royal Bank of Canada",
            "publication_date": "January 31, 2026",
        },
    )

    mod.summarize_document(result, mocks["llm"])

    content = mocks["call_log"][0]["messages"][1]["content"]
    assert "Pillar 3 Disclosures" in content
    assert "Royal Bank of Canada" in content
    assert "January 31, 2026" in content
    assert "document_metadata" in content


# ------------------------------------------------------------------
# test_format_toc
# ------------------------------------------------------------------


def test_format_toc():
    """Sections formatted with titles, summaries, keywords."""
    sections = [
        _make_section(
            section_id="1",
            title="Cover",
            page_start=1,
            page_end=1,
            summary="Cover page for RBC Pillar 3.",
            keywords=[],
            entities=["Royal Bank of Canada"],
        ),
        _make_section(
            section_id="3",
            title="KM1",
            page_start=3,
            page_end=5,
            summary="Key capital metrics CET1 13.7%.",
            keywords=["CET1", "Tier 1", "RWA"],
            entities=["OSFI"],
        ),
    ]

    toc = mod.format_toc(sections)

    assert "[1] Cover (p.1)" in toc
    assert "Cover page for RBC Pillar 3." in toc
    assert "[3] KM1 (p.3-5)" in toc
    assert "CET1" in toc
    assert "OSFI" in toc
    assert "Royal Bank of Canada" in toc


# ------------------------------------------------------------------
# test_parse_doc_summary_response_valid
# ------------------------------------------------------------------


def test_parse_doc_summary_response_valid():
    """Valid response parsed into summary dict."""
    response = _make_llm_response(
        executive_summary="Strong capital position.",
        keywords=["CET1", "leverage"],
        entities=["OSFI"],
    )

    parsed = mod.parse_doc_summary_response(response)

    assert parsed["executive_summary"] == ("Strong capital position.")
    assert parsed["keywords"] == ["CET1", "leverage"]
    assert parsed["entities"] == ["OSFI"]


# ------------------------------------------------------------------
# test_parse_doc_summary_response_malformed
# ------------------------------------------------------------------


def test_parse_doc_summary_response_no_choices():
    """Raises ValueError when response has no choices."""
    with pytest.raises(ValueError, match="no choices"):
        mod.parse_doc_summary_response({"choices": []})


def test_parse_doc_summary_response_no_tool_calls():
    """Raises ValueError when no tool calls present."""
    response = {
        "choices": [{"message": {"tool_calls": None}}],
    }
    with pytest.raises(ValueError, match="no tool calls"):
        mod.parse_doc_summary_response(response)


def test_parse_doc_summary_response_bad_json():
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
        mod.parse_doc_summary_response(response)


def test_parse_doc_summary_response_missing_summary():
    """Raises ValueError when executive_summary missing."""
    args = json.dumps({"keywords": [], "entities": [], "rationale": "x"})
    response = {
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
    with pytest.raises(ValueError, match="executive_summary"):
        mod.parse_doc_summary_response(response)


def test_parse_doc_summary_response_missing_keywords():
    """Raises ValueError when keywords missing."""
    args = json.dumps(
        {
            "executive_summary": "Sum.",
            "entities": [],
            "rationale": "x",
        }
    )
    response = {
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
    with pytest.raises(ValueError, match="keywords"):
        mod.parse_doc_summary_response(response)


def test_parse_doc_summary_response_missing_entities():
    """Raises ValueError when entities missing."""
    args = json.dumps(
        {
            "executive_summary": "Sum.",
            "keywords": [],
            "rationale": "x",
        }
    )
    response = {
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
    with pytest.raises(ValueError, match="entities"):
        mod.parse_doc_summary_response(response)


# ------------------------------------------------------------------
# test_summarize_document_preserves_other_metadata
# ------------------------------------------------------------------


def test_summarize_document_preserves_other_metadata(
    monkeypatch,
):
    """Existing metadata fields remain unchanged."""
    mocks = _patch_dependencies(monkeypatch)
    sections = [
        _make_section(
            section_id="1",
            summary="Sum.",
            chunk_ids=["1"],
        ),
    ]
    result = _make_result(
        sections=sections,
        metadata={
            "title": "Report",
            "authors": "BMO",
            "structure_type": "sheet_based",
            "toc_entries": [{"section_id": "1"}],
        },
    )

    enriched = mod.summarize_document(result, mocks["llm"])

    assert enriched.document_metadata["title"] == "Report"
    assert enriched.document_metadata["authors"] == "BMO"
    assert enriched.document_metadata["structure_type"] == "sheet_based"
    assert enriched.document_metadata["toc_entries"] == [{"section_id": "1"}]


# ------------------------------------------------------------------
# test_summarize_document_preserves_pages
# ------------------------------------------------------------------


def test_summarize_document_preserves_pages(monkeypatch):
    """Pages unchanged after document summarization."""
    mocks = _patch_dependencies(monkeypatch)
    pages = [
        _make_page(
            "Original content",
            page_number=1,
            layout_type="text",
        ),
    ]
    sections = [
        _make_section(
            section_id="1",
            summary="Sum.",
            chunk_ids=["1"],
        ),
    ]
    result = _make_result(pages=pages, sections=sections)

    enriched = mod.summarize_document(result, mocks["llm"])

    assert enriched.pages[0].raw_content == "Original content"
    assert enriched.pages[0].layout_type == "text"
    assert enriched.pages[0].page_number == 1


# ------------------------------------------------------------------
# test_summarize_document_empty_sections
# ------------------------------------------------------------------


def test_summarize_document_empty_sections(monkeypatch):
    """No sections still makes LLM call with metadata."""
    mocks = _patch_dependencies(monkeypatch)
    result = _make_result(
        sections=[],
        metadata={"title": "Empty Doc"},
    )

    enriched = mod.summarize_document(result, mocks["llm"])

    assert len(mocks["call_log"]) == 1
    content = mocks["call_log"][0]["messages"][1]["content"]
    assert "table_of_contents" in content
    assert "executive_summary" in enriched.document_metadata


# ------------------------------------------------------------------
# Additional helper tests
# ------------------------------------------------------------------


def test_get_primary_sections_filters():
    """Only level=section items returned, sorted."""
    sections = [
        _make_section(section_id="1", level="section", sequence=2),
        _make_section(
            section_id="1.1",
            level="subsection",
            sequence=1,
        ),
        _make_section(section_id="2", level="section", sequence=1),
    ]

    primaries = mod.get_primary_sections(sections)

    assert len(primaries) == 2
    assert primaries[0]["section_id"] == "2"
    assert primaries[1]["section_id"] == "1"


def test_format_doc_metadata():
    """Metadata formatted with key-value lines."""
    metadata = {
        "title": "Pillar 3 Report",
        "authors": "RBC",
        "publication_date": "2026-01-31",
        "data_source": "pillar3",
        "filter_1": "2026_Q1",
        "filter_2": "RBC",
    }

    formatted = mod.format_doc_metadata(metadata)

    assert 'title: "Pillar 3 Report"' in formatted
    assert 'authors: "RBC"' in formatted
    assert 'data_source: "pillar3"' in formatted
    assert 'filter_1: "2026_Q1"' in formatted
    assert 'filter_2: "RBC"' in formatted


def test_format_doc_metadata_empty():
    """Empty metadata returns empty string."""
    formatted = mod.format_doc_metadata({})

    assert formatted == ""


def test_format_user_input_includes_xml_tags():
    """User input has document_metadata and TOC tags."""
    prompt = _stub_prompt()
    user_msg = mod.format_user_input(
        'title: "Test"',
        "[1] Cover (p.1)",
        prompt,
    )

    assert "<document_metadata>" in user_msg
    assert "</document_metadata>" in user_msg
    assert "<table_of_contents>" in user_msg
    assert "</table_of_contents>" in user_msg
    assert 'title: "Test"' in user_msg


def test_format_user_input_no_metadata():
    """Empty metadata omits document_metadata tag."""
    prompt = _stub_prompt()
    user_msg = mod.format_user_input(
        "",
        "[1] Cover (p.1)",
        prompt,
    )

    assert "document_metadata" not in user_msg
    assert "table_of_contents" in user_msg


def test_update_metadata_sets_fields():
    """Update applies all three fields."""
    result = _make_result(
        metadata={"title": "Report"},
    )
    summary_data = {
        "executive_summary": "Summary text.",
        "keywords": ["a", "b"],
        "entities": ["E1"],
    }

    mod.update_metadata(result, summary_data)

    meta = result.document_metadata
    assert meta["executive_summary"] == "Summary text."
    assert meta["keywords"] == ["a", "b"]
    assert meta["entities"] == ["E1"]
    assert meta["title"] == "Report"


def test_update_metadata_null_metadata():
    """Creates document_metadata when it is None."""
    result = _make_result()
    result.document_metadata = None
    summary_data = {
        "executive_summary": "Sum.",
        "keywords": [],
        "entities": [],
    }

    mod.update_metadata(result, summary_data)

    assert result.document_metadata is not None
    assert result.document_metadata["executive_summary"] == ("Sum.")


def test_format_toc_empty_sections():
    """Empty sections list returns empty string."""
    toc = mod.format_toc([])

    assert toc == ""


def test_format_toc_single_page_range():
    """Single-page section shows just the page number."""
    sections = [
        _make_section(
            section_id="5",
            title="Table",
            page_start=5,
            page_end=5,
            summary="A table.",
        ),
    ]

    toc = mod.format_toc(sections)

    assert "(p.5)" in toc
    assert "5-5" not in toc
