"""Tests for the section_summary enrichment module."""

import json

import pytest

from ingestion.stages.enrichment import (
    section_summary as mod,
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
        page_end, chunk_ids, level.
    Returns: dict.
    """
    section = {
        "section_id": section_id,
        "parent_section_id": "",
        "level": level,
        "title": title,
        "sequence": sequence,
        "page_start": page_start,
        "page_end": page_end,
        "chunk_ids": chunk_ids if chunk_ids else [],
        "summary": "",
        "keywords": [],
        "entities": [],
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


def _make_llm_response(items, rationale="test"):
    """Build a mock LLM tool-call response dict.

    Params: items, rationale. Returns: dict.
    """
    args = json.dumps({"items": items, "rationale": rationale})
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
        "stage": "section_summary",
        "version": "1.0",
        "description": "test",
        "system_prompt": "You are a test agent.",
        "user_prompt": ("Summarize:\n\n{user_input}\n\n" "Rules:\n1. Do it."),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "summarize_section",
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
    items_per_call=None,
    budget=50000,
):
    """Patch load_prompt and config for all tests.

    Params: monkeypatch, items_per_call, budget.
    Returns: dict of mocks.
    """
    prompt = _stub_prompt()

    monkeypatch.setattr(
        mod,
        "load_prompt",
        lambda name, prompts_dir=None: prompt,
    )
    monkeypatch.setattr(
        mod,
        "get_section_summary_batch_budget",
        lambda: budget,
    )
    monkeypatch.setattr(
        mod,
        "count_message_tokens",
        lambda messages, tools=None: len(messages[1]["content"]),
    )

    call_log = []
    call_count = [0]

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
        idx = call_count[0]
        call_count[0] += 1
        if items_per_call and idx < len(items_per_call):
            return _make_llm_response(items_per_call[idx])
        if items_per_call:
            return _make_llm_response(items_per_call[-1])
        return _make_llm_response([])

    fake_llm = type(
        "FakeLLM",
        (),
        {"call": staticmethod(fake_call), "__doc__": "Fake."},
    )()

    return {"llm": fake_llm, "call_log": call_log}


# ------------------------------------------------------------------
# test_summarize_sections_populates_summaries
# ------------------------------------------------------------------


def test_summarize_sections_populates_summaries(
    monkeypatch,
):
    """Each primary section gets a summary after processing."""
    items = [
        [
            {
                "section_id": "1",
                "summary": "First section summary.",
                "keywords": ["k1"],
                "entities": ["e1"],
            },
            {
                "section_id": "2",
                "summary": "Second section summary.",
                "keywords": ["k2"],
                "entities": ["e2"],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    pages = [
        _make_page("Page one content", page_number=1),
        _make_page("Page two content", page_number=2),
    ]
    sections = [
        _make_section(
            section_id="1",
            title="S1",
            sequence=1,
            page_start=1,
            page_end=1,
            chunk_ids=["1"],
        ),
        _make_section(
            section_id="2",
            title="S2",
            sequence=2,
            page_start=2,
            page_end=2,
            chunk_ids=["2"],
        ),
    ]
    result = _make_result(pages=pages, sections=sections)

    enriched = mod.summarize_sections(result, mocks["llm"])

    assert enriched.sections[0]["summary"] == "First section summary."
    assert enriched.sections[1]["summary"] == "Second section summary."


# ------------------------------------------------------------------
# test_summarize_sections_populates_keywords
# ------------------------------------------------------------------


def test_summarize_sections_populates_keywords(
    monkeypatch,
):
    """Section keywords refined from unit keywords."""
    items = [
        [
            {
                "section_id": "1",
                "summary": "Summary.",
                "keywords": ["CET1", "RWA", "capital"],
                "entities": ["OSFI"],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    pages = [
        _make_page(
            "Capital data",
            page_number=1,
            keywords=["CET1", "Tier 1", "RWA"],
        ),
    ]
    sections = [
        _make_section(
            section_id="1",
            title="Capital",
            chunk_ids=["1"],
        ),
    ]
    result = _make_result(pages=pages, sections=sections)

    enriched = mod.summarize_sections(result, mocks["llm"])

    assert enriched.sections[0]["keywords"] == [
        "CET1",
        "RWA",
        "capital",
    ]
    assert enriched.sections[0]["entities"] == ["OSFI"]


# ------------------------------------------------------------------
# test_summarize_sections_sequential_with_toc
# ------------------------------------------------------------------


def test_summarize_sections_sequential_with_toc(
    monkeypatch,
):
    """Progressive TOC grows with each batch."""
    batch_1_items = [
        {
            "section_id": "1",
            "summary": "Summary of section 1.",
            "keywords": ["a"],
            "entities": [],
        },
    ]
    batch_2_items = [
        {
            "section_id": "2",
            "summary": "Summary of section 2.",
            "keywords": ["b"],
            "entities": [],
        },
    ]
    items = [batch_1_items, batch_2_items]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items, budget=150)
    pages = [
        _make_page(
            "Content 1",
            page_number=1,
            raw_token_count=100,
        ),
        _make_page(
            "Content 2",
            page_number=2,
            raw_token_count=100,
        ),
    ]
    sections = [
        _make_section(
            section_id="1",
            title="S1",
            sequence=1,
            page_start=1,
            page_end=1,
            chunk_ids=["1"],
        ),
        _make_section(
            section_id="2",
            title="S2",
            sequence=2,
            page_start=2,
            page_end=2,
            chunk_ids=["2"],
        ),
    ]
    result = _make_result(pages=pages, sections=sections)

    mod.summarize_sections(result, mocks["llm"])

    assert len(mocks["call_log"]) == 2
    first_user = mocks["call_log"][0]["messages"][1]["content"]
    assert "table_of_contents_so_far" not in first_user
    second_user = mocks["call_log"][1]["messages"][1]["content"]
    assert "table_of_contents_so_far" in second_user
    assert "S1" in second_user
    assert "Summary of section 1." in second_user


# ------------------------------------------------------------------
# test_summarize_sections_xlsx_chunked_section
# ------------------------------------------------------------------


def test_summarize_sections_xlsx_chunked_section(
    monkeypatch,
):
    """Chunked XLSX section includes passthrough and entities."""
    items = [
        [
            {
                "section_id": "1",
                "summary": "XLSX summary.",
                "keywords": ["PD"],
                "entities": [],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    chunk_header = "# Sheet: CR6_A\n| B | C | D |"
    passthrough = "| 2 | CR6: A-IRB |"
    section_passthrough = "| Total exposures | 1,000 |"
    pages = [
        _make_page(
            "Chunk 1 data rows",
            page_number=24,
            chunk_id="24.1",
            chunk_header=chunk_header,
            sheet_passthrough_content=passthrough,
            section_passthrough_content=section_passthrough,
            chunk_context="Rows 2-9",
            keywords=["PD scale", "CCF"],
            entities=["OSFI"],
        ),
        _make_page(
            "Chunk 2 data rows",
            page_number=24,
            chunk_id="24.2",
            chunk_header=chunk_header,
            sheet_passthrough_content=passthrough,
            section_passthrough_content=section_passthrough,
            chunk_context="Rows 10-19",
            keywords=["Sovereigns"],
            entities=["Canada"],
        ),
    ]
    sections = [
        _make_section(
            section_id="1",
            title="CR6_A",
            page_start=24,
            page_end=24,
            chunk_ids=["24.1", "24.2"],
        ),
    ]
    result = _make_result(
        pages=pages,
        sections=sections,
        filetype="xlsx",
    )

    mod.summarize_sections(result, mocks["llm"])

    user_msg = mocks["call_log"][0]["messages"][1]["content"]
    assert "sheet_header" in user_msg
    assert "sheet_context" in user_msg
    assert "section_context" in user_msg
    assert "Chunk 1 data rows" in user_msg
    assert "Chunk 2 data rows" in user_msg
    assert '"OSFI"' in user_msg
    assert '"Canada"' in user_msg
    assert user_msg.count("sheet_header") == 2
    assert user_msg.count("section_context") == 2


# ------------------------------------------------------------------
# test_summarize_sections_xlsx_unchunked
# ------------------------------------------------------------------


def test_summarize_sections_xlsx_unchunked(monkeypatch):
    """Unchunked XLSX section includes full raw_content."""
    items = [
        [
            {
                "section_id": "1",
                "summary": "Short sheet.",
                "keywords": ["KM1"],
                "entities": [],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    pages = [
        _make_page(
            "Full sheet content here",
            page_number=3,
            keywords=["CET1"],
        ),
    ]
    sections = [
        _make_section(
            section_id="1",
            title="KM1",
            page_start=3,
            page_end=3,
            chunk_ids=["3"],
        ),
    ]
    result = _make_result(
        pages=pages,
        sections=sections,
        filetype="xlsx",
    )

    mod.summarize_sections(result, mocks["llm"])

    user_msg = mocks["call_log"][0]["messages"][1]["content"]
    assert "Full sheet content here" in user_msg
    assert "sheet_header" not in user_msg


# ------------------------------------------------------------------
# test_summarize_sections_skips_subsections
# ------------------------------------------------------------------


def test_summarize_sections_also_summarizes_subsections(
    monkeypatch,
):
    """Primary sections and subsections without summaries are processed."""
    items = [
        [
            {
                "section_id": "1",
                "summary": "Primary summary.",
                "keywords": [],
                "entities": [],
            },
        ],
        [
            {
                "section_id": "1.1",
                "summary": "Sub summary.",
                "keywords": [],
                "entities": [],
            },
        ],
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    pages = [
        _make_page("Content", page_number=1),
        _make_page("Sub content", page_number=2),
    ]
    sections = [
        _make_section(
            section_id="1",
            title="Main",
            sequence=1,
            page_start=1,
            page_end=2,
            chunk_ids=["1"],
        ),
        _make_section(
            section_id="1.1",
            title="Sub",
            sequence=1,
            page_start=2,
            page_end=2,
            chunk_ids=["2"],
            level="subsection",
            parent_section_id="1",
        ),
    ]
    result = _make_result(pages=pages, sections=sections)

    enriched = mod.summarize_sections(result, mocks["llm"])

    primary = [s for s in enriched.sections if s["level"] == "section"]
    subs = [s for s in enriched.sections if s["level"] == "subsection"]
    assert primary[0]["summary"] == "Primary summary."
    assert subs[0]["summary"] == "Sub summary."


# ------------------------------------------------------------------
# test_summarize_sections_batches_by_budget
# ------------------------------------------------------------------


def test_summarize_sections_batches_by_budget(
    monkeypatch,
):
    """Small sections grouped into batches by budget."""
    items = [
        [
            {
                "section_id": "1",
                "summary": "S1.",
                "keywords": [],
                "entities": [],
            },
        ],
        [
            {
                "section_id": "2",
                "summary": "S2.",
                "keywords": [],
                "entities": [],
            },
        ],
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items, budget=150)
    pages = [
        _make_page("A", page_number=1, raw_token_count=100),
        _make_page("B", page_number=2, raw_token_count=100),
    ]
    sections = [
        _make_section(
            section_id="1",
            sequence=1,
            page_start=1,
            page_end=1,
            chunk_ids=["1"],
        ),
        _make_section(
            section_id="2",
            sequence=2,
            page_start=2,
            page_end=2,
            chunk_ids=["2"],
        ),
    ]
    result = _make_result(pages=pages, sections=sections)

    mod.summarize_sections(result, mocks["llm"])

    assert len(mocks["call_log"]) == 2


# ------------------------------------------------------------------
# test_gather_section_content_unchunked
# ------------------------------------------------------------------


def test_gather_section_content_unchunked():
    """Correct content for single-page section."""
    pages = [
        _make_page(
            "Revenue analysis content",
            page_number=3,
            keywords=["revenue", "Q4"],
            entities=["BMO"],
        ),
    ]
    section = _make_section(
        section_id="3",
        title="KM1",
        page_start=3,
        page_end=3,
        chunk_ids=["3"],
    )

    content = mod.gather_section_content(section, pages)

    assert 'id="3"' in content
    assert 'title="KM1"' in content
    assert "Revenue analysis content" in content
    assert '"revenue"' in content
    assert '"BMO"' in content


# ------------------------------------------------------------------
# test_gather_xlsx_section_content
# ------------------------------------------------------------------


def test_gather_xlsx_section_content():
    """Header + passthrough once + per-chunk entities."""
    chunk_header = "# Sheet: CR6\n| A | B |"
    passthrough = "| 1 | Title row |"
    section_passthrough = "| Totals | 200 |"
    pages = [
        _make_page(
            "Chunk one rows",
            page_number=10,
            chunk_id="10.1",
            chunk_header=chunk_header,
            sheet_passthrough_content=passthrough,
            section_passthrough_content=section_passthrough,
            chunk_context="Rows 2-5",
            entities=["OSFI"],
        ),
        _make_page(
            "Chunk two rows",
            page_number=10,
            chunk_id="10.2",
            chunk_header=chunk_header,
            sheet_passthrough_content=passthrough,
            section_passthrough_content=section_passthrough,
            chunk_context="Rows 6-10",
            entities=["Canada"],
        ),
    ]
    section = _make_section(
        section_id="5",
        title="CR6",
        page_start=10,
        page_end=10,
        chunk_ids=["10.1", "10.2"],
    )

    content = mod.gather_xlsx_section_content(section, pages)

    assert "sheet_header" in content
    assert "sheet_context" in content
    assert "section_context" in content
    assert "Chunk one rows" in content
    assert "Chunk two rows" in content
    assert '"OSFI"' in content
    assert '"Canada"' in content
    assert content.count("# Sheet: CR6") == 1
    assert content.count("Title row") == 1
    assert content.count("Totals | 200") == 1


def test_first_chunk_page_value_returns_empty_when_missing():
    """Missing page attributes return an empty string."""
    chunk_pages = [("10.1", _make_page("Chunk", page_number=10))]

    value = mod.first_chunk_page_value(chunk_pages, "chunk_header")

    assert value == ""


def test_format_xlsx_context_block_omits_empty_value():
    """Empty context values produce no XML block."""
    block = mod.format_xlsx_context_block("sheet_context", "")

    assert block == ""


# ------------------------------------------------------------------
# test_build_progressive_toc
# ------------------------------------------------------------------


def test_build_progressive_toc():
    """Formats completed sections as TOC entries."""
    completed = [
        {
            "section_id": "1",
            "title": "Cover",
            "page_start": 1,
            "summary": "Cover page for RBC Pillar 3.",
        },
        {
            "section_id": "3",
            "title": "KM1",
            "page_start": 3,
            "summary": "Key capital metrics: CET1 13.7%.",
        },
    ]

    toc = mod.build_progressive_toc(completed)

    assert "table_of_contents_so_far" in toc
    assert "[1] Cover (p.1)" in toc
    assert "[3] KM1 (p.3)" in toc
    assert "CET1 13.7%" in toc


def test_build_progressive_toc_empty():
    """Empty completed list returns empty string."""
    toc = mod.build_progressive_toc([])

    assert toc == ""


# ------------------------------------------------------------------
# test_parse_summary_response_valid
# ------------------------------------------------------------------


def test_parse_summary_response_valid():
    """Valid response parsed into section_id map."""
    items = [
        {
            "section_id": "1",
            "summary": "First summary.",
            "keywords": ["k1"],
            "entities": ["e1"],
        },
        {
            "section_id": "2",
            "summary": "Second summary.",
            "keywords": ["k2"],
            "entities": [],
        },
    ]
    response = _make_llm_response(items)

    result = mod.parse_summary_response(response)

    assert "1" in result
    assert result["1"]["summary"] == "First summary."
    assert result["1"]["keywords"] == ["k1"]
    assert "2" in result
    assert result["2"]["summary"] == "Second summary."


# ------------------------------------------------------------------
# test_parse_summary_response_malformed
# ------------------------------------------------------------------


def test_parse_summary_response_no_choices():
    """Raises ValueError when response has no choices."""
    with pytest.raises(ValueError, match="no choices"):
        mod.parse_summary_response({"choices": []})


def test_parse_summary_response_no_tool_calls():
    """Raises ValueError when no tool calls present."""
    response = {
        "choices": [{"message": {"tool_calls": None}}],
    }
    with pytest.raises(ValueError, match="no tool calls"):
        mod.parse_summary_response(response)


def test_parse_summary_response_bad_json():
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
        mod.parse_summary_response(response)


def test_parse_summary_response_missing_items():
    """Raises ValueError when items field is missing."""
    args = json.dumps({"rationale": "test"})
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
    with pytest.raises(ValueError, match="items"):
        mod.parse_summary_response(response)


def test_parse_summary_response_duplicate_section_id():
    """Duplicate section ids are rejected."""
    items = [
        {
            "section_id": "1",
            "summary": "A",
            "keywords": [],
            "entities": [],
        },
        {
            "section_id": "1",
            "summary": "B",
            "keywords": [],
            "entities": [],
        },
    ]
    response = _make_llm_response(items)

    with pytest.raises(ValueError, match="duplicate section_id"):
        mod.parse_summary_response(response)


def test_validate_batch_results_requires_exact_section_ids():
    """Batch validation rejects missing or extra section ids."""
    batch = [
        {"section_id": "1"},
        {"section_id": "2"},
    ]

    with pytest.raises(ValueError, match="missing section_ids"):
        mod.validate_batch_results(
            batch,
            {"1": {"summary": "A", "keywords": [], "entities": []}},
        )

    with pytest.raises(ValueError, match="unexpected section_ids"):
        mod.validate_batch_results(
            batch,
            {
                "1": {"summary": "A", "keywords": [], "entities": []},
                "2": {"summary": "B", "keywords": [], "entities": []},
                "3": {"summary": "C", "keywords": [], "entities": []},
            },
        )


def test_parse_summary_response_rejects_non_object_item():
    """Each response item must be an object."""
    response = _make_llm_response(["bad-item"])

    with pytest.raises(ValueError, match="must be an object"):
        mod.parse_summary_response(response)


def test_parse_summary_response_rejects_invalid_summary():
    """Summary must be a string."""
    items = [
        {
            "section_id": "1",
            "summary": [],
            "keywords": [],
            "entities": [],
        },
    ]
    response = _make_llm_response(items)

    with pytest.raises(ValueError, match="invalid summary"):
        mod.parse_summary_response(response)


def test_parse_summary_response_rejects_invalid_keywords():
    """Keywords must be a list of strings."""
    items = [
        {
            "section_id": "1",
            "summary": "ok",
            "keywords": ["valid", 2],
            "entities": [],
        },
    ]
    response = _make_llm_response(items)

    with pytest.raises(ValueError, match="invalid keywords"):
        mod.parse_summary_response(response)


def test_parse_summary_response_rejects_invalid_entities():
    """Entities must be a list of strings."""
    items = [
        {
            "section_id": "1",
            "summary": "ok",
            "keywords": ["valid"],
            "entities": ["ok", 2],
        },
    ]
    response = _make_llm_response(items)

    with pytest.raises(ValueError, match="invalid entities"):
        mod.parse_summary_response(response)


# ------------------------------------------------------------------
# test_summarize_sections_preserves_pages
# ------------------------------------------------------------------


def test_summarize_sections_preserves_pages(
    monkeypatch,
):
    """Pages unchanged after section summarization."""
    items = [
        [
            {
                "section_id": "1",
                "summary": "Sum.",
                "keywords": [],
                "entities": [],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
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
            chunk_ids=["1"],
        ),
    ]
    result = _make_result(pages=pages, sections=sections)

    enriched = mod.summarize_sections(result, mocks["llm"])

    assert enriched.pages[0].raw_content == "Original content"
    assert enriched.pages[0].layout_type == "text"
    assert enriched.pages[0].page_number == 1


# ------------------------------------------------------------------
# test_summarize_sections_empty_sections
# ------------------------------------------------------------------


def test_summarize_sections_empty_sections(monkeypatch):
    """No sections means no LLM calls."""
    mocks = _patch_dependencies(monkeypatch)
    result = _make_result(sections=[])

    enriched = mod.summarize_sections(result, mocks["llm"])

    assert len(mocks["call_log"]) == 0
    assert enriched.sections == []


def test_summarize_sections_rejects_missing_batch_results(
    monkeypatch,
):
    """Stage fails when the LLM omits requested sections."""
    items = [
        [
            {
                "section_id": "1",
                "summary": "Summary one.",
                "keywords": [],
                "entities": [],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    pages = [
        _make_page("A", page_number=1),
        _make_page("B", page_number=2),
    ]
    sections = [
        _make_section(section_id="1", chunk_ids=["1"]),
        _make_section(section_id="2", chunk_ids=["2"], sequence=2),
    ]
    result = _make_result(pages=pages, sections=sections)

    with pytest.raises(ValueError, match="missing section_ids"):
        mod.summarize_sections(result, mocks["llm"])


def test_summarize_sections_rejects_unexpected_batch_results(
    monkeypatch,
):
    """Stage fails when the LLM returns extra section ids."""
    items = [
        [
            {
                "section_id": "1",
                "summary": "Summary one.",
                "keywords": [],
                "entities": [],
            },
            {
                "section_id": "99",
                "summary": "Summary extra.",
                "keywords": [],
                "entities": [],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    pages = [_make_page("A", page_number=1)]
    sections = [_make_section(section_id="1", chunk_ids=["1"])]
    result = _make_result(pages=pages, sections=sections)

    with pytest.raises(ValueError, match="unexpected section_ids"):
        mod.summarize_sections(result, mocks["llm"])


# ------------------------------------------------------------------
# Additional helper tests
# ------------------------------------------------------------------


def test_get_primary_sections_filters():
    """Only level=section items returned."""
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


def test_batch_sections_empty():
    """Empty sections list returns empty batches."""
    batches = mod.batch_sections([], [], budget=1000)

    assert not batches


def test_batch_sections_single_over_budget():
    """Single large section gets its own batch."""
    pages = [_make_page("data", page_number=1, raw_token_count=200)]
    sections = [
        _make_section(section_id="1", chunk_ids=["1"]),
    ]

    batches = mod.batch_sections(sections, pages, budget=100)

    assert len(batches) == 1
    assert batches[0][0]["section_id"] == "1"


def test_batch_sections_counts_formatted_request_budget(
    monkeypatch,
):
    """Prompt-aware batching uses rendered section payload size."""
    pages = [
        _make_page("Page one", page_number=1, raw_token_count=10),
        _make_page("Page two", page_number=2, raw_token_count=10),
    ]
    sections = [
        _make_section(
            section_id="1",
            title="First section",
            chunk_ids=["1"],
        ),
        _make_section(
            section_id="2",
            title="Second section",
            sequence=2,
            page_start=2,
            page_end=2,
            chunk_ids=["2"],
        ),
    ]
    monkeypatch.setattr(
        mod,
        "count_message_tokens",
        lambda messages, tools=None: (
            200 if "Second section" in messages[1]["content"] else 100
        ),
    )

    batches = mod.batch_sections(
        sections,
        pages,
        budget=150,
        doc_context="",
        toc_so_far="",
        prompt=_stub_prompt(),
    )

    assert len(batches) == 2
    assert batches[0][0]["section_id"] == "1"
    assert batches[1][0]["section_id"] == "2"


def test_update_sections_applies_data():
    """Summaries applied to matching sections."""
    result = _make_result(
        sections=[
            _make_section(section_id="1"),
            _make_section(section_id="2"),
        ]
    )
    summaries = {
        "1": {
            "summary": "Sum 1.",
            "keywords": ["a"],
            "entities": ["e"],
        },
    }

    mod.update_sections(result, summaries)

    assert result.sections[0]["summary"] == "Sum 1."
    assert result.sections[0]["keywords"] == ["a"]
    assert result.sections[1]["summary"] == ""


def test_find_page_for_unit_by_chunk_id():
    """Finds page by chunk_id match."""
    pages = [
        _make_page("data", page_number=5, chunk_id="5.1"),
    ]

    found = mod.find_page_for_unit("5.1", pages)

    assert found is not None
    assert found.chunk_id == "5.1"


def test_find_page_for_unit_by_page_number():
    """Finds page by page_number string match."""
    pages = [
        _make_page("data", page_number=3),
    ]

    found = mod.find_page_for_unit("3", pages)

    assert found is not None
    assert found.page_number == 3


def test_find_page_for_unit_not_found():
    """Returns None when no match."""
    pages = [_make_page("data", page_number=1)]

    found = mod.find_page_for_unit("99", pages)

    assert found is None


def test_section_is_xlsx_chunked_true():
    """Returns True when page has chunk_id."""
    pages = [
        _make_page("data", page_number=5, chunk_id="5.1"),
    ]
    section = _make_section(chunk_ids=["5.1"])

    assert mod.section_is_xlsx_chunked(section, pages)


def test_section_is_xlsx_chunked_false():
    """Returns False when no page has chunk_id."""
    pages = [_make_page("data", page_number=5)]
    section = _make_section(chunk_ids=["5"])

    assert not mod.section_is_xlsx_chunked(section, pages)


def test_build_doc_context_with_metadata():
    """Context includes title, source context, and structure type."""
    result = _make_result(
        metadata={
            "title": "Pillar 3 Report",
            "structure_type": "sheet_based",
        },
    )
    result.data_source = "pillar3"
    result.filter_1 = "2026"

    ctx = mod.build_doc_context(result)

    assert "Pillar 3 Report" in ctx
    assert 'data_source: "pillar3"' in ctx
    assert 'filter_1: "2026"' in ctx
    assert "sheet_based" in ctx
    assert "document_context" in ctx


def test_build_doc_context_empty():
    """Empty metadata returns empty string."""
    result = _make_result(
        metadata={},
        file_path="/doc.pdf",
    )

    ctx = mod.build_doc_context(result)

    assert ctx == ""


def test_generated_toc_entries_built_from_sections(monkeypatch):
    """Generated TOC entries are stored separately."""
    items = [
        [
            {
                "section_id": "1",
                "summary": "First.",
                "keywords": [],
                "entities": [],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    pages = [_make_page("data", page_number=1)]
    sections = [
        _make_section(
            section_id="1",
            title="Intro",
            page_start=1,
            chunk_ids=["1"],
        ),
    ]
    result = _make_result(
        pages=pages,
        sections=sections,
        metadata={
            "title": "Report",
            "toc_entries": [{"name": "Source TOC", "page_number": 1}],
            "source_toc_entries": [{"name": "Source TOC", "page_number": 1}],
        },
    )

    enriched = mod.summarize_sections(result, mocks["llm"])

    toc = enriched.document_metadata["generated_toc_entries"]
    assert len(toc) == 1
    assert toc[0]["section_id"] == "1"
    assert toc[0]["title"] == "Intro"
    assert toc[0]["summary"] == "First."
    assert enriched.document_metadata["source_toc_entries"] == [
        {"name": "Source TOC", "page_number": 1}
    ]


def test_estimate_section_tokens():
    """Token estimate sums matching page counts."""
    pages = [
        _make_page("a", page_number=1, raw_token_count=50),
        _make_page(
            "b",
            page_number=1,
            raw_token_count=75,
            chunk_id="1.1",
        ),
    ]
    section = _make_section(chunk_ids=["1", "1.1"])

    tokens = mod.estimate_section_tokens(section, pages)

    assert tokens == 125


def test_gather_section_content_missing_page():
    """Pages not found in chunk_ids are skipped."""
    pages = [
        _make_page("data", page_number=1),
    ]
    section = _make_section(
        section_id="1",
        title="Test",
        chunk_ids=["1", "99"],
    )

    content = mod.gather_section_content(section, pages)

    assert "data" in content
    assert "99" not in content


def test_build_progressive_toc_long_summary():
    """Summary exceeding 80 chars is truncated with ellipsis."""
    long_summary = "A" * 100
    completed = [
        {
            "section_id": "1",
            "title": "Long",
            "page_start": 1,
            "summary": long_summary,
        },
    ]

    toc = mod.build_progressive_toc(completed)

    assert "..." in toc
    assert "A" * 80 in toc
    assert "A" * 81 not in toc


def test_parse_summary_response_rejects_empty_section_id():
    """Empty section_id is invalid."""
    items = [
        {
            "section_id": "",
            "summary": "Skip me.",
            "keywords": [],
            "entities": [],
        },
        {
            "section_id": "1",
            "summary": "Keep me.",
            "keywords": [],
            "entities": [],
        },
    ]
    response = _make_llm_response(items)

    with pytest.raises(ValueError, match="valid section_id"):
        mod.parse_summary_response(response)


def test_summarize_sections_null_metadata(monkeypatch):
    """Sets document_metadata when it is None."""
    items = [
        [
            {
                "section_id": "1",
                "summary": "Sum.",
                "keywords": [],
                "entities": [],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    pages = [_make_page("data", page_number=1)]
    sections = [
        _make_section(
            section_id="1",
            chunk_ids=["1"],
        ),
    ]
    result = _make_result(pages=pages, sections=sections)
    result.document_metadata = None

    enriched = mod.summarize_sections(result, mocks["llm"])

    assert enriched.document_metadata is not None
    assert "generated_toc_entries" in enriched.document_metadata
