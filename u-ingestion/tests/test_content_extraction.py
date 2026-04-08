"""Tests for the content_extraction enrichment module."""

import json

import pytest

from ingestion.stages.enrichment import (
    content_extraction as mod,
)
from ingestion.utils import llm_retry
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
        "stage": "content_extraction",
        "version": "1.0",
        "description": "test",
        "system_prompt": "You are a test agent.",
        "user_prompt": ("Analyze:\n\n{user_input}\n\nRules:\n1. Do it."),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "extract_content_metadata",
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
    max_retries=1,
):
    """Patch load_prompt and config for all tests.

    Params: monkeypatch, items_per_call, budget, max_retries.
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
        "get_content_extraction_batch_budget",
        lambda: budget,
    )
    monkeypatch.setattr(
        mod,
        "get_content_extraction_max_retries",
        lambda: max_retries,
    )
    monkeypatch.setattr(
        mod,
        "get_content_extraction_retry_delay",
        lambda: 0.0,
    )
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        mod,
        "count_message_tokens",
        lambda messages, tools=None: messages[1]["content"].count("<unit ")
        * 100,
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
# test_extract_content_sets_keywords_on_pages
# ------------------------------------------------------------------


def test_extract_content_sets_keywords_on_pages(
    monkeypatch,
):
    """Keywords populated on each page after extraction."""
    items = [
        [
            {
                "unit_id": "1",
                "keywords": ["revenue", "Q4"],
                "entities": ["BMO"],
            },
            {
                "unit_id": "2",
                "keywords": ["CET1", "ratio"],
                "entities": ["OSFI"],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    pages = [
        _make_page("Revenue data", page_number=1),
        _make_page("Capital ratios", page_number=2),
    ]
    result = _make_result(pages=pages)

    enriched = mod.extract_content(result, mocks["llm"])

    assert enriched.pages[0].keywords == ["revenue", "Q4"]
    assert enriched.pages[1].keywords == ["CET1", "ratio"]


# ------------------------------------------------------------------
# test_extract_content_sets_entities_on_pages
# ------------------------------------------------------------------


def test_extract_content_sets_entities_on_pages(
    monkeypatch,
):
    """Entities populated on each page after extraction."""
    items = [
        [
            {
                "unit_id": "1",
                "keywords": ["revenue"],
                "entities": ["BMO", "Toronto"],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    result = _make_result()

    enriched = mod.extract_content(result, mocks["llm"])

    assert enriched.pages[0].entities == ["BMO", "Toronto"]


# ------------------------------------------------------------------
# test_extract_content_uses_raw_content
# ------------------------------------------------------------------


def test_extract_content_uses_raw_content(monkeypatch):
    """Verifies raw_content is sent to LLM, not assembled."""
    items = [
        [
            {
                "unit_id": "1",
                "keywords": ["test"],
                "entities": [],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    page = _make_page(
        raw_content="RAW_CONTENT_MARKER",
        page_number=1,
    )
    result = _make_result(pages=[page])

    mod.extract_content(result, mocks["llm"])

    user_msg = mocks["call_log"][0]["messages"][1]["content"]
    assert "RAW_CONTENT_MARKER" in user_msg


# ------------------------------------------------------------------
# test_extract_content_batches_by_budget
# ------------------------------------------------------------------


def test_extract_content_batches_by_budget(monkeypatch):
    """Multiple batches when units exceed budget."""
    items = [
        [
            {
                "unit_id": "1",
                "keywords": ["a"],
                "entities": [],
            },
        ],
        [
            {
                "unit_id": "2",
                "keywords": ["b"],
                "entities": [],
            },
        ],
    ]
    mocks = _patch_dependencies(
        monkeypatch,
        items_per_call=items,
        budget=150,
    )
    pages = [
        _make_page(
            "Page one",
            page_number=1,
            raw_token_count=100,
        ),
        _make_page(
            "Page two",
            page_number=2,
            raw_token_count=100,
        ),
    ]
    result = _make_result(pages=pages)

    enriched = mod.extract_content(result, mocks["llm"])

    assert len(mocks["call_log"]) == 2
    assert enriched.pages[0].keywords == ["a"]
    assert enriched.pages[1].keywords == ["b"]


# ------------------------------------------------------------------
# test_extract_content_includes_document_context
# ------------------------------------------------------------------


def test_extract_content_includes_document_context(
    monkeypatch,
):
    """Document title and data_source appear in LLM input."""
    items = [
        [
            {
                "unit_id": "1",
                "keywords": [],
                "entities": [],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    result = _make_result(
        metadata={"title": "Pillar 3 Report"},
        file_path="/tmp/pillar3/2026/doc.pdf",
    )

    mod.extract_content(result, mocks["llm"])

    user_msg = mocks["call_log"][0]["messages"][1]["content"]
    assert "Pillar 3 Report" in user_msg
    assert "document_context" in user_msg


# ------------------------------------------------------------------
# test_extract_content_includes_chunk_context
# ------------------------------------------------------------------


def test_extract_content_includes_chunk_context(
    monkeypatch,
):
    """Chunks include chunk_context label in LLM input."""
    items = [
        [
            {
                "unit_id": "1.1",
                "keywords": ["chunk"],
                "entities": [],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    page = _make_page(
        "Chunk data",
        page_number=1,
        chunk_id="1.1",
        chunk_context="Rows 10-19",
    )
    result = _make_result(pages=[page])

    mod.extract_content(result, mocks["llm"])

    user_msg = mocks["call_log"][0]["messages"][1]["content"]
    assert "Rows 10-19" in user_msg


# ------------------------------------------------------------------
# test_build_unit_list
# ------------------------------------------------------------------


def test_build_unit_list_unchunked_and_chunked():
    """Correct unit_ids for unchunked and chunked pages."""
    pages = [
        _make_page("Page data", page_number=1),
        _make_page(
            "Chunk data",
            page_number=2,
            chunk_id="2.1",
        ),
    ]
    result = _make_result(pages=pages)

    units = mod.build_unit_list(result)

    assert len(units) == 2
    assert units[0]["unit_id"] == "1"
    assert units[1]["unit_id"] == "2.1"


# ------------------------------------------------------------------
# test_get_unit_name_xlsx
# ------------------------------------------------------------------


def test_get_unit_name_xlsx():
    """Extracts sheet name from XLSX header."""
    page = _make_page("# Sheet: KM1\ndata")

    name = mod.get_unit_name(page)

    assert name == "KM1"


# ------------------------------------------------------------------
# test_get_unit_name_pdf
# ------------------------------------------------------------------


def test_get_unit_name_pdf():
    """Extracts heading from PDF markdown heading."""
    page = _make_page("# Revenue Analysis\ntext")

    name = mod.get_unit_name(page)

    assert name == "Revenue Analysis"


# ------------------------------------------------------------------
# test_batch_units_splits_correctly
# ------------------------------------------------------------------


def test_batch_units_splits_correctly():
    """Budget boundary creates separate batches."""
    units = [
        {"unit_id": "1", "raw_token_count": 100},
        {"unit_id": "2", "raw_token_count": 100},
        {"unit_id": "3", "raw_token_count": 100},
    ]

    batches = mod.batch_units(units, budget=250)

    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1


def test_batch_units_empty():
    """Empty units list returns empty batches."""
    batches = mod.batch_units([], budget=1000)

    assert not batches


def test_batch_units_single_over_budget():
    """Single unit exceeding budget gets its own batch."""
    units = [
        {"unit_id": "1", "raw_token_count": 200},
    ]

    batches = mod.batch_units(units, budget=100)

    assert len(batches) == 1
    assert batches[0][0]["unit_id"] == "1"


def test_batch_units_counts_formatted_request_budget(monkeypatch):
    """Prompt-aware batching uses rendered request size."""
    units = [
        {
            "unit_id": "1",
            "page_number": 1,
            "name": "",
            "raw_content": "Page one",
            "raw_token_count": 10,
            "context": "",
            "section_id": "",
        },
        {
            "unit_id": "2",
            "page_number": 2,
            "name": "",
            "raw_content": "Page two",
            "raw_token_count": 10,
            "context": "",
            "section_id": "",
        },
    ]
    monkeypatch.setattr(
        mod,
        "count_message_tokens",
        lambda messages, tools=None: (
            200 if "Page two" in messages[1]["content"] else 100
        ),
    )

    batches = mod.batch_units(
        units,
        budget=150,
        doc_context={"title": "", "data_source": ""},
        prompt=_stub_prompt(),
    )

    assert len(batches) == 2
    assert batches[0][0]["unit_id"] == "1"
    assert batches[1][0]["unit_id"] == "2"


# ------------------------------------------------------------------
# test_parse_extraction_response_valid
# ------------------------------------------------------------------


def test_parse_extraction_response_valid():
    """Valid response parsed into unit_id map."""
    items = [
        {
            "unit_id": "1",
            "keywords": ["revenue"],
            "entities": ["BMO"],
        },
        {
            "unit_id": "2",
            "keywords": ["CET1"],
            "entities": [],
        },
    ]
    response = _make_llm_response(items)

    result = mod.parse_extraction_response(response)

    assert "1" in result
    assert result["1"]["keywords"] == ["revenue"]
    assert result["1"]["entities"] == ["BMO"]
    assert "2" in result
    assert result["2"]["keywords"] == ["CET1"]


# ------------------------------------------------------------------
# test_parse_extraction_response_malformed
# ------------------------------------------------------------------


def test_parse_extraction_response_no_choices():
    """Raises ValueError when response has no choices."""
    with pytest.raises(ValueError, match="no choices"):
        mod.parse_extraction_response({"choices": []})


def test_parse_extraction_response_no_tool_calls():
    """Raises ValueError when no tool calls present."""
    response = {
        "choices": [{"message": {"tool_calls": None}}],
    }
    with pytest.raises(ValueError, match="no tool calls"):
        mod.parse_extraction_response(response)


def test_parse_extraction_response_bad_json():
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
        mod.parse_extraction_response(response)


def test_parse_extraction_response_rejects_empty_unit_id():
    """Empty unit_id is invalid."""
    items = [
        {
            "unit_id": "",
            "keywords": ["skip"],
            "entities": [],
        },
        {
            "unit_id": "1",
            "keywords": ["keep"],
            "entities": [],
        },
    ]
    response = _make_llm_response(items)

    with pytest.raises(ValueError, match="valid unit_id"):
        mod.parse_extraction_response(response)


def test_parse_extraction_response_missing_items():
    """Raises ValueError when items field is missing."""
    args = json.dumps({"rationale": "test"})
    response = {
        "choices": [
            {"message": {"tool_calls": [{"function": {"arguments": args}}]}}
        ],
    }
    with pytest.raises(ValueError, match="items"):
        mod.parse_extraction_response(response)


def test_parse_extraction_response_duplicate_unit_id():
    """Duplicate unit ids are rejected."""
    items = [
        {
            "unit_id": "1",
            "keywords": ["a"],
            "entities": [],
        },
        {
            "unit_id": "1",
            "keywords": ["b"],
            "entities": [],
        },
    ]
    response = _make_llm_response(items)

    with pytest.raises(ValueError, match="duplicate unit_id"):
        mod.parse_extraction_response(response)


def test_parse_extraction_response_rejects_non_object_item():
    """Each response item must be an object."""
    response = _make_llm_response(["bad-item"])

    with pytest.raises(ValueError, match="must be an object"):
        mod.parse_extraction_response(response)


def test_parse_extraction_response_rejects_invalid_keywords():
    """Keywords must be a list of strings."""
    items = [
        {
            "unit_id": "1",
            "keywords": ["valid", 2],
            "entities": [],
        },
    ]
    response = _make_llm_response(items)

    with pytest.raises(ValueError, match="invalid keywords"):
        mod.parse_extraction_response(response)


def test_parse_extraction_response_rejects_invalid_entities():
    """Entities must be a list of strings."""
    items = [
        {
            "unit_id": "1",
            "keywords": ["valid"],
            "entities": ["ok", 2],
        },
    ]
    response = _make_llm_response(items)

    with pytest.raises(ValueError, match="invalid entities"):
        mod.parse_extraction_response(response)


def test_validate_batch_results_requires_exact_unit_ids():
    """Batch validation rejects missing or extra ids."""
    batch = [
        {"unit_id": "1"},
        {"unit_id": "2"},
    ]

    with pytest.raises(ValueError, match="missing unit_ids"):
        mod.validate_batch_results(
            batch, {"1": {"keywords": [], "entities": []}}
        )

    with pytest.raises(ValueError, match="unexpected unit_ids"):
        mod.validate_batch_results(
            batch,
            {
                "1": {"keywords": [], "entities": []},
                "2": {"keywords": [], "entities": []},
                "3": {"keywords": [], "entities": []},
            },
        )


# ------------------------------------------------------------------
# test_extract_content_preserves_existing_fields
# ------------------------------------------------------------------


def test_extract_content_preserves_existing_fields(
    monkeypatch,
):
    """Other PageResult fields unchanged after extraction."""
    items = [
        [
            {
                "unit_id": "1",
                "keywords": ["test"],
                "entities": ["Org"],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    page = _make_page(
        "Content here",
        page_number=1,
        layout_type="text",
        section_id="1",
    )
    result = _make_result(pages=[page])

    enriched = mod.extract_content(result, mocks["llm"])

    assert enriched.pages[0].layout_type == "text"
    assert enriched.pages[0].section_id == "1"
    assert enriched.pages[0].raw_content == "Content here"
    assert enriched.pages[0].page_number == 1


# ------------------------------------------------------------------
# test_extract_content_empty_document
# ------------------------------------------------------------------


def test_extract_content_empty_document(monkeypatch):
    """No pages means no LLM calls."""
    mocks = _patch_dependencies(monkeypatch)
    result = _make_result(pages=[])

    enriched = mod.extract_content(result, mocks["llm"])

    assert len(mocks["call_log"]) == 0
    assert not enriched.pages


# ------------------------------------------------------------------
# get_unit_name edge cases
# ------------------------------------------------------------------


def test_get_unit_name_no_heading():
    """Returns empty string when no heading found."""
    page = _make_page("Plain text without heading")

    name = mod.get_unit_name(page)

    assert name == ""


# ------------------------------------------------------------------
# format_batch
# ------------------------------------------------------------------


def test_format_batch_includes_unit_ids():
    """Formatted batch includes unit id attributes."""
    batch = [
        {
            "unit_id": "3",
            "page_number": 3,
            "name": "KM1",
            "raw_content": "data here",
            "raw_token_count": 100,
            "context": "",
            "section_id": "",
        },
    ]
    doc_ctx = {
        "title": "Report",
        "data_source": "pillar3",
        "filter_1": "2026",
        "filter_2": "",
        "filter_3": "",
    }
    prompt = _stub_prompt()

    output = mod.format_batch(batch, doc_ctx, prompt)

    assert 'id="3"' in output
    assert 'page="3"' in output
    assert 'name="KM1"' in output
    assert "data here" in output
    assert 'filter_1: "2026"' in output


def test_format_batch_includes_context_attr():
    """Context attribute appears when set."""
    batch = [
        {
            "unit_id": "24.2",
            "page_number": 24,
            "name": "CR6_A",
            "raw_content": "rows",
            "raw_token_count": 50,
            "context": "Rows 10-19",
            "section_id": "",
        },
    ]
    doc_ctx = {"title": "", "data_source": ""}
    prompt = _stub_prompt()

    output = mod.format_batch(batch, doc_ctx, prompt)

    assert 'context="Rows 10-19"' in output


def test_format_batch_no_doc_context():
    """Batch with empty doc context omits context block."""
    batch = [
        {
            "unit_id": "1",
            "page_number": 1,
            "name": "",
            "raw_content": "data",
            "raw_token_count": 50,
            "context": "",
            "section_id": "",
        },
    ]
    doc_ctx = {"title": "", "data_source": ""}
    prompt = _stub_prompt()

    output = mod.format_batch(batch, doc_ctx, prompt)

    assert "document_context" not in output
    assert "content_units" in output


# ------------------------------------------------------------------
# build_doc_context
# ------------------------------------------------------------------


def test_build_doc_context_extracts_data_source():
    """Data source and filters come from stable source context."""
    result = _make_result(
        metadata={"title": "Test Title"},
    )
    result.data_source = "pillar3"
    result.filter_1 = "2026"

    ctx = mod.build_doc_context(result)

    assert ctx["title"] == "Test Title"
    assert ctx["data_source"] == "pillar3"
    assert ctx["filter_1"] == "2026"


def test_build_doc_context_no_metadata():
    """Works when document_metadata is empty."""
    result = _make_result(metadata={})

    ctx = mod.build_doc_context(result)

    assert ctx["title"] == ""


# ------------------------------------------------------------------
# apply_to_pages
# ------------------------------------------------------------------


def test_apply_to_pages_sets_values():
    """Keywords and entities applied to matching pages."""
    pages = [
        _make_page("A", page_number=1),
        _make_page("B", page_number=2, chunk_id="2.1"),
    ]
    result = _make_result(pages=pages)
    extractions = {
        "1": {
            "keywords": ["alpha"],
            "entities": ["Org1"],
        },
        "2.1": {
            "keywords": ["beta"],
            "entities": ["Org2"],
        },
    }

    mod.apply_to_pages(result, extractions)

    assert result.pages[0].keywords == ["alpha"]
    assert result.pages[0].entities == ["Org1"]
    assert result.pages[1].keywords == ["beta"]
    assert result.pages[1].entities == ["Org2"]


def test_apply_to_pages_no_match():
    """Pages without matching extraction keep defaults."""
    pages = [_make_page("A", page_number=1)]
    result = _make_result(pages=pages)

    mod.apply_to_pages(result, {})

    assert result.pages[0].keywords == []
    assert result.pages[0].entities == []


def test_extract_content_materializes_content_units(monkeypatch):
    """Stage output includes normalized content-unit records."""
    items = [
        [
            {
                "unit_id": "1.1",
                "keywords": ["pd"],
                "entities": ["OSFI"],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    page = _make_page(
        "Chunk data",
        page_number=1,
        chunk_id="1.1",
        section_id="9.2",
        chunk_context="Rows 10-19",
    )
    result = _make_result(pages=[page])

    enriched = mod.extract_content(result, mocks["llm"])

    assert enriched.content_units == [
        {
            "content_unit_id": "1.1",
            "chunk_id": "1.1",
            "section_id": "9.2",
            "page_number": 1,
            "parent_page_number": 0,
            "raw_content": "Chunk data",
            "chunk_context": "Rows 10-19",
            "chunk_header": "",
            "sheet_passthrough_content": "",
            "section_passthrough_content": "",
            "keywords": ["pd"],
            "entities": ["OSFI"],
            "raw_token_count": 100,
            "embedding_token_count": 0,
            "token_count": 0,
        }
    ]


def test_extract_content_rejects_missing_batch_results(monkeypatch):
    """Stage fails when the LLM omits requested units."""
    items = [
        [
            {
                "unit_id": "1",
                "keywords": ["alpha"],
                "entities": [],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    result = _make_result(
        pages=[
            _make_page("Page one", page_number=1),
            _make_page("Page two", page_number=2),
        ]
    )

    with pytest.raises(ValueError, match="missing unit_ids"):
        mod.extract_content(result, mocks["llm"])


def test_extract_content_rejects_unexpected_batch_results(monkeypatch):
    """Stage fails when the LLM returns unit ids not in the batch."""
    items = [
        [
            {
                "unit_id": "1",
                "keywords": ["alpha"],
                "entities": [],
            },
            {
                "unit_id": "99",
                "keywords": ["beta"],
                "entities": [],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    result = _make_result(pages=[_make_page("Page one", page_number=1)])

    with pytest.raises(ValueError, match="unexpected unit_ids"):
        mod.extract_content(result, mocks["llm"])


# ------------------------------------------------------------------
# extract_content stage and context
# ------------------------------------------------------------------


def test_extract_content_calls_with_correct_stage(
    monkeypatch,
):
    """LLM is called with stage='content_extraction'."""
    items = [
        [
            {
                "unit_id": "1",
                "keywords": [],
                "entities": [],
            },
        ]
    ]
    mocks = _patch_dependencies(monkeypatch, items_per_call=items)
    result = _make_result()

    mod.extract_content(result, mocks["llm"])

    assert mocks["call_log"][0]["stage"] == ("content_extraction")


# ------------------------------------------------------------------
# Retry behavior on structural LLM failures
# ------------------------------------------------------------------


def _bad_response(finish_reason="", content=""):
    """Build a response with no tool_calls for retry tests."""
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"tool_calls": None, "content": content},
            }
        ]
    }


def test_parse_extraction_response_missing_tool_calls_includes_diagnostics():
    """Enriched missing-tool-calls error includes finish_reason and preview."""
    response = _bad_response(
        finish_reason="length",
        content="truncated thinking...",
    )

    with pytest.raises(ValueError) as exc_info:
        mod.parse_extraction_response(response)

    message = str(exc_info.value)
    assert "no tool calls" in message
    assert "finish_reason=length" in message
    assert "truncated thinking" in message


def test_extract_content_retries_on_missing_tool_calls(monkeypatch):
    """Retry when the LLM drops tool_calls, then succeed on next attempt."""
    good_items = [
        {
            "unit_id": "1",
            "keywords": ["alpha"],
            "entities": [],
        },
    ]

    responses = [
        _bad_response(finish_reason="length", content="oops"),
        _make_llm_response(good_items),
    ]
    call_count = [0]

    def fake_call(**_kwargs):
        idx = call_count[0]
        call_count[0] += 1
        return responses[idx]

    prompt = _stub_prompt()
    monkeypatch.setattr(
        mod, "load_prompt", lambda name, prompts_dir=None: prompt
    )
    monkeypatch.setattr(
        mod, "get_content_extraction_batch_budget", lambda: 50000
    )
    monkeypatch.setattr(mod, "get_content_extraction_max_retries", lambda: 2)
    monkeypatch.setattr(mod, "get_content_extraction_retry_delay", lambda: 0.0)
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        mod,
        "count_message_tokens",
        lambda messages, tools=None: messages[1]["content"].count("<unit ")
        * 100,
    )

    llm = type(
        "FakeLLM",
        (),
        {"call": staticmethod(fake_call), "__doc__": "Fake."},
    )()
    result = _make_result(pages=[_make_page("Page one", page_number=1)])

    enriched = mod.extract_content(result, llm)

    assert call_count[0] == 2
    assert enriched.pages[0].keywords == ["alpha"]


def test_extract_content_retries_on_duplicate_unit_id(monkeypatch):
    """Retry when the LLM emits duplicate unit_ids, then succeed."""
    bad_items = [
        {"unit_id": "1", "keywords": ["a"], "entities": []},
        {"unit_id": "1", "keywords": ["b"], "entities": []},
    ]
    good_items = [
        {"unit_id": "1", "keywords": ["ok"], "entities": []},
    ]

    responses = [
        _make_llm_response(bad_items),
        _make_llm_response(good_items),
    ]
    call_count = [0]

    def fake_call(**_kwargs):
        idx = call_count[0]
        call_count[0] += 1
        return responses[idx]

    prompt = _stub_prompt()
    monkeypatch.setattr(
        mod, "load_prompt", lambda name, prompts_dir=None: prompt
    )
    monkeypatch.setattr(
        mod, "get_content_extraction_batch_budget", lambda: 50000
    )
    monkeypatch.setattr(mod, "get_content_extraction_max_retries", lambda: 2)
    monkeypatch.setattr(mod, "get_content_extraction_retry_delay", lambda: 0.0)
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        mod,
        "count_message_tokens",
        lambda messages, tools=None: messages[1]["content"].count("<unit ")
        * 100,
    )

    llm = type(
        "FakeLLM",
        (),
        {"call": staticmethod(fake_call), "__doc__": "Fake."},
    )()
    result = _make_result(pages=[_make_page("Page one", page_number=1)])

    enriched = mod.extract_content(result, llm)

    assert call_count[0] == 2
    assert enriched.pages[0].keywords == ["ok"]


def test_extract_content_raises_after_exhausted_retries(monkeypatch):
    """Raise ValueError when every retry returns the same bad response."""
    bad_items = [
        {"unit_id": "1", "keywords": ["a"], "entities": []},
        {"unit_id": "1", "keywords": ["b"], "entities": []},
    ]
    mocks = _patch_dependencies(
        monkeypatch,
        items_per_call=[bad_items],
        max_retries=2,
    )
    result = _make_result(pages=[_make_page("Page one", page_number=1)])

    with pytest.raises(ValueError, match="duplicate unit_id"):
        mod.extract_content(result, mocks["llm"])
    assert len(mocks["call_log"]) == 2
