"""Tests for the markdown chunker module."""

import json
from unittest.mock import Mock

import pytest

from ingestion.stages.chunkers import markdown_chunker as md_chunker_module
from ingestion.utils.file_types import PageResult


def _make_page(
    content="line one\nline two\nline three",
    page_number=1,
    layout_type="standard",
    **overrides,
):
    """Build a PageResult with sensible defaults."""
    page = PageResult(
        page_number=page_number,
        raw_content=content,
        token_count=100,
        token_tier="high",
        layout_type=layout_type,
    )
    for key, value in overrides.items():
        setattr(page, key, value)
    return page


def _make_llm_response(breakpoints):
    """Build a mock LLM tool-call response dict."""
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps(
                                    {"breakpoints": (breakpoints)}
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }


# ---- _index_lines ----


def test_index_lines_numbers_correctly():
    """Lines are 1-indexed and paired with their text."""
    index_lines = md_chunker_module.index_lines
    result = index_lines("alpha\nbeta\ngamma")
    assert result == [
        (1, "alpha"),
        (2, "beta"),
        (3, "gamma"),
    ]


# ---- _format_batch ----


def test_format_batch_produces_numbered_lines():
    """Output uses [N] prefix format for each line."""
    format_batch = md_chunker_module.format_batch
    indexed = [(1, "hello"), (2, "world")]
    assert format_batch(indexed) == "[1] hello\n[2] world"


# ---- _parse_breakpoints_response ----


def test_parse_breakpoints_response_extracts_line_numbers():
    """Valid tool-call response is parsed into sorted ints."""
    parse = md_chunker_module.parse_breakpoints_response
    response = _make_llm_response([10, 5])
    assert parse(response) == [5, 10]


def test_parse_breakpoints_response_rejects_malformed():
    """Missing keys in the response raise ValueError."""
    parse = md_chunker_module.parse_breakpoints_response

    with pytest.raises(ValueError, match="no choices"):
        parse({})

    with pytest.raises(ValueError, match="no message"):
        parse({"choices": [{}]})

    with pytest.raises(ValueError, match="no tool_calls"):
        parse({"choices": [{"message": {}}]})

    with pytest.raises(ValueError, match="no arguments"):
        parse({"choices": [{"message": {"tool_calls": [{"function": {}}]}}]})


def test_parse_breakpoints_response_rejects_bad_json():
    """Non-JSON arguments string raises ValueError."""
    parse = md_chunker_module.parse_breakpoints_response
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"function": {"arguments": "not-json"}}]
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="Cannot parse"):
        parse(response)


def test_parse_breakpoints_response_rejects_missing_key():
    """Parsed JSON without 'breakpoints' key raises ValueError."""
    parse = md_chunker_module.parse_breakpoints_response
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps({"other": 1}),
                            }
                        }
                    ]
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="missing"):
        parse(response)


# ---- _collect_breakpoints ----


def test_collect_breakpoints_batches_correctly():
    """Lines are split into batches and LLM called per batch."""
    collect = md_chunker_module.collect_breakpoints
    index_lines = md_chunker_module.index_lines

    content = "\n".join(f"line {i}" for i in range(1, 11))
    indexed = index_lines(content)

    mock_llm = Mock()
    call_count = {"n": 0}

    def fake_call(**_kwargs):
        """Return breakpoints based on batch number."""
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _make_llm_response([3])
        return _make_llm_response([8])

    mock_llm.call = fake_call

    prompt = {
        "stage": "chunking",
        "system_prompt": "system",
        "user_prompt": "test {batch_content} {embedding_limit}",
        "tools": [],
        "tool_choice": "required",
    }

    result = collect(
        indexed_lines=indexed,
        llm=mock_llm,
        prompt=prompt,
        batch_size=5,
        embedding_limit=8192,
        context="test",
    )

    assert call_count["n"] == 2
    assert result == [3, 8]


def test_collect_breakpoints_handles_parse_failure():
    """Malformed LLM response logs warning, continues."""
    collect = md_chunker_module.collect_breakpoints
    index_lines = md_chunker_module.index_lines

    content = "\n".join(f"line {i}" for i in range(1, 6))
    indexed = index_lines(content)

    mock_llm = Mock()
    mock_llm.call.return_value = {"choices": [{}]}

    prompt = {
        "stage": "chunking",
        "system_prompt": "sys",
        "user_prompt": "{batch_content} {embedding_limit}",
        "tools": [],
        "tool_choice": "required",
    }

    result = collect(
        indexed_lines=indexed,
        llm=mock_llm,
        prompt=prompt,
        batch_size=100,
        embedding_limit=8192,
        context="test",
    )

    assert result == []


# ---- _assemble_chunks ----


def test_assemble_chunks_creates_correct_results():
    """Content is split at breakpoints with proper metadata."""
    assemble = md_chunker_module.assemble_chunks
    content = "A\nB\nC\nD\nE\nF"
    page = _make_page(content=content, page_number=2)

    chunks = assemble(content, [3, 5], page)

    assert len(chunks) == 3
    assert chunks[0].raw_content == "A\nB"
    assert chunks[0].chunk_id == "2.1"
    assert chunks[0].chunk_context == "Lines 1-2"
    assert chunks[1].raw_content == "C\nD"
    assert chunks[1].chunk_id == "2.2"
    assert chunks[1].chunk_context == "Lines 3-4"
    assert chunks[2].raw_content == "E\nF"
    assert chunks[2].chunk_id == "2.3"
    assert chunks[2].chunk_context == "Lines 5-6"


def test_assemble_chunks_single_breakpoint():
    """One breakpoint creates exactly 2 chunks."""
    assemble = md_chunker_module.assemble_chunks
    content = "A\nB\nC\nD"
    page = _make_page(content=content)

    chunks = assemble(content, [3], page)

    assert len(chunks) == 2
    assert chunks[0].raw_content == "A\nB"
    assert chunks[1].raw_content == "C\nD"


# ---- chunk_markdown_page ----


def test_chunk_markdown_page_splits_content(monkeypatch):
    """End-to-end: LLM breakpoints produce multiple chunks."""
    monkeypatch.setenv("CHUNKING_MD_BATCH_SIZE", "100")

    content = "\n".join(f"line {i}" for i in range(1, 11))
    page = _make_page(content=content, page_number=3)

    mock_llm = Mock()
    mock_llm.call.return_value = _make_llm_response([5])

    monkeypatch.setattr(
        md_chunker_module,
        "load_prompt",
        lambda name, prompts_dir: {
            "stage": "chunking",
            "system_prompt": "system",
            "user_prompt": ("{batch_content} {embedding_limit}"),
            "tools": [],
            "tool_choice": "required",
        },
    )

    chunks = md_chunker_module.chunk_markdown_page(
        page=page,
        llm=mock_llm,
        embedding_limit=8192,
    )

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "3.1"
    assert chunks[1].chunk_id == "3.2"
    assert chunks[0].parent_page_number == 3
    assert chunks[1].parent_page_number == 3


def test_chunk_markdown_page_no_breakpoints_single_chunk(
    monkeypatch,
):
    """When LLM returns no breakpoints a single chunk is produced."""
    monkeypatch.setenv("CHUNKING_MD_BATCH_SIZE", "100")

    content = "only\ntwo\nlines"
    page = _make_page(content=content, page_number=7)

    mock_llm = Mock()
    mock_llm.call.return_value = _make_llm_response([])

    monkeypatch.setattr(
        md_chunker_module,
        "load_prompt",
        lambda name, prompts_dir: {
            "stage": "chunking",
            "system_prompt": "system",
            "user_prompt": ("{batch_content} {embedding_limit}"),
            "tools": [],
            "tool_choice": "required",
        },
    )

    chunks = md_chunker_module.chunk_markdown_page(
        page=page,
        llm=mock_llm,
        embedding_limit=8192,
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "7.1"
    assert chunks[0].raw_content == content
    assert chunks[0].chunk_context == "Lines 1-3"


def test_chunk_markdown_page_preserves_layout_type(
    monkeypatch,
):
    """Layout type from the source page carries to all chunks."""
    monkeypatch.setenv("CHUNKING_MD_BATCH_SIZE", "100")

    content = "A\nB\nC\nD"
    page = _make_page(
        content=content,
        page_number=1,
        layout_type="dense_table",
    )

    mock_llm = Mock()
    mock_llm.call.return_value = _make_llm_response([3])

    monkeypatch.setattr(
        md_chunker_module,
        "load_prompt",
        lambda name, prompts_dir: {
            "stage": "chunking",
            "system_prompt": "system",
            "user_prompt": ("{batch_content} {embedding_limit}"),
            "tools": [],
            "tool_choice": "required",
        },
    )

    chunks = md_chunker_module.chunk_markdown_page(
        page=page,
        llm=mock_llm,
        embedding_limit=8192,
    )

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.layout_type == "dense_table"


def test_chunk_markdown_page_nests_ids_for_rechunked_page(monkeypatch):
    """Nested chunk IDs stay unique when a chunk is split again."""
    monkeypatch.setenv("CHUNKING_MD_BATCH_SIZE", "100")

    page = _make_page(
        content="A\nB\nC\nD",
        page_number=1,
        chunk_id="1.2",
        parent_page_number=1,
    )

    mock_llm = Mock()
    mock_llm.call.return_value = _make_llm_response([3])

    monkeypatch.setattr(
        md_chunker_module,
        "load_prompt",
        lambda name, prompts_dir: {
            "stage": "chunking",
            "system_prompt": "system",
            "user_prompt": ("{batch_content} {embedding_limit}"),
            "tools": [],
            "tool_choice": "required",
        },
    )

    chunks = md_chunker_module.chunk_markdown_page(
        page=page,
        llm=mock_llm,
        embedding_limit=8192,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["1.2.1", "1.2.2"]
    assert all(chunk.parent_page_number == 1 for chunk in chunks)
