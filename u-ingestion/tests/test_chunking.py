"""Tests for the chunking stage orchestrator."""

import logging

import pytest

from ingestion.stages import chunking as chunking_module
from ingestion.stages.chunking import chunk_result
from ingestion.utils.file_types import (
    ExtractionResult,
    PageResult,
)


def _make_page(
    content="x" * 100,
    page_number=1,
    token_count=100,
    layout_type="standard",
):
    """Build a PageResult with sensible defaults."""
    return PageResult(
        page_number=page_number,
        raw_content=content,
        token_count=token_count,
        token_tier="low",
        layout_type=layout_type,
    )


def _make_result(pages, filetype="pdf"):
    """Build an ExtractionResult wrapping the given pages."""
    return ExtractionResult(
        file_path="/tmp/doc.pdf",
        filetype=filetype,
        pages=pages,
        total_pages=len(pages),
        document_token_count=sum(p.token_count for p in pages),
    )


class _FakeEncoder:
    """Fake tiktoken encoder for deterministic tests."""

    def encode(self, text):
        """Encode text as list of chars."""
        return list(text)

    def decode(self, tokens):
        """Decode token list back to string."""
        return "".join(tokens)


def _patch_config(monkeypatch, overrides=None):
    """Patch config functions on chunking_module."""
    defaults = {
        "embedding_limit": 50,
        "truncation_limit": 200,
        "max_retries": 2,
    }
    if overrides:
        defaults.update(overrides)

    monkeypatch.setattr(
        chunking_module,
        "get_chunking_embedding_token_limit",
        lambda: defaults["embedding_limit"],
    )
    monkeypatch.setattr(
        chunking_module,
        "get_chunking_truncation_token_limit",
        lambda: defaults["truncation_limit"],
    )
    monkeypatch.setattr(
        chunking_module,
        "get_chunking_max_retries",
        lambda: defaults["max_retries"],
    )
    monkeypatch.setattr(
        chunking_module,
        "get_tokenizer_model",
        lambda: "test-model",
    )
    monkeypatch.setattr(
        chunking_module,
        "_get_encoder",
        lambda model: _FakeEncoder(),
    )


def _patch_chunkers(monkeypatch, md_fn=None, xlsx_fn=None):
    """Patch chunker functions on chunking_module."""
    if md_fn is not None:
        monkeypatch.setattr(
            chunking_module,
            "chunk_markdown_page",
            md_fn,
        )
    if xlsx_fn is not None:
        monkeypatch.setattr(
            chunking_module,
            "chunk_xlsx_page",
            xlsx_fn,
        )


def _default_md_chunker(page, _llm, _embedding_limit):
    """Return two small chunks for any page."""
    half = len(page.raw_content) // 2
    return [
        _make_page(
            content=page.raw_content[:half],
            page_number=page.page_number,
            token_count=0,
        ),
        _make_page(
            content=page.raw_content[half:],
            page_number=page.page_number,
            token_count=0,
        ),
    ]


def _default_xlsx_chunker(page, _llm, _embedding_limit):
    """Return two small chunks for any xlsx page."""
    half = len(page.raw_content) // 2
    return [
        _make_page(
            content=page.raw_content[:half],
            page_number=page.page_number,
            token_count=0,
        ),
        _make_page(
            content=page.raw_content[half:],
            page_number=page.page_number,
            token_count=0,
        ),
    ]


def test_chunk_result_passes_through_small_pages(
    monkeypatch,
):
    """Pages under the embedding limit pass through unchanged."""
    _patch_config(monkeypatch)
    _patch_chunkers(
        monkeypatch,
        md_fn=_default_md_chunker,
    )

    page = _make_page(
        content="small",
        token_count=10,
    )
    result = _make_result([page])
    out = chunk_result(result, None)

    assert len(out.pages) == 1
    assert out.pages[0].raw_content == "small"
    assert out.pages[0].raw_token_count == 10
    assert out.pages[0].embedding_token_count == 10
    assert out.pages[0].token_count == 10
    assert out.raw_document_token_count == 10
    assert out.embedding_document_token_count == 10


def test_chunk_result_truncates_oversized_content(
    monkeypatch,
    caplog,
):
    """Content over the truncation limit gets truncated."""
    _patch_config(
        monkeypatch,
        overrides={
            "embedding_limit": 50,
            "truncation_limit": 80,
        },
    )
    _patch_chunkers(
        monkeypatch,
        md_fn=_default_md_chunker,
    )

    big_content = "a" * 300
    page = _make_page(
        content=big_content,
        token_count=300,
    )
    result = _make_result([page])

    with caplog.at_level(logging.WARNING):
        out = chunk_result(result, None)

    assert "truncated" in caplog.text.lower()
    total = sum(p.token_count for p in out.pages)
    assert total <= 80


def test_chunk_result_routes_to_markdown_chunker(
    monkeypatch,
):
    """PDF pages route to the markdown chunker."""
    _patch_config(monkeypatch)

    md_called = []
    xlsx_called = []

    def mock_md(page, _llm, _limit):
        md_called.append(page.page_number)
        return _default_md_chunker(page, _llm, _limit)

    def mock_xlsx(page, _llm, _limit):
        xlsx_called.append(page.page_number)
        return _default_xlsx_chunker(page, _llm, _limit)

    _patch_chunkers(
        monkeypatch,
        md_fn=mock_md,
        xlsx_fn=mock_xlsx,
    )

    page = _make_page(
        content="x" * 100,
        token_count=100,
    )
    result = _make_result([page], filetype="pdf")
    chunk_result(result, None)

    assert len(md_called) == 1
    assert len(xlsx_called) == 0


def test_chunk_result_routes_to_xlsx_chunker(
    monkeypatch,
):
    """XLSX pages route to the xlsx chunker."""
    _patch_config(monkeypatch)

    md_called = []
    xlsx_called = []

    def mock_md(page, _llm, _limit):
        md_called.append(page.page_number)
        return _default_md_chunker(page, _llm, _limit)

    def mock_xlsx(page, _llm, _limit):
        xlsx_called.append(page.page_number)
        return _default_xlsx_chunker(page, _llm, _limit)

    _patch_chunkers(
        monkeypatch,
        md_fn=mock_md,
        xlsx_fn=mock_xlsx,
    )

    page = _make_page(
        content="x" * 100,
        token_count=100,
    )
    result = _make_result([page], filetype="xlsx")
    chunk_result(result, None)

    assert len(xlsx_called) == 1
    assert len(md_called) == 0


def test_chunk_result_recounts_tokens_after_chunking(
    monkeypatch,
):
    """Chunks get token_count and token_tier set correctly."""
    _patch_config(monkeypatch)
    _patch_chunkers(
        monkeypatch,
        md_fn=_default_md_chunker,
    )

    page = _make_page(
        content="x" * 100,
        token_count=100,
    )
    result = _make_result([page])
    out = chunk_result(result, None)

    for chunk in out.pages:
        assert chunk.raw_token_count > 0
        assert chunk.embedding_token_count >= chunk.raw_token_count
        assert chunk.token_count > 0
        assert chunk.token_tier in ("low", "medium", "high")


def test_chunk_result_retries_on_oversized_chunks(
    monkeypatch,
):
    """Re-chunks oversized chunks up to max_retries."""
    _patch_config(
        monkeypatch,
        overrides={"max_retries": 3},
    )

    call_count = []

    def retry_chunker(page, _llm, _limit):
        call_count.append(1)
        content = page.raw_content
        if len(content) > 50:
            half = len(content) // 2
            return [
                _make_page(
                    content=content[:half],
                    page_number=page.page_number,
                    token_count=0,
                ),
                _make_page(
                    content=content[half:],
                    page_number=page.page_number,
                    token_count=0,
                ),
            ]
        return [
            _make_page(
                content=content,
                page_number=page.page_number,
                token_count=0,
            )
        ]

    _patch_chunkers(monkeypatch, md_fn=retry_chunker)

    page = _make_page(
        content="y" * 200,
        page_number=1,
        token_count=200,
    )
    result = _make_result([page])
    out = chunk_result(result, None)

    assert len(call_count) > 1
    for chunk in out.pages:
        assert chunk.token_count <= 50


def test_chunk_result_uses_tokenizer_model_for_encoder(monkeypatch):
    """Chunking should derive its encoder from TOKENIZER_MODEL."""
    seen = []
    _patch_config(monkeypatch)
    monkeypatch.setattr(
        chunking_module,
        "get_tokenizer_model",
        lambda: "text-embedding-3-small",
    )
    monkeypatch.setattr(
        chunking_module,
        "_get_encoder",
        lambda model: seen.append(model) or _FakeEncoder(),
    )
    _patch_chunkers(monkeypatch, md_fn=_default_md_chunker)

    result = _make_result([_make_page(content="small", token_count=10)])
    chunk_result(result, None)

    assert seen == ["text-embedding-3-small"]


def test_chunk_result_raises_after_max_retries(
    monkeypatch,
):
    """RuntimeError raised after exhausting retries."""
    _patch_config(
        monkeypatch,
        overrides={"max_retries": 2},
    )

    def stubborn_chunker(page, _llm, _limit):
        return [
            _make_page(
                content=page.raw_content,
                page_number=page.page_number,
                token_count=0,
            )
        ]

    _patch_chunkers(monkeypatch, md_fn=stubborn_chunker)

    page = _make_page(
        content="z" * 100,
        token_count=100,
    )
    result = _make_result([page])

    with pytest.raises(RuntimeError, match="still exceed"):
        chunk_result(result, None)


def test_chunk_result_empty_document(monkeypatch):
    """Empty document returns unchanged."""
    _patch_config(monkeypatch)

    result = _make_result([])
    out = chunk_result(result, None)

    assert len(out.pages) == 0
    assert out.document_token_count == 0
    assert out.total_pages == 0


def test_chunk_result_mixed_small_and_large(
    monkeypatch,
):
    """Mix of pass-through and chunked pages."""
    _patch_config(monkeypatch)
    _patch_chunkers(
        monkeypatch,
        md_fn=_default_md_chunker,
    )

    small = _make_page(
        content="tiny",
        page_number=1,
        token_count=10,
    )
    large = _make_page(
        content="b" * 100,
        page_number=2,
        token_count=100,
    )
    result = _make_result([small, large])
    out = chunk_result(result, None)

    assert len(out.pages) == 3
    assert out.pages[0].raw_content == "tiny"
    assert out.total_pages == 2


def test_select_chunker_routes_correctly():
    """Correct chunker type for each file type."""
    assert chunking_module.select_chunker("xlsx") == "xlsx"
    assert chunking_module.select_chunker("pdf") == "markdown"
    assert chunking_module.select_chunker("docx") == "markdown"
    assert chunking_module.select_chunker("pptx") == "markdown"


def test_get_encoder_falls_back():
    """Unknown model falls back to o200k_base encoding."""
    encoder = chunking_module.get_encoder("unknown-model-xyz")
    assert encoder is not None
    assert len(encoder.encode("hello")) > 0


def test_count_tokens_positive():
    """Returns positive int for non-empty text."""
    encoder = chunking_module.get_encoder("gpt-4o")
    count = chunking_module.count_tokens("hello world", encoder)
    assert isinstance(count, int)
    assert count > 0


def test_truncate_content_within_limit():
    """Truncated text is within the token limit."""
    encoder = chunking_module.get_encoder("gpt-4o")
    long_text = "word " * 1000
    result = chunking_module.truncate_content(long_text, 10, encoder)
    token_count = chunking_module.count_tokens(result, encoder)
    assert token_count <= 10


def test_truncate_content_noop_when_within_limit():
    """Content already within limit is returned unchanged."""
    encoder = chunking_module.get_encoder("gpt-4o")
    short_text = "hello"
    result = chunking_module.truncate_content(short_text, 1000, encoder)
    assert result == short_text


def test_classify_tier_medium():
    """Token count between 5001 and 10000 is medium tier."""
    assert chunking_module.classify_tier(6000) == "medium"


def test_classify_tier_high():
    """Token count above 10000 is high tier."""
    assert chunking_module.classify_tier(15000) == "high"


def test_retry_keeps_good_chunks_alongside_oversized(
    monkeypatch,
):
    """Non-oversized chunks are kept during retry passes."""
    _patch_config(
        monkeypatch,
        overrides={"max_retries": 3},
    )

    call_count = []

    def mixed_chunker(page, _llm, _limit):
        call_count.append(1)
        content = page.raw_content
        if len(content) > 50:
            return [
                _make_page(
                    content="ok",
                    page_number=page.page_number,
                    token_count=0,
                ),
                _make_page(
                    content=content,
                    page_number=page.page_number,
                    token_count=0,
                ),
            ]
        return [
            _make_page(
                content=content,
                page_number=page.page_number,
                token_count=0,
            )
        ]

    _patch_chunkers(monkeypatch, md_fn=mixed_chunker)

    page = _make_page(
        content="a" * 51,
        page_number=1,
        token_count=51,
    )
    result = _make_result([page])

    with pytest.raises(RuntimeError, match="still exceed"):
        chunk_result(result, None)

    assert len(call_count) >= 2


def test_truncation_brings_page_under_embedding_limit(
    monkeypatch,
):
    """Page truncated to fit under embedding limit passes through."""
    _patch_config(
        monkeypatch,
        overrides={
            "embedding_limit": 80,
            "truncation_limit": 80,
        },
    )
    _patch_chunkers(
        monkeypatch,
        md_fn=_default_md_chunker,
    )

    page = _make_page(
        content="c" * 100,
        page_number=1,
        token_count=100,
    )
    result = _make_result([page])
    out = chunk_result(result, None)

    assert len(out.pages) == 1
    assert out.pages[0].token_count <= 80


def test_recount_chunk_tokens_assembles_header_chunks(
    monkeypatch,
):
    """Chunks with chunk_header count assembled content."""
    _patch_config(
        monkeypatch,
        overrides={
            "embedding_limit": 500,
            "truncation_limit": 1000,
        },
    )

    def header_chunker(page, _llm, _limit):
        """Return chunks with header and passthrough fields."""
        return [
            PageResult(
                page_number=page.page_number,
                raw_content="d",
                chunk_id="1.1",
                parent_page_number=page.page_number,
                chunk_header="hdr",
                sheet_passthrough_content="spt",
                section_passthrough_content="sec",
            ),
            PageResult(
                page_number=page.page_number,
                raw_content="e",
                chunk_id="1.2",
                parent_page_number=page.page_number,
                chunk_header="hdr",
                sheet_passthrough_content="spt",
            ),
        ]

    _patch_chunkers(monkeypatch, md_fn=header_chunker)

    page = _make_page(
        content="x" * 501,
        page_number=1,
        token_count=501,
    )
    result = _make_result([page])
    out = chunk_result(result, None)

    for chunk in out.pages:
        assert chunk.token_count > 0
        assert chunk.token_tier in ("low", "medium", "high")
    first = out.pages[0]
    assert first.chunk_header == "hdr"
    assembled_len = len("hdr\nspt\nsec\nd")
    assert first.token_count == assembled_len


def test_prepare_rechunk_restores_full_xlsx_context():
    """Restore header and passthrough rows before XLSX re-chunking."""
    chunk = PageResult(
        page_number=5,
        raw_content="| 10 | data |",
        chunk_id="5.1",
        parent_page_number=5,
        chunk_header="# Sheet: X\n| Row | A |\n| --- | --- |\n",
        sheet_passthrough_content="| 2 | Title |",
        section_passthrough_content="| 8 | Segment |",
    )
    restored = chunking_module.prepare_rechunk(chunk)

    assert restored.raw_content.startswith("# Sheet: X")
    assert "| 2 | Title |" in restored.raw_content
    assert "| 8 | Segment |" in restored.raw_content
    assert "| 10 | data |" in restored.raw_content
    assert restored.page_number == 5
    assert restored.chunk_id == "5.1"
    assert restored.sheet_passthrough_content == "| 2 | Title |"
    assert restored.section_passthrough_content == "| 8 | Segment |"


def test_prepare_rechunk_noop_without_header():
    """Return chunk unchanged when no chunk_header is set."""
    chunk = PageResult(
        page_number=3,
        raw_content="some content",
    )
    assert chunking_module.prepare_rechunk(chunk) is chunk
