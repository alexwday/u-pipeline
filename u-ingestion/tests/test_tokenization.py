"""Tests for the tokenization stage."""

from ingestion.stages import tokenization as tokenization_module
from ingestion.stages.tokenization import tokenize_result
from ingestion.utils.file_types import ExtractionResult, PageResult


def _make_page(page_number, content):
    """Build a PageResult with given content."""
    return PageResult(page_number=page_number, raw_content=content)


def _make_result(pages):
    """Build an ExtractionResult wrapping the given pages."""
    return ExtractionResult(
        file_path="/tmp/doc.pdf",
        filetype="pdf",
        pages=pages,
        total_pages=len(pages),
    )


def test_tokenize_sets_counts():
    """Token counts are set on pages and document."""
    pages = [
        _make_page(1, "Hello world"),
        _make_page(2, "Short"),
    ]
    result = _make_result(pages)
    tokenize_result(result)

    assert result.pages[0].raw_token_count > 0
    assert result.pages[0].embedding_token_count > 0
    assert result.pages[0].token_count > 0
    assert result.pages[1].token_count > 0
    assert (
        result.pages[0].raw_token_count
        == result.pages[0].embedding_token_count
        == result.pages[0].token_count
    )
    assert (
        result.raw_document_token_count
        == result.embedding_document_token_count
        == result.document_token_count
    )
    assert result.document_token_count == (
        result.pages[0].token_count + result.pages[1].token_count
    )


def test_tokenize_tier_low():
    """Small pages get tier 'low'."""
    pages = [_make_page(1, "Small content")]
    result = _make_result(pages)
    tokenize_result(result)

    assert result.pages[0].token_tier == "low"


def test_tokenize_tier_medium():
    """Pages between 5k-10k tokens get tier 'medium'."""
    content = "word " * 6000
    pages = [_make_page(1, content)]
    result = _make_result(pages)
    tokenize_result(result)

    assert result.pages[0].token_tier == "medium"
    assert result.pages[0].token_count > 5000
    assert result.pages[0].token_count <= 10000


def test_tokenize_tier_high():
    """Pages over 10k tokens get tier 'high'."""
    content = "word " * 12000
    pages = [_make_page(1, content)]
    result = _make_result(pages)
    tokenize_result(result)

    assert result.pages[0].token_tier == "high"
    assert result.pages[0].token_count > 10000


def test_tokenize_empty_document():
    """Empty document gets zero token count."""
    result = _make_result([])
    tokenize_result(result)

    assert result.document_token_count == 0


def test_tokenize_unknown_model_falls_back(monkeypatch):
    """Unknown model name falls back to o200k_base encoding."""
    monkeypatch.setenv("TOKENIZER_MODEL", "unknown-model-xyz")

    pages = [_make_page(1, "Test content")]
    result = _make_result(pages)
    tokenize_result(result)

    assert result.pages[0].token_count > 0
    assert result.pages[0].token_tier == "low"


def test_tokenize_uses_tokenizer_model_for_encoder(monkeypatch):
    """Tokenization should derive its encoder from TOKENIZER_MODEL."""
    seen = []

    monkeypatch.setattr(
        tokenization_module,
        "get_tokenizer_model",
        lambda: "text-embedding-3-small",
    )
    monkeypatch.setattr(
        tokenization_module,
        "_get_encoder",
        lambda model: seen.append(model)
        or tokenization_module.tiktoken.get_encoding("o200k_base"),
    )

    result = _make_result([_make_page(1, "hello")])
    tokenize_result(result)

    assert seen == ["text-embedding-3-small"]
