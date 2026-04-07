"""Tests for the embedding enrichment module."""

from ingestion.stages.enrichment import (
    embedding as mod,
)
from ingestion.utils.file_types import (
    ExtractionResult,
    PageResult,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

FAKE_DIM = 4
FAKE_MODEL = "text-embedding-3-large"
FAKE_BATCH = 2


def _vec(seed: int = 1) -> list[float]:
    """Build a fake embedding vector. Params: seed. Returns: list."""
    return [float(seed)] * FAKE_DIM


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
        page_end, chunk_ids, level, plus overrides.
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
    content_units=None,
):
    """Build an ExtractionResult for testing.

    Params: pages, sections, filetype, file_path,
        metadata, content_units.
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
    if content_units is not None:
        result.content_units = content_units
    return result


def _patch_config(monkeypatch):
    """Patch embedding config functions.

    Params: monkeypatch. Returns: None.
    """
    monkeypatch.setattr(mod, "get_embedding_model", lambda: FAKE_MODEL)
    monkeypatch.setattr(mod, "get_embedding_dimensions", lambda: FAKE_DIM)
    monkeypatch.setattr(mod, "get_embedding_batch_size", lambda: FAKE_BATCH)


class FakeLLM:
    """Mock LLM client that returns deterministic vectors."""

    def __init__(self):
        """Initialize call log."""
        self.embed_calls = []

    def embed(self, texts, model="", dimensions=0):
        """Record call and return fake vectors.

        Params: texts, model, dimensions.
        Returns: list[list[float]].
        """
        self.embed_calls.append(
            {
                "texts": list(texts),
                "model": model,
                "dimensions": dimensions,
            }
        )
        return [_vec(i + 1) for i in range(len(texts))]

    def reset(self):
        """Clear recorded calls. Returns: None."""
        self.embed_calls = []


# ------------------------------------------------------------------
# test_embed_content_creates_content_embeddings
# ------------------------------------------------------------------


def test_embed_content_creates_content_embeddings(
    monkeypatch,
):
    """Each content unit gets a content embedding."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    pages = [
        _make_page("Page one text", page_number=1),
        _make_page("Page two text", page_number=2),
    ]
    result = _make_result(pages=pages)

    enriched = mod.embed_content(result, llm)

    for unit in enriched.content_units:
        assert "content_embedding" in unit
        assert len(unit["content_embedding"]) == FAKE_DIM


# ------------------------------------------------------------------
# test_embed_content_creates_keyword_embeddings
# ------------------------------------------------------------------


def test_embed_content_creates_keyword_embeddings(
    monkeypatch,
):
    """Keyword embeddings from joined keywords."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    pages = [
        _make_page(
            "text",
            page_number=1,
            keywords=["CET1", "RWA"],
        ),
    ]
    result = _make_result(pages=pages)

    enriched = mod.embed_content(result, llm)

    unit = enriched.content_units[0]
    assert len(unit["keyword_embedding"]) == FAKE_DIM


# ------------------------------------------------------------------
# test_embed_content_creates_entity_embeddings
# ------------------------------------------------------------------


def test_embed_content_creates_entity_embeddings(
    monkeypatch,
):
    """Entity embeddings from joined entities."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    pages = [
        _make_page(
            "text",
            page_number=1,
            entities=["OSFI", "RBC"],
        ),
    ]
    result = _make_result(pages=pages)

    enriched = mod.embed_content(result, llm)

    unit = enriched.content_units[0]
    assert len(unit["entity_embedding"]) == FAKE_DIM


# ------------------------------------------------------------------
# test_embed_content_creates_section_embeddings
# ------------------------------------------------------------------


def test_embed_content_creates_section_embeddings(
    monkeypatch,
):
    """Primary sections get summary, keyword, entity embeddings."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    sections = [
        _make_section(
            section_id="1",
            title="KM1",
            summary="Key metrics.",
            keywords=["CET1"],
            entities=["OSFI"],
        ),
    ]
    result = _make_result(sections=sections)

    enriched = mod.embed_content(result, llm)

    section = enriched.sections[0]
    assert "summary_embedding" in section
    assert len(section["summary_embedding"]) == FAKE_DIM
    assert "keyword_embedding" in section
    assert len(section["keyword_embedding"]) == FAKE_DIM
    assert "entity_embedding" in section
    assert len(section["entity_embedding"]) == FAKE_DIM


# ------------------------------------------------------------------
# test_embed_content_creates_document_embedding
# ------------------------------------------------------------------


def test_embed_content_creates_document_embedding(
    monkeypatch,
):
    """Executive summary gets embedded."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    result = _make_result(
        metadata={
            "executive_summary": "Strong capital position.",
        },
    )

    enriched = mod.embed_content(result, llm)

    assert "summary_embedding" in enriched.document_metadata
    vec = enriched.document_metadata["summary_embedding"]
    assert len(vec) == FAKE_DIM


# ------------------------------------------------------------------
# test_embed_content_embeds_subsections_with_summaries
# ------------------------------------------------------------------


def test_embed_content_embeds_subsections_with_summaries(
    monkeypatch,
):
    """Subsections with summaries get embedded; without do not."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    sections = [
        _make_section(
            section_id="1",
            level="section",
            summary="Primary.",
        ),
        _make_section(
            section_id="1.1",
            level="subsection",
            summary="Sub.",
        ),
        _make_section(
            section_id="1.2",
            level="subsection",
            summary="",
        ),
    ]
    result = _make_result(sections=sections)

    enriched = mod.embed_content(result, llm)

    primary = enriched.sections[0]
    sub_with_summary = enriched.sections[1]
    sub_no_summary = enriched.sections[2]
    assert "summary_embedding" in primary
    assert "summary_embedding" in sub_with_summary
    assert "summary_embedding" not in sub_no_summary


# ------------------------------------------------------------------
# test_embed_content_skips_empty_keywords
# ------------------------------------------------------------------


def test_embed_content_skips_empty_keywords(monkeypatch):
    """No embedding for empty keyword list."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    pages = [
        _make_page("text", page_number=1, keywords=[]),
    ]
    result = _make_result(pages=pages)

    enriched = mod.embed_content(result, llm)

    unit = enriched.content_units[0]
    assert not unit["keyword_embedding"]


# ------------------------------------------------------------------
# test_embed_content_assembles_chunk_content
# ------------------------------------------------------------------


def test_embed_content_assembles_chunk_content(
    monkeypatch,
):
    """Chunked pages include header + passthrough."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    pages = [
        _make_page(
            "data rows",
            page_number=1,
            chunk_id="1.1",
            chunk_header="## Sheet: KM1\n",
            sheet_passthrough_content="header row\n",
            section_passthrough_content="section row\n",
        ),
    ]
    result = _make_result(pages=pages)

    mod.embed_content(result, llm)

    content_calls = [c for c in llm.embed_calls if c["texts"]]
    all_texts = []
    for call in content_calls:
        all_texts.extend(call["texts"])
    content_text = all_texts[0]
    assert "## Sheet: KM1" in content_text
    assert "header row" in content_text
    assert "section row" in content_text
    assert "data rows" in content_text


# ------------------------------------------------------------------
# test_embed_content_prefixes_section_title
# ------------------------------------------------------------------


def test_embed_content_prefixes_section_title(monkeypatch):
    """Content text prefixed with section title."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    sections = [
        _make_section(
            section_id="1",
            title="KM1",
            summary="Summary.",
        ),
    ]
    pages = [
        _make_page(
            "page content",
            page_number=1,
            section_id="1",
        ),
    ]
    result = _make_result(pages=pages, sections=sections)

    mod.embed_content(result, llm)

    content_calls = [c for c in llm.embed_calls if c["texts"]]
    all_texts = []
    for call in content_calls:
        all_texts.extend(call["texts"])
    assert any(t.startswith("Section: KM1.") for t in all_texts)


# ------------------------------------------------------------------
# test_embed_content_batches_correctly
# ------------------------------------------------------------------


def test_embed_content_batches_correctly(monkeypatch):
    """Respects batch_size config."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    pages = [_make_page(f"Page {i}", page_number=i) for i in range(1, 6)]
    result = _make_result(pages=pages)

    mod.embed_content(result, llm)

    content_batch_sizes = [len(c["texts"]) for c in llm.embed_calls]
    for size in content_batch_sizes:
        assert size <= FAKE_BATCH


# ------------------------------------------------------------------
# test_embed_content_preserves_pages
# ------------------------------------------------------------------


def test_embed_content_preserves_pages(monkeypatch):
    """Pages unchanged."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    pages = [
        _make_page(
            "Original content",
            page_number=1,
            layout_type="text",
        ),
    ]
    result = _make_result(pages=pages)

    enriched = mod.embed_content(result, llm)

    assert enriched.pages[0].raw_content == "Original content"
    assert enriched.pages[0].layout_type == "text"
    assert enriched.pages[0].page_number == 1


# ------------------------------------------------------------------
# test_embed_content_empty_document
# ------------------------------------------------------------------


def test_embed_content_empty_document(monkeypatch):
    """No pages = no embed calls."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    result = _make_result(pages=[])

    enriched = mod.embed_content(result, llm)

    assert not llm.embed_calls
    assert not enriched.content_units


# ------------------------------------------------------------------
# test_batch_embed_helper
# ------------------------------------------------------------------


def test_batch_embed_helper():
    """Batching splits and recombines correctly."""
    llm = FakeLLM()
    texts = ["a", "b", "c", "d", "e"]

    vectors = mod.batch_embed(llm, texts, FAKE_MODEL, FAKE_DIM, 2)

    assert len(vectors) == 5
    for vec in vectors:
        assert len(vec) == FAKE_DIM
    assert len(llm.embed_calls) == 3


# ------------------------------------------------------------------
# test_build_content_text_unchunked
# ------------------------------------------------------------------


def test_build_content_text_unchunked():
    """raw_content used for unchunked pages."""
    page = _make_page("Raw page content", page_number=1)

    text = mod.build_content_text(page, "")

    assert text == "Raw page content"


# ------------------------------------------------------------------
# test_build_content_text_chunked
# ------------------------------------------------------------------


def test_build_content_text_chunked():
    """Assembled content for chunks."""
    page = _make_page(
        "data rows",
        page_number=1,
        chunk_id="1.1",
        chunk_header="## Sheet\n",
        sheet_passthrough_content="header\n",
        section_passthrough_content="section\n",
    )

    text = mod.build_content_text(page, "KM1")

    assert text.startswith("Section: KM1. ")
    assert "## Sheet" in text
    assert "header" in text
    assert "section" in text
    assert "data rows" in text


# ------------------------------------------------------------------
# test_batch_embed_skips_empty_strings
# ------------------------------------------------------------------


def test_batch_embed_skips_empty_strings():
    """Empty strings get empty list, not sent to API."""
    llm = FakeLLM()
    texts = ["a", "", "c"]

    vectors = mod.batch_embed(llm, texts, FAKE_MODEL, FAKE_DIM, 10)

    assert len(vectors) == 3
    assert len(vectors[0]) == FAKE_DIM
    assert not vectors[1]
    assert len(vectors[2]) == FAKE_DIM
    assert len(llm.embed_calls) == 1
    assert llm.embed_calls[0]["texts"] == ["a", "c"]


# ------------------------------------------------------------------
# test_batch_embed_all_empty
# ------------------------------------------------------------------


def test_batch_embed_all_empty():
    """All empty strings means no API calls."""
    llm = FakeLLM()

    vectors = mod.batch_embed(llm, ["", ""], FAKE_MODEL, FAKE_DIM, 10)

    assert len(vectors) == 2
    assert not vectors[0]
    assert not vectors[1]
    assert not llm.embed_calls


# ------------------------------------------------------------------
# test_find_section_title_match
# ------------------------------------------------------------------


def test_find_section_title_match():
    """Returns title when section_id matches."""
    page = _make_page(section_id="2")
    sections = [
        _make_section(section_id="1", title="First"),
        _make_section(section_id="2", title="Second"),
    ]

    title = mod.find_section_title(page, sections)

    assert title == "Second"


# ------------------------------------------------------------------
# test_find_section_title_no_match
# ------------------------------------------------------------------


def test_find_section_title_no_match():
    """Returns empty when no section matches."""
    page = _make_page(section_id="99")
    sections = [
        _make_section(section_id="1", title="First"),
    ]

    title = mod.find_section_title(page, sections)

    assert title == ""


# ------------------------------------------------------------------
# test_find_section_title_no_section_id
# ------------------------------------------------------------------


def test_find_section_title_no_section_id():
    """Returns empty when page has no section_id."""
    page = _make_page()
    sections = [
        _make_section(section_id="1", title="First"),
    ]

    title = mod.find_section_title(page, sections)

    assert title == ""


# ------------------------------------------------------------------
# test_build_section_texts_includes_summarized_subsections
# ------------------------------------------------------------------


def test_build_section_texts_includes_summarized_subsections():
    """Primary sections and subsections with summaries are included."""
    sections = [
        _make_section(
            section_id="1",
            level="section",
            title="Main",
            summary="Main sum.",
        ),
        _make_section(
            section_id="1.1",
            level="subsection",
            title="Sub",
            summary="Sub sum.",
        ),
        _make_section(
            section_id="1.2",
            level="subsection",
            title="Empty",
            summary="",
        ),
    ]

    texts = mod.build_section_texts(sections)

    assert len(texts) == 2
    assert "Main" in texts[0][0]
    assert "Sub" in texts[1][0]


# ------------------------------------------------------------------
# test_ensure_content_units_builds_from_pages
# ------------------------------------------------------------------


def test_ensure_content_units_builds_from_pages():
    """Populates content_units from pages when empty."""
    pages = [
        _make_page(
            "content",
            page_number=1,
            keywords=["kw"],
            entities=["ent"],
        ),
    ]
    result = _make_result(pages=pages)

    mod.ensure_content_units(result)

    assert len(result.content_units) == 1
    assert result.content_units[0]["raw_content"] == "content"
    assert result.content_units[0]["keywords"] == ["kw"]


# ------------------------------------------------------------------
# test_ensure_content_units_preserves_existing
# ------------------------------------------------------------------


def test_ensure_content_units_preserves_existing():
    """Does not rebuild when content_units already set."""
    pages = [_make_page("content", page_number=1)]
    existing = [{"content_unit_id": "1", "custom": True}]
    result = _make_result(pages=pages, content_units=existing)

    mod.ensure_content_units(result)

    assert len(result.content_units) == 1
    assert result.content_units[0]["custom"] is True


# ------------------------------------------------------------------
# test_embed_content_no_doc_summary
# ------------------------------------------------------------------


def test_embed_content_no_doc_summary(monkeypatch):
    """No document embedding when executive_summary empty."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    result = _make_result(
        metadata={"executive_summary": ""},
    )

    enriched = mod.embed_content(result, llm)

    assert "summary_embedding" not in (enriched.document_metadata)


# ------------------------------------------------------------------
# test_embed_content_uses_correct_model
# ------------------------------------------------------------------


def test_embed_content_uses_correct_model(monkeypatch):
    """Passes configured model and dimensions to embed."""
    _patch_config(monkeypatch)
    llm = FakeLLM()
    pages = [_make_page("text", page_number=1)]
    result = _make_result(pages=pages)

    mod.embed_content(result, llm)

    for call in llm.embed_calls:
        assert call["model"] == FAKE_MODEL
        assert call["dimensions"] == FAKE_DIM


# ------------------------------------------------------------------
# test_store_section_embeddings_skips_subsections
# ------------------------------------------------------------------


def test_store_section_embeddings_skips_subsections():
    """Only primary sections receive vectors."""
    sections = [
        _make_section(section_id="1", level="section"),
        _make_section(section_id="1.1", level="subsection"),
    ]

    mod.store_section_embeddings(
        sections,
        [_vec(1)],
        [_vec(2)],
        [_vec(3)],
    )

    assert "summary_embedding" in sections[0]
    assert "summary_embedding" not in sections[1]


# ------------------------------------------------------------------
# test_build_content_unit_texts_empty_keywords_entities
# ------------------------------------------------------------------


def test_build_content_unit_texts_empty_keywords_entities():
    """Empty keywords and entities produce empty strings."""
    pages = [
        _make_page(
            "text",
            page_number=1,
            keywords=[],
            entities=[],
        ),
    ]
    result = _make_result(pages=pages, sections=[])

    texts = mod.build_content_unit_texts(result)

    assert texts[0][1] == ""
    assert texts[0][2] == ""
