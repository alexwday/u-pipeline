"""Tests for the classification stage."""

from ingestion.stages import classification as classification_module
from ingestion.utils.file_types import ExtractionResult, PageResult


def _make_page(page_number, content="Sample content", **kwargs):
    """Build a PageResult with given content and optional overrides."""
    return PageResult(page_number=page_number, raw_content=content, **kwargs)


def _make_result(pages, filetype="pdf"):
    """Build an ExtractionResult wrapping the given pages."""
    return ExtractionResult(
        file_path=f"/tmp/doc.{filetype}",
        filetype=filetype,
        pages=pages,
        total_pages=len(pages),
    )


def test_classify_result_sets_meaningful_layout_types():
    """Apply stable layout labels across different page shapes."""
    pages = [
        _make_page(1, content="Intro paragraph", token_tier="low"),
        _make_page(
            2,
            content="\n".join(
                [
                    "| Col A | Col B |",
                    "| --- | --- |",
                    "| 1 | 2 |",
                    "| 3 | 4 |",
                ]
            ),
            token_tier="medium",
        ),
        _make_page(
            3,
            content="- Agenda\n- Goals\n- Risks\n- Timeline",
            token_tier="low",
        ),
        _make_page(4, content="Long page", token_tier="high"),
    ]
    result = _make_result(pages)

    classification_module.classify_result(result)

    assert [page.layout_type for page in result.pages] == [
        "narrative_text",
        "table",
        "visual_heavy",
        "longform_text",
    ]


def test_classify_result_empty_document():
    """Empty pages list returns unchanged."""
    result = _make_result([])

    returned = classification_module.classify_result(result)

    assert not returned.pages
    assert returned is result


def test_classify_result_preserves_existing_fields():
    """Classification should only populate layout_type."""
    pages = [
        _make_page(1, token_count=500, token_tier="low"),
        _make_page(2, token_count=7000, token_tier="medium"),
    ]
    result = _make_result(pages)

    classification_module.classify_result(result)

    assert result.pages[0].token_count == 500
    assert result.pages[0].token_tier == "low"
    assert result.pages[1].token_count == 7000
    assert result.pages[1].token_tier == "medium"


def test_lookup_layout_type_uses_filetype_specific_labels():
    """Use direct layout labels for spreadsheets and slides."""
    assert (
        classification_module.lookup_layout_type("xlsx", "grid data", "low")
        == "spreadsheet"
    )
    assert (
        classification_module.lookup_layout_type("pptx", "- bullet", "low")
        == "presentation_slide"
    )


def test_lookup_layout_type_detects_empty_table_and_dense_table():
    """Classify empty, table, and dense-table content explicitly."""
    dense_table = "\n".join(
        [
            "| Col A | Col B |",
            "| --- | --- |",
            "| 1 | 2 |",
            "| 3 | 4 |",
            "| 5 | 6 |",
        ]
    )

    assert (
        classification_module.lookup_layout_type("pdf", "   ", "low")
        == "empty"
    )
    assert (
        classification_module.lookup_layout_type("pdf", dense_table, "medium")
        == "table"
    )
    assert (
        classification_module.lookup_layout_type("pdf", dense_table, "high")
        == "dense_table"
    )


def test_lookup_layout_type_detects_visual_markers():
    """Prefer visual classification when image-like markers are present."""
    assert (
        classification_module.lookup_layout_type(
            "pdf",
            "Figure 1\nThis page shows a chart.",
            "low",
        )
        == "visual_heavy"
    )
