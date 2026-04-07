"""Tests for file metadata helpers."""

from ingestion.utils import file_types


def test_file_record_support_flags(file_record_factory):
    """Compute the supported flag from ACCEPTED_FILETYPES."""
    supported = file_record_factory(filetype="pdf")
    unsupported = file_record_factory(
        filename="doc.txt",
        filetype="txt",
        file_path="/tmp/doc.txt",
    )

    assert supported.supported is True
    assert unsupported.supported is False


def test_discovery_models_and_hash(tmp_path, file_record_factory):
    """Instantiate the discovery dataclasses and hash file contents."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("hello", encoding="utf-8")

    diff = file_types.DiscoveryDiff(
        new=[file_record_factory()],
        modified=[],
        deleted=[],
    )
    scan = file_types.DiscoveryScan(
        supported=[file_record_factory()],
        unsupported=[
            file_record_factory(
                filename="skip.txt",
                filetype="txt",
                file_path="/tmp/skip.txt",
            )
        ],
    )

    assert len(diff.new) == 1
    assert len(scan.supported) == 1
    assert len(scan.unsupported) == 1
    assert (
        file_types.compute_file_hash(str(sample_file))
        == "2cf24dba5fb0a30e26e83b2ac5b9e29e"
        "1b161e5c1fa7425e73043362938b9824"
    )


def test_extraction_dataclasses(accepted_filetypes):
    """Instantiate the extraction types."""
    assert accepted_filetypes is None

    page = file_types.PageResult(
        page_number=1,
        raw_content="# Content",
    )
    assert page.page_number == 1
    assert page.raw_content == "# Content"
    assert page.raw_token_count == 0
    assert page.embedding_token_count == 0
    assert page.token_count == 0
    assert page.token_tier == ""

    result = file_types.ExtractionResult(
        file_path="/tmp/doc.pdf",
        filetype="pdf",
        pages=[page],
        total_pages=1,
    )
    assert result.total_pages == 1
    assert result.raw_document_token_count == 0
    assert result.embedding_document_token_count == 0
    assert result.document_token_count == 0
    assert len(result.pages) == 1


def test_get_content_unit_id_uses_chunk_id_when_present():
    """Prefer chunk_id for chunked content units."""
    page = file_types.PageResult(
        page_number=24,
        raw_content="chunk",
        chunk_id="24.2",
    )

    assert file_types.get_content_unit_id(page) == "24.2"


def test_get_content_unit_id_falls_back_to_page_number():
    """Use the page number string for unchunked pages."""
    page = file_types.PageResult(
        page_number=7,
        raw_content="page",
    )

    assert file_types.get_content_unit_id(page) == "7"
