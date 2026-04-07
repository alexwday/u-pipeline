"""Tests for the extraction stage router."""

from unittest.mock import Mock, patch

import pytest

from ingestion.stages import extraction as extraction_module
from ingestion.stages.extraction import extract_file
from ingestion.utils.file_types import ExtractionResult


def _make_result(filetype):
    """Build a minimal ExtractionResult."""
    return ExtractionResult(
        file_path=f"/tmp/doc.{filetype}",
        filetype=filetype,
        pages=[],
        total_pages=0,
    )


def test_routes_pdf(file_record_factory):
    """Route a PDF file to the PDF processor."""
    record = file_record_factory(filetype="pdf")
    expected = _make_result("pdf")
    with patch.object(
        extraction_module, "process_pdf", return_value=expected
    ) as mock_proc:
        result = extract_file(record, Mock())
    mock_proc.assert_called_once_with(
        record.file_path, mock_proc.call_args[0][1]
    )
    assert result is expected
    assert result.data_source == record.data_source
    assert result.filter_1 == record.filter_1


def test_routes_docx(file_record_factory):
    """Route a DOCX file to the DOCX processor."""
    record = file_record_factory(
        filename="doc.docx",
        filetype="docx",
        file_path="/tmp/doc.docx",
    )
    expected = _make_result("docx")
    with patch.object(
        extraction_module, "process_docx", return_value=expected
    ) as mock_proc:
        result = extract_file(record, Mock())
    mock_proc.assert_called_once()
    assert result is expected
    assert result.data_source == record.data_source


def test_routes_pptx(file_record_factory):
    """Route a PPTX file to the PPTX processor."""
    record = file_record_factory(
        filename="deck.pptx",
        filetype="pptx",
        file_path="/tmp/deck.pptx",
    )
    expected = _make_result("pptx")
    with patch.object(
        extraction_module, "process_pptx", return_value=expected
    ) as mock_proc:
        result = extract_file(record, Mock())
    mock_proc.assert_called_once()
    assert result is expected
    assert result.data_source == record.data_source


def test_routes_xlsx(file_record_factory):
    """Route an XLSX file to the XLSX processor."""
    record = file_record_factory(
        filename="data.xlsx",
        filetype="xlsx",
        file_path="/tmp/data.xlsx",
    )
    expected = _make_result("xlsx")
    with patch.object(
        extraction_module, "process_xlsx", return_value=expected
    ) as mock_proc:
        result = extract_file(record, Mock())
    mock_proc.assert_called_once()
    assert result is expected
    assert result.data_source == record.data_source


def test_raises_on_unsupported(file_record_factory):
    """Raise ValueError for an unsupported filetype."""
    record = file_record_factory(
        filename="doc.txt",
        filetype="txt",
        file_path="/tmp/doc.txt",
    )
    with pytest.raises(ValueError, match="Unsupported filetype: txt"):
        extract_file(record, Mock())
