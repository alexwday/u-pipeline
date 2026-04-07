"""Shared pytest fixtures for ingestion tests."""

import pytest

from ingestion.utils.file_types import FileRecord


@pytest.fixture(name="accepted_filetypes")
def accepted_filetypes_fixture(monkeypatch):
    """Set accepted file types for tests that create FileRecords."""
    monkeypatch.setenv("ACCEPTED_FILETYPES", "pdf,docx,pptx,xlsx")


@pytest.fixture
def file_record_factory(accepted_filetypes):
    """Create FileRecords with concise defaults."""
    assert accepted_filetypes is None

    def factory(**overrides: object) -> FileRecord:
        record = {
            "data_source": "source",
            "filter_1": "",
            "filter_2": "",
            "filter_3": "",
            "filename": "doc.pdf",
            "filetype": "pdf",
            "file_size": 10,
            "date_last_modified": 100.0,
            "file_hash": "",
            "file_path": "/tmp/doc.pdf",
        }
        record.update(overrides)
        return FileRecord(
            data_source=str(record["data_source"]),
            filter_1=str(record["filter_1"]),
            filter_2=str(record["filter_2"]),
            filter_3=str(record["filter_3"]),
            filename=str(record["filename"]),
            filetype=str(record["filetype"]),
            file_size=int(record["file_size"]),
            date_last_modified=float(record["date_last_modified"]),
            file_hash=str(record["file_hash"]),
            file_path=str(record["file_path"]),
        )

    return factory
