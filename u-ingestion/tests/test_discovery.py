"""Tests for filesystem discovery."""

import json

from ingestion.stages import discovery
from ingestion.stages.discovery import (
    _build_file_record as build_file_record,
    _parse_path_parts as parse_path_parts,
)
from ingestion.utils.file_types import DiscoveryScan


def test_parse_path_parts_and_build_file_record(
    tmp_path,
    file_record_factory,
    monkeypatch,
):
    """Split nested paths and build file records from disk."""
    data_dir = tmp_path / "policy" / "2026" / "Q1" / "RBC"
    data_dir.mkdir(parents=True)
    document = data_dir / "report.pdf"
    document.write_text("pdf-data", encoding="utf-8")

    assert parse_path_parts("policy/2026/Q1/RBC/extra") == (
        "policy",
        "2026",
        "Q1",
        "RBC/extra",
    )

    monkeypatch.setenv("ACCEPTED_FILETYPES", "pdf,xlsx")
    record = build_file_record(
        str(data_dir),
        "report.pdf",
        ("policy", "2026", "Q1", "RBC"),
    )

    assert record == file_record_factory(
        data_source="policy",
        filter_1="2026",
        filter_2="Q1",
        filter_3="RBC",
        filename="report.pdf",
        filetype="pdf",
        file_size=document.stat().st_size,
        date_last_modified=document.stat().st_mtime,
        file_path=str(document),
    )


def test_scan_filesystem_filters_hidden_and_unsupported(tmp_path, monkeypatch):
    """Collect supported and unsupported files without hidden entries."""
    monkeypatch.setenv("ACCEPTED_FILETYPES", "pdf,xlsx")
    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "skip.pdf").write_text("hidden", encoding="utf-8")

    visible_dir = tmp_path / "policy" / "2026" / "Q1"
    visible_dir.mkdir(parents=True)
    (visible_dir / "include.pdf").write_text("pdf", encoding="utf-8")
    (visible_dir / "skip.txt").write_text("txt", encoding="utf-8")
    (visible_dir / ".secret.pdf").write_text("hidden", encoding="utf-8")

    scan = discovery.scan_filesystem(str(tmp_path))

    assert [record.filename for record in scan.supported] == ["include.pdf"]
    assert [record.filename for record in scan.unsupported] == ["skip.txt"]


def test_scan_filesystem_returns_empty_for_missing_directory():
    """Log and return an empty scan when the root does not exist."""
    scan = discovery.scan_filesystem("/path/that/does/not/exist")

    assert scan == DiscoveryScan(supported=[], unsupported=[])


def test_compute_diff_detects_new_modified_and_deleted(
    monkeypatch,
    file_record_factory,
):
    """Compute filesystem deltas using size and lazy hashing."""
    disk_new = file_record_factory(
        filename="new.pdf",
        file_path="/tmp/new.pdf",
    )
    disk_size_changed = file_record_factory(
        filename="size.pdf",
        file_path="/tmp/size.pdf",
        file_size=200,
    )
    disk_hash_changed = file_record_factory(
        filename="hash.pdf",
        file_path="/tmp/hash.pdf",
        file_size=100,
        date_last_modified=200.0,
    )
    disk_same_hash = file_record_factory(
        filename="same.pdf",
        file_path="/tmp/same.pdf",
        file_size=100,
        date_last_modified=300.0,
    )
    disk_unchanged = file_record_factory(
        filename="ok.pdf",
        file_path="/tmp/ok.pdf",
        file_size=100,
        date_last_modified=400.0,
    )

    cat_size_changed = file_record_factory(
        filename="size.pdf",
        file_path="/tmp/size.pdf",
        file_size=100,
    )
    cat_hash_changed = file_record_factory(
        filename="hash.pdf",
        file_path="/tmp/hash.pdf",
        file_size=100,
        date_last_modified=150.0,
        file_hash="catalog-hash",
    )
    cat_same_hash = file_record_factory(
        filename="same.pdf",
        file_path="/tmp/same.pdf",
        file_size=100,
        date_last_modified=250.0,
        file_hash="same-hash",
    )
    cat_unchanged = file_record_factory(
        filename="ok.pdf",
        file_path="/tmp/ok.pdf",
        file_size=100,
        date_last_modified=400.0,
    )
    cat_deleted = file_record_factory(
        filename="gone.pdf",
        file_path="/tmp/gone.pdf",
    )

    monkeypatch.setattr(
        discovery,
        "compute_file_hash",
        lambda path: {
            "/tmp/hash.pdf": "disk-hash",
            "/tmp/same.pdf": "same-hash",
        }[path],
    )

    diff = discovery.compute_diff(
        [
            disk_new,
            disk_size_changed,
            disk_hash_changed,
            disk_same_hash,
            disk_unchanged,
        ],
        [
            cat_size_changed,
            cat_hash_changed,
            cat_same_hash,
            cat_unchanged,
            cat_deleted,
        ],
    )

    assert [record.filename for record in diff.new] == ["new.pdf"]
    assert [record.filename for record in diff.modified] == [
        "size.pdf",
        "hash.pdf",
    ]
    assert diff.modified[1].file_hash == "disk-hash"
    assert [record.filename for record in diff.deleted] == ["gone.pdf"]


def test_run_discovery_writes_trace_and_ignores_unsupported_catalog_records(
    tmp_path,
    monkeypatch,
    file_record_factory,
):
    """Write discovery output with skipped unsupported file details."""
    processing_dir = tmp_path / "processing"
    processing_dir.mkdir()
    monkeypatch.setattr(discovery, "PROCESSING_DIR", processing_dir)
    monkeypatch.setattr(
        discovery,
        "get_data_source_path",
        lambda: str(tmp_path / "sources"),
    )

    supported_disk = file_record_factory(
        filename="fresh.pdf",
        file_path="/tmp/fresh.pdf",
    )
    unsupported_disk = file_record_factory(
        filename="fresh.txt",
        filetype="txt",
        file_path="/tmp/fresh.txt",
    )
    unchanged_catalog = file_record_factory(
        filename="keep.pdf",
        file_path="/tmp/keep.pdf",
    )
    deleted_catalog = file_record_factory(
        filename="gone.pdf",
        file_path="/tmp/gone.pdf",
    )
    unsupported_catalog = file_record_factory(
        filename="skip.txt",
        filetype="txt",
        file_path="/tmp/skip.txt",
    )

    monkeypatch.setattr(
        discovery,
        "scan_filesystem",
        lambda _base: DiscoveryScan(
            supported=[supported_disk, unchanged_catalog],
            unsupported=[unsupported_disk],
        ),
    )
    monkeypatch.setattr(
        discovery,
        "fetch_catalog_records",
        lambda _conn: [
            unchanged_catalog,
            deleted_catalog,
            unsupported_catalog,
        ],
    )

    discovery_run = discovery.run_discovery(conn=object())
    diff = discovery_run.diff
    output = json.loads(
        (processing_dir / "discovery.json").read_text(encoding="utf-8")
    )

    assert [record.filename for record in discovery_run.scan.supported] == [
        "fresh.pdf",
        "keep.pdf",
    ]
    assert [record.filename for record in diff.new] == ["fresh.pdf"]
    assert [record.filename for record in diff.deleted] == ["gone.pdf"]
    assert output["unsupported_files"][0]["filename"] == "fresh.txt"
    assert output["ignored_catalog_records"][0]["filename"] == "skip.txt"
