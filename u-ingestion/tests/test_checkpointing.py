"""Tests for checkpoint planning and artifact persistence."""

import re
from types import SimpleNamespace

import pytest

from ingestion.utils import checkpointing
from ingestion.utils.file_types import (
    ExtractionResult,
    PageResult,
    StageCheckpoint,
)


def test_resolve_cli_options_defaults_and_validation():
    """Default target is the latest stage and start must precede target."""
    options = checkpointing.resolve_cli_options(
        SimpleNamespace(
            to_stage="",
            from_stage="",
            only_stage="",
            force_stage="",
            force_all=False,
            file_path=[],
            glob=[],
        )
    )

    assert options.target_stage == "persistence"
    assert options.forced_start_stage == ""
    assert options.force_all is False
    assert not options.file_paths
    assert not options.glob_patterns

    with pytest.raises(ValueError, match="Start stage cannot come after"):
        checkpointing.resolve_cli_options(
            SimpleNamespace(
                to_stage="extraction",
                from_stage="chunking",
                only_stage="",
                force_stage="",
                force_all=False,
                file_path=[],
                glob=[],
            )
        )


def test_resolve_cli_options_only_stage_and_force_stage():
    """Honor only-stage and force-stage overrides."""
    only_stage = checkpointing.resolve_cli_options(
        SimpleNamespace(
            to_stage="",
            from_stage="",
            only_stage="extraction",
            force_stage="",
            force_all=False,
            file_path=["/tmp/doc.pdf"],
            glob=["*.pdf"],
        )
    )
    force_stage = checkpointing.resolve_cli_options(
        SimpleNamespace(
            to_stage="chunking",
            from_stage="",
            only_stage="",
            force_stage="chunking",
            force_all=False,
            file_path=[],
            glob=[],
        )
    )

    assert only_stage.target_stage == "extraction"
    assert only_stage.forced_start_stage == "extraction"
    assert only_stage.file_paths == ("/tmp/doc.pdf",)
    assert only_stage.glob_patterns == ("*.pdf",)
    assert force_stage.forced_start_stage == "chunking"


def test_stage_navigation_helpers():
    """Return consistent ordered stage ranges and neighbors."""
    assert checkpointing.get_stage_range("extraction", "chunking") == (
        "extraction",
        "tokenization",
        "classification",
        "chunking",
    )
    assert checkpointing.get_previous_stage("extraction") == ""
    assert checkpointing.get_previous_stage("tokenization") == "extraction"
    assert checkpointing.get_previous_stage("classification") == "tokenization"
    assert checkpointing.get_previous_stage("chunking") == "classification"
    assert checkpointing.get_previous_stage("doc_metadata") == "chunking"
    assert (
        checkpointing.get_previous_stage("section_detection") == "doc_metadata"
    )
    assert (
        checkpointing.get_previous_stage("content_extraction")
        == "section_detection"
    )
    assert (
        checkpointing.get_previous_stage("section_summary")
        == "content_extraction"
    )
    assert checkpointing.get_previous_stage("doc_summary") == "section_summary"
    assert checkpointing.get_previous_stage("embedding") == "doc_summary"
    assert checkpointing.get_previous_stage("persistence") == "embedding"
    assert checkpointing.get_downstream_stages("classification") == (
        "classification",
        "chunking",
        "doc_metadata",
        "section_detection",
        "content_extraction",
        "section_summary",
        "doc_summary",
        "embedding",
        "persistence",
    )
    assert checkpointing.get_downstream_stages("persistence") == (
        "persistence",
    )


def test_stage_helpers_raise_for_invalid_stage_names():
    """Reject invalid stage names and reversed stage ranges."""
    assert checkpointing.get_default_target_stage() == "persistence"

    with pytest.raises(ValueError, match="Unknown stage"):
        checkpointing.get_downstream_stages("missing")

    with pytest.raises(ValueError, match="Start stage cannot come after"):
        checkpointing.get_stage_range("chunking", "extraction")


def test_build_stage_signature_uses_config(monkeypatch, tmp_path):
    """Hash relevant files and model config into a stable signature."""
    dependency = tmp_path / "stage.py"
    dependency.write_text("print('hello')\n", encoding="utf-8")
    monkeypatch.setattr(
        checkpointing,
        "_stage_dependency_paths",
        lambda _stage_name: [dependency],
    )
    monkeypatch.setattr(
        checkpointing,
        "_stage_config",
        lambda stage_name: {"stage": stage_name, "model": "gpt-test"},
    )

    signature_one = checkpointing.build_stage_signature("extraction")
    signature_two = checkpointing.build_stage_signature("extraction")

    assert signature_one == signature_two
    assert len(signature_one) == 64


def test_stage_dependency_helpers(monkeypatch):
    """Resolve dependency lists, config fingerprints, and all signatures."""
    monkeypatch.setattr(
        checkpointing,
        "get_stage_model_config",
        lambda stage_name: {"stage": stage_name},
    )
    monkeypatch.setattr(
        checkpointing,
        "get_tokenizer_model",
        lambda: "o200k_base",
    )
    monkeypatch.setattr(
        checkpointing,
        "get_vision_dpi_scale",
        lambda: 2.5,
    )
    monkeypatch.setattr(
        checkpointing,
        "get_chunking_embedding_token_limit",
        lambda: 8192,
    )
    monkeypatch.setattr(
        checkpointing,
        "get_chunking_truncation_token_limit",
        lambda: 80000,
    )
    monkeypatch.setattr(
        checkpointing,
        "get_chunking_max_retries",
        lambda: 3,
    )
    monkeypatch.setattr(
        checkpointing,
        "get_chunking_md_batch_size",
        lambda: 100,
    )
    monkeypatch.setattr(
        checkpointing,
        "get_chunking_xlsx_batch_size",
        lambda: 50,
    )
    monkeypatch.setattr(
        checkpointing,
        "get_chunking_xlsx_header_rows",
        lambda: 5,
    )
    monkeypatch.setattr(
        checkpointing,
        "get_chunking_xlsx_overlap_rows",
        lambda: 3,
    )
    monkeypatch.setattr(
        checkpointing,
        "get_embedding_model",
        lambda: "text-embedding-3-large",
    )
    monkeypatch.setattr(
        checkpointing,
        "get_embedding_dimensions",
        lambda: 3072,
    )
    monkeypatch.setattr(
        checkpointing,
        "get_embedding_batch_size",
        lambda: 20,
    )
    monkeypatch.setattr(
        checkpointing,
        "build_stage_signature",
        lambda stage_name: f"sig-{stage_name}",
    )

    extraction_paths = checkpointing.get_stage_dependency_paths("extraction")

    assert extraction_paths
    assert checkpointing.get_stage_config("extraction") == {
        "model_config": {"stage": "extraction"},
        "vision_dpi_scale": 2.5,
    }
    assert checkpointing.get_stage_config("tokenization") == {
        "tokenizer_model": "o200k_base"
    }
    classification_config = checkpointing.get_stage_config("classification")
    assert isinstance(classification_config, dict)
    assert not classification_config
    assert checkpointing.get_stage_config("chunking") == {
        "model_config": {"stage": "chunking"},
        "tokenizer_model": "o200k_base",
        "chunking_config": {
            "embedding_token_limit": 8192,
            "truncation_token_limit": 80000,
            "max_retries": 3,
            "markdown_batch_size": 100,
            "xlsx_batch_size": 50,
            "xlsx_header_rows": 5,
            "xlsx_overlap_rows": 3,
        },
    }
    assert checkpointing.get_stage_config("doc_metadata") == {
        "model_config": {"stage": "doc_metadata"},
    }
    assert checkpointing.get_stage_config("embedding") == {
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 3072,
        "embedding_batch_size": 20,
    }
    persistence_config = checkpointing.get_stage_config("persistence")
    assert isinstance(persistence_config, dict)
    assert not persistence_config
    assert checkpointing.build_stage_signatures() == {
        "extraction": "sig-extraction",
        "tokenization": "sig-tokenization",
        "classification": "sig-classification",
        "chunking": "sig-chunking",
        "doc_metadata": "sig-doc_metadata",
        "section_detection": "sig-section_detection",
        "content_extraction": "sig-content_extraction",
        "section_summary": "sig-section_summary",
        "doc_summary": "sig-doc_summary",
        "embedding": "sig-embedding",
        "persistence": "sig-persistence",
    }


def test_enrichment_stage_prompt_dependencies_are_scoped():
    """Each enrichment stage depends only on its own prompt files."""
    doc_summary_paths = checkpointing.get_stage_dependency_paths("doc_summary")
    section_detection_paths = checkpointing.get_stage_dependency_paths(
        "section_detection"
    )

    doc_summary_names = {path.name for path in doc_summary_paths}
    section_detection_names = {path.name for path in section_detection_paths}

    assert "doc_summary.yaml" in doc_summary_names
    assert "doc_metadata.yaml" not in doc_summary_names
    assert "content_extraction.yaml" not in doc_summary_names
    assert "section_detection.yaml" in section_detection_names
    assert "subsection_detection.yaml" in section_detection_names
    assert "doc_summary.yaml" not in section_detection_names


def test_save_and_load_extraction_result(monkeypatch, tmp_path):
    """Round-trip stage artifacts through JSON."""
    monkeypatch.setattr(
        checkpointing,
        "get_document_cache_root",
        lambda: tmp_path / "artifacts",
    )
    result = ExtractionResult(
        file_path="/tmp/doc.pdf",
        filetype="pdf",
        pages=[
            PageResult(
                page_number=1,
                raw_content="Revenue",
                raw_token_count=12,
                embedding_token_count=12,
                token_count=12,
                token_tier="low",
                section_id="1",
                keywords=["revenue"],
                entities=["RBC"],
            )
        ],
        total_pages=1,
        data_source="pillar3",
        filter_1="2026",
        filter_2="Q1",
        raw_document_token_count=12,
        embedding_document_token_count=12,
        document_token_count=12,
        document_metadata={
            "title": "Report",
            "data_source": "pillar3",
            "filter_1": "2026",
            "filter_2": "Q1",
            "has_toc": False,
        },
        sections=[
            {
                "section_id": "1",
                "level": "section",
                "title": "Intro",
                "chunk_ids": ["1"],
            }
        ],
        content_units=[
            {
                "content_unit_id": "1",
                "section_id": "1",
                "keywords": ["revenue"],
                "entities": ["RBC"],
            }
        ],
    )

    artifact_path, checksum = checkpointing.save_extraction_result(
        result,
        "file-hash",
        "tokenization",
    )
    checkpoint = StageCheckpoint(
        document_version_id=1,
        stage_name="tokenization",
        status="succeeded",
        stage_signature="sig",
        artifact_path=artifact_path,
        artifact_checksum=checksum,
        error_message="",
    )

    loaded = checkpointing.load_extraction_result(checkpoint)

    assert loaded == result


def test_load_extraction_result_accepts_legacy_content_payload(tmp_path):
    """Support older artifacts that stored page text in content."""
    artifact_path = tmp_path / "legacy.json"
    artifact_path.write_text(
        """
        {
          "file_path": "/tmp/doc.pdf",
          "filetype": "pdf",
          "pages": [{
            "page_number": 1,
            "content": "legacy text",
            "token_count": 7
          }],
          "total_pages": 1,
          "document_token_count": 7
        }
        """.strip(),
        encoding="utf-8",
    )
    checkpoint = StageCheckpoint(
        document_version_id=1,
        stage_name="tokenization",
        status="succeeded",
        stage_signature="sig",
        artifact_path=str(artifact_path),
        artifact_checksum="",
        error_message="",
    )

    loaded = checkpointing.load_extraction_result(checkpoint)

    assert loaded.file_path == "/tmp/doc.pdf"
    assert loaded.pages == [
        PageResult(
            page_number=1,
            raw_content="legacy text",
            raw_token_count=7,
            embedding_token_count=7,
            token_count=7,
        )
    ]
    assert loaded.raw_document_token_count == 7
    assert loaded.embedding_document_token_count == 7
    assert loaded.document_token_count == 7


def test_checkpoint_is_usable_validates_status_signature_and_artifact(
    monkeypatch, tmp_path
):
    """Reuse only successful checkpoints with intact matching artifacts."""
    monkeypatch.setattr(
        checkpointing,
        "get_document_cache_root",
        lambda: tmp_path / "artifacts",
    )
    result = ExtractionResult(
        file_path="/tmp/doc.pdf",
        filetype="pdf",
        pages=[PageResult(page_number=1, raw_content="ok")],
        total_pages=1,
    )
    artifact_path, checksum = checkpointing.save_extraction_result(
        result,
        "file-hash",
        "extraction",
    )
    checkpoint = StageCheckpoint(
        document_version_id=1,
        stage_name="extraction",
        status="succeeded",
        stage_signature="expected",
        artifact_path=artifact_path,
        artifact_checksum=checksum,
        error_message="",
    )

    assert checkpointing.checkpoint_is_usable(checkpoint, "expected") is True
    assert checkpointing.checkpoint_is_usable(checkpoint, "other") is False
    assert (
        checkpointing.checkpoint_is_usable(
            StageCheckpoint(
                document_version_id=1,
                stage_name="extraction",
                status="failed",
                stage_signature="expected",
                artifact_path=artifact_path,
                artifact_checksum=checksum,
                error_message="boom",
            ),
            "expected",
        )
        is False
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            """
            {
              "file_path": "/tmp/doc.pdf",
              "filetype": "pdf",
              "pages": {},
              "total_pages": 1
            }
            """.strip(),
            "Artifact payload.pages must be a list",
        ),
        (
            """
            {
              "file_path": "/tmp/doc.pdf",
              "filetype": "pdf",
              "pages": [{"page_number": "bad", "raw_content": "x"}],
              "total_pages": 1
            }
            """.strip(),
            "pages[0].page_number must be an integer",
        ),
        (
            """
            {
              "file_path": 123,
              "filetype": "pdf",
              "pages": [{"page_number": 1, "raw_content": "x"}],
              "total_pages": 1
            }
            """.strip(),
            "Artifact payload.file_path must be a string",
        ),
        (
            "[]",
            "Artifact payload must be an object",
        ),
        (
            """
            {
              "file_path": "/tmp/doc.pdf",
              "filetype": "pdf",
              "pages": [{"raw_content": "x"}],
              "total_pages": 1
            }
            """.strip(),
            "pages[0] missing 'page_number'",
        ),
        (
            """
            {
              "filetype": "pdf",
              "pages": [{"page_number": 1, "raw_content": "x"}],
              "total_pages": 1
            }
            """.strip(),
            "Artifact payload missing 'file_path'",
        ),
    ],
)
def test_load_extraction_result_rejects_malformed_payloads(
    tmp_path, payload, message
):
    """Normalize malformed artifact payloads to ValueError."""
    artifact_path = tmp_path / "broken.json"
    artifact_path.write_text(payload, encoding="utf-8")
    checkpoint = StageCheckpoint(
        document_version_id=1,
        stage_name="extraction",
        status="succeeded",
        stage_signature="sig",
        artifact_path=str(artifact_path),
        artifact_checksum="",
        error_message="",
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        checkpointing.load_extraction_result(checkpoint)


def test_load_extraction_result_and_checkpoint_usable_error_paths(
    monkeypatch, tmp_path
):
    """Raise on missing or corrupt artifacts and allow checksumless files."""
    missing = StageCheckpoint(
        document_version_id=1,
        stage_name="extraction",
        status="succeeded",
        stage_signature="sig",
        artifact_path=str(tmp_path / "missing.json"),
        artifact_checksum="checksum",
        error_message="",
    )

    with pytest.raises(FileNotFoundError, match="Missing stage artifact"):
        checkpointing.load_extraction_result(missing)

    monkeypatch.setattr(
        checkpointing,
        "get_document_cache_root",
        lambda: tmp_path / "artifacts",
    )
    result = ExtractionResult(
        file_path="/tmp/doc.pdf",
        filetype="pdf",
        pages=[PageResult(page_number=1, raw_content="ok")],
        total_pages=1,
    )
    artifact_path, _checksum = checkpointing.save_extraction_result(
        result,
        "hash-123",
        "extraction",
    )
    bad_checksum = StageCheckpoint(
        document_version_id=1,
        stage_name="extraction",
        status="succeeded",
        stage_signature="sig",
        artifact_path=artifact_path,
        artifact_checksum="wrong",
        error_message="",
    )
    no_checksum = StageCheckpoint(
        document_version_id=1,
        stage_name="extraction",
        status="succeeded",
        stage_signature="sig",
        artifact_path=artifact_path,
        artifact_checksum="",
        error_message="",
    )

    with pytest.raises(RuntimeError, match="Artifact checksum mismatch"):
        checkpointing.load_extraction_result(bad_checksum)

    assert checkpointing.checkpoint_is_usable(no_checksum, "sig") is True
    assert (
        checkpointing.checkpoint_is_usable(
            StageCheckpoint(
                document_version_id=1,
                stage_name="extraction",
                status="succeeded",
                stage_signature="sig",
                artifact_path=str(tmp_path / "gone.json"),
                artifact_checksum="",
                error_message="",
            ),
            "sig",
        )
        is False
    )
