"""Tests for the ingestion entry point."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ingestion import main as main_module
from ingestion.utils.checkpointing import FileExecutionPlan, PipelineCliOptions
from ingestion.utils.file_types import (
    DiscoveryDiff,
    DocumentVersion,
    ExtractionResult,
    PrunableDocumentVersion,
    StageCheckpoint,
)


def _all_stage_signatures():
    """Build a signature dict covering all pipeline stages."""
    return {
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


def _make_result(path="/tmp/doc.pdf"):
    """Build a minimal ExtractionResult."""
    return ExtractionResult(
        file_path=path,
        filetype="pdf",
        pages=[],
        total_pages=0,
    )


def _empty_diff():
    """Build an empty discovery diff."""
    return DiscoveryDiff(new=[], modified=[], deleted=[])


def _make_version(record, document_version_id=1):
    """Build a current DocumentVersion from a FileRecord."""
    return DocumentVersion(
        document_version_id=document_version_id,
        file_path=record.file_path,
        data_source=record.data_source,
        filter_1=record.filter_1,
        filter_2=record.filter_2,
        filter_3=record.filter_3,
        filename=record.filename,
        filetype=record.filetype,
        file_size=record.file_size,
        date_last_modified=record.date_last_modified,
        file_hash=record.file_hash,
        is_current=True,
    )


def _make_checkpoint(stage_name="extraction", status="succeeded"):
    """Build a stage checkpoint test double."""
    return StageCheckpoint(
        document_version_id=1,
        stage_name=stage_name,
        status=status,
        stage_signature=f"sig-{stage_name}",
        artifact_path="/tmp/artifact.json",
        artifact_checksum="checksum",
        error_message="",
    )


def test_plan_file_resumes_from_next_incomplete_stage(
    monkeypatch, file_record_factory
):
    """Resume from tokenization when extraction is already reusable."""
    record = file_record_factory(file_hash="hash-123")
    extraction_checkpoint = _make_checkpoint("extraction")
    monkeypatch.setattr(
        main_module,
        "checkpoint_is_usable",
        lambda checkpoint, signature: (
            checkpoint.status == "succeeded"
            and checkpoint.stage_signature == signature
        ),
    )

    plan = main_module.plan_file(
        record,
        {"extraction": extraction_checkpoint},
        {
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
        },
        PipelineCliOptions(target_stage="chunking"),
    )

    assert plan.start_stage == "tokenization"
    assert plan.resume_checkpoint == extraction_checkpoint
    assert plan.target_stage == "chunking"


def test_plan_file_skips_when_target_stage_is_complete(
    monkeypatch, file_record_factory
):
    """Skip files that already completed the requested target stage."""
    record = file_record_factory(file_hash="hash-123")
    monkeypatch.setattr(
        main_module,
        "checkpoint_is_usable",
        lambda checkpoint, signature: checkpoint.stage_signature == signature,
    )

    plan = main_module.plan_file(
        record,
        {
            "extraction": _make_checkpoint("extraction"),
            "tokenization": _make_checkpoint("tokenization"),
            "classification": _make_checkpoint("classification"),
            "chunking": _make_checkpoint("chunking"),
        },
        {
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
        },
        PipelineCliOptions(target_stage="chunking"),
    )

    assert plan.should_run is False
    assert plan.start_stage == ""


def test_plan_file_blocks_for_forced_downstream_without_upstream(
    monkeypatch, file_record_factory
):
    """Reject section reruns when the prerequisite checkpoint is missing."""
    record = file_record_factory(file_hash="hash-123")
    monkeypatch.setattr(
        main_module,
        "checkpoint_is_usable",
        lambda checkpoint, signature: checkpoint.stage_signature == signature,
    )

    plan = main_module.plan_file(
        record,
        {},
        {
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
        },
        PipelineCliOptions(
            target_stage="chunking",
            forced_start_stage="tokenization",
        ),
    )

    assert plan.should_run is False
    assert "Missing valid extraction checkpoint" in plan.error_message


def test_plan_file_force_all_and_forced_extraction(
    monkeypatch, file_record_factory
):
    """Allow full invalidation and explicit reruns from the first stage."""
    record = file_record_factory(file_hash="hash-123")
    monkeypatch.setattr(
        main_module,
        "checkpoint_is_usable",
        lambda checkpoint, signature: checkpoint.stage_signature == signature,
    )

    force_all_plan = main_module.plan_file(
        record,
        {},
        {
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
        },
        PipelineCliOptions(target_stage="chunking", force_all=True),
    )
    forced_extraction_plan = main_module.plan_file(
        record,
        {},
        {
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
        },
        PipelineCliOptions(
            target_stage="extraction",
            forced_start_stage="extraction",
        ),
    )

    assert force_all_plan.start_stage == "extraction"
    assert force_all_plan.invalidation_stage == "extraction"
    assert forced_extraction_plan.start_stage == "extraction"


def test_plan_file_forced_downstream_with_valid_upstream(
    monkeypatch, file_record_factory
):
    """Resume a forced downstream rerun with a valid upstream checkpoint."""
    record = file_record_factory(file_hash="hash-123")
    extraction_checkpoint = _make_checkpoint("extraction")
    monkeypatch.setattr(
        main_module,
        "checkpoint_is_usable",
        lambda checkpoint, signature: checkpoint.stage_signature == signature,
    )

    plan = main_module.plan_file(
        record,
        {"extraction": extraction_checkpoint},
        {
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
        },
        PipelineCliOptions(
            target_stage="chunking",
            forced_start_stage="tokenization",
        ),
    )

    assert plan.start_stage == "tokenization"
    assert plan.resume_checkpoint == extraction_checkpoint


def test_plan_file_restarts_when_checkpoint_is_not_reusable(
    monkeypatch, file_record_factory
):
    """Restart from extraction when the first checkpoint is stale."""
    record = file_record_factory(file_hash="hash-123")
    monkeypatch.setattr(
        main_module,
        "checkpoint_is_usable",
        lambda _checkpoint, _signature: False,
    )

    plan = main_module.plan_file(
        record,
        {
            "extraction": _make_checkpoint("extraction"),
            "tokenization": _make_checkpoint("tokenization"),
            "classification": _make_checkpoint("classification"),
            "chunking": _make_checkpoint("chunking"),
        },
        {
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
        },
        PipelineCliOptions(target_stage="chunking"),
    )

    assert plan.start_stage == "extraction"


def test_run_stage_routes_and_errors(monkeypatch, file_record_factory):
    """Dispatch stages, enforce inputs, and reject bad stage names."""
    record = file_record_factory()
    result = _make_result(record.file_path)
    monkeypatch.setattr(
        main_module,
        "extract_file",
        lambda _record, _llm: result,
    )
    monkeypatch.setattr(
        main_module,
        "tokenize_result",
        lambda incoming: incoming,
    )
    monkeypatch.setattr(
        main_module,
        "classify_result",
        lambda incoming: incoming,
    )
    monkeypatch.setattr(
        main_module,
        "chunk_result",
        lambda incoming, _llm: incoming,
    )
    monkeypatch.setattr(
        main_module,
        "enrich_doc_metadata",
        lambda incoming, _llm: incoming,
    )
    monkeypatch.setattr(
        main_module,
        "summarize_document",
        lambda incoming, _llm: incoming,
    )
    monkeypatch.setattr(
        main_module,
        "persist_enrichment",
        lambda incoming, _llm: incoming,
    )

    assert (
        main_module.run_stage("extraction", record, object(), None) == result
    )
    assert (
        main_module.run_stage("tokenization", record, object(), result)
        == result
    )
    with pytest.raises(RuntimeError, match="requires extraction output"):
        main_module.run_stage("tokenization", record, object(), None)

    assert (
        main_module.run_stage("classification", record, object(), result)
        == result
    )
    with pytest.raises(RuntimeError, match="requires tokenization output"):
        main_module.run_stage("classification", record, object(), None)

    assert (
        main_module.run_stage("chunking", record, object(), result) == result
    )
    with pytest.raises(RuntimeError, match="requires classification output"):
        main_module.run_stage("chunking", record, object(), None)

    assert (
        main_module.run_stage("doc_metadata", record, object(), result)
        == result
    )
    with pytest.raises(RuntimeError, match="requires chunking output"):
        main_module.run_stage("doc_metadata", record, object(), None)

    assert (
        main_module.run_stage("section_detection", record, object(), result)
        == result
    )
    with pytest.raises(RuntimeError, match="requires doc metadata output"):
        main_module.run_stage("section_detection", record, object(), None)

    assert (
        main_module.run_stage("content_extraction", record, object(), result)
        == result
    )
    with pytest.raises(
        RuntimeError, match="requires section detection output"
    ):
        main_module.run_stage("content_extraction", record, object(), None)

    assert (
        main_module.run_stage("section_summary", record, object(), result)
        == result
    )
    with pytest.raises(
        RuntimeError, match="requires content extraction output"
    ):
        main_module.run_stage("section_summary", record, object(), None)

    assert (
        main_module.run_stage("doc_summary", record, object(), result)
        == result
    )
    with pytest.raises(RuntimeError, match="requires section summary output"):
        main_module.run_stage("doc_summary", record, object(), None)

    assert (
        main_module.run_stage("embedding", record, object(), result) == result
    )
    with pytest.raises(RuntimeError, match="requires doc summary output"):
        main_module.run_stage("embedding", record, object(), None)

    assert (
        main_module.run_stage("persistence", record, object(), result)
        == result
    )
    with pytest.raises(RuntimeError, match="requires embedding output"):
        main_module.run_stage("persistence", record, object(), None)

    with pytest.raises(ValueError, match="Unsupported stage"):
        main_module.run_stage("bad-stage", record, object(), result)


def test_record_selected_applies_file_and_glob_filters(file_record_factory):
    """Honor exact file path and glob-based file selection."""
    record = file_record_factory(
        filename="deck.pdf",
        file_path="/tmp/source/deck.pdf",
    )

    assert (
        main_module.record_selected(
            record,
            PipelineCliOptions(
                target_stage="chunking",
                file_paths=("/tmp/source/deck.pdf",),
            ),
        )
        is True
    )
    assert (
        main_module.record_selected(
            record,
            PipelineCliOptions(
                target_stage="chunking",
                file_paths=("/tmp/source/other.pdf",),
            ),
        )
        is False
    )
    assert (
        main_module.record_selected(
            record,
            PipelineCliOptions(
                target_stage="chunking",
                glob_patterns=("*.pdf",),
            ),
        )
        is True
    )
    assert (
        main_module.record_selected(
            record,
            PipelineCliOptions(
                target_stage="chunking",
                glob_patterns=("*.xlsx",),
            ),
        )
        is False
    )


def test_build_parser_accepts_file_selection_flags():
    """Parse repeatable file-path and glob filters from the CLI."""
    parser = main_module.build_parser()

    args = parser.parse_args(
        [
            "--to-stage",
            "extraction",
            "--file-path",
            "/tmp/a.pdf",
            "--file-path",
            "/tmp/b.pdf",
            "--glob",
            "*.pdf",
            "--glob",
            "bank/*",
        ]
    )

    assert args.to_stage == "extraction"
    assert args.file_path == ["/tmp/a.pdf", "/tmp/b.pdf"]
    assert args.glob == ["*.pdf", "bank/*"]


def test_process_file_resumes_and_persists_checkpoints(
    monkeypatch, file_record_factory
):
    """Load upstream artifacts, run the stage, and persist success."""
    calls = []
    record = file_record_factory(file_hash="hash-123")
    checkpoint = _make_checkpoint("extraction")
    version = _make_version(record, document_version_id=9)
    conn = SimpleNamespace(close=lambda: calls.append("close"))

    monkeypatch.setattr(main_module, "get_connection", lambda: conn)
    monkeypatch.setattr(
        main_module,
        "register_document_version",
        lambda _conn, _record: version,
    )
    monkeypatch.setattr(
        main_module,
        "upsert_catalog_record",
        lambda _conn, _record: calls.append("catalog"),
    )
    monkeypatch.setattr(
        main_module,
        "clear_stage_checkpoints",
        lambda _conn, doc_id, stage: calls.append((doc_id, stage)),
    )
    monkeypatch.setattr(
        main_module,
        "load_extraction_result",
        lambda _checkpoint: _make_result(record.file_path),
    )
    monkeypatch.setattr(
        main_module,
        "tokenize_result",
        lambda result: result,
    )
    monkeypatch.setattr(
        main_module,
        "classify_result",
        lambda result: result,
    )
    monkeypatch.setattr(
        main_module,
        "chunk_result",
        lambda result, _llm: result,
    )
    monkeypatch.setattr(
        main_module,
        "save_extraction_result",
        lambda result, _file_hash, _stage_name: (
            "/tmp/tokenization.json",
            "artifact-checksum",
        ),
    )
    started = []
    monkeypatch.setattr(
        main_module,
        "mark_stage_checkpoint_started",
        lambda *args: started.append(args[2:]),
    )
    success = []
    monkeypatch.setattr(
        main_module,
        "mark_stage_checkpoint_succeeded",
        lambda *args: success.append(args[2:]),
    )
    monkeypatch.setattr(
        main_module,
        "mark_stage_checkpoint_failed",
        lambda *args: calls.append(("failed", args[2])),
    )

    result = main_module.process_file(
        record,
        object(),
        FileExecutionPlan(
            record_key=record.file_path,
            start_stage="tokenization",
            target_stage="tokenization",
            resume_checkpoint=checkpoint,
            invalidation_stage="tokenization",
        ),
        {
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
        },
    )

    assert result.file_path == record.file_path
    assert (9, "tokenization") in calls
    assert started == [("tokenization", "sig-tokenization")]
    assert success == [
        (
            "tokenization",
            "sig-tokenization",
            "/tmp/tokenization.json",
            "artifact-checksum",
        )
    ]
    assert "catalog" in calls
    assert "close" in calls


def test_process_file_marks_resume_failures(monkeypatch, file_record_factory):
    """Record a failed checkpoint when resume artifact loading fails."""
    calls = []
    record = file_record_factory(file_hash="hash-123")
    version = _make_version(record, document_version_id=5)
    conn = SimpleNamespace(close=lambda: calls.append("close"))

    monkeypatch.setattr(main_module, "get_connection", lambda: conn)
    monkeypatch.setattr(
        main_module,
        "register_document_version",
        lambda _conn, _record: version,
    )
    monkeypatch.setattr(
        main_module,
        "upsert_catalog_record",
        lambda _conn, _record: None,
    )
    monkeypatch.setattr(
        main_module,
        "load_extraction_result",
        lambda _checkpoint: (_ for _ in ()).throw(
            FileNotFoundError("missing artifact")
        ),
    )
    monkeypatch.setattr(
        main_module,
        "mark_stage_checkpoint_failed",
        lambda *args: calls.append(args[2:]),
    )

    with pytest.raises(RuntimeError, match="Failed to resume"):
        main_module.process_file(
            record,
            object(),
            FileExecutionPlan(
                record_key=record.file_path,
                start_stage="tokenization",
                target_stage="tokenization",
                resume_checkpoint=_make_checkpoint("extraction"),
            ),
            {
                "extraction": "sig-extraction",
                "tokenization": "sig-tokenization",
                "classification": "sig-classification",
                "chunking": "sig-chunking",
            },
        )

    assert calls[0] == (
        "tokenization",
        "sig-tokenization",
        "Failed to load upstream checkpoint: missing artifact",
    )
    assert "close" in calls


def test_process_file_marks_unexpected_resume_load_failures(
    monkeypatch, file_record_factory
):
    """Wrap any resume-load exception so one bad artifact does not abort."""
    calls = []
    record = file_record_factory(file_hash="hash-123")
    version = _make_version(record, document_version_id=15)
    conn = SimpleNamespace(close=lambda: calls.append("close"))

    monkeypatch.setattr(main_module, "get_connection", lambda: conn)
    monkeypatch.setattr(
        main_module,
        "register_document_version",
        lambda _conn, _record: version,
    )
    monkeypatch.setattr(
        main_module,
        "upsert_catalog_record",
        lambda _conn, _record: None,
    )
    monkeypatch.setattr(
        main_module,
        "load_extraction_result",
        lambda _checkpoint: (_ for _ in ()).throw(TypeError("bad payload")),
    )
    monkeypatch.setattr(
        main_module,
        "mark_stage_checkpoint_failed",
        lambda *args: calls.append(args[2:]),
    )

    with pytest.raises(RuntimeError, match="Failed to resume"):
        main_module.process_file(
            record,
            object(),
            FileExecutionPlan(
                record_key=record.file_path,
                start_stage="tokenization",
                target_stage="tokenization",
                resume_checkpoint=_make_checkpoint("extraction"),
            ),
            {
                "extraction": "sig-extraction",
                "tokenization": "sig-tokenization",
                "classification": "sig-classification",
                "chunking": "sig-chunking",
            },
        )

    assert calls[0] == (
        "tokenization",
        "sig-tokenization",
        "Failed to load upstream checkpoint: bad payload",
    )
    assert "close" in calls


def test_process_file_deletes_artifact_on_db_commit_failure(
    monkeypatch, file_record_factory, tmp_path
):
    """Roll back orphan artifact files when the DB checkpoint commit fails."""
    record = file_record_factory(file_hash="hash-123")
    version = _make_version(record, document_version_id=6)
    conn = SimpleNamespace(close=lambda: None)
    calls: list = []

    artifact_file = tmp_path / "artifact.json"
    artifact_file.write_text('{"stage": "extraction"}')

    monkeypatch.setattr(main_module, "get_connection", lambda: conn)
    monkeypatch.setattr(
        main_module,
        "register_document_version",
        lambda _conn, _record: version,
    )
    monkeypatch.setattr(
        main_module,
        "upsert_catalog_record",
        lambda _conn, _record: None,
    )
    monkeypatch.setattr(
        main_module,
        "mark_stage_checkpoint_started",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        main_module,
        "extract_file",
        lambda _record, _llm: SimpleNamespace(file_path=_record.file_path),
    )
    monkeypatch.setattr(
        main_module,
        "save_extraction_result",
        lambda _result, _hash, _stage: (str(artifact_file), "checksum-abc"),
    )

    def failing_commit(*_args, **_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(
        main_module, "mark_stage_checkpoint_succeeded", failing_commit
    )
    monkeypatch.setattr(
        main_module,
        "mark_stage_checkpoint_failed",
        lambda *args: calls.append(args[2:]),
    )

    with pytest.raises(RuntimeError, match="Stage extraction failed"):
        main_module.process_file(
            record,
            object(),
            FileExecutionPlan(
                record_key=record.file_path,
                start_stage="extraction",
                target_stage="extraction",
            ),
            {
                "extraction": "sig-extraction",
                "tokenization": "sig-tokenization",
                "classification": "sig-classification",
                "chunking": "sig-chunking",
            },
        )

    assert not artifact_file.exists()
    assert calls and calls[0][0] == "extraction"


def test_process_file_wraps_stage_failures(monkeypatch, file_record_factory):
    """Persist stage failure details and re-raise as RuntimeError."""
    calls = []
    record = file_record_factory(file_hash="hash-123")
    version = _make_version(record, document_version_id=6)
    conn = SimpleNamespace(close=lambda: calls.append("close"))

    monkeypatch.setattr(main_module, "get_connection", lambda: conn)
    monkeypatch.setattr(
        main_module,
        "register_document_version",
        lambda _conn, _record: version,
    )
    monkeypatch.setattr(
        main_module,
        "upsert_catalog_record",
        lambda _conn, _record: None,
    )
    monkeypatch.setattr(
        main_module,
        "mark_stage_checkpoint_started",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        main_module,
        "extract_file",
        lambda _record, _llm: (_ for _ in ()).throw(ValueError("bad extract")),
    )
    monkeypatch.setattr(
        main_module,
        "mark_stage_checkpoint_failed",
        lambda *args: calls.append(args[2:]),
    )

    with pytest.raises(RuntimeError, match="Stage extraction failed"):
        main_module.process_file(
            record,
            object(),
            FileExecutionPlan(
                record_key=record.file_path,
                start_stage="extraction",
                target_stage="extraction",
            ),
            {
                "extraction": "sig-extraction",
                "tokenization": "sig-tokenization",
                "classification": "sig-classification",
                "chunking": "sig-chunking",
            },
        )

    assert calls[0] == ("extraction", "sig-extraction", "bad extract")
    assert "close" in calls


def test_process_file_raises_when_no_stage_executes(
    monkeypatch, file_record_factory
):
    """Guard against empty execution ranges after planning."""
    calls = []
    record = file_record_factory(file_hash="hash-123")
    version = _make_version(record, document_version_id=8)
    conn = SimpleNamespace(close=lambda: calls.append("close"))

    monkeypatch.setattr(main_module, "get_connection", lambda: conn)
    monkeypatch.setattr(
        main_module,
        "register_document_version",
        lambda _conn, _record: version,
    )
    monkeypatch.setattr(
        main_module,
        "upsert_catalog_record",
        lambda _conn, _record: None,
    )
    monkeypatch.setattr(main_module, "get_stage_range", lambda *_args: ())

    with pytest.raises(RuntimeError, match="No stage executed"):
        main_module.process_file(
            record,
            object(),
            FileExecutionPlan(
                record_key=record.file_path,
                start_stage="extraction",
                target_stage="extraction",
            ),
            {
                "extraction": "sig-extraction",
                "tokenization": "sig-tokenization",
                "classification": "sig-classification",
                "chunking": "sig-chunking",
            },
        )

    assert "close" in calls


def test_prepare_execution_queue_handles_empty_and_blocked_files(
    monkeypatch, file_record_factory
):
    """Return empty queues for no files and count blocked plans."""
    logger = Mock()
    monkeypatch.setattr(
        main_module,
        "scan_filesystem",
        lambda _path: SimpleNamespace(supported=[]),
    )
    monkeypatch.setattr(main_module, "get_data_source_path", lambda: "/tmp")

    planned, skipped, blocked = main_module.prepare_execution_queue(
        object(),
        {
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
        },
        PipelineCliOptions(target_stage="chunking"),
        logger,
    )

    assert not planned
    assert skipped == 0
    assert blocked == 0
    logger.info.assert_called_once()

    blocked_record = file_record_factory(file_path="/tmp/blocked.pdf")
    monkeypatch.setattr(
        main_module,
        "scan_filesystem",
        lambda _path: SimpleNamespace(supported=[blocked_record]),
    )
    monkeypatch.setattr(
        main_module,
        "fetch_current_document_versions",
        lambda _conn: [],
    )
    monkeypatch.setattr(
        main_module,
        "fetch_stage_checkpoints",
        lambda _conn: [],
    )
    monkeypatch.setattr(
        main_module,
        "compute_file_hash",
        lambda _path: "hash-123",
    )
    monkeypatch.setattr(
        main_module,
        "plan_file",
        lambda *_args: FileExecutionPlan(
            record_key=blocked_record.file_path,
            start_stage="",
            target_stage="chunking",
            error_message="blocked",
        ),
    )
    logger = Mock()

    planned, skipped, blocked = main_module.prepare_execution_queue(
        object(),
        {
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
        },
        PipelineCliOptions(target_stage="chunking"),
        logger,
    )

    assert not planned
    assert skipped == 0
    assert blocked == 1
    logger.error.assert_called_once()


def test_prepare_execution_queue_skips_unselected_files(
    monkeypatch, file_record_factory
):
    """Ignore discovered files that do not match CLI selection filters."""
    logger = Mock()
    record = file_record_factory(file_path="/tmp/skip-me.pdf")
    monkeypatch.setattr(
        main_module,
        "scan_filesystem",
        lambda _path: SimpleNamespace(supported=[record]),
    )
    monkeypatch.setattr(main_module, "get_data_source_path", lambda: "/tmp")
    monkeypatch.setattr(
        main_module,
        "fetch_current_document_versions",
        lambda _conn: [],
    )
    monkeypatch.setattr(
        main_module,
        "fetch_stage_checkpoints",
        lambda _conn: [],
    )
    compute_calls = []
    monkeypatch.setattr(
        main_module,
        "compute_file_hash",
        lambda _path: compute_calls.append(True) or "hash-123",
    )

    planned, skipped, blocked = main_module.prepare_execution_queue(
        object(),
        {
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
        },
        PipelineCliOptions(
            target_stage="chunking",
            file_paths=("/tmp/other.pdf",),
        ),
        logger,
    )

    assert not planned
    assert skipped == 0
    assert blocked == 0
    assert not compute_calls


def test_prepare_execution_queue_reuses_discovery_files_and_hashes(
    monkeypatch, file_record_factory
):
    """Reuse discovery scan input and stored hashes for unchanged files."""
    logger = Mock()
    record = file_record_factory(
        file_path="/tmp/keep.pdf",
        file_size=123,
        date_last_modified=456.0,
    )
    version = _make_version(record, document_version_id=7)
    version.file_hash = "stored-hash"
    scan_calls = []
    compute_calls = []

    monkeypatch.setattr(
        main_module,
        "scan_filesystem",
        lambda _path: scan_calls.append(True) or SimpleNamespace(supported=[]),
    )
    monkeypatch.setattr(
        main_module,
        "fetch_current_document_versions",
        lambda _conn: [version],
    )
    monkeypatch.setattr(
        main_module,
        "fetch_stage_checkpoints",
        lambda _conn: [],
    )
    monkeypatch.setattr(
        main_module,
        "compute_file_hash",
        lambda _path: compute_calls.append(True) or "computed-hash",
    )
    monkeypatch.setattr(
        main_module,
        "plan_file",
        lambda *_args: FileExecutionPlan(
            record_key=record.file_path,
            start_stage="extraction",
            target_stage="chunking",
        ),
    )

    planned, skipped, blocked = main_module.prepare_execution_queue(
        object(),
        {
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
        },
        PipelineCliOptions(target_stage="chunking"),
        logger,
        discovered_files=[record],
    )

    assert [item[0].file_hash for item in planned] == ["stored-hash"]
    assert skipped == 0
    assert blocked == 0
    assert not scan_calls
    assert not compute_calls


def test_prune_non_current_versions_deletes_artifact_directories(
    monkeypatch, tmp_path
):
    """Remove stale version rows and their artifact directories."""
    logger = Mock()
    directory = tmp_path / "ab" / "doc-key"
    directory.mkdir(parents=True)
    (directory / "extraction.json").write_text("{}", encoding="utf-8")
    (directory / "tokenization.json").write_text("{}", encoding="utf-8")

    stale_version = PrunableDocumentVersion(
        document_version_id=7,
        file_path="/tmp/doc.pdf",
        artifact_paths=[
            str(directory / "extraction.json"),
            str(directory / "tokenization.json"),
        ],
    )
    deleted = []
    monkeypatch.setattr(
        main_module,
        "get_non_current_version_retention_count",
        lambda: 2,
    )
    monkeypatch.setattr(
        main_module,
        "fetch_prunable_document_versions",
        lambda _conn, retain_non_current: (
            deleted.append(("fetch", retain_non_current)) or [stale_version]
        ),
    )
    monkeypatch.setattr(
        main_module,
        "delete_document_versions",
        lambda _conn, ids: deleted.append(("delete", ids)),
    )

    main_module.prune_non_current_versions(object(), logger)

    assert deleted == [("fetch", 2), ("delete", [7])]
    assert directory.exists() is False
    logger.info.assert_called_once_with(
        "Pruned %d non-current document versions",
        1,
    )


def test_prune_non_current_versions_noops_when_nothing_is_stale(monkeypatch):
    """Skip deletion work when retention pruning finds no old versions."""
    logger = Mock()
    deleted = []
    monkeypatch.setattr(
        main_module,
        "get_non_current_version_retention_count",
        lambda: 1,
    )
    monkeypatch.setattr(
        main_module,
        "fetch_prunable_document_versions",
        lambda _conn, _retain_non_current: [],
    )
    monkeypatch.setattr(
        main_module,
        "delete_document_versions",
        lambda _conn, _ids: deleted.append(True),
    )

    main_module.prune_non_current_versions(object(), logger)

    assert not deleted
    logger.info.assert_not_called()


def test_main_processes_queued_files_and_skips_completed(
    monkeypatch, file_record_factory
):
    """Queue only incomplete files and still archive and release the lock."""
    calls = []
    conn = SimpleNamespace(close=lambda: calls.append("close"))
    llm = object()
    queued = file_record_factory(
        filename="queued.pdf", file_path="/tmp/queued.pdf"
    )
    skipped = file_record_factory(
        filename="skipped.pdf", file_path="/tmp/skipped.pdf"
    )

    monkeypatch.setattr(
        main_module,
        "run_startup",
        Mock(return_value=(conn, llm)),
    )
    monkeypatch.setattr(
        main_module,
        "run_discovery",
        Mock(return_value=_empty_diff()),
    )
    monkeypatch.setattr(
        main_module,
        "scan_filesystem",
        lambda _path: SimpleNamespace(supported=[queued, skipped]),
    )
    monkeypatch.setattr(main_module, "get_data_source_path", lambda: "/tmp")
    monkeypatch.setattr(
        main_module,
        "compute_file_hash",
        lambda path: "queued-hash" if "queued" in path else "skipped-hash",
    )
    monkeypatch.setattr(
        main_module,
        "build_stage_signatures",
        lambda: {
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
        },
    )
    skipped_version = _make_version(skipped, document_version_id=2)
    skipped_version.file_hash = "skipped-hash"
    monkeypatch.setattr(
        main_module,
        "fetch_current_document_versions",
        lambda _conn: [skipped_version],
    )
    monkeypatch.setattr(
        main_module,
        "fetch_stage_checkpoints",
        lambda _conn: [
            StageCheckpoint(
                document_version_id=2,
                stage_name="extraction",
                status="succeeded",
                stage_signature="sig-extraction",
                artifact_path="/tmp/extraction.json",
                artifact_checksum="checksum",
                error_message="",
            ),
            StageCheckpoint(
                document_version_id=2,
                stage_name="tokenization",
                status="succeeded",
                stage_signature="sig-tokenization",
                artifact_path="/tmp/tokenization.json",
                artifact_checksum="checksum",
                error_message="",
            ),
            StageCheckpoint(
                document_version_id=2,
                stage_name="classification",
                status="succeeded",
                stage_signature="sig-classification",
                artifact_path="/tmp/classification.json",
                artifact_checksum="checksum",
                error_message="",
            ),
            StageCheckpoint(
                document_version_id=2,
                stage_name="chunking",
                status="succeeded",
                stage_signature="sig-chunking",
                artifact_path="/tmp/chunking.json",
                artifact_checksum="checksum",
                error_message="",
            ),
            StageCheckpoint(
                document_version_id=2,
                stage_name="doc_metadata",
                status="succeeded",
                stage_signature="sig-doc_metadata",
                artifact_path="/tmp/doc_metadata.json",
                artifact_checksum="checksum",
                error_message="",
            ),
            StageCheckpoint(
                document_version_id=2,
                stage_name="section_detection",
                status="succeeded",
                stage_signature="sig-section_detection",
                artifact_path="/tmp/section_detection.json",
                artifact_checksum="checksum",
                error_message="",
            ),
            StageCheckpoint(
                document_version_id=2,
                stage_name="content_extraction",
                status="succeeded",
                stage_signature="sig-content_extraction",
                artifact_path="/tmp/content_extraction.json",
                artifact_checksum="checksum",
                error_message="",
            ),
            StageCheckpoint(
                document_version_id=2,
                stage_name="section_summary",
                status="succeeded",
                stage_signature="sig-section_summary",
                artifact_path="/tmp/section_summary.json",
                artifact_checksum="checksum",
                error_message="",
            ),
            StageCheckpoint(
                document_version_id=2,
                stage_name="doc_summary",
                status="succeeded",
                stage_signature="sig-doc_summary",
                artifact_path="/tmp/doc_summary.json",
                artifact_checksum="checksum",
                error_message="",
            ),
            StageCheckpoint(
                document_version_id=2,
                stage_name="embedding",
                status="succeeded",
                stage_signature="sig-embedding",
                artifact_path="/tmp/embedding.json",
                artifact_checksum="checksum",
                error_message="",
            ),
            StageCheckpoint(
                document_version_id=2,
                stage_name="persistence",
                status="succeeded",
                stage_signature="sig-persistence",
                artifact_path="/tmp/persistence.json",
                artifact_checksum="checksum",
                error_message="",
            ),
        ],
    )
    monkeypatch.setattr(
        main_module,
        "checkpoint_is_usable",
        lambda checkpoint, signature: (
            checkpoint.stage_signature == signature
        ),
    )
    monkeypatch.setattr(
        main_module,
        "get_max_workers",
        Mock(return_value=1),
    )
    processed = []
    monkeypatch.setattr(
        main_module,
        "process_file",
        lambda record, _llm, plan, _sigs: (
            processed.append(
                (
                    record.filename,
                    plan.start_stage,
                    plan.target_stage,
                )
            )
            or _make_result(record.file_path)
        ),
    )
    monkeypatch.setattr(
        main_module,
        "archive_run",
        lambda: calls.append("archive"),
    )
    monkeypatch.setattr(
        main_module,
        "_prune_non_current_versions",
        lambda _conn, _logger: calls.append("prune"),
    )
    monkeypatch.setattr(
        main_module,
        "release_lock",
        lambda: calls.append("release"),
    )

    main_module.main([])

    assert processed == [("queued.pdf", "extraction", "persistence")]
    assert "archive" in calls
    assert "close" in calls
    assert "prune" in calls
    assert "release" in calls


def test_main_counts_failed_worker_results(monkeypatch, file_record_factory):
    """Count RuntimeError worker failures without aborting the run."""
    calls = []
    conn = SimpleNamespace(close=lambda: calls.append("close"))
    queued = file_record_factory(
        filename="queued.pdf", file_path="/tmp/queued.pdf"
    )

    monkeypatch.setattr(
        main_module,
        "run_startup",
        Mock(return_value=(conn, object())),
    )
    monkeypatch.setattr(
        main_module,
        "run_discovery",
        Mock(return_value=_empty_diff()),
    )
    monkeypatch.setattr(
        main_module,
        "_prepare_execution_queue",
        lambda *_args: (
            [
                (
                    queued,
                    FileExecutionPlan(
                        record_key=queued.file_path,
                        start_stage="extraction",
                        target_stage="chunking",
                    ),
                )
            ],
            0,
            0,
        ),
    )
    monkeypatch.setattr(
        main_module,
        "build_stage_signatures",
        lambda: {
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
        },
    )
    monkeypatch.setattr(
        main_module,
        "get_max_workers",
        Mock(return_value=1),
    )
    monkeypatch.setattr(
        main_module,
        "process_file",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        main_module,
        "archive_run",
        lambda: calls.append("archive"),
    )
    monkeypatch.setattr(
        main_module,
        "_prune_non_current_versions",
        lambda _conn, _logger: calls.append("prune"),
    )
    monkeypatch.setattr(
        main_module,
        "release_lock",
        lambda: calls.append("release"),
    )

    main_module.main([])

    assert "archive" in calls
    assert "close" in calls
    assert "prune" in calls
    assert "release" in calls


def test_main_passes_discovery_scan_into_planning(
    monkeypatch, file_record_factory
):
    """Use the discovery-stage scan directly when planning the run."""
    calls = []
    conn = SimpleNamespace(close=lambda: calls.append("close"))
    discovered = [file_record_factory(file_path="/tmp/queued.pdf")]
    captured = {}

    monkeypatch.setattr(
        main_module,
        "run_startup",
        Mock(return_value=(conn, object())),
    )
    monkeypatch.setattr(
        main_module,
        "run_discovery",
        Mock(
            return_value=SimpleNamespace(
                diff=_empty_diff(),
                scan=SimpleNamespace(supported=discovered),
            )
        ),
    )
    monkeypatch.setattr(
        main_module,
        "build_stage_signatures",
        lambda: {
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
        },
    )

    def fake_prepare_execution_queue(
        _conn,
        _signatures,
        _options,
        _logger,
        discovered_files=None,
    ):
        captured["files"] = discovered_files
        return [], 0, 0

    monkeypatch.setattr(
        main_module,
        "_prepare_execution_queue",
        fake_prepare_execution_queue,
    )
    monkeypatch.setattr(
        main_module,
        "archive_run",
        lambda: calls.append("archive"),
    )
    monkeypatch.setattr(
        main_module,
        "_prune_non_current_versions",
        lambda _conn, _logger: calls.append("prune"),
    )
    monkeypatch.setattr(
        main_module,
        "release_lock",
        lambda: calls.append("release"),
    )

    main_module.main([])

    assert captured["files"] == discovered
    assert "archive" in calls
    assert "close" in calls
    assert "prune" in calls
    assert "release" in calls


def test_main_counts_non_runtime_worker_failures(
    monkeypatch, file_record_factory
):
    """Treat any worker exception as a failed file and continue."""
    calls = []
    conn = SimpleNamespace(close=lambda: calls.append("close"))
    queued = file_record_factory(
        filename="queued.pdf", file_path="/tmp/queued.pdf"
    )

    monkeypatch.setattr(
        main_module,
        "run_startup",
        Mock(return_value=(conn, object())),
    )
    monkeypatch.setattr(
        main_module,
        "run_discovery",
        Mock(return_value=_empty_diff()),
    )
    monkeypatch.setattr(
        main_module,
        "_prepare_execution_queue",
        lambda *_args: (
            [
                (
                    queued,
                    FileExecutionPlan(
                        record_key=queued.file_path,
                        start_stage="extraction",
                        target_stage="chunking",
                    ),
                )
            ],
            0,
            0,
        ),
    )
    monkeypatch.setattr(
        main_module,
        "build_stage_signatures",
        lambda: {
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
        },
    )
    monkeypatch.setattr(
        main_module,
        "get_max_workers",
        Mock(return_value=1),
    )
    monkeypatch.setattr(
        main_module,
        "process_file",
        lambda *_args: (_ for _ in ()).throw(ValueError("boom")),
    )
    monkeypatch.setattr(
        main_module,
        "archive_run",
        lambda: calls.append("archive"),
    )
    monkeypatch.setattr(
        main_module,
        "_prune_non_current_versions",
        lambda _conn, _logger: calls.append("prune"),
    )
    monkeypatch.setattr(
        main_module,
        "release_lock",
        lambda: calls.append("release"),
    )

    main_module.main([])

    assert "archive" in calls
    assert "close" in calls
    assert "prune" in calls
    assert "release" in calls


def test_main_releases_lock_on_discovery_failure(monkeypatch):
    """Release the lock even when discovery fails."""
    calls = []
    conn = SimpleNamespace(close=lambda: calls.append("close"))

    monkeypatch.setattr(
        main_module,
        "run_startup",
        Mock(return_value=(conn, object())),
    )

    def fail(_conn):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module, "run_discovery", fail)
    monkeypatch.setattr(
        main_module,
        "archive_run",
        lambda: calls.append("archive"),
    )
    monkeypatch.setattr(
        main_module,
        "release_lock",
        lambda: calls.append("release"),
    )

    with pytest.raises(RuntimeError, match="boom"):
        main_module.main([])

    assert "close" in calls
    assert "release" in calls


def test_main_reconciles_deleted_files_before_planning(
    monkeypatch, file_record_factory
):
    """Remove deleted files from the current DB state after discovery."""
    calls = []
    conn = SimpleNamespace(close=lambda: calls.append("close"))
    deleted_record = file_record_factory(file_path="/tmp/deleted.pdf")

    monkeypatch.setattr(
        main_module,
        "run_startup",
        Mock(return_value=(conn, object())),
    )
    monkeypatch.setattr(
        main_module,
        "run_discovery",
        Mock(
            return_value=DiscoveryDiff(
                new=[],
                modified=[],
                deleted=[deleted_record],
            )
        ),
    )
    removed = []
    monkeypatch.setattr(
        main_module,
        "remove_deleted_files",
        lambda _conn, file_paths: removed.append(file_paths),
    )
    monkeypatch.setattr(
        main_module,
        "_prepare_execution_queue",
        lambda *_args: ([], 0, 0),
    )
    monkeypatch.setattr(
        main_module,
        "build_stage_signatures",
        lambda: {
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
        },
    )
    monkeypatch.setattr(
        main_module,
        "archive_run",
        lambda: calls.append("archive"),
    )
    monkeypatch.setattr(
        main_module,
        "_prune_non_current_versions",
        lambda _conn, _logger: calls.append("prune"),
    )
    monkeypatch.setattr(
        main_module,
        "release_lock",
        lambda: calls.append("release"),
    )

    main_module.main([])

    assert removed == [["/tmp/deleted.pdf"]]
    assert "close" in calls
    assert "prune" in calls
    assert "release" in calls
