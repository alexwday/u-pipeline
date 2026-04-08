"""Ingestion pipeline entry point."""

import argparse
import fnmatch
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .stages.discovery import run_discovery, scan_filesystem
from .stages.extraction import extract_file
from .stages.tokenization import tokenize_result
from .stages.classification import classify_result
from .stages.chunking import chunk_result
from .stages.enrichment.doc_metadata import enrich_doc_metadata
from .stages.enrichment.section_detection import detect_sections
from .stages.enrichment.content_extraction import extract_content
from .stages.enrichment.section_summary import summarize_sections
from .stages.enrichment.doc_summary import summarize_document
from .stages.enrichment.embedding import embed_content
from .stages.enrichment.persistence import persist_enrichment
from .stages.startup import archive_run, release_lock, run_startup
from .utils.llm_connector import LLMClient
from .utils.checkpointing import (
    PIPELINE_STAGES,
    FileExecutionPlan,
    PipelineCliOptions,
    build_stage_signatures,
    checkpoint_is_usable,
    get_previous_stage,
    get_stage_range,
    load_extraction_result,
    resolve_cli_options,
    save_extraction_result,
)
from .utils.config_setup import (
    get_data_source_path,
    get_max_workers,
    get_non_current_version_retention_count,
)
from .utils.file_types import (
    ExtractionResult,
    FileRecord,
    StageCheckpoint,
    compute_file_hash,
)
from .utils.logging_setup import get_stage_logger
from .utils.postgres_connector import (
    clear_stage_checkpoints,
    delete_document_versions,
    fetch_current_document_versions,
    fetch_prunable_document_versions,
    fetch_stage_checkpoints,
    get_connection,
    mark_stage_checkpoint_failed,
    mark_stage_checkpoint_started,
    mark_stage_checkpoint_succeeded,
    remove_deleted_files,
    register_document_version,
    upsert_catalog_record,
)

RESUME_LOAD_ERRORS = (
    FileNotFoundError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for pipeline stage controls."""
    parser = argparse.ArgumentParser(prog="ingestion")
    parser.add_argument(
        "--to-stage",
        choices=PIPELINE_STAGES,
        default="",
        help="Run files through this stage and stop.",
    )
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--from-stage",
        choices=PIPELINE_STAGES,
        default="",
        help="Rerun this stage and everything after it.",
    )
    start_group.add_argument(
        "--only-stage",
        choices=PIPELINE_STAGES,
        default="",
        help="Run only this stage.",
    )
    start_group.add_argument(
        "--force-stage",
        choices=PIPELINE_STAGES,
        default="",
        help="Invalidate this stage and rerun from it.",
    )
    start_group.add_argument(
        "--force-all",
        action="store_true",
        help="Invalidate all stage checkpoints and rerun from the start.",
    )
    parser.add_argument(
        "--file-path",
        action="append",
        default=[],
        help="Limit processing to an exact file path. May be repeated.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Limit processing to file paths or filenames matching a glob.",
    )
    return parser


build_parser = _build_parser


def _checkpoints_by_version(
    checkpoints: list[StageCheckpoint],
) -> dict[int, dict[str, StageCheckpoint]]:
    """Group stage checkpoints by document version. Returns: dict."""
    grouped: dict[int, dict[str, StageCheckpoint]] = {}
    for checkpoint in checkpoints:
        grouped.setdefault(checkpoint.document_version_id, {})[
            checkpoint.stage_name
        ] = checkpoint
    return grouped


def _record_selected(
    record: FileRecord,
    options: PipelineCliOptions,
) -> bool:
    """Check whether a file matches the CLI selection filters.

    Params:
        record: Current filesystem file record
        options: Resolved CLI options

    Returns:
        bool — True when the file should be considered for planning
    """
    if options.file_paths and record.file_path not in options.file_paths:
        return False
    if not options.glob_patterns:
        return True
    return any(
        fnmatch.fnmatch(record.file_path, pattern)
        or fnmatch.fnmatch(record.filename, pattern)
        for pattern in options.glob_patterns
    )


record_selected = _record_selected


def _contiguous_completed_stages(
    stage_checkpoints: dict[str, StageCheckpoint],
    stage_signatures: dict[str, str],
) -> tuple[str, ...]:
    """Return valid completed stages from the pipeline start.

    Params:
        stage_checkpoints: Stage name -> persisted checkpoint
        stage_signatures: Stage name -> current stage signature

    Returns:
        tuple[str, ...] — contiguous reusable stages
    """
    completed: list[str] = []
    for stage_name in PIPELINE_STAGES:
        checkpoint = stage_checkpoints.get(stage_name)
        if checkpoint is None:
            break
        if not checkpoint_is_usable(checkpoint, stage_signatures[stage_name]):
            break
        completed.append(stage_name)
    return tuple(completed)


def plan_file(
    record: FileRecord,
    stage_checkpoints: dict[str, StageCheckpoint],
    stage_signatures: dict[str, str],
    options: PipelineCliOptions,
) -> FileExecutionPlan:
    """Build a resumable execution plan for one file.

    Params:
        record: Current filesystem record with populated file_hash
        stage_checkpoints: Persisted checkpoints for the version
        stage_signatures: Stage name -> current stage signature
        options: Resolved CLI options

    Returns:
        FileExecutionPlan for the file
    """
    target_stage = options.target_stage
    target_index = PIPELINE_STAGES.index(target_stage)
    record_key = record.file_path
    completed = _contiguous_completed_stages(
        stage_checkpoints, stage_signatures
    )

    if options.force_all:
        return FileExecutionPlan(
            record_key=record_key,
            start_stage=PIPELINE_STAGES[0],
            target_stage=target_stage,
            invalidation_stage=PIPELINE_STAGES[0],
        )

    if options.forced_start_stage:
        start_stage = options.forced_start_stage
        previous_stage = get_previous_stage(start_stage)
        if previous_stage:
            checkpoint = stage_checkpoints.get(previous_stage)
            if checkpoint is None or not checkpoint_is_usable(
                checkpoint, stage_signatures[previous_stage]
            ):
                return FileExecutionPlan(
                    record_key=record_key,
                    start_stage="",
                    target_stage=target_stage,
                    error_message=(
                        f"Missing valid {previous_stage} checkpoint for "
                        f"requested start stage {start_stage}"
                    ),
                )
            return FileExecutionPlan(
                record_key=record_key,
                start_stage=start_stage,
                target_stage=target_stage,
                resume_checkpoint=checkpoint,
                invalidation_stage=start_stage,
            )
        return FileExecutionPlan(
            record_key=record_key,
            start_stage=start_stage,
            target_stage=target_stage,
            invalidation_stage=start_stage,
        )

    if len(completed) >= target_index + 1:
        return FileExecutionPlan(
            record_key=record_key,
            start_stage="",
            target_stage=target_stage,
        )

    start_stage = PIPELINE_STAGES[len(completed)]
    invalidation_stage = (
        start_stage if stage_checkpoints.get(start_stage) is not None else ""
    )

    previous_stage = get_previous_stage(start_stage)
    if previous_stage:
        checkpoint = stage_checkpoints[previous_stage]
        return FileExecutionPlan(
            record_key=record_key,
            start_stage=start_stage,
            target_stage=target_stage,
            resume_checkpoint=checkpoint,
            invalidation_stage=invalidation_stage,
        )
    return FileExecutionPlan(
        record_key=record_key,
        start_stage=start_stage,
        target_stage=target_stage,
        invalidation_stage=invalidation_stage,
    )


_STAGE_REQUIRES_RESULT = {
    "tokenization": "extraction",
    "classification": "tokenization",
    "chunking": "classification",
    "doc_metadata": "chunking",
    "section_detection": "doc metadata",
    "content_extraction": "section detection",
    "section_summary": "content extraction",
    "doc_summary": "section summary",
    "embedding": "doc summary",
    "persistence": "embedding",
}

_STAGE_DISPATCH: dict[
    str,
    Any,
] = {
    "extraction": lambda rec, llm, _res: extract_file(rec, llm),
    "tokenization": lambda _rec, _llm, res: tokenize_result(res),
    "classification": lambda _rec, _llm, res: classify_result(res),
    "chunking": lambda _rec, llm, res: chunk_result(res, llm),
    "doc_metadata": lambda _rec, llm, res: enrich_doc_metadata(res, llm),
    "section_detection": lambda _rec, llm, res: detect_sections(res, llm),
    "content_extraction": lambda _rec, llm, res: extract_content(res, llm),
    "section_summary": lambda _rec, llm, res: summarize_sections(res, llm),
    "doc_summary": lambda _rec, llm, res: summarize_document(res, llm),
    "embedding": lambda _rec, llm, res: embed_content(res, llm),
    "persistence": lambda _rec, llm, res: persist_enrichment(res, llm),
}


def _run_stage(
    stage_name: str,
    record: FileRecord,
    llm: LLMClient,
    result: ExtractionResult | None,
) -> ExtractionResult:
    """Execute one pipeline stage for a file.

    Params:
        stage_name: Pipeline stage name
        record: Current file record
        llm: Initialized LLM client
        result: Upstream stage result, if any

    Returns:
        ExtractionResult for the executed stage
    """
    handler = _STAGE_DISPATCH.get(stage_name)
    if handler is None:
        raise ValueError(f"Unsupported stage: {stage_name}")
    upstream = _STAGE_REQUIRES_RESULT.get(stage_name)
    if upstream is not None and result is None:
        raise RuntimeError(f"{stage_name} requires {upstream} output")
    return handler(record, llm, result)


run_stage = _run_stage


def process_file(
    record: FileRecord,
    llm: LLMClient,
    plan: FileExecutionPlan,
    stage_signatures: dict[str, str],
) -> ExtractionResult:
    """Run one file from its planned start stage to target stage.

    Params:
        record: Current filesystem file record
        llm: Initialized LLM client
        plan: Resume-aware execution plan
        stage_signatures: Stage name -> current stage signature

    Returns:
        ExtractionResult from the final executed stage
    """
    conn = get_connection()
    try:
        version = register_document_version(conn, record)
        upsert_catalog_record(conn, record)

        if plan.invalidation_stage:
            clear_stage_checkpoints(
                conn,
                version.document_version_id,
                plan.invalidation_stage,
            )

        result: ExtractionResult | None = None
        if plan.resume_checkpoint is not None:
            try:
                result = load_extraction_result(plan.resume_checkpoint)
            except RESUME_LOAD_ERRORS as exc:
                mark_stage_checkpoint_failed(
                    conn,
                    version.document_version_id,
                    plan.start_stage,
                    stage_signatures[plan.start_stage],
                    f"Failed to load upstream checkpoint: {exc}",
                )
                raise RuntimeError(
                    f"Failed to resume {record.filename}: {exc}"
                ) from exc

        for stage_name in get_stage_range(plan.start_stage, plan.target_stage):
            mark_stage_checkpoint_started(
                conn,
                version.document_version_id,
                stage_name,
                stage_signatures[stage_name],
            )
            try:
                result = _run_stage(stage_name, record, llm, result)
                artifact_path, artifact_checksum = save_extraction_result(
                    result,
                    record.file_hash,
                    stage_name,
                )
                try:
                    mark_stage_checkpoint_succeeded(
                        conn,
                        version.document_version_id,
                        stage_name,
                        stage_signatures[stage_name],
                        artifact_path,
                        artifact_checksum,
                    )
                except Exception:
                    Path(artifact_path).unlink(missing_ok=True)
                    raise
            except Exception as exc:
                mark_stage_checkpoint_failed(
                    conn,
                    version.document_version_id,
                    stage_name,
                    stage_signatures[stage_name],
                    str(exc),
                )
                raise RuntimeError(
                    f"Stage {stage_name} failed for {record.filename}"
                ) from exc

        if result is None:
            raise RuntimeError(f"No stage executed for {record.filename}")
        return result
    finally:
        conn.close()


def _prepare_execution_queue(
    conn: Any,
    stage_signatures: dict[str, str],
    options: PipelineCliOptions,
    logger: logging.Logger,
    discovered_files: list[FileRecord] | None = None,
) -> tuple[list[tuple[FileRecord, FileExecutionPlan]], int, int]:
    """Build the per-file execution queue for this run.

    Params:
        conn: psycopg2 connection
        stage_signatures: Stage name -> current stage signature
        options: Resolved CLI options
        logger: Pipeline logger
        discovered_files: Optional supported files from discovery

    Returns:
        tuple of (planned files, skipped count, blocked count)
    """
    if discovered_files is None:
        scan = scan_filesystem(get_data_source_path())
        files = sorted(scan.supported, key=lambda record: record.file_path)
    else:
        files = sorted(discovered_files, key=lambda record: record.file_path)
    if not files:
        logger.info("No supported files discovered")
        return [], 0, 0

    versions = fetch_current_document_versions(conn)
    checkpoints = fetch_stage_checkpoints(conn)
    versions_by_key = {
        (version.file_path, version.file_hash): version for version in versions
    }
    versions_by_metadata = {
        (
            version.file_path,
            version.file_size,
            version.date_last_modified,
        ): version
        for version in versions
    }
    checkpoints_by_version = _checkpoints_by_version(checkpoints)

    planned: list[tuple[FileRecord, FileExecutionPlan]] = []
    skipped = 0
    blocked = 0

    for record in files:
        if not _record_selected(record, options):
            continue
        if not record.file_hash:
            version = versions_by_metadata.get(
                (
                    record.file_path,
                    record.file_size,
                    record.date_last_modified,
                )
            )
            if version is not None:
                record.file_hash = version.file_hash
        if not record.file_hash:
            record.file_hash = compute_file_hash(record.file_path)
        version = versions_by_key.get((record.file_path, record.file_hash))
        stage_checkpoints = (
            checkpoints_by_version.get(version.document_version_id, {})
            if version is not None
            else {}
        )
        plan = plan_file(
            record,
            stage_checkpoints,
            stage_signatures,
            options,
        )
        if plan.error_message:
            blocked += 1
            logger.error(
                "Blocked: %s — %s",
                record.filename,
                plan.error_message,
            )
            continue
        if not plan.should_run:
            skipped += 1
            continue
        planned.append((record, plan))

    return planned, skipped, blocked


def _reconcile_deleted_files(
    conn: Any,
    deleted_records: list[FileRecord],
    logger: logging.Logger,
) -> None:
    """Remove deleted files from the current planning state.

    Params:
        conn: psycopg2 connection
        deleted_records: Discovery records missing from disk
        logger: Pipeline logger
    """
    deleted_paths = sorted({record.file_path for record in deleted_records})
    if not deleted_paths:
        return

    remove_deleted_files(conn, deleted_paths)
    logger.info(
        "Removed %d deleted files from current catalog state",
        len(deleted_paths),
    )


def _prune_non_current_versions(conn: Any, logger: logging.Logger) -> None:
    """Delete stale artifacts and non-current document versions.

    Params:
        conn: psycopg2 connection
        logger: Pipeline logger
    """
    stale_versions = fetch_prunable_document_versions(
        conn,
        get_non_current_version_retention_count(),
    )
    if not stale_versions:
        return

    artifact_dirs = {
        str(Path(artifact_path).parent)
        for version in stale_versions
        for artifact_path in version.artifact_paths
    }
    delete_document_versions(
        conn,
        [version.document_version_id for version in stale_versions],
    )
    for directory in artifact_dirs:
        shutil.rmtree(directory, ignore_errors=True)
    logger.info(
        "Pruned %d non-current document versions",
        len(stale_versions),
    )


prune_non_current_versions = _prune_non_current_versions


prepare_execution_queue = _prepare_execution_queue
reconcile_deleted_files = _reconcile_deleted_files


def main(argv: list[str] | None = None) -> None:
    """Run the ingestion pipeline.

    Startup initializes connections, discovery refreshes
    the filesystem view, then each file resumes from the
    earliest incomplete stage up to the requested target.

    Params:
        argv: Optional CLI arguments

    Returns:
        None

    Example:
        >>> main([])
    """
    args = _build_parser().parse_args(argv or [])
    options = resolve_cli_options(args)
    conn, llm = run_startup()
    try:
        logger = get_stage_logger(__name__, "PIPELINE")
        discovery_result = run_discovery(conn)
        discovery_diff = getattr(discovery_result, "diff", discovery_result)
        discovery_scan = getattr(discovery_result, "scan", None)
        _reconcile_deleted_files(conn, discovery_diff.deleted, logger)
        stage_signatures = build_stage_signatures()
        planned, skipped, blocked = _prepare_execution_queue(
            conn,
            stage_signatures,
            options,
            logger,
            None if discovery_scan is None else discovery_scan.supported,
        )

        logger.info(
            "Planning complete — target: %s, queued: %d, "
            "skipped: %d, blocked: %d",
            options.target_stage,
            len(planned),
            skipped,
            blocked,
        )

        succeeded = 0
        failed = blocked
        if planned:
            logger.info(
                "Processing %d files with %d workers",
                len(planned),
                get_max_workers(),
            )
            with ThreadPoolExecutor(max_workers=get_max_workers()) as pool:
                futures = {
                    pool.submit(
                        process_file,
                        record,
                        llm,
                        plan,
                        stage_signatures,
                    ): record
                    for record, plan in planned
                }
                for future in as_completed(futures):
                    record = futures[future]
                    error = future.exception()
                    if error is None:
                        future.result()
                        succeeded += 1
                    else:
                        failed += 1
                        logger.error(
                            "Failed: %s",
                            record.filename,
                            exc_info=(
                                type(error),
                                error,
                                error.__traceback__,
                            ),
                        )
        logger.info(
            "Pipeline complete — succeeded: %d, failed: %d, skipped: %d",
            succeeded,
            failed,
            skipped,
        )
        _prune_non_current_versions(conn, logger)

        archive_run()
    finally:
        try:
            conn.close()
        finally:
            release_lock()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
    sys.exit(0)
