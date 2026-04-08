"""Checkpoint planning, stage signatures, and artifact persistence."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config_setup import (
    PROJECT_ROOT,
    get_document_cache_root,
    get_chunking_embedding_token_limit,
    get_chunking_max_retries,
    get_chunking_md_batch_size,
    get_chunking_truncation_token_limit,
    get_chunking_xlsx_batch_size,
    get_chunking_xlsx_header_rows,
    get_chunking_xlsx_overlap_rows,
    get_embedding_batch_size,
    get_embedding_dimensions,
    get_embedding_model,
    get_tokenizer_model,
    get_stage_model_config,
    get_vision_dpi_scale,
)
from .file_types import ExtractionResult, PageResult, StageCheckpoint

PIPELINE_STAGES = (
    "extraction",
    "tokenization",
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
_STAGE_INDEX = {
    stage_name: index for index, stage_name in enumerate(PIPELINE_STAGES)
}

_STAGE_DEPENDENCIES = {
    "extraction": (
        "src/ingestion/stages/extraction.py",
        "src/ingestion/processors/pdf/processor.py",
        "src/ingestion/processors/docx/processor.py",
        "src/ingestion/processors/pptx/processor.py",
        "src/ingestion/processors/xlsx/processor.py",
        "src/ingestion/utils/fitz_rendering.py",
        "src/ingestion/utils/file_types.py",
        "src/ingestion/utils/prompt_loader.py",
    ),
    "tokenization": (
        "src/ingestion/stages/tokenization.py",
        "src/ingestion/utils/file_types.py",
    ),
    "classification": (
        "src/ingestion/stages/classification.py",
        "src/ingestion/utils/file_types.py",
    ),
    "chunking": (
        "src/ingestion/stages/chunking.py",
        "src/ingestion/stages/chunkers/markdown_chunker.py",
        "src/ingestion/stages/chunkers/xlsx_chunker.py",
        "src/ingestion/utils/file_types.py",
        "src/ingestion/utils/prompt_loader.py",
    ),
    "doc_metadata": (
        "src/ingestion/stages/enrichment/doc_metadata.py",
        "src/ingestion/utils/file_types.py",
        "src/ingestion/utils/llm_retry.py",
        "src/ingestion/utils/prompt_loader.py",
        "src/ingestion/utils/source_context.py",
        "src/ingestion/utils/token_counting.py",
    ),
    "section_detection": (
        "src/ingestion/stages/enrichment/section_detection.py",
        "src/ingestion/utils/file_types.py",
        "src/ingestion/utils/llm_retry.py",
        "src/ingestion/utils/prompt_loader.py",
        "src/ingestion/utils/token_counting.py",
    ),
    "content_extraction": (
        "src/ingestion/stages/enrichment/content_extraction.py",
        "src/ingestion/utils/file_types.py",
        "src/ingestion/utils/llm_retry.py",
        "src/ingestion/utils/prompt_loader.py",
        "src/ingestion/utils/source_context.py",
        "src/ingestion/utils/token_counting.py",
    ),
    "section_summary": (
        "src/ingestion/stages/enrichment/section_summary.py",
        "src/ingestion/utils/file_types.py",
        "src/ingestion/utils/llm_retry.py",
        "src/ingestion/utils/prompt_loader.py",
        "src/ingestion/utils/source_context.py",
        "src/ingestion/utils/token_counting.py",
    ),
    "doc_summary": (
        "src/ingestion/stages/enrichment/doc_summary.py",
        "src/ingestion/utils/file_types.py",
        "src/ingestion/utils/llm_retry.py",
        "src/ingestion/utils/prompt_loader.py",
    ),
    "embedding": (
        "src/ingestion/stages/enrichment/embedding.py",
        "src/ingestion/utils/file_types.py",
    ),
    "persistence": (
        "src/ingestion/stages/enrichment/persistence.py",
        "src/ingestion/utils/file_types.py",
        "src/ingestion/utils/postgres_connector.py",
    ),
}

_PROMPT_PATTERNS = {
    "extraction": ("src/ingestion/processors/*/prompts/*.yaml",),
    "tokenization": (),
    "classification": (),
    "chunking": ("src/ingestion/stages/chunkers/prompts/*.yaml",),
    "doc_metadata": (
        "src/ingestion/stages/enrichment/prompts/doc_metadata.yaml",
    ),
    "section_detection": (
        "src/ingestion/stages/enrichment/prompts/section_detection.yaml",
        "src/ingestion/stages/enrichment/prompts/subsection_detection.yaml",
    ),
    "content_extraction": (
        "src/ingestion/stages/enrichment/prompts/content_extraction.yaml",
    ),
    "section_summary": (
        "src/ingestion/stages/enrichment/prompts/section_summary.yaml",
    ),
    "doc_summary": (
        "src/ingestion/stages/enrichment/prompts/doc_summary.yaml",
    ),
    "embedding": (),
    "persistence": (),
}


@dataclass(frozen=True)
class PipelineCliOptions:
    """Resolved CLI options for a pipeline run.

    Params:
        target_stage: Last stage to execute for selected files
        forced_start_stage: Stage to rerun from, or empty when resuming
        force_all: Whether to invalidate the full pipeline path
        file_paths: Exact source file paths to include
        glob_patterns: Glob patterns matched against file path or filename
    """

    target_stage: str
    forced_start_stage: str = ""
    force_all: bool = False
    file_paths: tuple[str, ...] = ()
    glob_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class FileExecutionPlan:
    """Planned execution path for one file.

    Params:
        record_key: Stable identifier for logs
        start_stage: First stage to execute, or empty when skipped
        target_stage: Last stage to execute
        resume_checkpoint: Valid upstream checkpoint for stage resume
        invalidation_stage: Stage at which old checkpoints must be cleared
        error_message: Planning failure that blocks execution
    """

    record_key: str
    start_stage: str
    target_stage: str
    resume_checkpoint: StageCheckpoint | None = None
    invalidation_stage: str = ""
    error_message: str = ""

    @property
    def should_run(self) -> bool:
        """Return whether the file has planned work. Returns: bool."""
        return bool(self.start_stage) and not self.error_message


def _require_stage(stage_name: str) -> str:
    """Validate a pipeline stage name. Params: stage_name. Returns: str."""
    if stage_name not in _STAGE_INDEX:
        raise ValueError(f"Unknown stage: {stage_name}")
    return stage_name


def get_default_target_stage() -> str:
    """Get the latest available pipeline stage. Returns: str."""
    return PIPELINE_STAGES[-1]


def resolve_cli_options(args: Any) -> PipelineCliOptions:
    """Normalize CLI args into pipeline stage controls.

    Params:
        args: argparse namespace with to_stage, from_stage,
            only_stage, force_stage, force_all

    Returns:
        PipelineCliOptions with validated stage controls
    """
    target_stage = args.to_stage or get_default_target_stage()
    _require_stage(target_stage)
    file_paths = tuple(args.file_path or [])
    glob_patterns = tuple(args.glob or [])

    forced_start_stage = ""
    if args.only_stage:
        only_stage = _require_stage(args.only_stage)
        return PipelineCliOptions(
            target_stage=only_stage,
            forced_start_stage=only_stage,
            force_all=False,
            file_paths=file_paths,
            glob_patterns=glob_patterns,
        )
    if args.from_stage:
        forced_start_stage = _require_stage(args.from_stage)
    if args.force_stage:
        forced_start_stage = _require_stage(args.force_stage)

    if forced_start_stage:
        if _STAGE_INDEX[forced_start_stage] > _STAGE_INDEX[target_stage]:
            raise ValueError("Start stage cannot come after the target stage")
        return PipelineCliOptions(
            target_stage=target_stage,
            forced_start_stage=forced_start_stage,
            force_all=False,
            file_paths=file_paths,
            glob_patterns=glob_patterns,
        )

    return PipelineCliOptions(
        target_stage=target_stage,
        force_all=bool(args.force_all),
        file_paths=file_paths,
        glob_patterns=glob_patterns,
    )


def get_stage_range(start_stage: str, target_stage: str) -> tuple[str, ...]:
    """Return the inclusive ordered stage range.

    Params:
        start_stage: First stage to execute
        target_stage: Last stage to execute

    Returns:
        tuple[str, ...] — stages in execution order
    """
    start_name = _require_stage(start_stage)
    target_name = _require_stage(target_stage)
    if _STAGE_INDEX[start_name] > _STAGE_INDEX[target_name]:
        raise ValueError("Start stage cannot come after target stage")
    return PIPELINE_STAGES[
        _STAGE_INDEX[start_name] : _STAGE_INDEX[target_name] + 1
    ]


def get_previous_stage(stage_name: str) -> str:
    """Get the immediate upstream stage, if any. Returns: str."""
    index = _STAGE_INDEX[_require_stage(stage_name)]
    if index == 0:
        return ""
    return PIPELINE_STAGES[index - 1]


def get_downstream_stages(stage_name: str) -> tuple[str, ...]:
    """Get the named stage and all later stages. Returns: tuple[str, ...]."""
    start_name = _require_stage(stage_name)
    return PIPELINE_STAGES[_STAGE_INDEX[start_name] :]


def _file_sha256(path: Path) -> str:
    """Hash a file with SHA-256. Params: path (Path). Returns: str."""
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def _stage_dependency_paths(stage_name: str) -> list[Path]:
    """Resolve source and prompt paths for a stage.

    Params:
        stage_name: Pipeline stage name

    Returns:
        list[Path] — sorted dependency file paths
    """
    stage = _require_stage(stage_name)
    paths = [
        PROJECT_ROOT / relative_path
        for relative_path in _STAGE_DEPENDENCIES[stage]
    ]
    for pattern in _PROMPT_PATTERNS[stage]:
        paths.extend(sorted(PROJECT_ROOT.glob(pattern)))
    return sorted(paths)


def _stage_config(stage_name: str) -> dict[str, Any]:
    """Build the config fingerprint for a stage. Returns: dict[str, Any]."""
    stage = _require_stage(stage_name)
    if stage == "extraction":
        return {
            "model_config": get_stage_model_config(stage),
            "vision_dpi_scale": get_vision_dpi_scale(),
        }
    if stage == "tokenization":
        return {
            "tokenizer_model": get_tokenizer_model(),
        }
    if stage == "classification":
        return {}
    if stage == "chunking":
        return {
            "model_config": get_stage_model_config("chunking"),
            "tokenizer_model": get_tokenizer_model(),
            "chunking_config": {
                "embedding_token_limit": (
                    get_chunking_embedding_token_limit()
                ),
                "truncation_token_limit": (
                    get_chunking_truncation_token_limit()
                ),
                "max_retries": get_chunking_max_retries(),
                "markdown_batch_size": get_chunking_md_batch_size(),
                "xlsx_batch_size": get_chunking_xlsx_batch_size(),
                "xlsx_header_rows": get_chunking_xlsx_header_rows(),
                "xlsx_overlap_rows": get_chunking_xlsx_overlap_rows(),
            },
        }
    if stage == "embedding":
        return {
            "embedding_model": get_embedding_model(),
            "embedding_dimensions": get_embedding_dimensions(),
            "embedding_batch_size": get_embedding_batch_size(),
        }
    if stage == "persistence":
        return {}
    return {
        "model_config": get_stage_model_config(stage),
    }


get_stage_dependency_paths = _stage_dependency_paths
get_stage_config = _stage_config


def _dependency_label(path: Path) -> str:
    """Build a stable display label for a dependency path. Returns: str."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_stage_signature(stage_name: str) -> str:
    """Build a stable fingerprint for a stage.

    Params:
        stage_name: Pipeline stage name

    Returns:
        str — SHA-256 hex digest of relevant code and config
    """
    payload = {
        "stage": _require_stage(stage_name),
        "config": _stage_config(stage_name),
        "dependencies": {
            _dependency_label(path): _file_sha256(path)
            for path in _stage_dependency_paths(stage_name)
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_stage_signatures() -> dict[str, str]:
    """Build signatures for all current pipeline stages. Returns: dict."""
    return {
        stage_name: build_stage_signature(stage_name)
        for stage_name in PIPELINE_STAGES
    }


def _document_key(file_path: str, file_hash: str) -> str:
    """Build a stable artifact directory key. Returns: str."""
    payload = f"{file_path}\0{file_hash}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_artifact_path(file_path: str, file_hash: str, stage_name: str) -> Path:
    """Resolve the artifact path for a stage result.

    Params:
        file_path: Absolute source path
        file_hash: SHA-256 hex digest of the file contents
        stage_name: Pipeline stage name

    Returns:
        Path to the stage artifact JSON
    """
    stage = _require_stage(stage_name)
    artifact_root = get_document_cache_root()
    doc_key = _document_key(file_path, file_hash)
    return artifact_root / doc_key[:2] / doc_key / f"{stage}.json"


def save_extraction_result(
    result: ExtractionResult, file_hash: str, stage_name: str
) -> tuple[str, str]:
    """Persist a stage artifact and return its path and checksum.

    Params:
        result: ExtractionResult to serialize
        file_hash: SHA-256 hex digest of the file contents
        stage_name: Pipeline stage name

    Returns:
        tuple[str, str] of absolute path and artifact checksum
    """
    artifact_path = get_artifact_path(result.file_path, file_hash, stage_name)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = json.dumps(asdict(result), indent=2).encode("utf-8")
    artifact_path.write_bytes(payload_bytes)
    checksum = hashlib.sha256(payload_bytes).hexdigest()
    return str(artifact_path), checksum


def _require_payload_object(value: Any, label: str) -> dict[str, Any]:
    """Validate that an artifact value is a JSON object.

    Params: value, label. Returns: dict[str, Any].
    """
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_payload_list(value: Any, label: str) -> list[Any]:
    """Validate that an artifact value is a JSON array.

    Params: value, label. Returns: list[Any].
    """
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _read_required_int(
    payload: dict[str, Any], field_name: str, label: str
) -> int:
    """Read and validate a required integer artifact field.

    Params: payload, field_name, label. Returns: int.
    """
    if field_name not in payload:
        raise ValueError(f"{label} missing '{field_name}'")
    value = payload[field_name]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.{field_name} must be an integer") from exc


def _read_string(
    payload: dict[str, Any],
    field_name: str,
    label: str,
    default: str | None = None,
) -> str:
    """Read and validate a string artifact field.

    Params: payload, field_name, label, default. Returns: str.
    """
    if field_name not in payload:
        if default is None:
            raise ValueError(f"{label} missing '{field_name}'")
        return default
    value = payload[field_name]
    if not isinstance(value, str):
        raise ValueError(f"{label}.{field_name} must be a string")
    return value


def _page_result_from_payload(page: Any, index: int) -> PageResult:
    """Build one PageResult from artifact JSON.

    Params: page, index. Returns: PageResult.
    """
    page_payload = _require_payload_object(page, f"pages[{index}]")
    raw_content = _read_string(
        page_payload,
        "raw_content",
        f"pages[{index}]",
        default="",
    )
    if not raw_content and "content" in page_payload:
        raw_content = _read_string(
            page_payload,
            "content",
            f"pages[{index}]",
        )
    token_count = (
        _read_required_int(page_payload, "token_count", f"pages[{index}]")
        if "token_count" in page_payload
        else 0
    )
    return PageResult(
        page_number=_read_required_int(
            page_payload, "page_number", f"pages[{index}]"
        ),
        raw_content=raw_content,
        raw_token_count=(
            _read_required_int(
                page_payload,
                "raw_token_count",
                f"pages[{index}]",
            )
            if "raw_token_count" in page_payload
            else token_count
        ),
        embedding_token_count=(
            _read_required_int(
                page_payload,
                "embedding_token_count",
                f"pages[{index}]",
            )
            if "embedding_token_count" in page_payload
            else token_count
        ),
        token_count=token_count,
        token_tier=_read_string(
            page_payload,
            "token_tier",
            f"pages[{index}]",
            default="",
        ),
        chunk_id=_read_string(
            page_payload,
            "chunk_id",
            f"pages[{index}]",
            default="",
        ),
        parent_page_number=(
            _read_required_int(
                page_payload,
                "parent_page_number",
                f"pages[{index}]",
            )
            if "parent_page_number" in page_payload
            else 0
        ),
        layout_type=_read_string(
            page_payload,
            "layout_type",
            f"pages[{index}]",
            default="",
        ),
        chunk_context=_read_string(
            page_payload,
            "chunk_context",
            f"pages[{index}]",
            default="",
        ),
        chunk_header=_read_string(
            page_payload,
            "chunk_header",
            f"pages[{index}]",
            default="",
        ),
        sheet_passthrough_content=_read_string(
            page_payload,
            "sheet_passthrough_content",
            f"pages[{index}]",
            default="",
        ),
        section_passthrough_content=_read_string(
            page_payload,
            "section_passthrough_content",
            f"pages[{index}]",
            default="",
        ),
        section_id=str(page_payload.get("section_id", "")),
        keywords=list(page_payload.get("keywords", [])),
        entities=list(page_payload.get("entities", [])),
    )


def load_extraction_result(checkpoint: StageCheckpoint) -> ExtractionResult:
    """Load an ExtractionResult from a stage checkpoint artifact.

    Params:
        checkpoint: Successful persisted stage checkpoint

    Returns:
        ExtractionResult reconstructed from JSON
    """
    artifact_path = Path(checkpoint.artifact_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Missing stage artifact: {checkpoint.artifact_path}"
        )

    payload_bytes = artifact_path.read_bytes()
    checksum = hashlib.sha256(payload_bytes).hexdigest()
    if (
        checkpoint.artifact_checksum
        and checksum != checkpoint.artifact_checksum
    ):
        raise RuntimeError(
            f"Artifact checksum mismatch for {checkpoint.artifact_path}"
        )

    try:
        payload = _require_payload_object(
            json.loads(payload_bytes.decode("utf-8")),
            "Artifact payload",
        )
        pages_payload = _require_payload_list(
            payload.get("pages"),
            "Artifact payload.pages",
        )
        pages = [
            _page_result_from_payload(page_payload, index)
            for index, page_payload in enumerate(pages_payload)
        ]
        document_metadata = payload.get("document_metadata", {})
        sections = payload.get("sections", [])
        content_units = payload.get("content_units", [])
        document_token_count = (
            _read_required_int(
                payload,
                "document_token_count",
                "Artifact payload",
            )
            if "document_token_count" in payload
            else 0
        )
        return ExtractionResult(
            file_path=_read_string(
                payload,
                "file_path",
                "Artifact payload",
            ),
            filetype=_read_string(
                payload,
                "filetype",
                "Artifact payload",
            ),
            pages=pages,
            total_pages=_read_required_int(
                payload,
                "total_pages",
                "Artifact payload",
            ),
            data_source=(
                _read_string(
                    payload,
                    "data_source",
                    "Artifact payload",
                )
                if "data_source" in payload
                else ""
            ),
            filter_1=(
                _read_string(
                    payload,
                    "filter_1",
                    "Artifact payload",
                )
                if "filter_1" in payload
                else ""
            ),
            filter_2=(
                _read_string(
                    payload,
                    "filter_2",
                    "Artifact payload",
                )
                if "filter_2" in payload
                else ""
            ),
            filter_3=(
                _read_string(
                    payload,
                    "filter_3",
                    "Artifact payload",
                )
                if "filter_3" in payload
                else ""
            ),
            raw_document_token_count=(
                _read_required_int(
                    payload,
                    "raw_document_token_count",
                    "Artifact payload",
                )
                if "raw_document_token_count" in payload
                else document_token_count
            ),
            embedding_document_token_count=(
                _read_required_int(
                    payload,
                    "embedding_document_token_count",
                    "Artifact payload",
                )
                if "embedding_document_token_count" in payload
                else document_token_count
            ),
            document_token_count=(
                document_token_count
                if "document_token_count" in payload
                else 0
            ),
            document_metadata=document_metadata,
            sections=sections,
            content_units=content_units,
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid artifact payload in {checkpoint.artifact_path}: {exc}"
        ) from exc


def checkpoint_is_usable(
    checkpoint: StageCheckpoint, expected_signature: str
) -> bool:
    """Check whether a checkpoint can be reused.

    Params:
        checkpoint: Persisted stage checkpoint
        expected_signature: Current stage signature

    Returns:
        bool — True when status, signature, and artifact are valid
    """
    if checkpoint.status != "succeeded":
        return False
    if checkpoint.stage_signature != expected_signature:
        return False
    artifact_path = Path(checkpoint.artifact_path)
    if not artifact_path.is_file():
        return False
    if not checkpoint.artifact_checksum:
        return True
    return _file_sha256(artifact_path) == checkpoint.artifact_checksum
