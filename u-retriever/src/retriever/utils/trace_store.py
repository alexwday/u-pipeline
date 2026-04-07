"""Durable structured trace helpers for retriever runs."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from uuid import uuid4

from .config_setup import get_trace_root

TRACE_SCHEMA_VERSION = 1
_MAX_SLUG_LENGTH = 48
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class TraceSession:
    """Container for one retriever run's trace paths."""

    trace_id: str
    created_at: str
    run_dir: Path


def iso_utc_now() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    """Normalize text into a short filesystem-safe slug."""
    normalized = _NON_ALNUM_RE.sub("-", value.casefold()).strip("-")
    if not normalized:
        return "item"
    return normalized[:_MAX_SLUG_LENGTH].rstrip("-")


def start_trace_session() -> TraceSession:
    """Create a new trace session directory for one retrieval run."""
    created_at = iso_utc_now()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_id = f"{timestamp}_{uuid4().hex[:8]}"
    run_dir = get_trace_root() / trace_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return TraceSession(
        trace_id=trace_id,
        created_at=created_at,
        run_dir=run_dir,
    )


def get_run_trace_path(session: TraceSession) -> Path:
    """Return the run-level trace file path for a session."""
    return session.run_dir / "run_trace.json"


def get_source_trace_path(
    session: TraceSession,
    index: int,
    combo: dict,
    source: dict,
) -> Path:
    """Return the per-source trace file path for a session."""
    bank = _slugify(combo.get("bank", "bank"))
    period = _slugify(combo.get("period", "period"))
    data_source = _slugify(source.get("data_source", "source"))
    doc_id = int(source.get("document_version_id", 0))
    filename = (
        f"source_{index:02d}_{bank}_{period}_{data_source}_{doc_id}.json"
    )
    return session.run_dir / filename


def write_trace_json(path: Path, payload: dict) -> str:
    """Write a JSON trace payload to disk and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return str(path)


def _round_float(value) -> float:
    """Round numeric trace values for stable JSON output."""
    return round(float(value), 6)


def _listify(value) -> list:
    """Return a JSON-friendly list copy for iterable values."""
    if value is None:
        return []
    return list(value)


def snapshot_content_row(row: dict) -> dict:
    """Build a stable trace snapshot for one content row."""
    return {
        "content_unit_id": row.get("content_unit_id", ""),
        "chunk_id": row.get("chunk_id", ""),
        "section_id": row.get("section_id", ""),
        "page_number": int(row.get("page_number", 0)),
        "raw_content": row.get("raw_content", ""),
        "chunk_context": row.get("chunk_context", ""),
        "chunk_header": row.get("chunk_header", ""),
        "keywords": _listify(row.get("keywords")),
        "entities": _listify(row.get("entities")),
        "token_count": int(row.get("token_count", 0)),
    }


def snapshot_search_result(result: dict) -> dict:
    """Build a trace snapshot for one fused search result."""
    snapshot = snapshot_content_row(result)
    snapshot["score"] = _round_float(result.get("score", 0.0))
    snapshot["strategy_scores"] = {
        key: _round_float(value)
        for key, value in result.get("strategy_scores", {}).items()
    }
    return snapshot


def snapshot_expanded_chunk(chunk: dict) -> dict:
    """Build a trace snapshot for one expanded chunk."""
    return {
        "content_unit_id": chunk.get("content_unit_id", ""),
        "page_number": int(chunk.get("page_number", 0)),
        "section_id": chunk.get("section_id", ""),
        "section_title": chunk.get("section_title", ""),
        "raw_content": chunk.get("raw_content", ""),
        "chunk_context": chunk.get("chunk_context", ""),
        "chunk_header": chunk.get("chunk_header", ""),
        "sheet_passthrough_content": chunk.get(
            "sheet_passthrough_content",
            "",
        ),
        "section_passthrough_content": chunk.get(
            "section_passthrough_content",
            "",
        ),
        "is_original": bool(chunk.get("is_original", False)),
        "token_count": int(chunk.get("token_count", 0)),
        "score": _round_float(chunk.get("score", 0.0)),
    }
