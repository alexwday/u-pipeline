"""Trace file loading endpoints."""

import json

from flask import Blueprint, abort, jsonify

from retriever.utils.config_setup import get_trace_root

traces_bp = Blueprint("traces", __name__)


def _safe_trace_path(trace_id, filename=None):
    """Resolve and validate a trace file path.

    Params:
        trace_id: Trace session directory name
        filename: Optional filename within the trace directory

    Returns:
        Path to the validated file

    Raises:
        404 if the path does not exist or escapes the trace root
    """
    trace_root = get_trace_root()
    trace_dir = (trace_root / trace_id).resolve()

    if not str(trace_dir).startswith(str(trace_root.resolve())):
        abort(404, "Invalid trace ID")

    if filename is not None:
        target = (trace_dir / filename).resolve()
        if not str(target).startswith(str(trace_dir)):
            abort(404, "Invalid filename")
        if not target.is_file():
            abort(404, f"Trace file not found: {filename}")
        return target

    if not trace_dir.is_dir():
        abort(404, f"Trace not found: {trace_id}")
    return trace_dir


@traces_bp.route("/trace/<trace_id>")
def get_run_trace(trace_id):
    """Load a run trace JSON file.

    Params:
        trace_id: Trace session directory name

    Returns:
        JSON contents of run_trace.json
    """
    path = _safe_trace_path(trace_id, "run_trace.json")
    with open(path, encoding="utf-8") as handle:
        return jsonify(json.load(handle))


@traces_bp.route("/trace/<trace_id>/sources")
def list_source_traces(trace_id):
    """List available source trace files for a run.

    Params:
        trace_id: Trace session directory name

    Returns:
        JSON list of source trace filenames
    """
    trace_dir = _safe_trace_path(trace_id)
    source_files = sorted(p.name for p in trace_dir.glob("source_*.json"))
    return jsonify(source_files)


@traces_bp.route("/trace/<trace_id>/source/<filename>")
def get_source_trace(trace_id, filename):
    """Load a per-source trace JSON file.

    Params:
        trace_id: Trace session directory name
        filename: Source trace filename

    Returns:
        JSON contents of the source trace file
    """
    path = _safe_trace_path(trace_id, filename)
    with open(path, encoding="utf-8") as handle:
        return jsonify(json.load(handle))
