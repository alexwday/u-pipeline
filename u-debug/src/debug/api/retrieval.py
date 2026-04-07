"""Query execution with SSE streaming."""

import json
import logging
import queue
import threading

from flask import Blueprint, Response, request

from retriever.stages.orchestrator import run_retrieval
from retriever.stages.startup import run_startup

retrieval_bp = Blueprint("retrieval", __name__)

logger = logging.getLogger(__name__)

_state = {"conn": None, "llm": None, "initialized": False}
_init_lock = threading.Lock()


def _ensure_initialized():
    """Initialize retriever connections on first use."""
    with _init_lock:
        if _state["initialized"]:
            return
        logger.info("Initializing retriever connections")
        conn, llm = run_startup()
        _state["conn"] = conn
        _state["llm"] = llm
        _state["initialized"] = True
        logger.info("Retriever connections ready")


def _serialize_combo_results(combo_results):
    """Extract serializable fields from combo results.

    Params:
        combo_results: list of ComboSourceResult dicts

    Returns:
        list of serializable dicts
    """
    serialized = []
    for combo_result in combo_results:
        serialized.append(
            {
                "combo": dict(combo_result.get("combo", {})),
                "source": dict(combo_result.get("source", {})),
                "research_iterations": combo_result.get(
                    "research_iterations", []
                ),
                "chunk_count": combo_result.get("chunk_count", 0),
                "total_tokens": combo_result.get("total_tokens", 0),
                "findings": combo_result.get("findings", []),
                "metrics": combo_result.get("metrics", {}),
            }
        )
    return serialized


def _serialize_result(result):
    """Extract serializable fields from ConsolidatedResult.

    Params:
        result: ConsolidatedResult dict

    Returns:
        dict with JSON-safe fields
    """
    return {
        "query": result.get("query", ""),
        "consolidated_response": result.get("consolidated_response", ""),
        "key_findings": result.get("key_findings", []),
        "data_gaps": result.get("data_gaps", []),
        "citation_warnings": result.get("citation_warnings", []),
        "summary_answer": result.get("summary_answer", ""),
        "metrics_table": result.get("metrics_table", ""),
        "detailed_summary": result.get("detailed_summary", ""),
        "reference_index": result.get("reference_index", []),
        "metrics": result.get("metrics", {}),
        "trace_id": result.get("trace_id", ""),
        "trace_path": result.get("trace_path", ""),
        "combo_results": _serialize_combo_results(
            result.get("combo_results", [])
        ),
    }


@retrieval_bp.route("/query", methods=["POST"])
def run_query():
    """Execute a retrieval query and stream results via SSE.

    Expects JSON body with query, combos, and optional sources.

    Returns:
        SSE stream with chunk, result, and error events
    """
    _ensure_initialized()

    data = request.get_json()
    query_text = data["query"]
    combos = [
        {"bank": c["bank"], "period": c["period"]} for c in data["combos"]
    ]
    sources = data.get("sources") or None

    chunk_queue = queue.Queue()
    result_holder = [None]
    error_holder = [None]

    def on_chunk(text):
        chunk_queue.put(("chunk", text))

    def run_in_thread():
        try:
            result = run_retrieval(
                query_text,
                combos,
                sources,
                _state["conn"],
                _state["llm"],
                on_chunk=on_chunk,
            )
            result_holder[0] = result
        except (RuntimeError, ValueError, OSError) as exc:
            logger.exception("Retrieval failed")
            error_holder[0] = str(exc)
        finally:
            chunk_queue.put(("done", None))

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    def generate():
        while True:
            event_type, payload = chunk_queue.get()
            if event_type == "chunk":
                yield f"event: chunk\ndata: {json.dumps(payload)}\n\n"
            elif event_type == "done":
                if error_holder[0]:
                    yield (
                        f"event: error\n"
                        f"data: {json.dumps(error_holder[0])}\n\n"
                    )
                else:
                    serialized = _serialize_result(result_holder[0])
                    yield (
                        f"event: result\n"
                        f"data: {json.dumps(serialized)}\n\n"
                    )
                break

    return Response(
        generate(),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
