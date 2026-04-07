"""Tests for logging helpers."""

import logging
import sys

from ingestion.utils import logging_setup
from ingestion.utils.logging_setup import (
    _short_source_path as short_source_path,
)


def test_short_source_path_variants():
    """Shorten known pipeline paths and fall back to the filename."""
    assert (
        short_source_path("/tmp/project/ingestion/stages/startup.py")
        == "stages/startup.py"
    )
    assert short_source_path("/tmp/project/retriever/main.py") == "main.py"
    assert short_source_path("/tmp/plain.txt") == "plain.txt"


def test_console_formatter_formats_time_and_exception():
    """Render console log records with color and tracebacks."""
    formatter = logging_setup.ConsoleFormatter()
    record = logging.LogRecord(
        name="ingestion",
        level=logging.ERROR,
        pathname="/tmp/project/ingestion/stages/startup.py",
        lineno=10,
        msg="broken",
        args=(),
        exc_info=None,
    )
    record.stage = "STARTUP"

    assert formatter.formatTime(
        record, "%Y"
    ) == logging_setup.datetime.fromtimestamp(record.created).strftime("%Y")

    try:
        raise ValueError("bad")
    except ValueError:
        record.exc_info = sys.exc_info()

    rendered = formatter.format(record)

    assert "STARTUP" in rendered
    assert "broken" in rendered
    assert "ValueError: bad" in rendered


def test_file_formatter_formats_time_and_exception():
    """Render file log records with milliseconds and tracebacks."""
    formatter = logging_setup.FileFormatter()
    record = logging.LogRecord(
        name="ingestion",
        level=logging.WARNING,
        pathname="/tmp/project/retriever/main.py",
        lineno=12,
        msg="warn",
        args=(),
        exc_info=None,
    )
    record.stage = "DISCOVERY"

    assert formatter.formatTime(
        record, "%H"
    ) == logging_setup.datetime.fromtimestamp(record.created).strftime("%H")

    try:
        raise RuntimeError("ouch")
    except RuntimeError:
        record.exc_info = sys.exc_info()

    rendered = formatter.format(record)

    assert "DISCOVERY" in rendered
    assert "main.py:12" in rendered
    assert "RuntimeError: ouch" in rendered


def test_setup_logging_and_get_stage_logger(tmp_path, monkeypatch):
    """Install fresh root handlers and stage-aware adapters."""
    root = logging.getLogger()
    stale_handler = logging.StreamHandler()
    root.addHandler(stale_handler)

    monkeypatch.setattr(logging_setup, "LOGS_DIR", tmp_path / "logs")

    logging_setup.setup_logging(logging.INFO)
    adapter = logging_setup.get_stage_logger(__name__, "DISCOVERY")

    log_files = list((tmp_path / "logs").glob("pipeline_*.log"))

    assert len(root.handlers) == 2
    assert stale_handler not in root.handlers
    assert len(log_files) == 1
    assert adapter.extra == {"stage": "DISCOVERY"}
    assert logging.getLogger("openai").level == logging.WARNING

    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)
