"""Tests for startup, locking, and archive management."""

import json
import os
import time
import zipfile
from unittest.mock import Mock

import pytest

from ingestion.stages import startup
from ingestion.stages.startup import (
    _acquire_lock as acquire_lock,
    _activate_lock as activate_lock,
    _build_lock_data as build_lock_data,
    _clean_stale_processing as clean_stale_processing,
    _handle_existing_lock as handle_existing_lock,
    _handle_unreadable_lock as handle_unreadable_lock,
    _heartbeat_loop as heartbeat_loop,
    _lock_age_hours as lock_age_hours,
    _processing_contents as processing_contents,
    _prune_old_files as prune_old_files,
    _read_lock_data as read_lock_data,
    _refresh_lock as refresh_lock,
    _rewrite_lock_file as rewrite_lock_file,
    _safe_unlink as safe_unlink,
    _stop_heartbeat as stop_heartbeat,
    _write_new_lock_file as write_new_lock_file,
)


@pytest.fixture(name="startup_workspace")
def startup_workspace_fixture(tmp_path, monkeypatch):
    """Point startup module paths at a temporary test workspace."""
    processing_dir = tmp_path / "processing"
    archive_dir = tmp_path / "archive"
    logs_dir = tmp_path / "logs"
    processing_dir.mkdir()

    monkeypatch.setattr(startup, "PROCESSING_DIR", processing_dir)
    monkeypatch.setattr(startup, "LOCK_FILE", processing_dir / "pipeline.lock")
    monkeypatch.setattr(startup, "ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(startup, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(
        startup,
        "_LOCK_STATE",
        {"owner": None, "thread": None, "stop": None},
    )

    yield {
        "processing": processing_dir,
        "archive": archive_dir,
        "logs": logs_dir,
    }

    stop_heartbeat()
    lock_state = getattr(startup, "_LOCK_STATE")
    lock_state["owner"] = None
    lock_state["thread"] = None
    lock_state["stop"] = None


def _write_lock(lock_path, payload):
    """Write raw lock content for test setup."""
    if isinstance(payload, dict):
        lock_path.write_text(json.dumps(payload), encoding="utf-8")
        return
    lock_path.write_text(payload, encoding="utf-8")


def _lock_state():
    """Return the startup module lock state."""
    return getattr(startup, "_LOCK_STATE")


def _make_fake_event():
    """Return an event whose wait method yields False once, then True."""
    state = {"calls": 0}

    def wait(_self, _interval):
        """Advance the fake wait sequence."""
        state["calls"] += 1
        return state["calls"] > 1

    return type("FakeEvent", (), {"wait": wait})()


def test_lock_helpers_and_processing_contents(startup_workspace, monkeypatch):
    """Cover helper functions that inspect filesystem state."""
    extra_file = startup_workspace["processing"] / "extra.txt"
    extra_dir = startup_workspace["processing"] / "folder"
    staging_dir = startup_workspace["processing"] / ".staging_test"
    extra_file.write_text("data", encoding="utf-8")
    extra_dir.mkdir()
    staging_dir.mkdir()
    startup.LOCK_FILE.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(startup.time, "time", lambda: 7200.0)

    assert lock_age_hours(3600.0) == 1.0
    assert sorted(item.name for item in processing_contents()) == [
        "extra.txt",
        "folder",
    ]
    assert safe_unlink(extra_file) is True
    assert safe_unlink(extra_file) is False

    missing_dir = startup_workspace["processing"].parent / "missing"
    monkeypatch.setattr(startup, "PROCESSING_DIR", missing_dir)
    monkeypatch.setattr(startup, "LOCK_FILE", missing_dir / "pipeline.lock")
    assert processing_contents() == []


@pytest.mark.usefixtures("startup_workspace")
def test_acquire_lock_refreshes_heartbeat_and_release_removes_lock(
    monkeypatch,
):
    """Acquire an owned lock, refresh it, then release it safely."""
    monkeypatch.setattr(startup, "HEARTBEAT_INTERVAL_SECONDS", 0.01)

    acquire_lock()
    first_lock = read_lock_data()

    deadline = time.time() + 0.5
    latest_lock = first_lock
    while time.time() < deadline:
        latest_lock = read_lock_data()
        if latest_lock["timestamp"] > first_lock["timestamp"]:
            break
        time.sleep(0.01)

    assert latest_lock["run_id"] == first_lock["run_id"]
    assert latest_lock["pid"] == first_lock["pid"]
    assert latest_lock["timestamp"] > first_lock["timestamp"]
    assert _lock_state()["owner"] == {
        "run_id": first_lock["run_id"],
        "pid": first_lock["pid"],
    }

    startup.release_lock()

    assert startup.LOCK_FILE.exists() is False
    assert _lock_state()["owner"] is None


@pytest.mark.usefixtures("startup_workspace")
def test_acquire_lock_replaces_stale_lock_and_rejects_active_lock(monkeypatch):
    """Replace stale locks but fail fast on active ones."""
    monkeypatch.setattr(startup, "_start_heartbeat", lambda owner: None)

    stale_lock = build_lock_data()
    stale_lock["timestamp"] = time.time() - (
        startup.LOCK_EXPIRY_HOURS * 3600 + 10
    )
    _write_lock(startup.LOCK_FILE, stale_lock)

    acquire_lock()
    replacement = read_lock_data()
    assert replacement["run_id"] != stale_lock["run_id"]
    startup.release_lock()

    active_lock = build_lock_data()
    _write_lock(startup.LOCK_FILE, active_lock)

    with pytest.raises(RuntimeError, match="Another run may be in progress"):
        acquire_lock()


@pytest.mark.usefixtures("startup_workspace")
def test_acquire_lock_handles_unreadable_locks(monkeypatch):
    """Treat stale unreadable locks as recoverable and fresh ones as active."""
    monkeypatch.setattr(startup, "_start_heartbeat", lambda owner: None)

    _write_lock(startup.LOCK_FILE, "{bad json")
    old_time = time.time() - (startup.LOCK_EXPIRY_HOURS * 3600 + 10)
    os.utime(startup.LOCK_FILE, (old_time, old_time))

    acquire_lock()
    assert read_lock_data()["run_id"]
    startup.release_lock()

    _write_lock(startup.LOCK_FILE, "{bad json")
    with pytest.raises(RuntimeError, match="active but unreadable"):
        acquire_lock()


def test_handle_existing_and_unreadable_lock_missing_paths_return(
    startup_workspace,
    monkeypatch,
):
    """Return quietly when the lock disappears mid-check."""
    monkeypatch.setattr(
        startup,
        "_read_lock_timestamp",
        Mock(side_effect=FileNotFoundError()),
    )

    handle_existing_lock(FileExistsError("exists"))

    monkeypatch.setattr(
        startup,
        "LOCK_FILE",
        startup_workspace["processing"] / "missing.lock",
    )
    handle_unreadable_lock(ValueError("bad"))


@pytest.mark.usefixtures("startup_workspace")
def test_refresh_lock_failure_paths():
    """Stop heartbeating when the lock is missing, unreadable, or stolen."""
    owner = {"run_id": "run-1", "pid": 10}

    assert refresh_lock(owner) is False

    _write_lock(startup.LOCK_FILE, "{bad json")
    assert refresh_lock(owner) is False

    _write_lock(
        startup.LOCK_FILE,
        {
            "run_id": "other",
            "pid": 11,
            "started_at": "now",
            "timestamp": time.time(),
            "heartbeat_at": "now",
        },
    )
    assert refresh_lock(owner) is False


def test_heartbeat_loop_stops_on_false_or_exception(monkeypatch):
    """Exit the heartbeat loop on failed refreshes or exceptions."""
    monkeypatch.setattr(startup, "_refresh_lock", lambda owner: False)
    heartbeat_loop({"run_id": "run", "pid": 1}, _make_fake_event())

    def raise_refresh(_owner):
        """Raise a refresh failure."""
        raise OSError("boom")

    monkeypatch.setattr(startup, "_refresh_lock", raise_refresh)
    heartbeat_loop({"run_id": "run", "pid": 1}, _make_fake_event())


def test_activate_lock_releases_when_heartbeat_start_fails(monkeypatch):
    """Clean up ownership state when the heartbeat cannot start."""
    calls = []
    monkeypatch.setattr(
        startup,
        "_start_heartbeat",
        Mock(side_effect=RuntimeError("no thread")),
    )
    monkeypatch.setattr(
        startup, "release_lock", lambda: calls.append("release")
    )

    with pytest.raises(RuntimeError, match="no thread"):
        activate_lock({"run_id": "run", "pid": 1})

    assert calls == ["release"]


@pytest.mark.usefixtures("startup_workspace")
def test_release_lock_handles_non_owned_cases():
    """Leave the lock in place when ownership cannot be proven."""
    startup.release_lock()

    _lock_state()["owner"] = {"run_id": "run", "pid": 1}
    startup.release_lock()
    assert _lock_state()["owner"] is None

    _lock_state()["owner"] = {"run_id": "run", "pid": 1}
    _write_lock(startup.LOCK_FILE, "{bad json")
    startup.release_lock()
    assert startup.LOCK_FILE.exists() is True
    startup.LOCK_FILE.unlink()

    _lock_state()["owner"] = {"run_id": "run", "pid": 1}
    _write_lock(
        startup.LOCK_FILE,
        {
            "run_id": "other",
            "pid": 2,
            "started_at": "now",
            "timestamp": time.time(),
            "heartbeat_at": "now",
        },
    )
    startup.release_lock()
    assert startup.LOCK_FILE.exists() is True


def test_clean_stale_processing_archives_contents(startup_workspace):
    """Archive stale processing files from a previous crashed run."""
    clean_stale_processing()
    startup.archive_run()

    stale_file = startup_workspace["processing"] / "trace.json"
    stale_dir = startup_workspace["processing"] / "pages"
    stale_file.write_text("{}", encoding="utf-8")
    stale_dir.mkdir()
    (stale_dir / "page.txt").write_text("page", encoding="utf-8")

    clean_stale_processing()

    archives = list(startup_workspace["archive"].glob("crashed_*.zip"))
    assert len(archives) == 1
    assert stale_file.exists() is False
    assert stale_dir.exists() is False

    with zipfile.ZipFile(archives[0]) as archive:
        names = sorted(archive.namelist())

    assert "pages/" in names
    assert "pages/page.txt" in names
    assert "trace.json" in names


@pytest.mark.usefixtures("startup_workspace")
def test_lock_file_writers_remove_partial_files_on_failure(monkeypatch):
    """Clean up partial lock files when atomic writes fail."""
    lock_data = build_lock_data()

    monkeypatch.setattr(
        startup.os,
        "fdopen",
        Mock(side_effect=OSError("write failed")),
    )
    with pytest.raises(OSError, match="write failed"):
        write_new_lock_file(lock_data)
    assert startup.LOCK_FILE.exists() is False

    startup.LOCK_FILE.write_text("{}", encoding="utf-8")
    temp_suffix = f"{startup.LOCK_FILE.name}.{lock_data['run_id']}.tmp"

    def fail_replace(_src, _dst):
        """Raise during the final atomic replace."""
        raise OSError("replace failed")

    monkeypatch.setattr(startup.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        rewrite_lock_file(lock_data)
    assert not any(
        path.name == temp_suffix for path in startup.PROCESSING_DIR.iterdir()
    )


def test_prune_old_files_skips_missing_and_small_sets(
    startup_workspace,
    monkeypatch,
):
    """Skip pruning when directories are absent or within retention."""
    monkeypatch.setattr(startup, "get_retention_count", lambda: 2)

    prune_old_files(startup_workspace["archive"], "*.zip", "archives")

    startup_workspace["archive"].mkdir()
    only_file = startup_workspace["archive"] / "single.zip"
    only_file.write_text("zip", encoding="utf-8")

    prune_old_files(startup_workspace["archive"], "*.zip", "archives")

    assert only_file.exists() is True


def test_archive_run_prunes_old_archives_and_logs(
    startup_workspace,
    monkeypatch,
):
    """Archive the current run and prune old files beyond retention."""
    monkeypatch.setattr(startup, "get_retention_count", lambda: 1)
    startup_workspace["archive"].mkdir()
    startup_workspace["logs"].mkdir()

    old_archive = startup_workspace["archive"] / "old_1.zip"
    older_archive = startup_workspace["archive"] / "old_0.zip"
    old_archive.write_text("zip", encoding="utf-8")
    older_archive.write_text("zip", encoding="utf-8")
    old_log = startup_workspace["logs"] / "a.log"
    older_log = startup_workspace["logs"] / "b.log"
    old_log.write_text("log", encoding="utf-8")
    older_log.write_text("log", encoding="utf-8")

    current_file = startup_workspace["processing"] / "current.json"
    current_file.write_text("{}", encoding="utf-8")

    startup.archive_run()

    archives = sorted(startup_workspace["archive"].glob("*.zip"))
    logs = sorted(startup_workspace["logs"].glob("*.log"))

    assert len(archives) == 1
    assert archives[0].name.startswith("run_")
    assert len(logs) == 1
    assert current_file.exists() is False


def test_run_startup_success(monkeypatch):
    """Initialize config, LLM, and database on the happy path."""
    calls = []
    conn = Mock()
    llm = Mock()

    monkeypatch.setattr(startup, "load_config", lambda: calls.append("config"))
    monkeypatch.setattr(
        startup, "setup_logging", lambda: calls.append("logging")
    )
    monkeypatch.setattr(startup, "setup_ssl", lambda: calls.append("ssl"))
    monkeypatch.setattr(startup, "_acquire_lock", lambda: calls.append("lock"))
    monkeypatch.setattr(
        startup,
        "_clean_stale_processing",
        lambda: calls.append("clean"),
    )
    monkeypatch.setattr(startup, "LLMClient", lambda: llm)
    monkeypatch.setattr(startup, "get_connection", lambda: conn)
    monkeypatch.setattr(
        startup,
        "verify_connection",
        lambda db_conn: calls.append(("verify", db_conn)),
    )
    monkeypatch.setattr(
        startup,
        "ensure_schema_objects",
        lambda db_conn: calls.append(("ensure", db_conn)),
    )

    result_conn, result_llm = startup.run_startup()

    llm.test_connection.assert_called_once_with()
    assert result_conn is conn
    assert result_llm is llm
    assert calls == [
        "config",
        "logging",
        "ssl",
        "lock",
        "clean",
        ("verify", conn),
        ("ensure", conn),
    ]


def test_run_startup_failure_closes_connection_and_releases_lock(monkeypatch):
    """Release the lock and close the DB connection on startup failure."""
    conn = Mock()
    llm = Mock()
    release_lock = Mock()

    monkeypatch.setattr(startup, "load_config", lambda: None)
    monkeypatch.setattr(startup, "setup_logging", lambda: None)
    monkeypatch.setattr(startup, "setup_ssl", lambda: None)
    monkeypatch.setattr(startup, "_acquire_lock", lambda: None)
    monkeypatch.setattr(startup, "_clean_stale_processing", lambda: None)
    monkeypatch.setattr(startup, "LLMClient", lambda: llm)
    monkeypatch.setattr(startup, "get_connection", lambda: conn)
    monkeypatch.setattr(
        startup,
        "verify_connection",
        Mock(side_effect=RuntimeError("db down")),
    )
    monkeypatch.setattr(startup, "ensure_schema_objects", lambda _conn: None)
    monkeypatch.setattr(startup, "release_lock", release_lock)

    with pytest.raises(RuntimeError, match="db down"):
        startup.run_startup()

    conn.close.assert_called_once_with()
    llm.test_connection.assert_called_once_with()
    release_lock.assert_called_once_with()
