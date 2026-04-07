"""Stage 0: Pipeline startup, lock management, and cleanup."""

import json
import os
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.config_setup import get_retention_count, load_config
from ..utils.llm_connector import LLMClient
from ..utils.logging_setup import (
    LOGS_DIR,
    get_stage_logger,
    setup_logging,
)
from ..utils.postgres_connector import (
    ensure_schema_objects,
    get_connection,
    verify_connection,
)
from ..utils.ssl_setup import setup_ssl

STAGE = "0-STARTUP"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSING_DIR = PROJECT_ROOT / "processing"
LOCK_FILE = PROCESSING_DIR / "pipeline.lock"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

LOCK_EXPIRY_HOURS = 6
HEARTBEAT_INTERVAL_SECONDS = 300
HEARTBEAT_THREAD_NAME = "ingestion-lock-heartbeat"

_LOCK_STATE: dict[str, Any] = {
    "owner": None,
    "thread": None,
    "stop": None,
}


def _lock_age_hours(timestamp: float) -> float:
    """Get lock age in hours. Params: timestamp. Returns: float."""
    return (time.time() - timestamp) / 3600


def _safe_unlink(path: Path) -> bool:
    """Delete a path if it exists. Params: path. Returns: bool."""
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def _read_lock_data() -> dict[str, Any]:
    """Read the lock data from JSON. Returns: dict[str, Any]."""
    return json.loads(LOCK_FILE.read_text(encoding="utf-8"))


def _read_lock_timestamp() -> float:
    """Read the lock timestamp from JSON. Returns: float."""
    lock_data = _read_lock_data()
    return float(lock_data["timestamp"])


def _lock_owner_matches(
    lock_data: dict[str, Any],
    owner: dict[str, Any],
) -> bool:
    """Check whether lock data matches the active owner. Returns: bool."""
    return lock_data.get("run_id") == owner["run_id"] and int(
        lock_data.get("pid", -1)
    ) == int(owner["pid"])


def _build_lock_data() -> dict[str, Any]:
    """Build a new lock record for the current run. Returns: dict."""
    current_time = time.time()
    current_iso = datetime.now().isoformat()
    return {
        "run_id": uuid.uuid4().hex,
        "pid": os.getpid(),
        "started_at": current_iso,
        "timestamp": current_time,
        "heartbeat_at": current_iso,
    }


def _write_new_lock_file(lock_data: dict[str, Any]) -> None:
    """Atomically create a new lock file. Params: lock_data. Returns: None."""
    lock_json = json.dumps(lock_data)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(LOCK_FILE, flags)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
            lock_file.write(lock_json)
            lock_file.flush()
            os.fsync(lock_file.fileno())
    except Exception:
        _safe_unlink(LOCK_FILE)
        raise


def _rewrite_lock_file(lock_data: dict[str, Any]) -> None:
    """Atomically rewrite the lock file. Params: lock_data. Returns: None."""
    temp_path = LOCK_FILE.with_name(
        f"{LOCK_FILE.name}.{lock_data['run_id']}.tmp"
    )
    try:
        with temp_path.open("w", encoding="utf-8") as lock_file:
            lock_file.write(json.dumps(lock_data))
            lock_file.flush()
            os.fsync(lock_file.fileno())
        os.replace(temp_path, LOCK_FILE)
    except Exception:
        _safe_unlink(temp_path)
        raise


def _stop_heartbeat() -> None:
    """Stop the background heartbeat thread. Returns: None."""
    stop_event = _LOCK_STATE["stop"]
    thread = _LOCK_STATE["thread"]
    if stop_event is not None:
        stop_event.set()
    if (
        thread is not None
        and thread.is_alive()
        and threading.current_thread() is not thread
    ):
        thread.join(timeout=5)
    _LOCK_STATE["stop"] = None
    _LOCK_STATE["thread"] = None


def _refresh_lock(owner: dict[str, Any]) -> bool:
    """Refresh the current lock if it is still owned. Returns: bool."""
    logger = get_stage_logger(__name__, STAGE)
    try:
        lock_data = _read_lock_data()
    except FileNotFoundError:
        logger.error("Pipeline lock disappeared during heartbeat")
        return False
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.error("Pipeline lock became unreadable during heartbeat")
        return False

    if not _lock_owner_matches(lock_data, owner):
        logger.error("Pipeline lock ownership changed during heartbeat")
        return False

    current_time = time.time()
    lock_data["timestamp"] = current_time
    lock_data["heartbeat_at"] = datetime.now().isoformat()
    _rewrite_lock_file(lock_data)
    return True


def _heartbeat_loop(
    owner: dict[str, Any],
    stop_event: threading.Event,
) -> None:
    """Refresh the active lock until the stop event is set. Returns: None."""
    logger = get_stage_logger(__name__, STAGE)
    while not stop_event.wait(HEARTBEAT_INTERVAL_SECONDS):
        try:
            if not _refresh_lock(owner):
                return
        except OSError:
            logger.exception("Failed to refresh pipeline lock heartbeat")
            return


def _start_heartbeat(owner: dict[str, Any]) -> None:
    """Start the background lock heartbeat. Params: owner. Returns: None."""
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(owner, stop_event),
        name=HEARTBEAT_THREAD_NAME,
        daemon=True,
    )
    _LOCK_STATE["stop"] = stop_event
    _LOCK_STATE["thread"] = heartbeat_thread
    heartbeat_thread.start()


def _activate_lock(lock_data: dict[str, Any]) -> None:
    """Store ownership state and start heartbeating. Returns: None."""
    owner = {
        "run_id": lock_data["run_id"],
        "pid": lock_data["pid"],
    }
    _LOCK_STATE["owner"] = owner
    try:
        _start_heartbeat(owner)
    except Exception:
        release_lock()
        raise


def _handle_unreadable_lock(invalid_exc: Exception) -> None:
    """Raise or remove an unreadable lock based on age. Returns: None."""
    logger = get_stage_logger(__name__, STAGE)
    try:
        modified_at = LOCK_FILE.stat().st_mtime
    except FileNotFoundError:
        return

    lock_age = _lock_age_hours(modified_at)
    if lock_age < LOCK_EXPIRY_HOURS:
        raise RuntimeError(
            f"Pipeline lock is active but unreadable "
            f"(age: {lock_age:.1f}h). "
            f"Another run may be in progress."
        ) from invalid_exc

    logger.warning(
        "Unreadable stale lock found (%.1fh old), removing",
        lock_age,
    )
    _safe_unlink(LOCK_FILE)


def _handle_existing_lock(exc: FileExistsError) -> None:
    """Raise or remove the existing lock. Params: exc. Returns: None."""
    logger = get_stage_logger(__name__, STAGE)
    try:
        lock_timestamp = _read_lock_timestamp()
    except FileNotFoundError:
        return
    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as invalid_exc:
        _handle_unreadable_lock(invalid_exc)
        return

    lock_age = _lock_age_hours(lock_timestamp)
    if lock_age < LOCK_EXPIRY_HOURS:
        raise RuntimeError(
            f"Pipeline lock is active "
            f"(age: {lock_age:.1f}h). "
            f"Another run may be in progress."
        ) from exc
    logger.warning(
        "Stale lock found (%.1fh old), removing",
        lock_age,
    )
    _safe_unlink(LOCK_FILE)


def _acquire_lock() -> None:
    """Create a lock file or abort if one is active.

    Ensures the processing directory exists, then checks for
    an existing lock. Fresh locks cause the pipeline to abort.
    Stale locks (older than LOCK_EXPIRY_HOURS) are removed.

    Returns:
        None

    Example:
        >>> _acquire_lock()
    """
    logger = get_stage_logger(__name__, STAGE)

    PROCESSING_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        lock_data = _build_lock_data()
        try:
            _write_new_lock_file(lock_data)
            _activate_lock(lock_data)
            logger.info("Pipeline lock acquired")
            return
        except FileExistsError as exc:
            _handle_existing_lock(exc)


def release_lock() -> None:
    """Remove the current run's lock file if still owned. Returns: None."""
    logger = get_stage_logger(__name__, STAGE)
    owner = _LOCK_STATE["owner"]
    _stop_heartbeat()

    if owner is None:
        return

    try:
        lock_data = _read_lock_data()
    except FileNotFoundError:
        _LOCK_STATE["owner"] = None
        return
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning(
            "Lock file unreadable during release, leaving it in place"
        )
        _LOCK_STATE["owner"] = None
        return

    if not _lock_owner_matches(lock_data, owner):
        logger.warning(
            "Lock ownership changed before release, leaving it in place"
        )
        _LOCK_STATE["owner"] = None
        return

    if _safe_unlink(LOCK_FILE):
        logger.info("Pipeline lock released")
    _LOCK_STATE["owner"] = None


def _processing_contents() -> list[Path]:
    """List processing dir items, excluding lock and staging dirs."""
    if not PROCESSING_DIR.exists():
        return []
    return [
        item
        for item in PROCESSING_DIR.iterdir()
        if item.name != LOCK_FILE.name
        and not item.name.startswith(".staging_")
    ]


def _clean_stale_processing() -> None:
    """Remove leftover files from a crashed previous run.

    If the processing directory contains files other than
    the lock, a previous run crashed without archiving.
    Archives those files for debugging, then cleans the
    directory so the current run starts fresh.

    Returns:
        None

    Example:
        >>> _clean_stale_processing()
    """
    logger = get_stage_logger(__name__, STAGE)

    contents = _processing_contents()
    if not contents:
        return

    logger.warning("Stale processing files found, archiving")
    _archive_and_clean(contents, "crashed")


def archive_run() -> None:
    """Archive the current run's processing output.

    Zips the processing directory contents (excluding the
    lock file) into the archive directory with a timestamped
    name, then cleans the processing directory. Also prunes
    old archives beyond the retention limit.

    Returns:
        None

    Example:
        >>> archive_run()
    """
    logger = get_stage_logger(__name__, STAGE)

    contents = _processing_contents()
    if not contents:
        return

    _archive_and_clean(contents, "run")
    _prune_old_files(ARCHIVE_DIR, "*.zip", "archives")
    _prune_old_files(LOGS_DIR, "*.log", "logs")
    logger.info("Run archived")


def _archive_and_clean(contents: list[Path], prefix: str) -> None:
    """Zip processing contents to archive and remove them.

    Copies only the specified files to a temp directory
    before zipping, so the lock file is never included.

    Params:
        contents: List of Path objects to archive
        prefix: Archive filename prefix ("run" or "crashed")
    """
    logger = get_stage_logger(__name__, STAGE)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{prefix}_{timestamp}"
    archive_path = ARCHIVE_DIR / archive_name

    staging = PROCESSING_DIR / f".staging_{timestamp}"
    staging.mkdir()
    for item in contents:
        dest = staging / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    shutil.make_archive(str(archive_path), "zip", staging)
    shutil.rmtree(staging)
    logger.info("Archived to %s.zip", archive_name)

    for item in contents:
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def _prune_old_files(directory: Path, pattern: str, label: str) -> None:
    """Delete old files beyond the retention limit.

    Keeps the most recent files matching the pattern, sorted
    by name (which includes timestamp).

    Params:
        directory: Directory to prune
        pattern: Glob pattern to match files
        label: Label for log message (e.g. "archives", "logs")
    """
    if not directory.exists():
        return

    retention = get_retention_count()
    files = sorted(directory.glob(pattern))
    if len(files) <= retention:
        return

    logger = get_stage_logger(__name__, STAGE)
    to_remove = files[: len(files) - retention]
    for old_file in to_remove:
        old_file.unlink()
    logger.info("Pruned %d old %s", len(to_remove), label)


def run_startup() -> tuple[Any, LLMClient]:
    """Initialize the pipeline and return connections.

    Handles config loading, logging setup, SSL, lock
    acquisition, crash recovery cleanup, and verification
    of database and LLM connectivity.

    Returns:
        tuple of (psycopg2 connection, LLMClient)

    Example:
        >>> conn, llm = run_startup()
    """
    load_config()
    setup_logging()
    logger = get_stage_logger(__name__, STAGE)
    logger.info("Ingestion pipeline starting")

    setup_ssl()

    conn = None
    llm = None
    _acquire_lock()
    try:
        _clean_stale_processing()

        logger.info("Initializing LLM client")
        llm = LLMClient()
        llm.test_connection()

        logger.info("Connecting to database")
        conn = get_connection()
        verify_connection(conn)
        logger.info("Ensuring database schema objects")
        ensure_schema_objects(conn)
    except Exception:
        if conn is not None:
            conn.close()
        release_lock()
        raise

    logger.info("Startup complete")
    return conn, llm
