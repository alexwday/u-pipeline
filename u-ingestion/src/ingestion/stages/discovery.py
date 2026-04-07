"""Stage 1: Filesystem scanning and catalog diff."""

import json
import os
from dataclasses import dataclass
from dataclasses import asdict
from typing import List

from ..utils.config_setup import get_data_source_path
from ..utils.file_types import (
    DiscoveryDiff,
    DiscoveryScan,
    FileRecord,
    compute_file_hash,
)
from ..utils.logging_setup import get_stage_logger
from ..utils.postgres_connector import fetch_catalog_records
from ..utils.source_context import parse_relative_source_context
from .startup import PROCESSING_DIR

STAGE = "1-DISCOVERY"


@dataclass
class DiscoveryRun:
    """Complete discovery-stage output for downstream planning.

    Params:
        scan: Full filesystem scan result
        diff: Catalog diff derived from the scan
    """

    scan: DiscoveryScan
    diff: DiscoveryDiff


def _parse_path_parts(rel_path: str) -> tuple:
    """Split a relative path into data_source and filters.

    Params:
        rel_path: Path relative to base, e.g. "src/2026/Q1/RBC"

    Returns:
        tuple of (data_source, filter_1, filter_2, filter_3)
    """
    return parse_relative_source_context(rel_path)


def _build_file_record(
    dirpath: str, fname: str, path_parts: tuple
) -> FileRecord:
    """Build a FileRecord from directory context and filename.

    Params:
        dirpath: Directory containing the file
        fname: Filename
        path_parts: Tuple of (data_source, f1, f2, f3)

    Returns:
        FileRecord
    """
    full_path = os.path.join(dirpath, fname)
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    stat = os.stat(full_path)
    return FileRecord(
        data_source=path_parts[0],
        filter_1=path_parts[1],
        filter_2=path_parts[2],
        filter_3=path_parts[3],
        filename=fname,
        filetype=ext,
        file_size=stat.st_size,
        date_last_modified=stat.st_mtime,
        file_hash="",
        file_path=full_path,
    )


def scan_filesystem(base_path: str) -> DiscoveryScan:
    """Walk a data-source tree and classify discovered files.

    First subfolder under base_path is the data_source.
    The next 1-3 levels become filter_1, filter_2, filter_3.
    Deeper nesting is flattened into filter_3.
    Hidden files and directories are skipped.

    Params:
        base_path: Absolute path to the data sources root

    Returns:
        DiscoveryScan with supported and unsupported records

    Example:
        >>> scan = scan_filesystem("/data/sources")
        >>> scan.supported[0].data_source
        "policy_docs"
    """
    logger = get_stage_logger(__name__, STAGE)
    supported: List[FileRecord] = []
    unsupported: List[FileRecord] = []

    if not os.path.isdir(base_path):
        logger.error("Base path does not exist: %s", base_path)
        return DiscoveryScan(supported=supported, unsupported=unsupported)

    for dirpath, dirnames, filenames in os.walk(base_path):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        rel = os.path.relpath(dirpath, base_path)
        if rel == ".":
            continue

        path_parts = _parse_path_parts(rel)

        for fname in filenames:
            if fname.startswith("."):
                continue
            record = _build_file_record(dirpath, fname, path_parts)
            if record.supported:
                supported.append(record)
            else:
                unsupported.append(record)

    logger.info(
        "Scanned %d supported and %d unsupported files across %s",
        len(supported),
        len(unsupported),
        base_path,
    )
    return DiscoveryScan(supported=supported, unsupported=unsupported)


def compute_diff(
    discovered: List[FileRecord],
    cataloged: List[FileRecord],
) -> DiscoveryDiff:
    """Compare discovered files against the catalog.

    Keys on file_path. New = on disk only. Deleted = in DB only.
    Modified = path matches but size differs, or size matches
    but date differs and hash differs (lazy hash). Unchanged =
    path + size + date match.

    Params:
        discovered: Files found on the filesystem
        cataloged: Files from the database catalog

    Returns:
        DiscoveryDiff with new, modified, deleted lists

    Example:
        >>> diff = compute_diff(disk_files, db_files)
        >>> len(diff.new)
        5
    """
    catalog_map = {r.file_path: r for r in cataloged}
    disk_map = {r.file_path: r for r in discovered}

    new: List[FileRecord] = []
    modified: List[FileRecord] = []
    deleted: List[FileRecord] = []

    for path, disk_rec in disk_map.items():
        if path not in catalog_map:
            new.append(disk_rec)
            continue

        cat_rec = catalog_map[path]
        if disk_rec.file_size != cat_rec.file_size:
            modified.append(disk_rec)
        elif disk_rec.date_last_modified != cat_rec.date_last_modified:
            disk_hash = compute_file_hash(path)
            cat_hash = cat_rec.file_hash
            if disk_hash != cat_hash:
                disk_rec.file_hash = disk_hash
                modified.append(disk_rec)

    for path, cat_rec in catalog_map.items():
        if path not in disk_map:
            deleted.append(cat_rec)

    return DiscoveryDiff(new=new, modified=modified, deleted=deleted)


def run_discovery(conn) -> DiscoveryRun:
    """Orchestrate filesystem scan, catalog fetch, and diff.

    Reads DATA_SOURCE_PATH from config, scans the tree, fetches
    the existing catalog from the database, computes the diff,
    and logs a summary.

    Params:
        conn: psycopg2 database connection

    Returns:
        DiscoveryRun with both the scan and diff results

    Example:
        >>> discovery = run_discovery(conn)
        >>> print(f"{len(discovery.diff.new)} new files")
    """
    logger = get_stage_logger(__name__, STAGE)
    logger.info("Starting file discovery")

    base_path = get_data_source_path()
    discovered = scan_filesystem(base_path)
    cataloged = fetch_catalog_records(conn)
    supported_cataloged = [record for record in cataloged if record.supported]
    ignored_cataloged = [
        record for record in cataloged if not record.supported
    ]
    diff = compute_diff(discovered.supported, supported_cataloged)

    output = {
        "new": [asdict(r) for r in diff.new],
        "modified": [asdict(r) for r in diff.modified],
        "deleted": [asdict(r) for r in diff.deleted],
        "unsupported_files": [asdict(r) for r in discovered.unsupported],
        "ignored_catalog_records": [asdict(r) for r in ignored_cataloged],
    }
    output_path = PROCESSING_DIR / "discovery.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    logger.info(
        "Discovery complete — supported: %d, unsupported: %d, "
        "new: %d, modified: %d, deleted: %d, unchanged: %d",
        len(discovered.supported),
        len(discovered.unsupported),
        len(diff.new),
        len(diff.modified),
        len(diff.deleted),
        len(discovered.supported) - len(diff.new) - len(diff.modified),
    )
    return DiscoveryRun(scan=discovered, diff=diff)
