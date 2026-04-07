"""Helpers for data-source and filter path metadata."""

import os
from pathlib import Path
from typing import Any

from .config_setup import get_data_source_path


def parse_relative_source_context(rel_path: str) -> tuple[str, str, str, str]:
    """Split a relative path into data_source and filters.

    Params:
        rel_path: Path relative to DATA_SOURCE_PATH

    Returns:
        tuple[str, str, str, str] -- data_source and filters
    """
    normalized = rel_path.strip(os.sep)
    if not normalized or normalized == ".":
        return "", "", "", ""

    parts = [part for part in normalized.split(os.sep) if part]
    data_source = parts[0] if parts else ""
    filter_1 = parts[1] if len(parts) > 1 else ""
    filter_2 = parts[2] if len(parts) > 2 else ""
    filter_3 = os.sep.join(parts[3:]) if len(parts) > 3 else ""
    return data_source, filter_1, filter_2, filter_3


def get_source_context_from_path(file_path: str) -> dict[str, str]:
    """Derive source context relative to DATA_SOURCE_PATH.

    Params:
        file_path: Absolute path to the source file

    Returns:
        dict[str, str] -- data_source and filter fields
    """
    try:
        base_path = Path(get_data_source_path()).resolve()
    except ValueError:
        return {
            "data_source": "",
            "filter_1": "",
            "filter_2": "",
            "filter_3": "",
        }

    file_dir = Path(file_path).resolve().parent
    try:
        relative_dir = file_dir.relative_to(base_path)
    except ValueError:
        return {
            "data_source": "",
            "filter_1": "",
            "filter_2": "",
            "filter_3": "",
        }

    data_source, filter_1, filter_2, filter_3 = parse_relative_source_context(
        str(relative_dir)
    )
    return {
        "data_source": data_source,
        "filter_1": filter_1,
        "filter_2": filter_2,
        "filter_3": filter_3,
    }


def get_result_source_context(result: Any) -> dict[str, str]:
    """Read source context from a result, with path fallback.

    Params:
        result: ExtractionResult-like object

    Returns:
        dict[str, str] -- data_source and filter fields
    """
    context = {
        "data_source": getattr(result, "data_source", "") or "",
        "filter_1": getattr(result, "filter_1", "") or "",
        "filter_2": getattr(result, "filter_2", "") or "",
        "filter_3": getattr(result, "filter_3", "") or "",
    }
    if any(context.values()):
        return context
    return get_source_context_from_path(getattr(result, "file_path", ""))
