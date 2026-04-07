"""Stage 12: Enrichment persistence to database.

Writes all enrichment data (metadata, sections, content
units with embeddings) to PostgreSQL and creates indexes
for retrieval.
"""

import logging

from ...utils.file_types import ExtractionResult
from ...utils.llm_connector import LLMClient
from ...utils.logging_setup import get_stage_logger
from ...utils.postgres_connector import (
    delete_enrichment,
    fetch_current_version_id,
    get_connection,
    insert_document_content,
    insert_document_metadata,
    insert_document_sections,
)

STAGE = "12-PERSISTENCE"

logger = logging.getLogger(__name__)


def persist_enrichment(
    result: ExtractionResult,
    llm: LLMClient,
) -> ExtractionResult:
    """Persist enrichment data to database.

    Opens a DB connection, looks up the current
    document_version_id, deletes any existing enrichment
    rows for idempotency, then inserts metadata, sections,
    and content units with their embeddings and tsvectors.

    Params:
        result: ExtractionResult from upstream stage
        llm: Initialized LLM client (unused, matches
            stage function signature)

    Returns:
        ExtractionResult -- passed through unchanged
    """
    del llm
    stage_log = get_stage_logger(__name__, STAGE)

    conn = get_connection()
    try:
        version_id = fetch_current_version_id(conn, result.file_path)

        delete_enrichment(conn, version_id)

        metadata = result.document_metadata or {}
        insert_document_metadata(conn, version_id, metadata)

        insert_document_sections(conn, version_id, result.sections)

        insert_document_content(conn, version_id, result.content_units)

        stage_log.info(
            "Persisted enrichment for %s" " (sections=%d, content_units=%d)",
            result.file_path,
            len(result.sections),
            len(result.content_units),
        )
    finally:
        conn.close()

    return result
