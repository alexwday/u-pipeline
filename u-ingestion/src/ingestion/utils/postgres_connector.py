"""PostgreSQL connection and verification."""

import json
import logging
from typing import List

import psycopg2

from .config_setup import get_database_config, get_database_schema
from .checkpointing import get_downstream_stages
from .file_types import (
    DocumentVersion,
    FileRecord,
    PrunableDocumentVersion,
    StageCheckpoint,
)

logger = logging.getLogger(__name__)


def _table_ddl(schema: str) -> dict[str, str]:
    """Build CREATE TABLE DDL for pipeline tables. Returns: dict[str, str]."""
    return {
        "document_catalog": f"""
            CREATE TABLE IF NOT EXISTS {schema}.document_catalog (
                file_path           TEXT PRIMARY KEY,
                data_source         TEXT NOT NULL,
                filter_1            TEXT NOT NULL DEFAULT '',
                filter_2            TEXT NOT NULL DEFAULT '',
                filter_3            TEXT NOT NULL DEFAULT '',
                filename            TEXT NOT NULL,
                filetype            TEXT NOT NULL,
                file_size           BIGINT NOT NULL,
                date_last_modified  DOUBLE PRECISION NOT NULL,
                file_hash           TEXT NOT NULL DEFAULT ''
            );
        """,
        "document_versions": f"""
            CREATE TABLE IF NOT EXISTS {schema}.document_versions (
                id                  BIGSERIAL PRIMARY KEY,
                file_path           TEXT NOT NULL,
                data_source         TEXT NOT NULL,
                filter_1            TEXT NOT NULL DEFAULT '',
                filter_2            TEXT NOT NULL DEFAULT '',
                filter_3            TEXT NOT NULL DEFAULT '',
                filename            TEXT NOT NULL,
                filetype            TEXT NOT NULL,
                file_size           BIGINT NOT NULL,
                date_last_modified  DOUBLE PRECISION NOT NULL,
                file_hash           TEXT NOT NULL,
                is_current          BOOLEAN NOT NULL DEFAULT TRUE,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (file_path, file_hash)
            );
        """,
        "stage_checkpoints": f"""
            CREATE TABLE IF NOT EXISTS {schema}.stage_checkpoints (
                document_version_id BIGINT NOT NULL
                    REFERENCES {schema}.document_versions (id)
                    ON DELETE CASCADE,
                stage_name          TEXT NOT NULL,
                status              TEXT NOT NULL,
                stage_signature     TEXT NOT NULL,
                artifact_path       TEXT NOT NULL DEFAULT '',
                artifact_checksum   TEXT NOT NULL DEFAULT '',
                error_message       TEXT NOT NULL DEFAULT '',
                started_at          TIMESTAMPTZ,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at        TIMESTAMPTZ,
                PRIMARY KEY (document_version_id, stage_name)
            );
        """,
        "document_metadata": f"""
            CREATE TABLE IF NOT EXISTS {schema}.document_metadata (
                document_version_id BIGINT PRIMARY KEY
                    REFERENCES {schema}.document_versions (id)
                    ON DELETE CASCADE,
                structure_type      TEXT NOT NULL DEFAULT '',
                title               TEXT NOT NULL DEFAULT '',
                authors             TEXT NOT NULL DEFAULT '',
                publication_date    TEXT NOT NULL DEFAULT '',
                language            TEXT NOT NULL DEFAULT 'en',
                has_toc             BOOLEAN NOT NULL DEFAULT FALSE,
                source_toc          JSONB NOT NULL
                    DEFAULT '[]'::jsonb,
                generated_toc       JSONB NOT NULL
                    DEFAULT '[]'::jsonb,
                rationale           TEXT NOT NULL DEFAULT '',
                executive_summary   TEXT NOT NULL DEFAULT '',
                keywords            JSONB NOT NULL
                    DEFAULT '[]'::jsonb,
                entities            JSONB NOT NULL
                    DEFAULT '[]'::jsonb,
                summary_embedding   vector(3072),
                created_at          TIMESTAMPTZ NOT NULL
                    DEFAULT NOW()
            );
        """,
        "document_sections": f"""
            CREATE TABLE IF NOT EXISTS {schema}.document_sections (
                id                  BIGSERIAL PRIMARY KEY,
                document_version_id BIGINT NOT NULL
                    REFERENCES {schema}.document_versions (id)
                    ON DELETE CASCADE,
                section_id          TEXT NOT NULL,
                parent_section_id   TEXT NOT NULL DEFAULT '',
                level               TEXT NOT NULL
                    DEFAULT 'section',
                title               TEXT NOT NULL DEFAULT '',
                sequence            INTEGER NOT NULL DEFAULT 0,
                page_start          INTEGER NOT NULL DEFAULT 0,
                page_end            INTEGER NOT NULL DEFAULT 0,
                token_count         INTEGER NOT NULL DEFAULT 0,
                summary             TEXT NOT NULL DEFAULT '',
                keywords            JSONB NOT NULL
                    DEFAULT '[]'::jsonb,
                entities            JSONB NOT NULL
                    DEFAULT '[]'::jsonb,
                summary_embedding   vector(3072),
                keyword_embedding   vector(3072),
                entity_embedding    vector(3072),
                created_at          TIMESTAMPTZ NOT NULL
                    DEFAULT NOW(),
                UNIQUE (document_version_id, section_id)
            );
        """,
        "document_content": f"""
            CREATE TABLE IF NOT EXISTS {schema}.document_content (
                id                  BIGSERIAL PRIMARY KEY,
                document_version_id BIGINT NOT NULL
                    REFERENCES {schema}.document_versions (id)
                    ON DELETE CASCADE,
                content_unit_id     TEXT NOT NULL,
                chunk_id            TEXT NOT NULL DEFAULT '',
                section_id          TEXT NOT NULL DEFAULT '',
                page_number         INTEGER NOT NULL,
                parent_page_number  INTEGER NOT NULL DEFAULT 0,
                raw_content         TEXT NOT NULL DEFAULT '',
                chunk_context       TEXT NOT NULL DEFAULT '',
                chunk_header        TEXT NOT NULL DEFAULT '',
                sheet_passthrough_content TEXT NOT NULL DEFAULT '',
                section_passthrough_content TEXT NOT NULL DEFAULT '',
                keywords            JSONB NOT NULL
                    DEFAULT '[]'::jsonb,
                entities            JSONB NOT NULL
                    DEFAULT '[]'::jsonb,
                token_count         INTEGER NOT NULL DEFAULT 0,
                content_embedding   vector(3072),
                keyword_embedding   vector(3072),
                entity_embedding    vector(3072),
                content_tsvector    TSVECTOR,
                created_at          TIMESTAMPTZ NOT NULL
                    DEFAULT NOW(),
                UNIQUE (document_version_id, content_unit_id)
            );
        """,
    }


def get_connection():
    """Open a psycopg2 connection using database config.

    Returns:
        psycopg2 connection object

    Example:
        >>> conn = get_connection()
        >>> conn.closed
        0
    """
    return psycopg2.connect(**get_database_config())


def _index_ddl(schema: str) -> list[str]:
    """Build CREATE INDEX DDL for retrieval indexes.

    Params: schema (str). Returns: list[str].
    """
    return [
        (
            f"CREATE INDEX IF NOT EXISTS idx_dc_tsvector"
            f" ON {schema}.document_content"
            f" USING gin (content_tsvector)"
        ),
        (
            f"CREATE INDEX IF NOT EXISTS idx_dc_keywords"
            f" ON {schema}.document_content"
            f" USING gin (keywords)"
        ),
        (
            f"CREATE INDEX IF NOT EXISTS idx_dc_entities"
            f" ON {schema}.document_content"
            f" USING gin (entities)"
        ),
        (
            f"CREATE INDEX IF NOT EXISTS idx_ds_keywords"
            f" ON {schema}.document_sections"
            f" USING gin (keywords)"
        ),
        (
            f"CREATE INDEX IF NOT EXISTS idx_dc_doc_version"
            f" ON {schema}.document_content"
            f" (document_version_id)"
        ),
        (
            f"CREATE INDEX IF NOT EXISTS idx_dc_section"
            f" ON {schema}.document_content"
            f" (document_version_id, section_id)"
        ),
        (
            f"CREATE INDEX IF NOT EXISTS idx_ds_doc_version"
            f" ON {schema}.document_sections"
            f" (document_version_id)"
        ),
    ]


def _vector_alter_ddl(schema: str) -> list[str]:
    """Build ALTER TABLE DDL to add vector columns to existing tables.

    Params: schema (str). Returns: list[str].
    """
    return [
        (
            f"ALTER TABLE {schema}.document_metadata"
            f" ADD COLUMN IF NOT EXISTS"
            f" summary_embedding vector(3072)"
        ),
        (
            f"ALTER TABLE {schema}.document_sections"
            f" ADD COLUMN IF NOT EXISTS"
            f" summary_embedding vector(3072)"
        ),
        (
            f"ALTER TABLE {schema}.document_sections"
            f" ADD COLUMN IF NOT EXISTS"
            f" keyword_embedding vector(3072)"
        ),
        (
            f"ALTER TABLE {schema}.document_sections"
            f" ADD COLUMN IF NOT EXISTS"
            f" entity_embedding vector(3072)"
        ),
        (
            f"ALTER TABLE {schema}.document_content"
            f" ADD COLUMN IF NOT EXISTS"
            f" content_embedding vector(3072)"
        ),
        (
            f"ALTER TABLE {schema}.document_content"
            f" ADD COLUMN IF NOT EXISTS"
            f" keyword_embedding vector(3072)"
        ),
        (
            f"ALTER TABLE {schema}.document_content"
            f" ADD COLUMN IF NOT EXISTS"
            f" entity_embedding vector(3072)"
        ),
        (
            f"ALTER TABLE {schema}.document_content"
            f" ADD COLUMN IF NOT EXISTS"
            f" content_tsvector TSVECTOR"
        ),
    ]


def ensure_schema_objects(conn) -> None:
    """Create the pipeline schema and tables when missing.

    Params:
        conn: psycopg2 connection
    """
    schema = get_database_schema()
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
        for ddl in _table_ddl(schema).values():
            cur.execute(ddl)
        cur.execute(
            f"ALTER TABLE {schema}.stage_checkpoints"
            f" ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;"
        )
        cur.execute(
            f"ALTER TABLE {schema}.document_metadata"
            f" ADD COLUMN IF NOT EXISTS has_toc BOOLEAN"
            f" NOT NULL DEFAULT FALSE;"
        )
        cur.execute(
            f"ALTER TABLE {schema}.document_metadata"
            f" ADD COLUMN IF NOT EXISTS source_toc JSONB NOT NULL"
            f" DEFAULT '[]'::jsonb;"
        )
        cur.execute(
            f"ALTER TABLE {schema}.document_metadata"
            f" ADD COLUMN IF NOT EXISTS generated_toc JSONB NOT NULL"
            f" DEFAULT '[]'::jsonb;"
        )
        cur.execute(
            f"ALTER TABLE {schema}.document_metadata"
            f" ADD COLUMN IF NOT EXISTS rationale"
            f" TEXT NOT NULL DEFAULT '';"
        )
        cur.execute(
            f"ALTER TABLE {schema}.document_content"
            f" ADD COLUMN IF NOT EXISTS"
            f" sheet_passthrough_content TEXT NOT NULL DEFAULT '';"
        )
        cur.execute(
            f"ALTER TABLE {schema}.document_content"
            f" ADD COLUMN IF NOT EXISTS"
            f" section_passthrough_content TEXT NOT NULL DEFAULT '';"
        )
        for alter_ddl in _vector_alter_ddl(schema):
            cur.execute(alter_ddl)
        for idx_ddl in _index_ddl(schema):
            cur.execute(idx_ddl)
    conn.commit()


def verify_connection(conn) -> bool:
    """Validate database connectivity.

    Params:
        conn: psycopg2 connection

    Returns:
        bool — True if connection is alive

    Example:
        >>> verify_connection(conn)
        True
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        logger.info("Database connection test passed")
        return True
    except Exception:
        logger.error("Database connection test failed")
        raise


def fetch_catalog_records(conn) -> List:
    """Fetch all rows from document_catalog as FileRecords.

    Params:
        conn: psycopg2 connection

    Returns:
        list[FileRecord] — one per catalog row

    Example:
        >>> records = fetch_catalog_records(conn)
        >>> len(records)
        42
    """
    schema = get_database_schema()
    sql = f"""
        SELECT data_source, filter_1, filter_2, filter_3,
               filename, filetype, file_size, date_last_modified,
               file_hash, file_path
        FROM {schema}.document_catalog
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [
        FileRecord(
            data_source=row[0],
            filter_1=row[1],
            filter_2=row[2],
            filter_3=row[3],
            filename=row[4],
            filetype=row[5],
            file_size=row[6],
            date_last_modified=row[7],
            file_hash=row[8],
            file_path=row[9],
        )
        for row in rows
    ]


def upsert_catalog_record(conn, record: FileRecord) -> None:
    """Insert or update a document_catalog row.

    Params:
        conn: psycopg2 connection
        record: FileRecord to persist
    """
    schema = get_database_schema()
    sql = f"""
        INSERT INTO {schema}.document_catalog (
            file_path, data_source, filter_1, filter_2, filter_3,
            filename, filetype, file_size, date_last_modified, file_hash
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (file_path) DO UPDATE
        SET data_source = EXCLUDED.data_source,
            filter_1 = EXCLUDED.filter_1,
            filter_2 = EXCLUDED.filter_2,
            filter_3 = EXCLUDED.filter_3,
            filename = EXCLUDED.filename,
            filetype = EXCLUDED.filetype,
            file_size = EXCLUDED.file_size,
            date_last_modified = EXCLUDED.date_last_modified,
            file_hash = EXCLUDED.file_hash
    """
    params = (
        record.file_path,
        record.data_source,
        record.filter_1,
        record.filter_2,
        record.filter_3,
        record.filename,
        record.filetype,
        record.file_size,
        record.date_last_modified,
        record.file_hash,
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()


def remove_deleted_files(conn, file_paths: list[str]) -> None:
    """Remove deleted files from current catalog state.

    Deletes catalog rows for the file paths and marks any current
    document versions as non-current so deleted files no longer
    participate in planning or current-state queries.

    Params:
        conn: psycopg2 connection
        file_paths: Absolute source file paths removed from disk
    """
    normalized_paths = sorted(set(file_paths))
    if not normalized_paths:
        return

    schema = get_database_schema()
    deactivate_sql = f"""
        UPDATE {schema}.document_versions
        SET is_current = FALSE
        WHERE file_path = ANY(%s)
          AND is_current = TRUE
    """
    delete_catalog_sql = f"""
        DELETE FROM {schema}.document_catalog
        WHERE file_path = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(deactivate_sql, (normalized_paths,))
        cur.execute(delete_catalog_sql, (normalized_paths,))
    conn.commit()


def register_document_version(conn, record: FileRecord) -> DocumentVersion:
    """Ensure the current document version row exists and is current.

    Params:
        conn: psycopg2 connection
        record: FileRecord with populated file_hash

    Returns:
        DocumentVersion for the current file_path and file_hash
    """
    schema = get_database_schema()
    deactivate_sql = f"""
        UPDATE {schema}.document_versions
        SET is_current = FALSE
        WHERE file_path = %s
          AND file_hash <> %s
          AND is_current = TRUE
    """
    upsert_sql = f"""
        INSERT INTO {schema}.document_versions (
            file_path, data_source, filter_1, filter_2, filter_3,
            filename, filetype, file_size, date_last_modified,
            file_hash, is_current
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
        ON CONFLICT (file_path, file_hash) DO UPDATE
        SET data_source = EXCLUDED.data_source,
            filter_1 = EXCLUDED.filter_1,
            filter_2 = EXCLUDED.filter_2,
            filter_3 = EXCLUDED.filter_3,
            filename = EXCLUDED.filename,
            filetype = EXCLUDED.filetype,
            file_size = EXCLUDED.file_size,
            date_last_modified = EXCLUDED.date_last_modified,
            is_current = TRUE
        RETURNING id, file_path, data_source, filter_1, filter_2, filter_3,
                  filename, filetype, file_size, date_last_modified,
                  file_hash, is_current
    """
    params = (
        record.file_path,
        record.data_source,
        record.filter_1,
        record.filter_2,
        record.filter_3,
        record.filename,
        record.filetype,
        record.file_size,
        record.date_last_modified,
        record.file_hash,
    )
    with conn.cursor() as cur:
        cur.execute(deactivate_sql, (record.file_path, record.file_hash))
        cur.execute(upsert_sql, params)
        row = cur.fetchone()
    conn.commit()
    return DocumentVersion(
        document_version_id=row[0],
        file_path=row[1],
        data_source=row[2],
        filter_1=row[3],
        filter_2=row[4],
        filter_3=row[5],
        filename=row[6],
        filetype=row[7],
        file_size=row[8],
        date_last_modified=row[9],
        file_hash=row[10],
        is_current=bool(row[11]),
    )


def fetch_current_document_versions(conn) -> List[DocumentVersion]:
    """Fetch all current document version rows.

    Params:
        conn: psycopg2 connection

    Returns:
        list[DocumentVersion] — current version rows
    """
    schema = get_database_schema()
    sql = f"""
        SELECT id, file_path, data_source, filter_1, filter_2, filter_3,
               filename, filetype, file_size, date_last_modified,
               file_hash, is_current
        FROM {schema}.document_versions
        WHERE is_current = TRUE
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [
        DocumentVersion(
            document_version_id=row[0],
            file_path=row[1],
            data_source=row[2],
            filter_1=row[3],
            filter_2=row[4],
            filter_3=row[5],
            filename=row[6],
            filetype=row[7],
            file_size=row[8],
            date_last_modified=row[9],
            file_hash=row[10],
            is_current=bool(row[11]),
        )
        for row in rows
    ]


def fetch_stage_checkpoints(conn) -> List[StageCheckpoint]:
    """Fetch stage checkpoints for current document versions.

    Params:
        conn: psycopg2 connection

    Returns:
        list[StageCheckpoint] — persisted stage status rows
    """
    schema = get_database_schema()
    sql = f"""
        SELECT sc.document_version_id, sc.stage_name, sc.status,
               sc.stage_signature, sc.artifact_path,
               sc.artifact_checksum, sc.error_message
        FROM {schema}.stage_checkpoints AS sc
        INNER JOIN {schema}.document_versions AS dv
            ON dv.id = sc.document_version_id
        WHERE dv.is_current = TRUE
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [
        StageCheckpoint(
            document_version_id=row[0],
            stage_name=row[1],
            status=row[2],
            stage_signature=row[3],
            artifact_path=row[4],
            artifact_checksum=row[5],
            error_message=row[6],
        )
        for row in rows
    ]


def fetch_prunable_document_versions(
    conn,
    retain_non_current: int,
) -> List[PrunableDocumentVersion]:
    """Fetch non-current versions beyond the configured retention window.

    Params:
        conn: psycopg2 connection
        retain_non_current: Number of old versions to keep per file path

    Returns:
        list[PrunableDocumentVersion] — versions eligible for cleanup
    """
    schema = get_database_schema()
    sql = f"""
        WITH ranked_versions AS (
            SELECT id, file_path,
                   ROW_NUMBER() OVER (
                       PARTITION BY file_path
                       ORDER BY is_current DESC, created_at DESC, id DESC
                   ) AS version_rank
            FROM {schema}.document_versions
        )
        SELECT rv.id, rv.file_path, COALESCE(sc.artifact_path, '')
        FROM ranked_versions AS rv
        LEFT JOIN {schema}.stage_checkpoints AS sc
            ON sc.document_version_id = rv.id
        WHERE rv.version_rank > %s
        ORDER BY rv.id
    """
    keep_rank = max(retain_non_current, 0) + 1
    with conn.cursor() as cur:
        cur.execute(sql, (keep_rank,))
        rows = cur.fetchall()

    grouped: dict[int, PrunableDocumentVersion] = {}
    for row in rows:
        version_id = int(row[0])
        artifact_path = str(row[2])
        if version_id not in grouped:
            grouped[version_id] = PrunableDocumentVersion(
                document_version_id=version_id,
                file_path=str(row[1]),
                artifact_paths=[],
            )
        if artifact_path:
            grouped[version_id].artifact_paths.append(artifact_path)
    return list(grouped.values())


def delete_document_versions(conn, document_version_ids: list[int]) -> None:
    """Delete document versions and cascade their checkpoints.

    Params:
        conn: psycopg2 connection
        document_version_ids: Version identifiers to delete
    """
    if not document_version_ids:
        return
    schema = get_database_schema()
    sql = f"""
        DELETE FROM {schema}.document_versions
        WHERE id = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (document_version_ids,))
    conn.commit()


def clear_stage_checkpoints(
    conn,
    document_version_id: int,
    start_stage: str,
) -> None:
    """Delete a stage and all downstream checkpoints.

    Params:
        conn: psycopg2 connection
        document_version_id: Owning document version identifier
        start_stage: First stage to clear
    """
    schema = get_database_schema()
    sql = f"""
        DELETE FROM {schema}.stage_checkpoints
        WHERE document_version_id = %s
          AND stage_name = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (document_version_id, list(get_downstream_stages(start_stage))),
        )
    conn.commit()


def mark_stage_checkpoint_started(
    conn,
    document_version_id: int,
    stage_name: str,
    stage_signature: str,
) -> None:
    """Record that a stage has started processing.

    Params:
        conn: psycopg2 connection
        document_version_id: Owning document version identifier
        stage_name: Pipeline stage name
        stage_signature: Fingerprint of stage code/config
    """
    schema = get_database_schema()
    sql = f"""
        INSERT INTO {schema}.stage_checkpoints (
            document_version_id, stage_name, status, stage_signature,
            started_at
        )
        VALUES (%s, %s, 'running', %s, NOW())
        ON CONFLICT (document_version_id, stage_name) DO UPDATE
        SET status = 'running',
            stage_signature = EXCLUDED.stage_signature,
            started_at = NOW(),
            updated_at = NOW(),
            completed_at = NULL
    """
    params = (document_version_id, stage_name, stage_signature)
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()


def mark_stage_checkpoint_succeeded(
    conn,
    document_version_id: int,
    stage_name: str,
    stage_signature: str,
    artifact_path: str,
    artifact_checksum: str,
) -> None:
    """Persist a successful stage checkpoint.

    Params:
        conn: psycopg2 connection
        document_version_id: Owning document version identifier
        stage_name: Pipeline stage name
        stage_signature: Fingerprint of stage code/config
        artifact_path: Absolute artifact path
        artifact_checksum: SHA-256 checksum of artifact bytes
    """
    schema = get_database_schema()
    sql = f"""
        INSERT INTO {schema}.stage_checkpoints (
            document_version_id, stage_name, status, stage_signature,
            artifact_path, artifact_checksum, error_message, completed_at
        )
        VALUES (%s, %s, 'succeeded', %s, %s, %s, '', NOW())
        ON CONFLICT (document_version_id, stage_name) DO UPDATE
        SET status = EXCLUDED.status,
            stage_signature = EXCLUDED.stage_signature,
            artifact_path = EXCLUDED.artifact_path,
            artifact_checksum = EXCLUDED.artifact_checksum,
            error_message = '',
            updated_at = NOW(),
            completed_at = NOW()
    """
    params = (
        document_version_id,
        stage_name,
        stage_signature,
        artifact_path,
        artifact_checksum,
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()


def mark_stage_checkpoint_failed(
    conn,
    document_version_id: int,
    stage_name: str,
    stage_signature: str,
    error_message: str,
) -> None:
    """Persist a failed stage checkpoint.

    Params:
        conn: psycopg2 connection
        document_version_id: Owning document version identifier
        stage_name: Pipeline stage name
        stage_signature: Fingerprint of stage code/config
        error_message: Failure summary for the last attempt
    """
    schema = get_database_schema()
    sql = f"""
        INSERT INTO {schema}.stage_checkpoints (
            document_version_id, stage_name, status, stage_signature,
            artifact_path, artifact_checksum, error_message, completed_at
        )
        VALUES (%s, %s, 'failed', %s, '', '', %s, NULL)
        ON CONFLICT (document_version_id, stage_name) DO UPDATE
        SET status = EXCLUDED.status,
            stage_signature = EXCLUDED.stage_signature,
            artifact_path = '',
            artifact_checksum = '',
            error_message = EXCLUDED.error_message,
            updated_at = NOW(),
            completed_at = NULL
    """
    params = (
        document_version_id,
        stage_name,
        stage_signature,
        error_message,
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()


# -----------------------------------------------------------------
# Enrichment persistence helpers
# -----------------------------------------------------------------


def format_vector(embedding: list) -> str | None:
    """Format embedding list for pgvector insertion.

    Params:
        embedding: List of floats from the embedding model

    Returns:
        str formatted as "[0.1,0.2,...]" or None if empty
    """
    if not embedding:
        return None
    return "[" + ",".join(str(f) for f in embedding) + "]"


def fetch_current_version_id(conn, file_path: str) -> int:
    """Get document_version_id for the current version of a file.

    Params:
        conn: psycopg2 connection
        file_path: Absolute path to the source file

    Returns:
        int -- the document_version_id

    Example:
        >>> fetch_current_version_id(conn, "/data/doc.pdf")
        42
    """
    schema = get_database_schema()
    sql = f"""
        SELECT id FROM {schema}.document_versions
        WHERE file_path = %s AND is_current = TRUE
        ORDER BY id DESC LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (file_path,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"No current version found for {file_path}")
    return int(row[0])


def delete_enrichment(conn, document_version_id: int) -> None:
    """Delete all enrichment data for a document version.

    Removes rows from document_content, document_sections,
    and document_metadata so the version can be re-persisted
    idempotently.

    Params:
        conn: psycopg2 connection
        document_version_id: Owning document version id
    """
    schema = get_database_schema()
    tables = [
        "document_content",
        "document_sections",
        "document_metadata",
    ]
    with conn.cursor() as cur:
        for table in tables:
            cur.execute(
                f"DELETE FROM {schema}.{table}"
                f" WHERE document_version_id = %s",
                (document_version_id,),
            )
    conn.commit()


def insert_document_metadata(
    conn,
    document_version_id: int,
    metadata: dict,
) -> None:
    """Insert one document_metadata row with embedding.

    Params:
        conn: psycopg2 connection
        document_version_id: Owning document version id
        metadata: Document metadata dict from enrichment
    """
    schema = get_database_schema()
    sql = f"""
        INSERT INTO {schema}.document_metadata (
            document_version_id, structure_type, title,
            authors, publication_date, language,
            has_toc, source_toc, generated_toc, rationale,
            executive_summary, keywords, entities,
            summary_embedding
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s::vector
        )
    """
    params = (
        document_version_id,
        metadata.get("structure_type", ""),
        metadata.get("title", ""),
        metadata.get("authors", ""),
        metadata.get("publication_date", ""),
        metadata.get("language", "en"),
        metadata.get("has_toc", False),
        json.dumps(metadata.get("source_toc_entries", [])),
        json.dumps(metadata.get("generated_toc_entries", [])),
        metadata.get("rationale", ""),
        metadata.get("executive_summary", ""),
        json.dumps(metadata.get("keywords", [])),
        json.dumps(metadata.get("entities", [])),
        format_vector(metadata.get("summary_embedding", [])),
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()


def insert_document_sections(
    conn,
    document_version_id: int,
    sections: list,
) -> None:
    """Batch-insert section rows with embeddings.

    Primary sections include summary, keyword, and entity
    embeddings. Subsections have NULL embedding columns.

    Params:
        conn: psycopg2 connection
        document_version_id: Owning document version id
        sections: Section dicts from ExtractionResult
    """
    if not sections:
        return
    schema = get_database_schema()
    sql = f"""
        INSERT INTO {schema}.document_sections (
            document_version_id, section_id,
            parent_section_id, level, title,
            sequence, page_start, page_end,
            token_count, summary, keywords, entities,
            summary_embedding, keyword_embedding,
            entity_embedding
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s::vector, %s::vector, %s::vector
        )
    """
    with conn.cursor() as cur:
        for section in sections:
            params = (
                document_version_id,
                section.get("section_id", ""),
                section.get("parent_section_id", ""),
                section.get("level", "section"),
                section.get("title", ""),
                section.get("sequence", 0),
                section.get("page_start", 0),
                section.get("page_end", 0),
                section.get("token_count", 0),
                section.get("summary", ""),
                json.dumps(section.get("keywords", [])),
                json.dumps(section.get("entities", [])),
                format_vector(section.get("summary_embedding", [])),
                format_vector(section.get("keyword_embedding", [])),
                format_vector(section.get("entity_embedding", [])),
            )
            cur.execute(sql, params)
    conn.commit()


def insert_document_content(
    conn,
    document_version_id: int,
    content_units: list,
) -> None:
    """Batch-insert content unit rows with embeddings.

    Generates content_tsvector from raw_content using
    PostgreSQL to_tsvector('english', ...) for full-text
    search.

    Params:
        conn: psycopg2 connection
        document_version_id: Owning document version id
        content_units: Content unit dicts with embeddings
    """
    if not content_units:
        return
    schema = get_database_schema()
    sql = f"""
        INSERT INTO {schema}.document_content (
            document_version_id, content_unit_id,
            chunk_id, section_id,
            page_number, parent_page_number,
            raw_content, chunk_context, chunk_header,
            sheet_passthrough_content,
            section_passthrough_content,
            keywords, entities, token_count,
            content_embedding, keyword_embedding,
            entity_embedding,
            content_tsvector
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s::vector, %s::vector, %s::vector,
            to_tsvector('english', %s)
        )
    """
    with conn.cursor() as cur:
        for unit in content_units:
            raw_content = unit.get("raw_content", "")
            params = (
                document_version_id,
                unit.get("content_unit_id", ""),
                unit.get("chunk_id", ""),
                unit.get("section_id", ""),
                unit.get("page_number", 0),
                unit.get("parent_page_number", 0),
                raw_content,
                unit.get("chunk_context", ""),
                unit.get("chunk_header", ""),
                unit.get("sheet_passthrough_content", ""),
                unit.get("section_passthrough_content", ""),
                json.dumps(unit.get("keywords", [])),
                json.dumps(unit.get("entities", [])),
                unit.get("token_count", 0),
                format_vector(unit.get("content_embedding", [])),
                format_vector(unit.get("keyword_embedding", [])),
                format_vector(unit.get("entity_embedding", [])),
                raw_content,
            )
            cur.execute(sql, params)
    conn.commit()


# -----------------------------------------------------------------
# Enrichment verification and retrieval queries
# -----------------------------------------------------------------


def verify_enrichment_completeness(conn, document_version_id: int) -> dict:
    """Check that all enrichment data exists for a version.

    Params:
        conn: psycopg2 connection
        document_version_id: Document version to verify

    Returns:
        dict with metadata_exists (bool),
        section_count (int), content_count (int),
        content_with_embedding (int),
        sections_with_embedding (int)
    """
    schema = get_database_schema()
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM {schema}.document_metadata"
            f" WHERE document_version_id = %s",
            (document_version_id,),
        )
        meta_count = cur.fetchone()[0]

        cur.execute(
            f"SELECT COUNT(*) FROM {schema}.document_sections"
            f" WHERE document_version_id = %s",
            (document_version_id,),
        )
        section_count = cur.fetchone()[0]

        cur.execute(
            f"SELECT COUNT(*) FROM {schema}.document_content"
            f" WHERE document_version_id = %s",
            (document_version_id,),
        )
        content_count = cur.fetchone()[0]

        cur.execute(
            f"SELECT COUNT(*)"
            f" FROM {schema}.document_content"
            f" WHERE document_version_id = %s"
            f" AND content_embedding IS NOT NULL",
            (document_version_id,),
        )
        content_with_emb = cur.fetchone()[0]

        cur.execute(
            f"SELECT COUNT(*)"
            f" FROM {schema}.document_sections"
            f" WHERE document_version_id = %s"
            f" AND summary_embedding IS NOT NULL",
            (document_version_id,),
        )
        sections_with_emb = cur.fetchone()[0]

    return {
        "metadata_exists": meta_count > 0,
        "section_count": section_count,
        "content_count": content_count,
        "content_with_embedding": content_with_emb,
        "sections_with_embedding": sections_with_emb,
    }


def query_content_by_vector(
    conn,
    document_version_id: int,
    query_embedding: list,
    top_k: int = 5,
) -> list:
    """Cosine similarity search on content embeddings.

    Params:
        conn: psycopg2 connection
        document_version_id: Scope search to this version
        query_embedding: Query vector as list of floats
        top_k: Maximum results to return

    Returns:
        list of dicts with content_unit_id,
        raw_content, and distance
    """
    schema = get_database_schema()
    vec_str = format_vector(query_embedding)
    sql = f"""
        SELECT content_unit_id, raw_content,
               content_embedding <=> %s::vector AS distance
        FROM {schema}.document_content
        WHERE document_version_id = %s
          AND content_embedding IS NOT NULL
        ORDER BY distance
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (vec_str, document_version_id, top_k),
        )
        rows = cur.fetchall()
    return [
        {
            "content_unit_id": row[0],
            "raw_content": row[1],
            "distance": float(row[2]),
        }
        for row in rows
    ]


def query_content_by_keywords(
    conn,
    document_version_id: int,
    keywords: list,
) -> list:
    """Array containment search on content keywords.

    Params:
        conn: psycopg2 connection
        document_version_id: Scope search to this version
        keywords: Keywords to match via JSONB containment

    Returns:
        list of dicts with content_unit_id,
        raw_content, and keywords
    """
    schema = get_database_schema()
    sql = f"""
        SELECT content_unit_id, raw_content, keywords
        FROM {schema}.document_content
        WHERE document_version_id = %s
          AND keywords @> %s::jsonb
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (document_version_id, json.dumps(keywords)),
        )
        rows = cur.fetchall()
    return [
        {
            "content_unit_id": row[0],
            "raw_content": row[1],
            "keywords": row[2],
        }
        for row in rows
    ]


def query_content_by_text(
    conn,
    document_version_id: int,
    query: str,
) -> list:
    """Full-text BM25 search on content_tsvector.

    Params:
        conn: psycopg2 connection
        document_version_id: Scope search to this version
        query: Plain text query for ts_rank matching

    Returns:
        list of dicts with content_unit_id,
        raw_content, and rank
    """
    schema = get_database_schema()
    sql = f"""
        SELECT content_unit_id, raw_content,
               ts_rank(content_tsvector,
                       plainto_tsquery('english', %s)) AS rank
        FROM {schema}.document_content
        WHERE document_version_id = %s
          AND content_tsvector @@
              plainto_tsquery('english', %s)
        ORDER BY rank DESC
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (query, document_version_id, query),
        )
        rows = cur.fetchall()
    return [
        {
            "content_unit_id": row[0],
            "raw_content": row[1],
            "rank": float(row[2]),
        }
        for row in rows
    ]


def query_sections_by_summary(
    conn,
    document_version_id: int,
    query_embedding: list,
    top_k: int = 5,
) -> list:
    """Cosine similarity search on section summary embeddings.

    Params:
        conn: psycopg2 connection
        document_version_id: Scope search to this version
        query_embedding: Query vector as list of floats
        top_k: Maximum results to return

    Returns:
        list of dicts with section_id, title,
        summary, and distance
    """
    schema = get_database_schema()
    vec_str = format_vector(query_embedding)
    sql = f"""
        SELECT section_id, title, summary,
               summary_embedding <=> %s::vector AS distance
        FROM {schema}.document_sections
        WHERE document_version_id = %s
          AND summary_embedding IS NOT NULL
        ORDER BY distance
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (vec_str, document_version_id, top_k),
        )
        rows = cur.fetchall()
    return [
        {
            "section_id": row[0],
            "title": row[1],
            "summary": row[2],
            "distance": float(row[3]),
        }
        for row in rows
    ]
