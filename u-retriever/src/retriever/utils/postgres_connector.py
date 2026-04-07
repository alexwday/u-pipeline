"""PostgreSQL connection and query functions for retrieval."""

import logging

import psycopg2

from .config_setup import get_database_config, get_database_schema

logger = logging.getLogger(__name__)

_NUMERIC_CUID_ORDER = (
    "page_number,"
    " CAST(SPLIT_PART(content_unit_id, '.', 1) AS INTEGER),"
    " COALESCE(NULLIF(SPLIT_PART(content_unit_id, '.', 2),"
    " ''), '0')::INTEGER"
)


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


def verify_connection(conn) -> bool:
    """Validate database connectivity.

    Params:
        conn: psycopg2 connection

    Returns:
        bool -- True if connection is alive

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


def format_vector(embedding: list) -> str | None:
    """Serialize a float list to pgvector literal.

    Params: embedding (list). Returns: str or None.
    """
    if not embedding:
        return None
    return "[" + ",".join(str(f) for f in embedding) + "]"


# ------------------------------------------------------------------
# Document resolution
# ------------------------------------------------------------------


def resolve_document_version_ids(
    conn,
    bank: str,
    period: str,
    sources: list[str] | None = None,
) -> list[dict]:
    """Find document_version_ids matching bank+period.

    Params:
        conn: psycopg2 connection
        bank: Filter_2 value (e.g. "RBC")
        period: Filter_1 value (e.g. "2026_Q1")
        sources: Optional data_source whitelist

    Returns:
        list[dict] -- matching version rows

    Example:
        >>> resolve_document_version_ids(conn, "RBC", "2026_Q1")
        [{"document_version_id": 38, ...}]
    """
    schema = get_database_schema()
    sql = (
        f"SELECT id AS document_version_id,"
        f" data_source, filter_1, filter_2, filename"
        f" FROM {schema}.document_versions"
        f" WHERE is_current = TRUE"
        f" AND filter_2 = %s"
        f" AND filter_1 = %s"
    )
    params: list = [bank, period]
    if sources:
        sql += " AND data_source = ANY(%s)"
        params.append(sources)
    sql += " ORDER BY data_source"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


# ------------------------------------------------------------------
# Document size
# ------------------------------------------------------------------


def get_document_total_tokens(conn, document_version_id: int) -> int:
    """Sum token_count for all content in a document.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version

    Returns:
        int -- total token count

    Example:
        >>> get_document_total_tokens(conn, 38)
        12500
    """
    schema = get_database_schema()
    sql = (
        f"SELECT COALESCE(SUM(token_count), 0)"
        f" FROM {schema}.document_content"
        f" WHERE document_version_id = %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (document_version_id,))
        return int(cur.fetchone()[0])


# ------------------------------------------------------------------
# Full document load
# ------------------------------------------------------------------


def load_full_document(conn, document_version_id: int) -> list[dict]:
    """Load all content units for a document version.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version

    Returns:
        list[dict] -- content rows ordered by position

    Example:
        >>> rows = load_full_document(conn, 38)
        >>> rows[0]["content_unit_id"]
        "cu_001"
    """
    schema = get_database_schema()
    sql = (
        f"SELECT content_unit_id, chunk_id, section_id,"
        f" page_number, parent_page_number, raw_content,"
        f" chunk_context, chunk_header,"
        f" sheet_passthrough_content,"
        f" section_passthrough_content,"
        f" keywords, entities, token_count"
        f" FROM {schema}.document_content"
        f" WHERE document_version_id = %s"
        f" ORDER BY {_NUMERIC_CUID_ORDER}"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (document_version_id,))
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


# ------------------------------------------------------------------
# Vector search
# ------------------------------------------------------------------


def search_by_content_vector(
    conn,
    document_version_id: int,
    query_embedding: list,
    top_k: int = 20,
) -> list[dict]:
    """Cosine-similarity search on content embeddings.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version
        query_embedding: Query vector
        top_k: Maximum results to return

    Returns:
        list[dict] -- content rows with distance score

    Example:
        >>> results = search_by_content_vector(
        ...     conn, 38, embedding, top_k=10
        ... )
    """
    schema = get_database_schema()
    vec_str = format_vector(query_embedding)
    sql = (
        f"SELECT content_unit_id, chunk_id, section_id,"
        f" page_number, raw_content, chunk_context,"
        f" chunk_header, keywords, entities, token_count,"
        f" content_embedding <=> %s::vector AS distance"
        f" FROM {schema}.document_content"
        f" WHERE document_version_id = %s"
        f" AND content_embedding IS NOT NULL"
        f" ORDER BY distance"
        f" LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (vec_str, document_version_id, top_k))
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def search_by_keyword_vector(
    conn,
    document_version_id: int,
    query_embedding: list,
    top_k: int = 20,
) -> list[dict]:
    """Cosine-similarity search on keyword embeddings.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version
        query_embedding: Query vector
        top_k: Maximum results to return

    Returns:
        list[dict] -- content rows with distance score

    Example:
        >>> results = search_by_keyword_vector(
        ...     conn, 38, embedding, top_k=10
        ... )
    """
    schema = get_database_schema()
    vec_str = format_vector(query_embedding)
    sql = (
        f"SELECT content_unit_id, chunk_id, section_id,"
        f" page_number, raw_content, chunk_context,"
        f" chunk_header, keywords, entities, token_count,"
        f" keyword_embedding <=> %s::vector AS distance"
        f" FROM {schema}.document_content"
        f" WHERE document_version_id = %s"
        f" AND keyword_embedding IS NOT NULL"
        f" ORDER BY distance"
        f" LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (vec_str, document_version_id, top_k))
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def search_by_section_summary(
    conn,
    document_version_id: int,
    query_embedding: list,
    top_k: int = 20,
) -> list[dict]:
    """Cosine-similarity search on section summaries.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version
        query_embedding: Query vector
        top_k: Maximum results to return

    Returns:
        list[dict] -- section rows with distance score

    Example:
        >>> results = search_by_section_summary(
        ...     conn, 38, embedding, top_k=5
        ... )
    """
    schema = get_database_schema()
    vec_str = format_vector(query_embedding)
    sql = (
        f"SELECT section_id, title, summary,"
        f" summary_embedding <=> %s::vector AS distance"
        f" FROM {schema}.document_sections"
        f" WHERE document_version_id = %s"
        f" AND summary_embedding IS NOT NULL"
        f" ORDER BY distance"
        f" LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (vec_str, document_version_id, top_k))
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


# ------------------------------------------------------------------
# BM25 full-text search
# ------------------------------------------------------------------


def search_by_bm25(
    conn,
    document_version_id: int,
    query_text: str,
    top_k: int = 20,
) -> list[dict]:
    """Full-text search using tsvector ranking.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version
        query_text: Natural language query text
        top_k: Maximum results to return

    Returns:
        list[dict] -- content rows with ts_rank score

    Example:
        >>> results = search_by_bm25(conn, 38, "CET1 ratio")
    """
    schema = get_database_schema()
    sql = (
        f"SELECT content_unit_id, chunk_id, section_id,"
        f" page_number, raw_content, chunk_context,"
        f" chunk_header, keywords, entities, token_count,"
        f" ts_rank(content_tsvector,"
        f" websearch_to_tsquery('english', %s)) AS rank"
        f" FROM {schema}.document_content"
        f" WHERE document_version_id = %s"
        f" AND content_tsvector"
        f" @@ websearch_to_tsquery('english', %s)"
        f" ORDER BY rank DESC"
        f" LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                query_text,
                document_version_id,
                query_text,
                top_k,
            ),
        )
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


# ------------------------------------------------------------------
# Array containment search
# ------------------------------------------------------------------


def search_by_keyword_containment(
    conn,
    document_version_id: int,
    keywords: list[str],
    limit: int = 50,
) -> list[dict]:
    """Find chunks whose keyword array overlaps the query terms.

    Uses case-insensitive substring matching between stored
    keywords and query terms, with results ordered by the
    number of matching terms.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version
        keywords: Keywords to match against
        limit: Maximum results to return

    Returns:
        list[dict] -- matching content rows

    Example:
        >>> search_by_keyword_containment(
        ...     conn, 38, ["CET1", "Tier 1"]
        ... )
    """
    if not keywords:
        return []
    schema = get_database_schema()
    sql = (
        f"SELECT content_unit_id, chunk_id, section_id,"
        f" page_number, raw_content, chunk_context,"
        f" chunk_header, keywords, entities, token_count"
        f" FROM {schema}.document_content"
        f" WHERE document_version_id = %s"
        f" AND EXISTS ("
        f"   SELECT 1"
        f"   FROM jsonb_array_elements_text(keywords) k,"
        f"        unnest(%s::text[]) q"
        f"   WHERE k ILIKE '%%' || q || '%%'"
        f" )"
        f" ORDER BY ("
        f"   SELECT count(DISTINCT q)"
        f"   FROM jsonb_array_elements_text(keywords) k,"
        f"        unnest(%s::text[]) q"
        f"   WHERE k ILIKE '%%' || q || '%%'"
        f" ) DESC"
        f" LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (document_version_id, keywords, keywords, limit))
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def search_by_entity_containment(
    conn,
    document_version_id: int,
    entities: list[str],
    limit: int = 50,
) -> list[dict]:
    """Find chunks whose entity array overlaps the query terms.

    Uses case-insensitive substring matching between stored
    entities and query terms, with results ordered by the
    number of matching terms.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version
        entities: Entity names to match against
        limit: Maximum results to return

    Returns:
        list[dict] -- matching content rows

    Example:
        >>> search_by_entity_containment(
        ...     conn, 38, ["Royal Bank of Canada"]
        ... )
    """
    if not entities:
        return []
    schema = get_database_schema()
    sql = (
        f"SELECT content_unit_id, chunk_id, section_id,"
        f" page_number, raw_content, chunk_context,"
        f" chunk_header, keywords, entities, token_count"
        f" FROM {schema}.document_content"
        f" WHERE document_version_id = %s"
        f" AND EXISTS ("
        f"   SELECT 1"
        f"   FROM jsonb_array_elements_text(entities) e,"
        f"        unnest(%s::text[]) q"
        f"   WHERE e ILIKE '%%' || q || '%%'"
        f" )"
        f" ORDER BY ("
        f"   SELECT count(DISTINCT q)"
        f"   FROM jsonb_array_elements_text(entities) e,"
        f"        unnest(%s::text[]) q"
        f"   WHERE e ILIKE '%%' || q || '%%'"
        f" ) DESC"
        f" LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (document_version_id, entities, entities, limit))
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


# ------------------------------------------------------------------
# Expansion queries
# ------------------------------------------------------------------


def load_section_content(
    conn,
    document_version_id: int,
    section_id: str,
) -> list[dict]:
    """Load all content units in a section.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version
        section_id: Section identifier

    Returns:
        list[dict] -- content rows ordered by position

    Example:
        >>> load_section_content(conn, 38, "s_003")
        [{"content_unit_id": "cu_010", ...}]
    """
    schema = get_database_schema()
    sql = (
        f"SELECT content_unit_id, chunk_id, section_id,"
        f" page_number, parent_page_number, raw_content,"
        f" chunk_context, chunk_header,"
        f" sheet_passthrough_content,"
        f" section_passthrough_content,"
        f" keywords, entities, token_count"
        f" FROM {schema}.document_content"
        f" WHERE document_version_id = %s"
        f" AND section_id = %s"
        f" ORDER BY {_NUMERIC_CUID_ORDER}"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (document_version_id, section_id))
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_section_info(
    conn,
    document_version_id: int,
    section_id: str,
) -> dict | None:
    """Get section metadata from document_sections.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version
        section_id: Section identifier

    Returns:
        dict or None -- section row if found

    Example:
        >>> get_section_info(conn, 38, "s_003")
        {"section_id": "s_003", "title": "Capital", ...}
    """
    schema = get_database_schema()
    sql = (
        f"SELECT section_id, level, title,"
        f" parent_section_id, token_count,"
        f" page_start, page_end, summary"
        f" FROM {schema}.document_sections"
        f" WHERE document_version_id = %s"
        f" AND section_id = %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (document_version_id, section_id))
        row = cur.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))


def get_section_for_content(
    conn,
    document_version_id: int,
    content_unit_id: str,
) -> dict | None:
    """Look up section info for a content unit.

    Finds the section_id from document_content, then
    retrieves full section metadata.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version
        content_unit_id: Content unit to look up

    Returns:
        dict or None -- section row if found

    Example:
        >>> get_section_for_content(conn, 38, "cu_010")
        {"section_id": "s_003", "title": "Capital", ...}
    """
    schema = get_database_schema()
    sql = (
        f"SELECT section_id"
        f" FROM {schema}.document_content"
        f" WHERE document_version_id = %s"
        f" AND content_unit_id = %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (document_version_id, content_unit_id))
        row = cur.fetchone()
        if row is None:
            return None
    section_id = row[0]
    if not section_id:
        return None
    return get_section_info(conn, document_version_id, section_id)


def load_neighbor_chunks(
    conn,
    document_version_id: int,
    content_unit_id: str,
    count: int = 2,
) -> list[dict]:
    """Load chunks surrounding a content unit.

    Uses ROW_NUMBER to find the position of the target
    content unit, then selects rows within the
    [position - count, position + count] window.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version
        content_unit_id: Center content unit
        count: Number of neighbors on each side

    Returns:
        list[dict] -- neighbor rows ordered by position

    Example:
        >>> load_neighbor_chunks(conn, 38, "cu_010", 2)
        [{"content_unit_id": "cu_008", ...}, ...]
    """
    schema = get_database_schema()
    sql = (
        f"WITH numbered AS ("
        f" SELECT content_unit_id, chunk_id, section_id,"
        f" page_number, parent_page_number, raw_content,"
        f" chunk_context, chunk_header,"
        f" sheet_passthrough_content,"
        f" section_passthrough_content,"
        f" keywords, entities, token_count,"
        f" ROW_NUMBER() OVER ("
        f"   ORDER BY {_NUMERIC_CUID_ORDER}"
        f" ) AS rn"
        f" FROM {schema}.document_content"
        f" WHERE document_version_id = %s"
        f")"
        f" SELECT content_unit_id, chunk_id, section_id,"
        f" page_number, parent_page_number, raw_content,"
        f" chunk_context, chunk_header,"
        f" sheet_passthrough_content,"
        f" section_passthrough_content,"
        f" keywords, entities, token_count"
        f" FROM numbered"
        f" WHERE rn BETWEEN"
        f"   (SELECT rn FROM numbered"
        f"    WHERE content_unit_id = %s) - %s"
        f" AND"
        f"   (SELECT rn FROM numbered"
        f"    WHERE content_unit_id = %s) + %s"
        f" ORDER BY rn"
    )
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                document_version_id,
                content_unit_id,
                count,
                content_unit_id,
                count,
            ),
        )
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


# ------------------------------------------------------------------
# Section hierarchy
# ------------------------------------------------------------------


def get_child_sections(
    conn,
    document_version_id: int,
    parent_section_id: str,
) -> list[dict]:
    """Get child sections under a parent section.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version
        parent_section_id: Parent section identifier

    Returns:
        list[dict] -- child section rows

    Example:
        >>> get_child_sections(conn, 38, "s_003")
        [{"section_id": "s_003_1", ...}]
    """
    schema = get_database_schema()
    sql = (
        f"SELECT section_id, level, title,"
        f" parent_section_id, token_count,"
        f" page_start, page_end, summary"
        f" FROM {schema}.document_sections"
        f" WHERE document_version_id = %s"
        f" AND parent_section_id = %s"
        f" ORDER BY page_start, section_id"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (document_version_id, parent_section_id))
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def find_child_section_for_content(
    conn,
    document_version_id: int,
    parent_section_id: str,
    content_unit_id: str,
) -> dict | None:
    """Find which child section contains a content unit.

    Single JOIN query replacing the N+1 pattern of loading
    each child section's content separately.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version
        parent_section_id: Parent section identifier
        content_unit_id: Content unit to locate

    Returns:
        dict or None -- child section row if found
    """
    schema = get_database_schema()
    sql = (
        f"SELECT s.section_id, s.level, s.title,"
        f" s.parent_section_id, s.token_count,"
        f" s.page_start, s.page_end, s.summary"
        f" FROM {schema}.document_sections s"
        f" JOIN {schema}.document_content c"
        f" ON c.document_version_id = s.document_version_id"
        f" AND c.section_id = s.section_id"
        f" WHERE s.document_version_id = %s"
        f" AND s.parent_section_id = %s"
        f" AND c.content_unit_id = %s"
        f" LIMIT 1"
    )
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (document_version_id, parent_section_id, content_unit_id),
        )
        row = cur.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))


# ------------------------------------------------------------------
# Document metadata
# ------------------------------------------------------------------


def get_document_metadata(conn, document_version_id: int) -> dict | None:
    """Get document-level metadata.

    Params:
        conn: psycopg2 connection
        document_version_id: Target document version

    Returns:
        dict or None -- metadata row if found

    Example:
        >>> get_document_metadata(conn, 38)
        {"title": "RBC Q1 2026 Investor Slides", ...}
    """
    schema = get_database_schema()
    sql = (
        f"SELECT title, authors, executive_summary,"
        f" keywords, entities, structure_type"
        f" FROM {schema}.document_metadata"
        f" WHERE document_version_id = %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (document_version_id,))
        row = cur.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))
