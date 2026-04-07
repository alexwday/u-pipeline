"""Stage 0: Pipeline startup and connection verification."""

from typing import Any

from ..utils.config_setup import load_config
from ..utils.llm_connector import LLMClient
from ..utils.logging_setup import get_stage_logger, setup_logging
from ..utils.postgres_connector import get_connection, verify_connection
from ..utils.ssl_setup import setup_ssl

STAGE = "0-STARTUP"


def run_startup() -> tuple[Any, LLMClient]:
    """Initialize the pipeline and return connections.

    Loads configuration, sets up logging and SSL, then
    verifies LLM and database connectivity. Returns live
    connections for downstream stages.

    Returns:
        tuple of (psycopg2 connection, LLMClient)

    Example:
        >>> conn, llm = run_startup()
    """
    load_config()
    setup_logging()
    logger = get_stage_logger(__name__, STAGE)
    logger.info("Retriever pipeline starting")

    setup_ssl()

    llm = None
    conn = None
    try:
        logger.info("Initializing LLM client")
        llm = LLMClient()
        llm.test_connection()

        logger.info("Connecting to database")
        conn = get_connection()
        verify_connection(conn)
    except Exception:
        if conn is not None:
            conn.close()
        raise

    logger.info("Startup complete")
    return conn, llm
