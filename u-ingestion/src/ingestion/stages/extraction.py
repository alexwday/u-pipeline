"""Stage 2: Route a single file to its filetype processor."""

from ..processors.docx import process_docx
from ..processors.pdf import process_pdf
from ..processors.pptx import process_pptx
from ..processors.xlsx import process_xlsx
from ..utils.file_types import ExtractionResult, FileRecord
from ..utils.llm_connector import LLMClient
from ..utils.logging_setup import get_stage_logger

STAGE = "2-EXTRACTION"


def extract_file(record: FileRecord, llm: LLMClient) -> ExtractionResult:
    """Extract content from a single file.

    Routes to the appropriate processor based on filetype.
    The processor handles rendering, LLM calls, retries,
    and error recovery internally.

    Params:
        record: FileRecord from discovery
        llm: Initialized LLM client

    Returns:
        ExtractionResult with per-page content

    Example:
        >>> result = extract_file(record, llm)
        >>> result.total_pages
        12
    """
    logger = get_stage_logger(__name__, STAGE)
    logger.info("Extracting: %s", record.filename)

    if record.filetype == "pdf":
        result = process_pdf(record.file_path, llm)
    elif record.filetype == "docx":
        result = process_docx(record.file_path, llm)
    elif record.filetype == "pptx":
        result = process_pptx(record.file_path, llm)
    elif record.filetype == "xlsx":
        result = process_xlsx(record.file_path, llm)
    else:
        raise ValueError(f"Unsupported filetype: {record.filetype}")

    result.data_source = record.data_source
    result.filter_1 = record.filter_1
    result.filter_2 = record.filter_2
    result.filter_3 = record.filter_3
    return result
