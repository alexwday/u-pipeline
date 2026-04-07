"""Stage 5: Chunk oversized pages to fit embedding token limits."""

import logging

import tiktoken

from .chunkers.markdown_chunker import chunk_markdown_page
from .chunkers.xlsx_chunker import chunk_xlsx_page
from ..utils.config_setup import (
    get_chunking_embedding_token_limit,
    get_chunking_max_retries,
    get_chunking_truncation_token_limit,
    get_tokenizer_model,
)
from ..utils.file_types import ExtractionResult, PageResult
from ..utils.llm_connector import LLMClient
from ..utils.logging_setup import get_stage_logger

STAGE = "5-CHUNKING"

_TIER_LOW_MAX = 5_000
_TIER_MEDIUM_MAX = 10_000


def _get_encoder(model: str) -> tiktoken.Encoding:
    """Get a tiktoken encoder for the given model.

    Falls back to o200k_base if the model name is not
    recognized by tiktoken.

    Params:
        model: Model name from stage config

    Returns:
        tiktoken.Encoding — cached encoder instance
    """
    try:
        encoding_name = tiktoken.encoding_name_for_model(model)
    except KeyError:
        encoding_name = "o200k_base"
    return tiktoken.get_encoding(encoding_name)


def _count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    """Count tokens in text. Params: text, encoder. Returns: int."""
    return len(encoder.encode(text))


def _truncate_content(
    content: str,
    limit: int,
    encoder: tiktoken.Encoding,
) -> str:
    """Truncate content to fit within a token limit.

    Encodes the content, slices to the limit, and decodes
    back to a string.

    Params:
        content: Text to truncate
        limit: Maximum number of tokens to keep
        encoder: Tiktoken encoder instance

    Returns:
        str — truncated text within the token limit
    """
    tokens = encoder.encode(content)
    if len(tokens) <= limit:
        return content
    return encoder.decode(tokens[:limit])


def _classify_tier(token_count: int) -> str:
    """Classify token count into a size tier.

    Params: token_count (int). Returns: str.
    """
    if token_count <= _TIER_LOW_MAX:
        return "low"
    if token_count <= _TIER_MEDIUM_MAX:
        return "medium"
    return "high"


def _select_chunker(filetype: str) -> str:
    """Select the chunker type for a given file type.

    Params: filetype (str). Returns: str — "xlsx" or "markdown".
    """
    if filetype == "xlsx":
        return "xlsx"
    return "markdown"


def _synchronize_page_token_fields(page: PageResult) -> None:
    """Backfill explicit token fields from legacy token_count values.

    Params:
        page: PageResult to normalize in place

    Returns:
        None
    """
    if page.raw_token_count == 0 and page.token_count > 0:
        page.raw_token_count = page.token_count
    if page.embedding_token_count == 0 and page.token_count > 0:
        page.embedding_token_count = page.token_count


def _chunk_page(
    page: PageResult,
    llm: LLMClient,
    embedding_limit: int,
    filetype: str,
) -> list[PageResult]:
    """Route a page to the appropriate chunker.

    Params:
        page: Oversized PageResult to chunk
        llm: Configured LLM client
        embedding_limit: Target token limit per chunk
        filetype: Source file type for chunker selection

    Returns:
        list[PageResult] — chunked page results
    """
    chunker = _select_chunker(filetype)
    if chunker == "xlsx":
        return chunk_xlsx_page(page, llm, embedding_limit)
    return chunk_markdown_page(page, llm, embedding_limit)


def _recount_chunk_tokens(
    chunks: list[PageResult],
    encoder: tiktoken.Encoding,
) -> None:
    """Recount tokens and classify tiers for each chunk.

    For chunks with a chunk_header, assembles the full
    embedding content (header + passthrough + raw) for an
    accurate token count. Unchunked pages count raw_content
    directly.

    Params:
        chunks: List of PageResult chunks to update in place
        encoder: Tiktoken encoder instance

    Returns:
        None
    """
    for chunk in chunks:
        chunk.raw_token_count = _count_tokens(chunk.raw_content, encoder)
        if chunk.chunk_header:
            parts = [chunk.chunk_header]
            if chunk.sheet_passthrough_content:
                parts.append(chunk.sheet_passthrough_content)
            if chunk.section_passthrough_content:
                parts.append(chunk.section_passthrough_content)
            parts.append(chunk.raw_content)
            assembled = "\n".join(parts)
            chunk.embedding_token_count = _count_tokens(assembled, encoder)
        else:
            chunk.embedding_token_count = chunk.raw_token_count
        chunk.token_count = chunk.embedding_token_count
        chunk.token_tier = _classify_tier(chunk.embedding_token_count)


def _find_oversized_chunks(chunks: list[PageResult], limit: int) -> list[int]:
    """Find indices of chunks exceeding the token limit.

    Params:
        chunks: List of PageResult chunks
        limit: Maximum allowed token count

    Returns:
        list[int] — indices of oversized chunks
    """
    return [i for i, chunk in enumerate(chunks) if chunk.token_count > limit]


def _prepare_rechunk(chunk: PageResult) -> PageResult:
    """Rebuild raw_content with structural prefix for re-chunking.

    When an XLSX chunk needs re-splitting, its raw_content
    contains only data rows. The chunker needs the heading,
    table header, and any passthrough rows to parse and
    re-split the chunk with the same context. This
    reconstructs raw_content by prepending chunk_header and
    any passthrough content.

    Params:
        chunk: Oversized chunk PageResult

    Returns:
        PageResult — copy with raw_content restored for parsing
    """
    if not chunk.chunk_header:
        return chunk
    restored_parts = [chunk.chunk_header.rstrip("\n")]
    if chunk.sheet_passthrough_content:
        restored_parts.append(chunk.sheet_passthrough_content)
    if chunk.section_passthrough_content:
        restored_parts.append(chunk.section_passthrough_content)
    restored_parts.append(chunk.raw_content)
    restored = "\n".join(part for part in restored_parts if part)
    return PageResult(
        page_number=chunk.page_number,
        raw_content=restored,
        raw_token_count=chunk.raw_token_count,
        embedding_token_count=chunk.embedding_token_count,
        token_count=chunk.token_count,
        token_tier=chunk.token_tier,
        chunk_id=chunk.chunk_id,
        parent_page_number=chunk.parent_page_number,
        layout_type=chunk.layout_type,
        chunk_context=chunk.chunk_context,
        chunk_header=chunk.chunk_header,
        sheet_passthrough_content=chunk.sheet_passthrough_content,
        section_passthrough_content=chunk.section_passthrough_content,
    )


prepare_rechunk = _prepare_rechunk


def _process_oversized_page(
    page: PageResult,
    llm: LLMClient,
    embedding_limit: int,
    max_retries: int,
    filetype: str,
    encoder: tiktoken.Encoding,
    logger: logging.LoggerAdapter,
) -> list[PageResult]:
    """Chunk an oversized page with retry logic.

    Calls the appropriate chunker, recounts tokens, and
    retries oversized chunks up to max_retries times.
    Raises RuntimeError if chunks remain oversized after
    all retries are exhausted.

    Params:
        page: Oversized PageResult to chunk
        llm: Configured LLM client
        embedding_limit: Target token limit per chunk
        max_retries: Maximum re-chunk attempts
        filetype: Source file type
        encoder: Tiktoken encoder instance
        logger: Stage logger adapter

    Returns:
        list[PageResult] — all chunks within token limit

    Raises:
        RuntimeError: When chunks remain oversized after retries
    """
    chunks = _chunk_page(page, llm, embedding_limit, filetype)
    _recount_chunk_tokens(chunks, encoder)

    for attempt in range(max_retries):
        oversized = _find_oversized_chunks(chunks, embedding_limit)
        if not oversized:
            break

        logger.warning(
            "Page %d: %d oversized chunks, retry %d/%d",
            page.page_number,
            len(oversized),
            attempt + 1,
            max_retries,
        )

        new_chunks: list[PageResult] = []
        for i, chunk in enumerate(chunks):
            if i in oversized:
                rechunk_page = _prepare_rechunk(chunk)
                sub = _chunk_page(rechunk_page, llm, embedding_limit, filetype)
                _recount_chunk_tokens(sub, encoder)
                new_chunks.extend(sub)
            else:
                new_chunks.append(chunk)
        chunks = new_chunks

    oversized = _find_oversized_chunks(chunks, embedding_limit)
    if oversized:
        raise RuntimeError(
            f"Page {page.page_number}: "
            f"{len(oversized)} chunks still exceed "
            f"{embedding_limit} tokens after "
            f"{max_retries} retries"
        )

    return chunks


def chunk_result(result: ExtractionResult, llm: LLMClient) -> ExtractionResult:
    """Chunk oversized pages to fit embedding token limits.

    Pages under the embedding limit pass through unchanged.
    Pages over the truncation limit are truncated first.
    Remaining oversized pages are split by the appropriate
    chunker (markdown or xlsx) with retry logic.

    Params:
        result: ExtractionResult from the tokenization stage
        llm: Configured LLM client for chunking calls

    Returns:
        ExtractionResult — with chunked pages and updated
        token counts

    Example:
        >>> chunked = chunk_result(result, llm)
        >>> all(p.token_count <= 8192 for p in chunked.pages)
        True
    """
    logger = get_stage_logger(__name__, STAGE)

    encoder = _get_encoder(get_tokenizer_model())

    embedding_limit = get_chunking_embedding_token_limit()
    truncation_limit = get_chunking_truncation_token_limit()
    max_retries = get_chunking_max_retries()

    output_pages: list[PageResult] = []
    passed_through = 0
    chunked = 0
    truncated = 0

    for page in result.pages:
        _synchronize_page_token_fields(page)
        if page.token_count <= embedding_limit:
            output_pages.append(page)
            passed_through += 1
            continue

        if page.token_count > truncation_limit:
            page.raw_content = _truncate_content(
                page.raw_content, truncation_limit, encoder
            )
            page.raw_token_count = _count_tokens(page.raw_content, encoder)
            page.embedding_token_count = page.raw_token_count
            page.token_count = page.embedding_token_count
            page.token_tier = _classify_tier(page.embedding_token_count)
            truncated += 1
            logger.warning(
                "Page %d truncated to %d tokens",
                page.page_number,
                page.token_count,
            )

        if page.token_count > embedding_limit:
            chunks = _process_oversized_page(
                page,
                llm,
                embedding_limit,
                max_retries,
                result.filetype,
                encoder,
                logger,
            )
            output_pages.extend(chunks)
            chunked += 1
        else:
            output_pages.append(page)
            passed_through += 1

    raw_document_token_count = sum(p.raw_token_count for p in output_pages)
    embedding_document_token_count = sum(
        p.embedding_token_count for p in output_pages
    )

    logger.info(
        "%s — %d pages: %d passed, %d chunked, %d truncated",
        result.file_path,
        len(result.pages),
        passed_through,
        chunked,
        truncated,
    )

    return ExtractionResult(
        file_path=result.file_path,
        filetype=result.filetype,
        pages=output_pages,
        total_pages=result.total_pages,
        raw_document_token_count=raw_document_token_count,
        embedding_document_token_count=embedding_document_token_count,
        document_token_count=embedding_document_token_count,
    )


# ---- Expose internals for testing ----
get_encoder = _get_encoder
count_tokens = _count_tokens
truncate_content = _truncate_content
classify_tier = _classify_tier
select_chunker = _select_chunker
chunk_page = _chunk_page
recount_chunk_tokens = _recount_chunk_tokens
find_oversized_chunks = _find_oversized_chunks
process_oversized_page = _process_oversized_page
