"""Stage 3: Count tokens per page and classify by size tier."""

import tiktoken

from ..utils.config_setup import get_tokenizer_model
from ..utils.file_types import ExtractionResult
from ..utils.logging_setup import get_stage_logger

STAGE = "3-TOKENIZATION"

_TIER_LOW_MAX = 5_000
_TIER_MEDIUM_MAX = 10_000


def _classify_tier(token_count: int) -> str:
    """Classify a page token count into a size tier.

    Params: token_count (int). Returns: str — "low", "medium", or "high".
    """
    if token_count <= _TIER_LOW_MAX:
        return "low"
    if token_count <= _TIER_MEDIUM_MAX:
        return "medium"
    return "high"


def _get_encoder(model: str) -> tiktoken.Encoding:
    """Get a tiktoken encoder for the configured model.

    Falls back to o200k_base if the model name is not recognized.

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


def tokenize_result(result: ExtractionResult) -> ExtractionResult:
    """Count tokens for each page and the full document.

    Uses tiktoken with the encoding matching the configured
    tokenizer model. Sets raw and embedding token counts on
    each page, along with token_tier. At this stage raw and
    embedding counts are identical because chunk assembly has
    not happened yet.

    Params:
        result: ExtractionResult from the extraction stage

    Returns:
        ExtractionResult — same object, mutated with token counts

    Example:
        >>> tokenize_result(result)
        >>> result.pages[0].token_count
        1234
    """
    logger = get_stage_logger(__name__, STAGE)

    encoder = _get_encoder(get_tokenizer_model())

    document_tokens = 0
    tier_counts = {"low": 0, "medium": 0, "high": 0}

    for page in result.pages:
        page.raw_token_count = len(encoder.encode(page.raw_content))
        page.embedding_token_count = page.raw_token_count
        page.token_count = page.embedding_token_count
        page.token_tier = _classify_tier(page.embedding_token_count)
        document_tokens += page.embedding_token_count
        tier_counts[page.token_tier] += 1

    result.raw_document_token_count = document_tokens
    result.embedding_document_token_count = document_tokens
    result.document_token_count = document_tokens

    logger.info(
        "%s — %d pages, %d tokens (low: %d, medium: %d, high: %d)",
        result.file_path,
        result.total_pages,
        document_tokens,
        tier_counts["low"],
        tier_counts["medium"],
        tier_counts["high"],
    )

    return result
