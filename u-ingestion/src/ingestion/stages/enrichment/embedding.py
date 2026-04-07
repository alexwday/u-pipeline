"""Stage 11: Embedding generation.

Generates vector embeddings for all searchable content
using the configured embedding model. Embeds content,
keywords, and entities for content units and sections,
plus the document executive summary.
"""

import logging

from ...utils.config_setup import (
    get_embedding_batch_size,
    get_embedding_dimensions,
    get_embedding_model,
)
from ...utils.file_types import (
    ExtractionResult,
    get_content_unit_id,
)
from ...utils.llm_connector import LLMClient
from ...utils.logging_setup import get_stage_logger

STAGE = "11-EMBEDDING"

logger = logging.getLogger(__name__)


def _find_section_title(
    page,
    sections: list[dict],
) -> str:
    """Find the section title for a page.

    Params: page, sections (list[dict]). Returns: str.
    """
    section_id = getattr(page, "section_id", "")
    if not section_id:
        return ""
    for section in sections:
        if section.get("section_id") == section_id:
            return section.get("title", "")
    return ""


def _build_content_text(page, section_title: str) -> str:
    """Assemble embedding text for a content unit.

    For chunked pages, includes chunk_header,
    sheet_passthrough_content, section_passthrough_content,
    and raw_content. For unchunked pages, uses raw_content.
    Prefixes with section title when available.

    Params:
        page: PageResult to build text from
        section_title: Section title for prefix

    Returns:
        str -- assembled text for embedding
    """
    chunk_header = getattr(page, "chunk_header", "")
    if chunk_header:
        sheet = getattr(page, "sheet_passthrough_content", "")
        section_pass = getattr(page, "section_passthrough_content", "")
        text = chunk_header + sheet + section_pass
        text = text + "\n" + page.raw_content
    else:
        text = page.raw_content
    if section_title:
        text = f"Section: {section_title}. {text}"
    return text


def _build_content_unit_texts(
    result: ExtractionResult,
) -> list[tuple]:
    """Build (content, keywords, entities) text per unit.

    Params:
        result: ExtractionResult with pages and sections

    Returns:
        list[tuple] -- (content_text, keyword_text,
            entity_text) for each page
    """
    texts: list[tuple] = []
    for page in result.pages:
        section_title = _find_section_title(page, result.sections)
        content_text = _build_content_text(page, section_title)

        keywords = getattr(page, "keywords", [])
        keyword_text = " ".join(keywords) if keywords else ""

        entities = getattr(page, "entities", [])
        entity_text = " ".join(entities) if entities else ""

        texts.append((content_text, keyword_text, entity_text))
    return texts


def _is_embeddable_section(section: dict) -> bool:
    """Check if a section should receive embeddings.

    Params: section (dict). Returns: bool.
    """
    level = section.get("level", "")
    if level == "section":
        return True
    return level == "subsection" and bool(section.get("summary"))


def _build_section_texts(
    sections: list[dict],
) -> list[tuple]:
    """Build (summary, keywords, entities) per embeddable section.

    Includes primary sections and subsections that have a
    non-empty summary.

    Params:
        sections: Section dicts from ExtractionResult

    Returns:
        list[tuple] -- (summary_text, keyword_text,
            entity_text) per embeddable section
    """
    texts: list[tuple] = []
    for section in sections:
        if not _is_embeddable_section(section):
            continue
        title = section.get("title", "")
        summary = section.get("summary", "")
        summary_text = f"Section: {title}. {summary}"

        keywords = section.get("keywords", [])
        keyword_text = " ".join(keywords) if keywords else ""

        entities = section.get("entities", [])
        entity_text = " ".join(entities) if entities else ""

        texts.append((summary_text, keyword_text, entity_text))
    return texts


def _batch_embed(
    llm: LLMClient,
    texts: list[str],
    model: str,
    dimensions: int,
    batch_size: int,
) -> list[list[float]]:
    """Embed texts in batches, skipping empty strings.

    Returns vectors in the same order as input texts.
    Empty strings receive an empty list placeholder.

    Params:
        llm: Initialized LLM client
        texts: Strings to embed
        model: Embedding model name
        dimensions: Output vector dimensions
        batch_size: Max texts per API call

    Returns:
        list[list[float]] -- one vector per input text
    """
    results: list[list[float]] = [[] for _ in texts]

    non_empty_indices: list[int] = []
    non_empty_texts: list[str] = []
    for idx, text in enumerate(texts):
        if text:
            non_empty_indices.append(idx)
            non_empty_texts.append(text)

    if not non_empty_texts:
        return results

    all_vectors: list[list[float]] = []
    for start in range(0, len(non_empty_texts), batch_size):
        batch = non_empty_texts[start : start + batch_size]
        vectors = llm.embed(batch, model=model, dimensions=dimensions)
        all_vectors.extend(vectors)

    for vec_idx, orig_idx in enumerate(non_empty_indices):
        results[orig_idx] = all_vectors[vec_idx]

    return results


def _ensure_content_units(
    result: ExtractionResult,
) -> None:
    """Build content_units list from pages if empty.

    Params:
        result: ExtractionResult to update

    Returns:
        None
    """
    if result.content_units:
        return
    for page in result.pages:
        result.content_units.append(
            {
                "content_unit_id": get_content_unit_id(page),
                "chunk_id": page.chunk_id,
                "section_id": page.section_id,
                "page_number": page.page_number,
                "parent_page_number": page.parent_page_number,
                "raw_content": page.raw_content,
                "chunk_context": page.chunk_context,
                "chunk_header": page.chunk_header,
                "sheet_passthrough_content": (page.sheet_passthrough_content),
                "section_passthrough_content": (
                    page.section_passthrough_content
                ),
                "keywords": list(page.keywords),
                "entities": list(page.entities),
                "raw_token_count": page.raw_token_count,
                "embedding_token_count": (page.embedding_token_count),
                "token_count": page.token_count,
            }
        )


def _store_content_embeddings(
    result: ExtractionResult,
    content_vecs: list[list[float]],
    keyword_vecs: list[list[float]],
    entity_vecs: list[list[float]],
) -> None:
    """Apply embedding vectors to content_units.

    Params:
        result: ExtractionResult with content_units
        content_vecs: Content embedding per unit
        keyword_vecs: Keyword embedding per unit
        entity_vecs: Entity embedding per unit

    Returns:
        None
    """
    for idx, unit in enumerate(result.content_units):
        unit["content_embedding"] = content_vecs[idx]
        unit["keyword_embedding"] = keyword_vecs[idx]
        unit["entity_embedding"] = entity_vecs[idx]


def _store_section_embeddings(
    sections: list[dict],
    summary_vecs: list[list[float]],
    keyword_vecs: list[list[float]],
    entity_vecs: list[list[float]],
) -> None:
    """Apply embedding vectors to embeddable sections.

    Params:
        sections: All section dicts
        summary_vecs: Summary embedding per embeddable section
        keyword_vecs: Keyword embedding per embeddable section
        entity_vecs: Entity embedding per embeddable section

    Returns:
        None
    """
    vec_idx = 0
    for section in sections:
        if not _is_embeddable_section(section):
            continue
        section["summary_embedding"] = summary_vecs[vec_idx]
        section["keyword_embedding"] = keyword_vecs[vec_idx]
        section["entity_embedding"] = entity_vecs[vec_idx]
        vec_idx += 1


def _embed_content_units(
    result: ExtractionResult,
    content_texts: list[tuple],
    llm: LLMClient,
    model: str,
    dimensions: int,
    batch_size: int,
) -> None:
    """Embed and store vectors for content units.

    Params:
        result: ExtractionResult with content_units
        content_texts: Tuples of (content, keyword, entity)
        llm: Initialized LLM client
        model: Embedding model name
        dimensions: Output vector dimensions
        batch_size: Max texts per API call

    Returns:
        None
    """
    c_texts = [t[0] for t in content_texts]
    k_texts = [t[1] for t in content_texts]
    e_texts = [t[2] for t in content_texts]

    content_vecs = _batch_embed(llm, c_texts, model, dimensions, batch_size)
    keyword_vecs = _batch_embed(llm, k_texts, model, dimensions, batch_size)
    entity_vecs = _batch_embed(llm, e_texts, model, dimensions, batch_size)

    _store_content_embeddings(result, content_vecs, keyword_vecs, entity_vecs)


def _embed_sections(
    sections: list[dict],
    section_texts: list[tuple],
    llm: LLMClient,
    model: str,
    dimensions: int,
    batch_size: int,
) -> None:
    """Embed and store vectors for primary sections.

    Params:
        sections: All section dicts
        section_texts: Tuples of (summary, keyword, entity)
        llm: Initialized LLM client
        model: Embedding model name
        dimensions: Output vector dimensions
        batch_size: Max texts per API call

    Returns:
        None
    """
    s_texts = [t[0] for t in section_texts]
    sk_texts = [t[1] for t in section_texts]
    se_texts = [t[2] for t in section_texts]

    summary_vecs = _batch_embed(llm, s_texts, model, dimensions, batch_size)
    sec_kw_vecs = _batch_embed(llm, sk_texts, model, dimensions, batch_size)
    sec_ent_vecs = _batch_embed(llm, se_texts, model, dimensions, batch_size)

    _store_section_embeddings(
        sections,
        summary_vecs,
        sec_kw_vecs,
        sec_ent_vecs,
    )


def embed_content(
    result: ExtractionResult,
    llm: LLMClient,
) -> ExtractionResult:
    """Generate embeddings for content units, sections, doc.

    Reads embedding config from environment. Builds text
    payloads for all embeddable content, batches them
    through the embedding API, and stores vectors on
    content_units, sections, and document_metadata.

    Params:
        result: ExtractionResult from upstream stage
        llm: Initialized LLM client

    Returns:
        ExtractionResult with embedding vectors stored
    """
    stage_log = get_stage_logger(__name__, STAGE)

    model = get_embedding_model()
    dimensions = get_embedding_dimensions()
    batch_size = get_embedding_batch_size()

    _ensure_content_units(result)

    content_texts = _build_content_unit_texts(result)
    section_texts = _build_section_texts(result.sections)

    metadata = result.document_metadata or {}
    doc_summary = metadata.get("executive_summary", "")

    msg = "Embedding %d units, %d sections, %d doc summary"
    stage_log.info(
        msg,
        len(content_texts),
        len(section_texts),
        1 if doc_summary else 0,
    )

    if content_texts:
        _embed_content_units(
            result,
            content_texts,
            llm,
            model,
            dimensions,
            batch_size,
        )

    if section_texts:
        _embed_sections(
            result.sections,
            section_texts,
            llm,
            model,
            dimensions,
            batch_size,
        )

    if doc_summary:
        doc_vecs = _batch_embed(
            llm,
            [doc_summary],
            model,
            dimensions,
            batch_size,
        )
        result.document_metadata["summary_embedding"] = doc_vecs[0]

    stage_log.info("Embedding complete")

    return result


# ------------------------------------------------------------------
# Public aliases for testing
# ------------------------------------------------------------------
find_section_title = _find_section_title
build_content_text = _build_content_text
build_content_unit_texts = _build_content_unit_texts
build_section_texts = _build_section_texts
batch_embed = _batch_embed
ensure_content_units = _ensure_content_units
store_content_embeddings = _store_content_embeddings
store_section_embeddings = _store_section_embeddings
embed_content_units = _embed_content_units
embed_sections = _embed_sections
