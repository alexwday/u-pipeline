"""Data structures for the retriever pipeline."""

from typing import NotRequired, TypedDict


class ResearchFinding(TypedDict):
    """One discrete fact extracted by the research stage.

    Required fields locate the fact in the source document.
    Optional metric fields are populated only for quantitative
    findings. Metric names are verbatim from the source —
    normalization happens at consolidation.
    """

    finding: str
    page: int
    location_detail: str
    metric_name: NotRequired[str]
    metric_value: NotRequired[str]
    unit: NotRequired[str]
    period: NotRequired[str]
    segment: NotRequired[str]


class ComboSpec(TypedDict):
    """Bank and period combination for a query scope."""

    bank: str
    period: str


class SourceSpec(TypedDict):
    """Resolved document source within a combo."""

    data_source: str
    document_version_id: int
    filename: str


class QueryEmbeddings(TypedDict):
    """Embedding vectors for each query facet."""

    rewritten: list[float]
    sub_queries: list[list[float]]
    keywords: list[float]
    hyde: list[float]


class PreparedQuery(TypedDict):
    """Fully decomposed and embedded query."""

    original_query: str
    rewritten_query: str
    sub_queries: list[str]
    keywords: list[str]
    entities: list[str]
    hyde_answer: str
    embeddings: QueryEmbeddings


class SearchResult(TypedDict):
    """Single content unit returned by multi-strategy search."""

    content_unit_id: str
    raw_content: str
    chunk_id: str
    section_id: str
    page_number: int
    chunk_context: str
    chunk_header: str
    keywords: list[str]
    entities: list[str]
    token_count: int
    score: float
    strategy_scores: dict[str, float]


class ExpandedChunk(TypedDict):
    """Content unit with section context after expansion."""

    content_unit_id: str
    raw_content: str
    page_number: int
    section_id: str
    section_title: str
    chunk_context: str
    chunk_header: str
    sheet_passthrough_content: str
    section_passthrough_content: str
    is_original: bool
    token_count: int
    score: NotRequired[float]


class ResearchIteration(TypedDict):
    """One pass of the iterative research loop."""

    iteration: int
    additional_queries: list[str]
    confidence: float
    findings: list[ResearchFinding]


class ComboSourceResult(TypedDict):
    """Research output for one combo + source pair."""

    combo: ComboSpec
    source: SourceSpec
    research_iterations: list[ResearchIteration]
    chunk_count: int
    total_tokens: int
    findings: list[ResearchFinding]
    metrics: NotRequired[dict]
    trace_path: NotRequired[str]


class ConsolidatedResult(TypedDict):
    """Final consolidated answer across all combos."""

    query: str
    combo_results: list[ComboSourceResult]
    consolidated_response: str
    key_findings: list[str]
    data_gaps: list[str]
    summary_answer: NotRequired[str]
    metrics_table: NotRequired[str]
    detailed_summary: NotRequired[str]
    reference_index: NotRequired[list[dict]]
    coverage_audit: NotRequired[str]
    uncited_ref_ids: NotRequired[list[int]]
    unincorporated_findings: NotRequired[list[dict]]
    metrics: NotRequired[dict]
    citation_warnings: NotRequired[list[str]]
    trace_id: NotRequired[str]
    trace_path: NotRequired[str]
