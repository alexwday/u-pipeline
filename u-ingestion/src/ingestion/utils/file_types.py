"""Shared dataclasses and file utilities for the pipeline."""

import hashlib
from dataclasses import dataclass, field
from typing import Any, List

from .config_setup import get_accepted_filetypes


@dataclass
class FileRecord:
    """A discovered file with its metadata and path components.

    Params:
        data_source: Top-level folder name (first subfolder under base)
        filter_1: Second-level subfolder or empty string
        filter_2: Third-level subfolder or empty string
        filter_3: Fourth-level subfolder or empty string
        filename: Basename of the file
        filetype: Lowercase file extension without dot
        file_size: Size in bytes
        date_last_modified: Raw mtime from os.stat
        file_hash: SHA-256 hex digest, empty until computed
        file_path: Full absolute path to the file
        supported: Whether the filetype is accepted

    Example:
        >>> r = FileRecord(
        ...     data_source="policy", filter_1="2026",
        ...     filter_2="", filter_3="", filename="doc.pdf",
        ...     filetype="pdf", file_size=1024,
        ...     date_last_modified=1700000000.0,
        ...     file_hash="", file_path="/data/policy/2026/doc.pdf",
        ... )
        >>> r.supported
        True
    """

    data_source: str
    filter_1: str
    filter_2: str
    filter_3: str
    filename: str
    filetype: str
    file_size: int
    date_last_modified: float
    file_hash: str
    file_path: str
    supported: bool = field(init=False)

    def __post_init__(self) -> None:
        """Compute supported flag from filetype."""
        self.supported = self.filetype in get_accepted_filetypes()


@dataclass
class DiscoveryDiff:
    """Result of comparing filesystem against the catalog.

    Params:
        new: Files on disk but not in the catalog
        modified: Files on disk whose size or hash changed
        deleted: Files in catalog but no longer on disk

    Example:
        >>> diff = DiscoveryDiff(new=[], modified=[], deleted=[])
        >>> len(diff.new)
        0
    """

    new: List[FileRecord]
    modified: List[FileRecord]
    deleted: List[FileRecord]


@dataclass
class DiscoveryScan:
    """Result of scanning the filesystem for candidate files.

    Params:
        supported: Files whose extensions are accepted for ingestion
        unsupported: Files skipped because their extensions are unsupported

    Example:
        >>> scan = DiscoveryScan(supported=[], unsupported=[])
        >>> len(scan.supported)
        0
    """

    supported: List[FileRecord]
    unsupported: List[FileRecord]


@dataclass
class PageResult:
    """Extraction result for a single page or chunk.

    Params:
        page_number: 1-indexed page number
        raw_content: Primary content field. For unchunked pages
            this is the full page content. For chunks it holds
            the data rows only (without passthrough context).
        raw_token_count: Token count of raw_content only.
            Set by tokenization for pages, and recalculated
            by chunking for final chunks.
        embedding_token_count: Token count of the fully
            assembled embedding payload. For unchunked pages
            this matches raw_token_count. For chunks it
            includes header and passthrough content.
        token_count: Token count (set by tokenization stage).
            Compatibility field matching embedding_token_count.
        token_tier: Size classification — "low" (under 5k),
            "medium" (5k-10k), "high" (over 10k)
        chunk_id: Chunk identifier within parent page
            (e.g. "3.1"), empty for unchunked pages
        parent_page_number: Original page_number before
            chunking, 0 when the page was not split
        layout_type: Classification label assigned by the
            classification stage
        chunk_context: Human-readable context label for the
            chunk
        chunk_header: Structural prefix (sheet heading, table
            header, separator). Empty for unchunked pages.
        sheet_passthrough_content: Sheet-wide passthrough rows
            present in every chunk. Empty for unchunked pages.
        section_passthrough_content: Section passthrough rows
            specific to this chunk position. Empty for
            unchunked pages.

    Example:
        >>> page = PageResult(
        ...     page_number=1, raw_content="### Revenue\\n..."
        ... )
    """

    page_number: int
    raw_content: str
    raw_token_count: int = 0
    embedding_token_count: int = 0
    token_count: int = 0
    token_tier: str = ""
    chunk_id: str = ""
    parent_page_number: int = 0
    layout_type: str = ""
    chunk_context: str = ""
    chunk_header: str = ""
    sheet_passthrough_content: str = ""
    section_passthrough_content: str = ""
    section_id: str = ""
    keywords: List = field(default_factory=list)
    entities: List = field(default_factory=list)


@dataclass
class DocumentMetadata:
    """Document-level metadata from enrichment.

    Params:
        title: Document title
        authors: Authors or publishing organization
        publication_date: Date of publication
        language: ISO 639-1 language code
        structure_type: How the document is organized
        data_source: Source folder from discovery
        filter_1: First discovery filter
        filter_2: Second discovery filter
        filter_3: Third discovery filter
        has_toc: Whether a table of contents was detected
        source_toc_entries: TOC entries extracted from the source
        generated_toc_entries: TOC entries synthesized from section summaries
        rationale: Metadata extraction rationale
        executive_summary: From doc_summary stage
        keywords: Document-wide refined keywords
        entities: Document-wide refined entities
    """

    title: str = ""
    authors: str = ""
    publication_date: str = ""
    language: str = "en"
    structure_type: str = ""
    data_source: str = ""
    filter_1: str = ""
    filter_2: str = ""
    filter_3: str = ""
    has_toc: bool = False
    source_toc_entries: List = field(default_factory=list)
    generated_toc_entries: List = field(default_factory=list)
    rationale: str = ""
    executive_summary: str = ""
    keywords: List = field(default_factory=list)
    entities: List = field(default_factory=list)


@dataclass
class SectionResult:
    """One detected section or subsection in the document.

    Params:
        section_id: Identifier (e.g. "1", "2", "2.1")
        parent_section_id: Parent section id, empty for
            primary sections
        level: "section" or "subsection"
        title: Section title
        sequence: Order in document
        page_start: First page number
        page_end: Last page number
        chunk_ids: Content unit identifiers in this section
        summary: Section summary from section_summary stage
        keywords: Refined section keywords
        entities: Refined section entities
        token_count: Sum of content unit tokens
    """

    section_id: str = ""
    parent_section_id: str = ""
    level: str = "section"
    title: str = ""
    sequence: int = 0
    page_start: int = 0
    page_end: int = 0
    chunk_ids: List = field(default_factory=list)
    summary: str = ""
    keywords: List = field(default_factory=list)
    entities: List = field(default_factory=list)
    token_count: int = 0


@dataclass
class ExtractionResult:
    """Extraction result for an entire file.

    All pages must succeed — any page failure fails the file.

    Params:
        file_path: Absolute path to the source file
        filetype: Lowercase extension without dot
        pages: Per-page extraction results
        total_pages: Total number of pages in the file
        data_source: Source folder from discovery
        filter_1: First discovery filter
        filter_2: Second discovery filter
        filter_3: Third discovery filter
        raw_document_token_count: Sum of page raw_token_count
        embedding_document_token_count: Sum of page
            embedding_token_count
        document_token_count: Sum of all page token counts
            for compatibility, matching
            embedding_document_token_count
        document_metadata: Document-level metadata from
            enrichment (title, authors, structure_type, etc.)
        sections: Section/subsection hierarchy from enrichment
        content_units: Per-chunk enrichment data
            (keywords, entities, embeddings)

    Example:
        >>> result = ExtractionResult(
        ...     file_path="/data/doc.pdf", filetype="pdf",
        ...     pages=[], total_pages=0,
        ... )
    """

    file_path: str
    filetype: str
    pages: List[PageResult]
    total_pages: int
    data_source: str = ""
    filter_1: str = ""
    filter_2: str = ""
    filter_3: str = ""
    raw_document_token_count: int = 0
    embedding_document_token_count: int = 0
    document_token_count: int = 0
    document_metadata: dict = field(default_factory=dict)
    sections: List = field(default_factory=list)
    content_units: List = field(default_factory=list)


@dataclass
class DocumentVersion:
    """Tracked version record for a source file.

    Params:
        document_version_id: Database identifier for the file version
        file_path: Absolute path to the source file
        data_source: Top-level folder name
        filter_1: Second-level folder or empty string
        filter_2: Third-level folder or empty string
        filter_3: Fourth-level folder or empty string
        filename: Basename of the file
        filetype: Lowercase extension without dot
        file_size: Size in bytes
        date_last_modified: Raw mtime from os.stat
        file_hash: SHA-256 hex digest of the file contents
        is_current: Whether this is the active version for file_path
    """

    document_version_id: int
    file_path: str
    data_source: str
    filter_1: str
    filter_2: str
    filter_3: str
    filename: str
    filetype: str
    file_size: int
    date_last_modified: float
    file_hash: str
    is_current: bool


@dataclass
class StageCheckpoint:
    """Persisted status for a document version at a pipeline stage.

    Params:
        document_version_id: Owning document version identifier
        stage_name: Pipeline stage name
        status: "succeeded" or "failed"
        stage_signature: Fingerprint of stage code/config
        artifact_path: Absolute path to persisted stage artifact
        artifact_checksum: SHA-256 checksum of artifact contents
        error_message: Last recorded failure for the stage
    """

    document_version_id: int
    stage_name: str
    status: str
    stage_signature: str
    artifact_path: str
    artifact_checksum: str
    error_message: str


@dataclass
class PrunableDocumentVersion:
    """A non-current document version eligible for cleanup.

    Params:
        document_version_id: Database identifier for the file version
        file_path: Absolute source path tracked by the version
        artifact_paths: Persisted artifact files linked to the version
    """

    document_version_id: int
    file_path: str
    artifact_paths: List[str]


def get_content_unit_id(page: Any) -> str:
    """Return the stable content-unit id for a page or chunk.

    Params: page (Any). Returns: str.
    """
    chunk_id = getattr(page, "chunk_id", "")
    if isinstance(chunk_id, str) and chunk_id:
        return chunk_id
    return str(getattr(page, "page_number"))


def compute_file_hash(path: str) -> str:
    """Compute SHA-256 hex digest of a file using 8KB chunks.

    Params:
        path: Absolute path to the file

    Returns:
        str — hex digest of the file contents

    Example:
        >>> compute_file_hash("/tmp/test.txt")
        'e3b0c44298fc1c149afb...'
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()
