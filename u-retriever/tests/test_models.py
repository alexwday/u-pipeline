"""Tests for retriever data structures."""

from retriever.models import (
    ComboSourceResult,
    ComboSpec,
    ConsolidatedResult,
    ExpandedChunk,
    PreparedQuery,
    QueryEmbeddings,
    ResearchFinding,
    ResearchIteration,
    SearchResult,
    SourceSpec,
)


def test_combo_spec_instantiation():
    """Create a ComboSpec with bank and period."""
    combo: ComboSpec = {"bank": "RBC", "period": "2026_Q1"}
    assert combo["bank"] == "RBC"
    assert combo["period"] == "2026_Q1"


def test_source_spec_instantiation():
    """Create a SourceSpec with document details."""
    source: SourceSpec = {
        "data_source": "investor-slides",
        "document_version_id": 38,
        "filename": "rbc_q1_2026_investor_slides.pdf",
    }
    assert source["document_version_id"] == 38


def test_query_embeddings_instantiation():
    """Create QueryEmbeddings with all vector fields."""
    embeddings: QueryEmbeddings = {
        "rewritten": [0.1, 0.2],
        "sub_queries": [[0.3, 0.4], [0.5, 0.6]],
        "keywords": [0.7, 0.8],
        "hyde": [0.9, 1.0],
    }
    assert len(embeddings["sub_queries"]) == 2


def test_prepared_query_instantiation():
    """Create a full PreparedQuery with embeddings."""
    query: PreparedQuery = {
        "original_query": "What is CET1?",
        "rewritten_query": "CET1 capital ratio",
        "sub_queries": ["CET1 definition"],
        "keywords": ["CET1"],
        "entities": ["Basel III"],
        "hyde_answer": "CET1 is...",
        "embeddings": {
            "rewritten": [0.1],
            "sub_queries": [[0.2]],
            "keywords": [0.3],
            "hyde": [0.4],
        },
    }
    assert query["original_query"] == "What is CET1?"


def test_search_result_instantiation():
    """Create a SearchResult with all score fields."""
    result: SearchResult = {
        "content_unit_id": "cu_001",
        "raw_content": "CET1 ratio is 13.5%",
        "chunk_id": "ch_001",
        "section_id": "s_003",
        "page_number": 5,
        "chunk_context": "Capital section",
        "chunk_header": "CET1 Ratio",
        "keywords": ["CET1"],
        "entities": ["Basel III"],
        "token_count": 25,
        "score": 0.92,
        "strategy_scores": {"content_vector": 0.95},
    }
    assert result["score"] == 0.92


def test_expanded_chunk_instantiation():
    """Create an ExpandedChunk with section context."""
    chunk: ExpandedChunk = {
        "content_unit_id": "cu_001",
        "raw_content": "text here",
        "page_number": 5,
        "section_id": "s_003",
        "section_title": "Capital",
        "chunk_context": "Capital overview",
        "is_original": True,
        "token_count": 20,
    }
    assert chunk["is_original"] is True


def test_expanded_chunk_with_score():
    """Create an ExpandedChunk carrying a fusion score."""
    chunk: ExpandedChunk = {
        "content_unit_id": "cu_002",
        "raw_content": "text",
        "page_number": 3,
        "section_id": "s_001",
        "section_title": "Overview",
        "chunk_context": "",
        "chunk_header": "",
        "sheet_passthrough_content": "",
        "section_passthrough_content": "",
        "is_original": False,
        "token_count": 10,
        "score": 0.87,
    }
    assert chunk["score"] == 0.87


def test_research_finding_required_only():
    """Create a ResearchFinding with required fields only."""
    finding: ResearchFinding = {
        "finding": "CET1 ratio was 13.5%",
        "page": 3,
        "location_detail": "Sheet KM1",
    }
    assert finding["finding"] == "CET1 ratio was 13.5%"
    assert finding["page"] == 3
    assert finding["location_detail"] == "Sheet KM1"


def test_research_finding_with_metric_fields():
    """Create a ResearchFinding with all optional metric fields."""
    finding: ResearchFinding = {
        "finding": "CET1 ratio was 13.5%",
        "page": 3,
        "location_detail": "Sheet KM1",
        "metric_name": "CET1 Ratio",
        "metric_value": "13.5%",
        "period": "Q1 2026",
        "segment": "Enterprise",
    }
    assert finding["metric_name"] == "CET1 Ratio"
    assert finding["segment"] == "Enterprise"


def test_research_finding_with_unit_field():
    """Create a ResearchFinding with the new unit field (F10)."""
    finding: ResearchFinding = {
        "finding": "CET1 capital was 100,415 million CAD",
        "page": 3,
        "location_detail": "Sheet KM1",
        "metric_name": "Common Equity Tier 1 (CET1)",
        "metric_value": "100,415",
        "unit": "$MM",
        "period": "Q1 2026",
        "segment": "Enterprise",
    }
    assert finding["unit"] == "$MM"
    assert finding["metric_value"] == "100,415"
    # Unit is separate from the numeric value
    assert "$" not in finding["metric_value"]


def test_research_iteration_instantiation():
    """Create a ResearchIteration with confidence."""
    iteration: ResearchIteration = {
        "iteration": 1,
        "additional_queries": ["Tier 1 ratio"],
        "confidence": 0.85,
        "findings": [],
    }
    assert iteration["confidence"] == 0.85


def test_research_iteration_with_findings():
    """Create a ResearchIteration with structured findings."""
    iteration: ResearchIteration = {
        "iteration": 1,
        "additional_queries": [],
        "confidence": 0.90,
        "findings": [
            {
                "finding": "CET1 ratio was 13.5%",
                "page": 3,
                "location_detail": "Sheet KM1",
                "metric_name": "CET1 Ratio",
                "metric_value": "13.5%",
                "period": "Q1 2026",
                "segment": "Enterprise",
            },
        ],
    }
    assert len(iteration["findings"]) == 1
    assert iteration["findings"][0]["page"] == 3


def test_combo_source_result_instantiation():
    """Create a ComboSourceResult with nested types."""
    result: ComboSourceResult = {
        "combo": {"bank": "RBC", "period": "2026_Q1"},
        "source": {
            "data_source": "pillar3",
            "document_version_id": 40,
            "filename": "rbc_q1_2026_pillar3.xlsx",
        },
        "research_iterations": [],
        "chunk_count": 5,
        "total_tokens": 1200,
        "findings": [],
    }
    assert result["chunk_count"] == 5


def test_combo_source_result_with_findings():
    """Create a ComboSourceResult with structured findings."""
    result: ComboSourceResult = {
        "combo": {"bank": "RBC", "period": "2026_Q1"},
        "source": {
            "data_source": "pillar3",
            "document_version_id": 40,
            "filename": "rbc_q1_2026_pillar3.xlsx",
        },
        "research_iterations": [],
        "chunk_count": 1,
        "total_tokens": 50,
        "findings": [
            {
                "finding": "Tier 1 ratio was 15.2%",
                "page": 5,
                "location_detail": "Section Capital Ratios",
            },
        ],
    }
    assert len(result["findings"]) == 1


def test_consolidated_result_instantiation():
    """Create a ConsolidatedResult with all fields."""
    result: ConsolidatedResult = {
        "query": "What is CET1?",
        "combo_results": [],
        "consolidated_response": "CET1 is...",
        "key_findings": ["CET1 ratio is 13.5%"],
        "data_gaps": ["Missing Q4 data"],
    }
    assert result["query"] == "What is CET1?"
    assert len(result["key_findings"]) == 1


def test_consolidated_result_with_structured_fields():
    """Create a ConsolidatedResult with new structured fields."""
    result: ConsolidatedResult = {
        "query": "What is CET1?",
        "combo_results": [],
        "consolidated_response": "Full response text",
        "key_findings": ["CET1 ratio is 13.5%"],
        "data_gaps": [],
        "summary_answer": "CET1 ratio was 13.5% [REF:1].",
        "metrics_table": "| Entity | CET1 |\n| RBC | 13.5% |",
        "detailed_summary": "Detailed analysis...",
        "reference_index": [
            {
                "ref_id": 1,
                "finding": "CET1 ratio was 13.5%",
                "page": 3,
                "source": "pillar3",
            },
        ],
    }
    assert result["summary_answer"].startswith("CET1")
    assert len(result["reference_index"]) == 1
