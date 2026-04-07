"""Retriever pipeline entry point with test harness CLI."""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from openai import OpenAIError
from psycopg2 import Error as PgError

from .models import ComboSpec
from .stages.orchestrator import run_retrieval
from .stages.startup import run_startup

_YAML_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "docs"
    / "test_queries.yaml"
)
_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

_SEP = "=" * 70


def _normalize_text(value: str) -> str:
    """Normalize assertion text. Params: value (str). Returns: str."""
    return value.casefold().replace(",", "")


def _evaluate_assertions(case: dict, result: dict) -> list[str]:
    """Validate a benchmark result against optional case assertions.

    Params:
        case: Test case dict from test_queries.yaml
        result: Retrieval result dict

    Returns:
        List of human-readable assertion failure messages
    """
    assertions = case.get("assertions") or {}
    if not assertions:
        return []

    failures: list[str] = []
    response = _normalize_text(result.get("consolidated_response", ""))
    findings = result.get("key_findings", [])
    findings_text = _normalize_text("\n".join(findings))
    data_gap_count = len(result.get("data_gaps", []))
    warning_count = len(result.get("citation_warnings", []))

    for term in assertions.get("response_includes", []):
        if _normalize_text(term) not in response:
            failures.append(f"response missing required term: {term}")

    for options in assertions.get("response_includes_any", []):
        normalized_options = [_normalize_text(option) for option in options]
        if not any(option in response for option in normalized_options):
            failures.append(
                "response missing one of required terms: " + ", ".join(options)
            )

    for term in assertions.get("key_findings_include", []):
        if _normalize_text(term) not in findings_text:
            failures.append(f"key findings missing required term: {term}")

    min_key_findings = assertions.get("min_key_findings")
    if isinstance(min_key_findings, int) and len(findings) < min_key_findings:
        failures.append(
            "key findings count below minimum: "
            f"{len(findings)} < {min_key_findings}"
        )

    max_data_gaps = assertions.get("max_data_gaps")
    if isinstance(max_data_gaps, int) and data_gap_count > max_data_gaps:
        failures.append(
            f"data gaps exceeded maximum: {data_gap_count} > "
            f"{max_data_gaps}"
        )

    max_citation_warnings = assertions.get("max_citation_warnings")
    if (
        isinstance(max_citation_warnings, int)
        and warning_count > max_citation_warnings
    ):
        failures.append(
            "citation warnings exceeded maximum: "
            f"{warning_count} > {max_citation_warnings}"
        )

    max_rerank_fallbacks = assertions.get("max_rerank_fallbacks")
    if isinstance(max_rerank_fallbacks, int):
        combo_results = result.get("combo_results", [])
        fallback_count = sum(
            1
            for cr in combo_results
            if cr.get("metrics", {})
            .get("rerank", {})
            .get("fallback_keep_all", False)
        )
        if fallback_count > max_rerank_fallbacks:
            failures.append(
                "rerank fallbacks exceeded maximum: "
                f"{fallback_count} > {max_rerank_fallbacks}"
            )

    return failures


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Params:
        (none)

    Returns:
        Configured ArgumentParser

    Example:
        >>> parser = _build_parser()
        >>> args = parser.parse_args(["--test"])
    """
    parser = argparse.ArgumentParser(prog="retriever")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test queries from test_queries.yaml",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Ad-hoc query text",
    )
    parser.add_argument(
        "--bank",
        type=str,
        action="append",
        default=[],
        help="Bank code (repeatable)",
    )
    parser.add_argument(
        "--period",
        type=str,
        action="append",
        default=[],
        help="Period (repeatable)",
    )
    parser.add_argument(
        "--source",
        type=str,
        action="append",
        default=[],
        help="Data source filter (repeatable)",
    )
    return parser


def _load_test_queries() -> list[dict]:
    """Load test cases from test_queries.yaml.

    Params:
        (none)

    Returns:
        list of test case dicts with name, query, combos,
        and optional sources keys

    Example:
        >>> cases = _load_test_queries()
        >>> cases[0]["name"]
        'RBC CET1 ratio'
    """
    with open(_YAML_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("test_cases", [])


def _print_result(
    label: str, query: str, elapsed: float, result: dict
) -> None:
    """Print a retrieval result to stdout.

    Params:
        label: Header label (test name or "QUERY")
        query: Original query text
        elapsed: Wall-clock seconds
        result: ConsolidatedResult dict

    Returns:
        None
    """
    print(f"\n{_SEP}")
    print(f"TEST: {label}")
    print(f"QUERY: {query}")
    print(f"TIME: {elapsed:.1f}s")
    print(_SEP)
    summary = result.get("summary_answer", "")
    if summary:
        print("\n## Summary")
        print(summary)
        metrics_table = result.get("metrics_table", "")
        if metrics_table:
            print("\n## Metrics")
            print(metrics_table)
        detail = result.get("detailed_summary", "")
        if detail:
            print("\n## Detail")
            print(detail)
    else:
        response = result.get("consolidated_response", "(no response)")
        print(response)
    gaps = result.get("data_gaps", [])
    if gaps:
        print("\n## Gaps")
        for gap in gaps:
            print(f"  * {gap}")
    warnings = result.get("citation_warnings", [])
    if warnings:
        print("\nCITATION WARNINGS:")
        for warning in warnings:
            print(f"  * {warning}")
    assertion_failures = result.get("assertion_failures", [])
    if assertion_failures:
        print("\nASSERTION FAILURES:")
        for failure in assertion_failures:
            print(f"  * {failure}")
    print()


def _run_test_case(
    case: dict, index: int, total: int, conn, llm, logger
) -> dict:
    """Run one benchmark case and return the saved result payload.

    Params:
        case: Test case dict from test_queries.yaml
        index: 1-based case index
        total: Total number of test cases
        conn: psycopg2 connection
        llm: Configured LLM client
        logger: Logger instance

    Returns:
        Result payload to persist in the test artifact
    """
    name = case["name"]
    query = case["query"]
    combos = [
        ComboSpec(bank=combo["bank"], period=combo["period"])
        for combo in case["combos"]
    ]
    sources = case.get("sources")

    logger.info("Test %d/%d: %s", index, total, name)
    start = time.time()

    try:
        result = run_retrieval(query, combos, sources, conn, llm)
        elapsed = time.time() - start
        findings = result.get("key_findings", [])
        gaps = result.get("data_gaps", [])
        combo_results = result.get("combo_results", [])
        assertion_failures = _evaluate_assertions(case, result)
        status = "success" if not assertion_failures else "quality_failed"

        logger.info("  Completed in %.1fs", elapsed)
        logger.info("  Key findings: %d", len(findings))
        logger.info("  Data gaps: %d", len(gaps))
        if assertion_failures:
            logger.error(
                "  Quality assertions failed: %d",
                len(assertion_failures),
            )
            for failure in assertion_failures:
                logger.error("    %s", failure)

        _print_result(
            name,
            query,
            elapsed,
            {**result, "assertion_failures": assertion_failures},
        )

        return {
            "name": name,
            "query": query,
            "elapsed_seconds": round(elapsed, 1),
            "consolidated_response": result.get(
                "consolidated_response",
                "",
            ),
            "key_findings": findings,
            "data_gaps": gaps,
            "citation_warnings": result.get(
                "citation_warnings",
                [],
            ),
            "coverage_audit": result.get("coverage_audit", ""),
            "uncited_ref_ids": result.get("uncited_ref_ids", []),
            "unincorporated_findings": result.get(
                "unincorporated_findings",
                [],
            ),
            "assertion_failures": assertion_failures,
            "combo_count": len(combo_results),
            "combo_results": combo_results,
            "metrics": result.get("metrics", {}),
            "trace_id": result.get("trace_id", ""),
            "trace_path": result.get("trace_path", ""),
            "status": status,
        }
    except (
        RuntimeError,
        PgError,
        OpenAIError,
        ValueError,
        KeyError,
        OSError,
    ) as exc:
        elapsed = time.time() - start
        logger.error(
            "  Failed in %.1fs: %s",
            elapsed,
            exc,
        )
        return {
            "name": name,
            "query": query,
            "elapsed_seconds": round(elapsed, 1),
            "status": "failed",
            "error": str(exc),
        }


def _run_test_suite(conn, llm, logger) -> None:
    """Run all test cases from test_queries.yaml.

    Loads test cases, runs each through the retrieval
    pipeline, prints results to console, and saves a
    JSON summary to the logs directory.

    Params:
        conn: psycopg2 connection
        llm: Configured LLM client
        logger: Logger instance

    Returns:
        None

    Example:
        >>> _run_test_suite(conn, llm, logger)
    """
    test_cases = _load_test_queries()
    logger.info("Loaded %d test cases", len(test_cases))

    results = []
    total = len(test_cases)
    for idx, case in enumerate(test_cases, 1):
        results.append(_run_test_case(case, idx, total, conn, llm, logger))

    _save_results(results, logger)
    _log_summary(results, logger)


def _save_results(results: list[dict], logger) -> None:
    """Write test results to a timestamped JSON file.

    Params:
        results: list of result dicts
        logger: Logger instance

    Returns:
        None
    """
    _LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = _LOGS_DIR / f"test_results_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved to %s", output_path)


def _log_summary(results: list[dict], logger) -> None:
    """Log pass/fail counts for the test suite.

    Params:
        results: list of result dicts
        logger: Logger instance

    Returns:
        None
    """
    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] != "success")
    logger.info(
        "Test suite complete: %d succeeded, %d failed",
        succeeded,
        failed,
    )


def _run_adhoc_query(
    conn, llm, query, banks, periods, sources, logger
) -> None:
    """Run a single ad-hoc query against the database.

    Builds combo specs from bank x period combinations,
    runs the retrieval pipeline, and prints the result.

    Params:
        conn: psycopg2 connection
        llm: Configured LLM client
        query: User query text
        banks: list of bank codes
        periods: list of period strings
        sources: list of source filters (may be empty)
        logger: Logger instance

    Returns:
        None

    Example:
        >>> _run_adhoc_query(
        ...     conn, llm, "CET1?",
        ...     ["RBC"], ["2026_Q1"], [], logger,
        ... )
    """
    if not banks or not periods:
        logger.error("--bank and --period required for ad-hoc queries")
        sys.exit(1)

    combos = [
        ComboSpec(bank=bank, period=period)
        for bank in banks
        for period in periods
    ]
    source_filter = sources if sources else None

    logger.info("Query: %s", query)
    logger.info("Combos: %s", combos)
    if source_filter:
        logger.info("Sources: %s", source_filter)

    start = time.time()
    result = run_retrieval(
        query,
        combos,
        source_filter,
        conn,
        llm,
    )
    elapsed = time.time() - start
    trace_path = result.get("trace_path", "")
    if trace_path:
        logger.info("Trace saved to %s", trace_path)

    _print_result(query, query, elapsed, result)


def main(argv=None) -> None:
    """Run the retriever pipeline CLI.

    Supports two modes: --test runs all queries from
    test_queries.yaml, or --query with --bank and --period
    runs a single ad-hoc query. Prints help if neither is
    provided.

    Params:
        argv: Command-line arguments (default sys.argv[1:])

    Returns:
        None

    Example:
        >>> main(["--test"])
        >>> main(["--query", "CET1?", "--bank", "RBC",
        ...        "--period", "2026_Q1"])
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    conn, llm = run_startup()
    logger = logging.getLogger(__name__)

    try:
        if args.test:
            _run_test_suite(conn, llm, logger)
        elif args.query:
            _run_adhoc_query(
                conn,
                llm,
                args.query,
                args.bank,
                args.period,
                args.source,
                logger,
            )
        else:
            parser.print_help()
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
