# Trace Review — Findings & Investigation Log

**Trace under review:** `u-debug/traces/20260407T025300Z_a2ef222f/`
**Query:** "Explain RBC's Q1 2026 earnings strength, capital composition, and credit losses. Include PPPT, CET1 capital or retained earnings, and PCL on performing versus impaired loans."
**Sources:** `investor-slides` (19 findings, 1 iter, conf 0.88) · `rts` (10 findings, 1 iter, conf 0.90) · `pillar3` (23 findings, 3 iters, conf 0.78)
**Catalog:** 27 refs total, 23 cited (gaps at 12, 22, 23, 24)

---

## How to use this doc

Each finding is a discrete unit of work with:
- **Status:** `Open` → `Investigating` → `Fix proposed` → `Accepted` → `Implemented`
- **Symptom**, **Evidence** (with quoted snippets and trace paths), **Hypothesized root cause**
- **Investigation notes** and **Proposed fix** are filled in as we work the item
- **Resolution** captures what was actually changed and where

Work order: top-to-bottom unless we choose otherwise. Do not advance to the next finding until the current one is `Accepted`.

---

## Index

| ID | Title | Category | Severity | Status |
|---|---|---|---|---|
| F01 | CET1 movement bridge math doesn't reconcile (15 vs 20 bps) | Response accuracy | High | Implemented (re-fix R1) |
| F02 | Multi-value cells in Metrics table (`8497; 8500`, retained earnings 3-way) | Metrics table | Medium | Implemented |
| F03 | ACL claim mixes loans-only vs all-exposures scopes | Response accuracy | Medium | Implemented |
| F04 | Garbled PPPT calculation finding (REF:12) | Research extraction | Low | Implemented |
| F05 | Capital Markets +$122M QoQ outlier — no analytic note | Response analysis | Medium | Implemented |
| F06 | Adjusted PPPT 12% YoY finding dropped from response | Coverage / dropped findings | Medium | Implemented |
| F07 | Net write-offs ($634M) and impaired exposures ($9,294M) dropped | Coverage / dropped findings | Medium-High | Implemented |
| F08 | Qualitative segment color (CNB release) underused | Coverage / dropped findings | Medium | Implemented (via F06 + re-fix R2) |
| F09 | Inconsistent units, signs, and formats across Metrics rows | Metrics table | Medium | Implemented |
| F10 | Research findings missing unit context from surrounding chunks | Research extraction | Medium | Implemented |
| F11 | Repeated `RBC` in Entity column (deferred) | Metrics table | Low | Deferred |
| F12 | Source-language differences (`PCL on impaired loans` vs `… assets`) | Metrics table | Medium | Open |
| F13 | Visible numbering gaps in reference table (12, 22, 23, 24) | Reference table | Low | Open |
| F14 | Page-level "duplication" — investor-slides p8 produces 4 refs | Reference index design | Medium | Open |
| F15 | Real dedup failure: pillar3 CR1 split into 2 refs by 1-word location_detail diff | Reference index | Medium | Open |
| F16 | Reference appendix lacks evidence preview (no metric/value column) | Reference table | Low | Open |
| F17 | Severe within-source duplication in pillar3 (23 findings → ~14 unique facts) | Research stage | High | Open |
| F18 | Pillar3 wasted 2 extra iterations chasing data not in source | Research stage | High | Open |
| F19 | Empty structured fields where hard metrics exist (REF:9, REF:19) | Research extraction | Medium | Open |
| F20 | `location_detail` pinned to slide title, not panel (REF:2) | Research extraction | Medium | Open |
| F21 | Over-citation by end-of-bullet clustering | Citation usage | Medium | Open |
| F22 | `[REF:4]` filler co-citation in CET1 YoY claim | Citation usage | Low | Open |
| F23 | Same physical evidence cited as multiple refs (visual diversity overstating) | Citation usage | Low | Open |
| F24 | `citation_validation` only checks ref existence, not factual support | Pipeline validation | Medium | Open |
| F25 | `key_findings` is naive regex split of summary bullets | Pipeline output | Low | Open |
| F26 | Investor-slides "ACL to loans ratio = 73 bps" finding dropped | Coverage / dropped findings | Low | Implemented |
| F27 | Footnote markers `(1)(2)` polluting `metric_name` | Research extraction | Low | Open |
| F28 | HTML anchor `href` not URL-encoded in `consolidate.py:161` | Future fragility | Low | Open |
| F29 | Period columns in Metrics table drift across runs | Metrics table | Low-Medium | Open |
| F30 | Research stage instability on methodology footnotes (PPPT calc) | Research extraction | Low | Open |

---

## Bugs found during validation re-runs

### B1 — F06 audit fields stripped from trace output by citation validator

**Category:** Trace serialization / cross-stage schema drift
**Severity:** Medium (visibility loss, not correctness)
**Status:** Implemented
**Discovered in:** re-run `20260407T131653Z_043b7c9f` (cluster-boundary validation for F04/F05/F06/F09/F10)

**Symptom:**
The consolidate stage log line correctly reports `uncited_refs=0, unincorporated=0`, proving the F06 coverage audit runs and computes results. But the run-level trace JSON (`outputs` dict) has **zero hits** on any of the three new fields (`coverage_audit`, `uncited_ref_ids`, `unincorporated_findings`). Post-hoc inspection of audit results from the trace is impossible — the data exists only in the log file, not in the serialized trace.

**Root cause (two stacked gaps):**

1. **`citation_validator.py:99-107`** — `validate_consolidated_citations` rebuilds a new `ConsolidatedResult` from the input and copies a hardcoded list of optional keys: `summary_answer, metrics_table, detailed_summary, reference_index, metrics`. The F06 cluster added `coverage_audit`, `uncited_ref_ids`, `unincorporated_findings` to the TypedDict and populated them in `consolidate_results`, but never updated this reconstructor. Every call to the validator silently strips the audit fields. This runs in `orchestrator._finalize_run` between `consolidate_results` (which sets the fields) and `_build_run_trace_payload` (which reads them), so the fields are gone by the time the trace is serialized. **This was the upstream leak.**

2. **`orchestrator.py:774-779`** — `_build_run_trace_payload` builds the trace `outputs` dict with only the 4 original fields. Even if the validator preserved the new fields, this serializer would not have propagated them to the trace JSON.

3. **`main.py:288-309`** — the CLI `_print_result` return path has the same gap (orthogonal to the debug API but worth fixing for consistency).

All three paths needed fixing. The validator was the hidden root cause — without that fix, the orchestrator edit alone wouldn't surface the fields because they'd be gone by the time the serializer ran.

**Resolution:**

Three-file patch:
- `u-retriever/src/retriever/utils/citation_validator.py` — extended the `optional_key` tuple in `validate_consolidated_citations` to include `coverage_audit`, `uncited_ref_ids`, `unincorporated_findings`. `if optional_key in result` guard preserves backwards compatibility: absent fields stay absent, present fields round-trip.
- `u-retriever/src/retriever/stages/orchestrator.py` — extended the `outputs` dict in `_build_run_trace_payload` to include the three audit fields with `.get(key, default)` safety.
- `u-retriever/src/retriever/main.py` — same three-field extension to the CLI return dict at lines 288-309.

**Files touched:**
- `u-retriever/src/retriever/utils/citation_validator.py` — 3-line addition to the `optional_key` tuple
- `u-retriever/src/retriever/stages/orchestrator.py` — 5 lines added to `outputs` dict
- `u-retriever/src/retriever/main.py` — 5 lines added to CLI return dict
- `u-retriever/tests/test_citation_validator.py`:
  - New `test_validate_preserves_coverage_audit_fields` — populated case round-trips
  - New `test_validate_omits_absent_coverage_audit_fields` — backwards-compat: absent fields stay absent
- `u-retriever/tests/test_orchestrator.py`:
  - Extended `test_run_retrieval_writes_trace_files` to assert empty-default audit fields present in trace `outputs`
  - New `test_run_retrieval_trace_includes_populated_audit_fields` — full end-to-end: consolidate stub returns populated audit fields; assert they round-trip through validator + serializer into the final trace JSON

**Quality gate:**
- `pytest --cov=src` — **254 passed** (was 251, +3 new tests), coverage **86%** unchanged
- `black --check` — clean
- `flake8` — clean
- `pylint` — **10.00/10 with zero warnings**
- `citation_validator.py` coverage stays at 100%

**Live verification:**
At the next re-run, the trace JSON's `outputs` block should contain `coverage_audit`, `uncited_ref_ids`, `unincorporated_findings` regardless of whether the audit produced any hits. For a clean run these will be `""`, `[]`, `[]` respectively — the empty values are themselves meaningful (they prove the audit ran and found nothing).

**Update (2026-04-07 second validation cycle):** the first cluster-boundary re-run (trace `20260407T131653Z_043b7c9f`) did not show the audit fields in the trace `outputs`, and neither did the second re-run (trace `20260407T135735Z_4197c116`). Root cause: the debug server (PID 25350 on port 5001) was started before the B1 Python edits were made. The Flask app runs with `debug=False` so there is no auto-reload. Python modules are loaded at import time; the running server was still holding the pre-B1 in-memory code. YAML prompts load fresh from disk on every request (no caching), which is why the R1 and R2 prompt changes took effect in the same two traces while the B1 code changes did not. Fix: server restart — commands and verification in the accompanying plan file. After restart the next re-run produces the audit fields in the trace, completing B1 live verification.

**Related findings:**
- **F06 cluster** — this bug was introduced by the F06 cluster work and missed during that cluster's implementation. The lesson: when adding new `NotRequired` fields to a TypedDict, enumerate every place in the codebase that reconstructs that TypedDict and update each. A grep for `ConsolidatedResult(` call sites would have caught the validator gap.
- **B2 (if it arises)** — the debug UI at `u-debug/src/debug/` does not currently render the audit fields. With the fields now surviving into the trace, the debug UI could surface them in a new panel. Intentional non-scope for B1.

---

## Findings (detail)

### F01 — CET1 movement bridge math doesn't reconcile

**Category:** Response accuracy
**Severity:** High
**Status:** Implemented

**Symptom:**
The Detail section reports a 5-component CET1 ratio walk that sums to +15 bps, but the headline change is +20 bps (13.5% → 13.7%). Off by 5 bps. Visible arithmetic error in the published response.

**Evidence:**
`outputs.consolidated_response`, Detail / capital section:
> "CET1 movement drivers from 13.5% to 13.7% included net income (+79 bps), dividends (-33 bps), RWA growth (-26 bps), share repurchases (-13 bps), and other (+8 bps) [REF:6]"

`+79 − 33 − 26 − 13 + 8 = +15 bps`, not +20.

Underlying research finding (`source_01_…investor-slides_38.json` → `stages.research.iterations[0].findings`, the CET1(1) Movement entry):
```json
{
  "finding": "CET1 movement drivers included Net Income (+79 bps), Dividends (33) bps, RWA Growth (26) bps, Share Repurchases (13) bps, and Other +8 bps, moving CET1 from 13.5% to 13.7%.",
  "metric_name": "CET1(1) Movement",
  "metric_value": "Q4/2025 13.5% to Q1/2026 13.7%",
  "page": 9
}
```
Same numbers verbatim — the discrepancy was already present in the extracted finding.

**Investigation notes:**

Initial hypothesis was wrong. I thought research had extracted 5 of N components and silently dropped some. Reading the actual chunk the research LLM saw (`source_01_…investor-slides_38.json` → `stages.research.final_chunks` → page 9), the slide only has 5 waterfall bars. The discrepancy is baked into the source itself.

What the page-9 chunk actually contains:
- **Waterfall**: 5 bars matching the extracted finding exactly (`+79`, `(33)`, `(26)`, `(13)`, `+8`).
- **Callout beside the chart** (not extracted): *"CET1 ratio of 13.7%, up 20 bps QoQ, reflecting: + Strong net internal capital generation; − Higher RWA mainly from strong client-driven business growth; − Repurchase of 4.2MM shares for $960MM"*.
- **Footnote (2)** (not extracted): *"Represents rounded figures. For more information, refer to the Capital Management section of our Q1/2026 Report to Shareholders."*
- **Footnote (4)** (not extracted): *"Other includes regulatory and model updates (+5 bps), fair value OCI adjustments (+5 bps), the impact of foreign exchange translation and other movements."*

The 5-bp gap is a rounding artifact: footnote (2) explicitly flags the endpoints as rounded figures, so the true QoQ change is probably ~15-18 bps, displayed as "up 20 bps QoQ" for simplicity. Research extracted the 5 bars but skipped the callout and footnotes, so consolidation had no context to either reconcile the bridge or explain the gap.

So this is a two-layer issue:
1. Research stage didn't capture the callout or footnotes alongside the waterfall.
2. Consolidation stage didn't double-check the bridge math before writing it.

**Proposed fix:**

1. **`u-retriever/src/retriever/stages/prompts/research.yaml`** — add rule 20: when extracting a movement walk / waterfall / component bridge, extract the visible components AND the stated endpoints/headline change AND any footnote or callout content that explains the components (sub-composition of "Other", rounding disclosures, residuals). These findings travel together so downstream stages have the context to reconcile.
2. **`u-retriever/src/retriever/stages/prompts/consolidation.yaml`** — add rule 7 (Reconciliation check): whenever presenting a component breakdown or movement walk in Detail, verify components sum to the stated total or endpoint change; if not, note the gap inline with any available explanation (rounding, residuals, scope), and surface to Gaps if no explanation is available.
3. **`consolidation.yaml`** Gaps section charter — widened to include internal consistency issues (component sums that don't reconcile, scope conflicts, unexplained residuals) in addition to the existing "information the query asked for that wasn't found".
4. **Structural tests** in `tests/test_consolidate.py` and `tests/test_research.py` loading each prompt via `load_prompt` and asserting the new rule text is present.

**Resolution:**

Files touched:
- `u-retriever/src/retriever/stages/prompts/consolidation.yaml` — new rule 7 (Reconciliation check), widened Gaps section charter. Gaps section name kept (not renamed to `Gaps & Caveats`) per minimal-change discipline.
- `u-retriever/src/retriever/stages/prompts/research.yaml` — new rule 20 (bridge extraction with endpoints and footnote context).
- `u-retriever/tests/test_consolidate.py` — added `test_consolidation_prompt_has_reconciliation_rule` and `test_consolidation_prompt_gaps_section_widened`; added `load_prompt` import and `_CONSOLIDATION_PROMPTS_DIR` constant.
- `u-retriever/tests/test_research.py` — added `test_research_prompt_has_bridge_extraction_rule`, `test_research_prompt_bridge_rule_requires_endpoints`, `test_research_prompt_bridge_rule_requires_footnotes`; added `load_prompt` import and `_RESEARCH_PROMPTS_DIR` constant.

Quality gate results (run from `u-retriever/`):
- `pytest --cov=src --cov-report=term-missing` — **217 passed** (baseline 212, +5 new). Coverage unchanged at **86%** (no new code paths, only prompt content + tests).
- `black --check src/ tests/` — clean, 38 files unchanged.
- `flake8 src/ tests/` — clean, zero warnings.
- `pylint src/ tests/` — **10.00/10**. Pre-existing `R0914` warning on `consolidate.py:418` (too-many-locals) is unchanged and unrelated to this fix.

LLM-behaviour verification is deferred to the next cluster-boundary re-run (after F02 and F03 are also completed), per the re-run policy in the plan.

**Related findings:**
- F03 (ACL scope conflation) — shares the "verify before juxtaposing" discipline. The widened Gaps charter in this fix will help F03's presentation layer, but F03 still needs its own research-side fix to capture population scope in structured fields. Kept separate.
- F24 (validation framework) — a future programmatic reconciliation validator could mechanically check what this rule asks the LLM to do. Flagged as a separate scope decision.

---

#### Post-validation re-fix (R1 — 2026-04-07 cluster-boundary re-run)

**Symptom of the regression:**
In the F04/F05/F06/F09/F10 cluster-boundary re-run (trace `20260407T131653Z_043b7c9f`), the investor-slides p9 chunk contained the full CET1 waterfall in its `raw_content` figure description (bars labeled `79 bps / (33) / (26) / (13) / +8 bps`, plus a dashed-box overlay `+46 bps "Net internal capital generation"`), but the research stage extracted ONLY the aggregated overlay. The 5 component bars were entirely missing from `source_01 ... investor-slides_38.json → stages.research.findings`. Consolidation therefore had no bridge to render and produced generic prose with no quantitative breakdown.

**Root cause:**
Cross-rule vocabulary collision between F01's rule 20 and F04's rule 22.

- Rule 20 (permissive, F01): *"Each visible component (bar, line item) as an individual quantitative finding..."*
- Rule 22 (prohibitive, F04): *"Do not extract the calculation inputs themselves (the A, B, C component values inside the parenthetical definition)."*

Both rules use the word **"component"**. Prohibitive framing tends to win against permissive framing for LLMs, especially when the same vocabulary appears in both. The waterfall bars can be reasonably reclassified as "calculation inputs" to the net +20 bps change, so the LLM applied rule 22's prohibition and skipped all bar-level findings.

This was not caught during F04 implementation because F04's cluster re-run (`20260407T041015Z_2dc7e8e0`) validated only F01/F02/F03. The F01 rule 20 vs F04 rule 22 interaction was first observable when F04 rule 22 was stable AND the waterfall slide was hit on the next re-run.

**Evidence from the regression trace:**
- p9 findings extracted: `CET1 ratio 13.7%`, `CET1 ratio 13.5%`, `Net internal capital generation 46 bps`, `Q1/2026 RWA 735 $B`
- p9 findings NOT extracted: `CET1 movement Net Income +79 bps`, `CET1 movement Dividends -33 bps`, `CET1 movement RWA Growth -26 bps`, `CET1 movement Share Repurchases -13 bps`, `CET1 movement Other +8 bps`, `CET1 QoQ change +20 bps`
- Research LLM confidence was 0.88 with 1 iteration — it felt complete
- Rule 20 had no worked example; rule 22 had a specific pattern (`"Income (A) before income taxes (B) and PCL (C)"`) that the LLM generalized over-broadly

**Re-fix:**

Two coordinated edits to `u-retriever/src/retriever/stages/prompts/research.yaml`:

1. **Rule 22 waterfall carve-out.** Added explicit narrowing after *"Do not extract the calculation inputs themselves..."*:
   > "This prohibition applies ONLY to parenthetical component definitions inside non-GAAP calculation footnotes (pattern: `X is calculated as measure_1 (value_1) operator measure_2 (value_2) ...`). It does NOT apply to waterfall bars, bridge steps, or decomposition line items from rule 20 — those are presentation of already-disaggregated data and MUST be extracted per rule 20, regardless of whether individual bar values appear inside parentheses (e.g., `(33) bps` for a negative CET1 movement component is standard accounting notation for a negative number, not a parenthetical calculation input)."

   The carve-out scopes rule 22 to the specific "X is calculated as..." footnote pattern and explicitly says parenthesized negatives in waterfalls are NOT what the rule targets.

2. **Rule 20 worked example.** Added a verbatim worked example using the exact RBC CET1 waterfall, immediately before the closing "The goal is..." sentence:
   > "Worked example — a CET1 movement waterfall figure with endpoints `"Q4/2025 13.5%"` and `"Q1/2026 13.7%"` and bars labeled `"79 bps"` (Net Income), `"(33) bps"` (Dividends), `"(26) bps"` (RWA Growth), `"(13) bps"` (Share Repurchases), `"8 bps"` (Other), plus a dashed-box overlay labeling the Net Income + Dividends combination as `"+46 bps"` (Net internal capital generation). The required findings are: (1) metric_name `"CET1 movement — Net Income"`, metric_value `"79"`, unit `"bps"`; (2) metric_name `"CET1 movement — Dividends"`, metric_value `"-33"`, unit `"bps"`; ... [all 5 bars enumerated] ... plus the two endpoints (13.5% and 13.7%) and the stated headline change (`"+20 bps QoQ"`). The aggregated `"+46 bps"` Net internal capital generation overlay is ALSO extracted as a separate finding — it is a visible data point on the figure. Parenthesized negatives like `"(33) bps"` are standard accounting notation for negative values — emit the negative sign in metric_value (`"-33"`). DO NOT collapse the 5 component bars into the aggregated overlay, and DO NOT skip extraction because the components do not reconcile to the headline change (that is by design — downstream stages use footnote context to explain the gap)."

   This quotes the failure case verbatim (session pattern #14) and makes rule 20 pattern-based rather than judgment-based (session pattern #10). The explicit `DO NOT collapse` and `DO NOT skip extraction because the components do not reconcile` clauses are the anti-retreat safety net.

**Files touched:**
- `u-retriever/src/retriever/stages/prompts/research.yaml` — rule 20 worked example added; rule 22 waterfall carve-out added
- `u-retriever/tests/test_research.py`:
  - New `test_research_prompt_rule22_has_waterfall_carveout` — asserts the key carve-out phrases including `"(33) bps"` and `"standard accounting notation for a negative number"`
  - New `test_research_prompt_rule20_has_waterfall_worked_example` — asserts all 5 `"CET1 movement — X"` metric_names, the 3 negative metric_values, and the `DO NOT` clauses
  - Existing `test_research_prompt_has_bridge_extraction_rule`, `test_research_prompt_has_methodology_footnote_rule`, and siblings still pass (key-phrase assertions unaffected by the additions)

**Quality gate:**
- `pytest --cov=src` — **256 passed** (was 254 after B1, +2 new tests for R1), coverage **86%** unchanged
- `black --check` — clean after one auto-reformat (test line-length wrap)
- `flake8` — clean (zero warnings)
- `pylint` — **10.00/10 with zero warnings**

**Live verification (pending next re-run):**
1. Investor-slides p9 research findings contain ≥7 entries: the 5 `CET1 movement — *` components + 2 endpoints, plus optionally the aggregated `Net internal capital generation` overlay
2. Each of the 5 bar findings has the correct sign in metric_value (`79`, `-33`, `-26`, `-13`, `8`)
3. unit field populated with `"bps"` on all bar findings
4. The consolidated response's Detail / Capital paragraph includes a quantitative bridge description with the 5 bps values (e.g., "Net Income +79 bps, Dividends -33 bps, RWA Growth -26 bps, Share Repurchases -13 bps, Other +8 bps")
5. F04 PPPT footnote stability preserved: rts p2 finding [1] still has clean `metric_value="8,497"` with no formula garble in future runs
6. F01's consolidation-side rule 7 (Reconciliation check) and Gaps charter still fire when the bridge is extracted but doesn't reconcile (the F01 original fix behavior)

**Cross-reference:**
- **F04** — the vocabulary collision originated when F04's rule 22 was added. The R1 re-fix preserves F04's core functionality (the PPPT footnote garble is still blocked) while scoping the prohibition to its specific pattern. See F04 Resolution for the full F04 rule 22 context.
- **F06 cluster** — F06's consolidation-side anti-retreat clause (rule 6 sub-clause e) was the model for R1's research-side anti-retreat language. F01 / F04 / F06 now all follow the same anti-retreat pattern.
- **R1 lesson for future prompt work**: when adding a new prohibitive rule, grep the rest of the prompt for shared vocabulary with existing permissive rules. Cross-rule vocabulary collisions are a recurring LLM failure mode; they should be caught at implementation time, not discovered through re-run regressions.

---

### F02 — Multi-value cells in Metrics table

**Category:** Metrics table
**Severity:** Medium
**Status:** Implemented

**Symptom:**
The Metrics table has cells containing multiple semicolon-separated values for what should be a single value column.

**Evidence:**
`outputs.consolidated_response`, Metrics table:
- `PPPT (Pre-provision, pre-tax earnings) | 8497; 8500 | 8% | 14%` — `8,497MM` and `8,500MM` are the same value rounded differently (investor-slides uses MM, rts uses "$8.5 billion")
- `Retained earnings ($MM) | 99265; 99023; 100247 | — | +9%` — three legitimately distinct concepts:
  - `99,265` = balance sheet (rts p48 REF:15)
  - `99,023` = CET1 composition input (pillar3 CC1 REF:21)
  - `100,247` = regulatory consolidation scope (pillar3 CC2 REF:25)

**Investigation notes:**

Root cause is a single misleading rule at `consolidation.yaml:44`:
> *"When values conflict, show both side by side in the value column."*

This one line conflates three distinct multi-value scenarios that deserve different handling:

1. **Same economic value at different precisions** — e.g. `$8,497MM` (investor-slides) and `"$8.5 billion"` (rts) are the SAME fact at different rounding. Current rule produces `8497; 8500` — garbage.
2. **Same metric label under different scopes or bases** — e.g. retained earnings at three legitimately distinct accounting scopes (balance sheet, CET1 composition, regulatory consolidation). These should be ONE ROW per scope.
3. **Genuinely conflicting values** — same metric, scope, period, different values across sources. Should be flagged via Gaps.

Pure prompt fix. No code change needed. The LLM is faithfully following the current instruction.

**Proposed fix:**

Replace `consolidation.yaml:44` with an explicit "ONE VALUE PER CELL" rule covering the three cases, instructing the LLM to:
1. Collapse rounding variants to the most precise value (cite both refs).
2. Split scope variants into separate rows with a distinguishing Line Item suffix (e.g. `"Retained earnings — balance sheet"`).
3. Surface genuine conflicts via per-source rows + a Gaps entry.

Add structural tests asserting the new rule text and three-case handling are present in the prompt.

**Resolution:**

Files touched:
- `u-retriever/src/retriever/stages/prompts/consolidation.yaml` — replaced the `When values conflict, show both side by side` line with a multi-paragraph rule covering the three scenarios. `## Metrics` section now begins with an explicit `ONE VALUE PER CELL` rule and three bulleted case handlers with illustrative examples ($8,497MM / $8.5B rounding collapse; retained earnings scope split; conflict → source suffix + Gaps).
- `u-retriever/tests/test_consolidate.py` — added `test_consolidation_prompt_metrics_one_value_per_cell` and `test_consolidation_prompt_metrics_scope_suffix_rule`, asserting the new rule text, the `"semicolon-separated"` forbidden phrase, and the three case keywords (`"different precisions"`, `"different scopes"`, `"Genuinely conflicting"`) are present.

Quality gate results (run from `u-retriever/`):
- `pytest --cov=src --cov-report=term-missing` — **219 passed** (217 prior + 2 new). Coverage unchanged at **86%** (pure prompt content + test additions, no new code paths).
- `black --check src/ tests/` — clean.
- `flake8 src/ tests/` — clean.
- `pylint src/ tests/` — **10.00/10**. Pre-existing R0914 on `consolidate.py:418` untouched.

**Iteration 1 (after F01/F02/F03 re-run, trace `20260407T041015Z_2dc7e8e0`):**

The first re-run validated the retained earnings split correctly into 4 scope rows (IFRS balance sheet / CET1 composition / regulatory scope / capital purposes), but PPPT duplicated into two rows with **identical** Entity + Segment + Line Item — one from investor-slides (`7,483 / 7,835 / 8,497`) and one from rts (`7,500 / 7,800 / 8,500`) — a rounding-variant case the rule failed to collapse. The "Do not create separate rows" clause in the first-precision bullet wasn't strong enough as a final check.

Iteration applied: added an explicit `DUPLICATE ROW CHECK` sub-section after the three case bullets in the Metrics rule. Instructs the LLM to scan the table for identical Entity + Segment + Line Item tuples as a final step before finalizing the table, and merge them using the most precise value + combined refs. Added test `test_consolidation_prompt_metrics_duplicate_row_check`.

**Iteration 2 (design change: Source column + preserve-transparency default):**

User pushback on iteration 1's "collapse rounding variants by picking most precise value" default: *"we can't just assume"*. When two sources report similar-but-not-identical values, the safer default is to preserve source transparency and only collapse when the rounding relationship is unambiguous from context. This is especially important because the consolidation LLM cannot always distinguish rounding from a genuine data difference.

Full `## Metrics` section restructure:

1. **New `Source` column** added between `Line Item` and the period columns. Every row carries an explicit data-source label (e.g. `"investor-slides"`, `"rts"`, `"pillar3"`, or a semicolon-separated list when multiple sources agree).

2. **Multi-source handling rewritten as 4 ordered rules with source-transparency default:**
   - Rule 1: Same metric, exact same value across sources → ONE ROW, Source lists all agreeing sources.
   - Rule 2: Rounding variant detectable unambiguously from context (e.g. `"$8.5 billion"` vs `"$8,497MM"`, or 1 decimal vs integer MM) → ONE ROW with the most precise value, Source lists all. Explicit fallthrough: *"When in doubt about whether rounding explains the gap, DO NOT collapse — fall through to rule 3."*
   - Rule 3: Values genuinely differ OR rounding cannot be determined → ONE ROW PER SOURCE, each row's Source column shows that single source's label. Discrepancy surfaced as a Gaps item.
   - Rule 4: Different scopes / bases / consolidation perimeters → split by scope with Line Item suffix (existing pattern), composes with rules 1–3.

3. **`DUPLICATE ROW CHECK` uniqueness tuple updated** from `(Entity, Segment, Line Item)` to `(Entity, Segment, Line Item, Source)`. Rows with same metric but different Source labels are legitimate rule-3 splits.

Tests updated/added:
- `test_consolidation_prompt_metrics_scope_suffix_rule` — replaced `"Genuinely conflicting"` assertion with `"genuinely differ"` (new phrasing), adjusted for new structure.
- `test_consolidation_prompt_metrics_duplicate_row_check` — updated tuple to `"identical Entity + Segment + Line Item + Source"`; removed the now-gone `"forbidden"` sentence.
- `test_consolidation_prompt_metrics_has_source_column` (new) — asserts `"| Source |"` column header, `"preserve source transparency"`, `"ONE ROW PER SOURCE"`, and the Gaps-surfacing instruction.

Quality gate after iteration 2: **223 passed** (222 prior + 1 new test; the updated tests needed no line-length fixes). Black, flake8, pylint all clean at 10.00/10.

**Iteration 2 validation (trace `20260407T042647Z_6ca4e478`):** ✅ clean win.

- `Source` column is present in the Metrics header: `| Entity | Segment | Line Item | Source | Q1 2026 | Ref |`.
- PPPT correctly merged into ONE row: `Source = "investor-slides; rts"`, value `8497` (precise investor-slides value), refs `[11]; [1]`. Rule 2 fired — the LLM judged rts's `$8.5B` as a rounded form of investor-slides's `$8,497MM`. The F02 iteration 1 duplication (two identical Line Item rows with different values) is gone.
- Retained earnings correctly split into 3 scope rows (`— balance sheet`, `— CET1 composition`, `— regulatory scope`) each with its own single-source label in the new Source column. (The prior re-run had a 4th "— capital purposes" row; not present in this re-run. Acceptable — the LLM found the same 3 distinct scope variants that were in the underlying findings.)
- Rule 1 (same value → merge with multi-source label) working widely: CET1 capital `rts; pillar3`, CET1 ratio `rts; investor-slides; pillar3`, CET1 bridge components all show both sources, PCL metrics all show `rts; investor-slides`.
- No duplicate rows. Baseline had 15 rows with semicolon-cells and dup Line Items; iteration 1 had 24 rows with dupes; iteration 2 has 21 rows with none.
- F01 CET1 reconciliation Gaps entry still present and working. Additionally, the 5 bridge components are now in the Metrics table as individual rows (`CET1 ratio bridge — Net income (bps) = 79`, etc.), so the reader can add them up in the table and see the `+15 vs +20` gap directly.
- F03 coverage loss unchanged (ACL loans-only totals still absent; logged for F06 cluster).

Side observations (not regressions, but behavioural drift worth noting):
- **Period columns reduced** from `Q1 2025 | Q4 2025 | Q1 2026` (iteration 1 re-run) to just `Q1 2026` in this trace. The LLM is relying on Detail prose for comparatives. F02's rules don't say anything about period coverage, so this is a separate concern — noted for F09 (formatting consistency).
- **REF:12 / REF:2 simplification**: the garbled PPPT calculation footnote is now only extracted as `Net income = 5785` instead of the confused three-row decomposition the earlier re-run had. Cleaner, but the original F04 fix hasn't run yet — this may be the LLM avoiding the confusion rather than resolving it. F04 investigation will address directly.

**F02 status: Implemented + validated.**

**Related findings:**
- F09 (formatting inconsistencies) — independent. F02 dictates cell content; F09 handles per-value units/signs.
- F12 (source-verbatim terminology footnoting) — will reuse F02's "suffix in the Line Item column" pattern. F02 lands first as the foundation.

---

### F03 — ACL scope conflation (loans-only vs all-exposures)

**Category:** Response accuracy
**Severity:** Medium
**Status:** Implemented

**Symptom:**
The Detail section juxtaposes a loans-only ACL total with all-exposure Stage 1&2 and Stage 3 figures as if they share the same population. They don't.

**Evidence:**
`outputs.consolidated_response`, Detail credit-losses bullet:
> "the allowance for credit losses on loans and acceptances was $7,752MM in total, up $293MM QoQ; **Stage 1 & 2 allowances across exposures totaled $7,258MM**, and Stage 3 (impaired) allowances totaled $2,238MM [REF:9][REF:26][REF:27]"

- `$7,752MM` = loans + acceptances ONLY (REF:9, investor-slides p18)
- `$7,258MM` = loans + debt securities + off-balance-sheet exposures (pillar3 CR1 REF:26)
- `$2,238MM` = loans + securities (pillar3 CRB_f_a REF:27)

`$7,258 + $2,238 ≠ $7,752` — and the reader cannot infer that the scopes differ from the surrounding text.

**Investigation notes:**

Two-layer bug confirmed by cross-referencing finding text against metric_name fields:

Scope info is captured in metric_name **inconsistently** across the pillar3 findings:
- `Q1/26 ACL on Loans & Acceptances | $ MM = 7,752` (investor-slides REF:9) — scope IN metric_name ✓
- `Allowances/impairments - Loans = 7401` (pillar3 CR1) — scope IN metric_name ✓
- `Off-Balance Sheet exposures - Allowances/impairments = 388` — scope IN metric_name ✓
- `Allowances/impairments = 7401` (pillar3 CR1, duplicate) — scope NOT in metric_name ✗
- `Total - Allowances/impairments; ... = 7803; ...; 7258` — ambiguous "Total" ✗
- `Regulatory category of General (Stage 1 & 2) - Total = 7258` — ambiguous "Total" ✗
- `Allowance (Stage 3 IFRS 9) - Total impaired exposures = 2238` — partial (IFRS stage but no population) ✗

Three of seven relevant findings have scope in metric_name; four do not. The four that don't bury the scope in the `finding` prose text (e.g. *"Stage 1 & 2 (performing) allowances across loans, debt securities, and off-balance sheet exposures totaled 7258 million"*). Consolidation reads the structured fields, juxtaposes values, and misses the scope qualifier when it's only in prose.

Layer 1: research stage has no rule forcing scope consistency in metric_name.
Layer 2: consolidation has no rule forcing a scope-comparability check before juxtaposing.

**Proposed fix:**

Two-layer fix, belt-and-suspenders:

1. **`research.yaml`** — new rule 21 instructing the LLM to include scope qualifiers verbatim in metric_name for any allowance / provision / exposure / balance / scope-dependent metric, with concrete examples (loans + acceptances; loans + debt securities + off-balance-sheet; drawn vs. EAD; regulatory consolidation scope; IFRS 9 stage group).
2. **`consolidation.yaml`** — new rule 8 (Scope comparability) instructing the LLM to verify entity + period + segment + population scope before juxtaposing values. If scopes differ, either split into separate rows per F02's scope-suffix rule, or name the scope difference explicitly in Detail prose. Includes the worked example from this finding as an anchor.
3. **Tests** — one structural test per file asserting the new rule text is present.

No schema change to `ResearchFinding`. Scope is embedded in the existing `metric_name` field as a verbatim suffix.

**Resolution:**

Files touched:
- `u-retriever/src/retriever/stages/prompts/research.yaml` — added rule 21 (scope qualifier in metric_name) after the F01 bridge rule. Includes five concrete examples covering loans, off-balance-sheet, drawn vs EAD, regulatory consolidation scope, and IFRS 9 stage grouping.
- `u-retriever/src/retriever/stages/prompts/consolidation.yaml` — added rule 8 (Scope comparability) after the F01 reconciliation rule. Includes the ACL loans-only vs. all-exposures worked example from this finding as the anchor case.
- `u-retriever/tests/test_research.py` — added `test_research_prompt_has_scope_qualifier_rule`. Uses `" ".join(prompt["user_prompt"].split())` to normalize YAML folded-scalar line breaks before asserting phrases that may span continuation lines.
- `u-retriever/tests/test_consolidate.py` — added `test_consolidation_prompt_has_scope_comparability_rule`. Same whitespace normalization pattern.

Quality gate results (run from `u-retriever/`):
- `pytest --cov=src --cov-report=term-missing` — **221 passed** (219 prior + 2 new). Coverage unchanged at **86%**.
- `black --check src/ tests/` — clean.
- `flake8 src/ tests/` — clean.
- `pylint src/ tests/` — **10.00/10**. Pre-existing R0914 on `consolidate.py:418` untouched.

LLM-behaviour verification: **ready for cluster-boundary re-run**. F01 + F02 + F03 are all in place, covering the CET1 bridge reconciliation, multi-value cells, and ACL scope conflation — the three highest-impact content issues in the trace baseline.

**Related findings:**
- F02 — F03 relies on F02's "scope suffix in Line Item" pattern for the row-splitting branch of the comparability rule. F02 landed first as prerequisite.
- F12 (source-verbatim terminology footnoting) — will benefit from F03's scope-in-metric_name mechanism for the PCL loans vs. assets case. May need minor F12-specific addendum later.
- F24 (validation framework) — F03's scope check is an LLM-level instruction; a programmatic cross-check could enforce it mechanically. Separate scope decision.

**Notes:**
- The scope-comparability rule in consolidation.yaml explicitly cites the `$7,752MM loans vs $7,258MM all-exposures` case as the worked example, which anchors the LLM's understanding of the trap.
- The F03 test assertions use `" ".join(prompt["user_prompt"].split())` to normalize YAML folded-scalar line breaks. Existing F01 and F02 tests don't use this pattern because their asserted phrases happen not to cross line boundaries; the F03 phrases do, so normalization is required. Future prompt-structural tests should default to this pattern for robustness.

**Known side effect — coverage loss (logged for F06/F07/F08/F26 cluster):**

The F01/F02/F03 cluster-boundary re-run (trace `20260407T041015Z_2dc7e8e0`) confirmed F03's scope conflation is **gone** — but the consolidation LLM avoided the trap by **dropping the investor-slides REF:9 loans-only ACL data entirely** rather than surfacing it with a scope qualifier. The new response contains no mention of:
- `$7,752MM` ACL on loans and acceptances (total ACL on loans)
- `$293MM` QoQ increase in loans ACL
- `$5.5B` performing ACL on loans
- The qualitative driver: *"We took $28MM of provisions on performing loans this quarter, mainly in Personal Banking, Capital Markets and Commercial Banking, partially offset by a release in City National Bank."*

All four of these were in the baseline response. The new response stuck to PCL-level metrics (income-statement basis) and net write-offs, avoiding ACL balances entirely.

This is a coverage regression in a topic (credit losses) the query explicitly asked about. Common pattern: a "be careful about X" rule induces the LLM to retreat from X altogether. Logged as adjacent to the F06/F07/F08/F26 coverage-drops cluster — that cluster's fix (a coverage-audit step) should ensure scope-qualified findings still get surfaced, not silently dropped when the LLM is uncertain about comparability. The fix is NOT to loosen F03's scope rule; the fix is to make F06's coverage audit detect when a finding with a clear scope qualifier was dropped without justification.

---

### F04 — Garbled PPPT calculation finding (REF:12)

**Category:** Research extraction
**Severity:** Low (consolidator already declined to cite it)
**Status:** Implemented

**Symptom:**
The research stage extracted a derived/computed finding from the PPPT methodology footnote whose `metric_value` doesn't actually compute as written.

**Evidence:**
`source_03_…rts_39.json` → finding 1:
```json
{
  "finding": "PPPT earnings are calculated as income before income taxes and PCL; for Q1 2026, PPPT equals income before income taxes of $1,622 million plus...",
  "metric_name": "Pre-provision, pre-tax (PPPT) earnings",
  "metric_value": "Income ($5,785m) before income taxes ($1,622m) and PCL ($1,090m)",
  "page": 2
}
```
The numbers look like (Net income $5,785M, Income tax $1,622M, PCL $1,090M), where PPPT = NI + tax + PCL = $8,497M ✓ — but the finding's prose mis-labels `$1,622M` as "income before income taxes" (it's the tax provision).

The consolidation LLM correctly chose not to cite this, so the confusion didn't propagate. But the finding occupies a catalog slot ([REF:12]) and is wrong.

**Hypothesized root cause:**
LLM tried to "extract" a methodology footnote that describes a calculation, then attempted to fill `metric_value` with the formula instead of recognizing it as qualitative methodology commentary. No rule against extracting derived/computed values.

**Investigation notes:**

The page-2 chunk (baseline `source_03_…rts_39.json` → `stages.research.final_chunks[1]`, `content_unit_id=2`, `section_title="Management's Discussion and Analysis"`) contains a reported/adjusted comparison table, the MD&A table of contents, and footnote (5) — the *only* place on the page where the PPPT calculation is shown:

> "Pre-provision, pre-tax (PPPT) earnings is calculated as income (January 31, 2026: $5,785 million; October 31, 2025: $5,434 million; January 31, 2025: $5,131 million) before income taxes (January 31, 2026: $1,622 million; October 31, 2025: $1,394 million; January 31, 2025: $1,302 million) and PCL (January 31, 2026: $1,090 million; October 31, 2025: $1,007 million; January 31, 2025: $1,050 million). This is a non-GAAP measure. … refer to the Key performance and non-GAAP measures section …"

The footnote grammar is a parsing trap. The intended read is:
```
PPPT = Net Income + Income Taxes + PCL
     = $5,785 + $1,622 + $1,090
     = $8,497M  ✓  (matches the $8.5B PPPT headline extracted from page 1)
```
i.e. "income (A) before income taxes (B) and PCL (C)" is a sentence whose `before` clause is an add-back instruction — `A` is Net Income and `B`/`C` are being added back. But "income before income taxes" reads *syntactically* as a compound noun phrase meaning "pretax income", so every run of the research LLM mis-attaches `$1,622M` to "income before income taxes" in the finding prose.

**Critical observation:** footnote (5) adds no new business fact. The primary PPPT value (`$8.5 billion, up $1.0 billion or 14% YoY`) is captured from page 1 by `source_03_…rts_39.json` finding 0, and from investor-slides page 8 by REF:1. The `$1,090M` PCL component is independently captured from rts page 4 as a direct PCL reading (finding 5). Only the `$1,622M` income-tax provision is footnote-only, and it's ancillary to the query ("PPPT, CET1 capital or retained earnings, and PCL on performing versus impaired loans" — tax provision isn't asked for).

**Cross-run instability** (from F30 evidence + verification against the three traces):
- **Baseline** (`20260407T025300Z_a2ef222f`): single garbled finding on page 2; consolidator declined to cite REF:12; no PPPT component rows in the Metrics table. Garble contained in research-stage output.
- **Iter 1 re-run** (`20260407T041015Z_2dc7e8e0`): the research LLM broke the footnote into 3 "PPPT components — Income/Income taxes/PCL" rows in the Metrics table (all marked Q1 2026), AND a separate `Income before income taxes | 6,400 |  | 7,400` row whose `$6,400` and `$7,400` values are **nowhere in the page-2 chunk** — invented numbers. Prose cited all three components with slightly cleaner arithmetic but still referred to `$5,785M` as "income".
- **Iter 2 re-run** (`20260407T042647Z_6ca4e478`): the research LLM retreated to a single `Net income | rts | 5,785` row and dropped the `$1,622` / `$1,090` footnote components entirely. The prose correctly labels `$5,785M` as "Net income" but the methodology context is gone.

Three runs, three structurally different failure modes on identical source content. The drift tracks with F01/F02/F03 rule additions — unrelated prompt changes shift the LLM's balance of "how aggressively should I extract from footnotes", because there is no explicit rule governing methodology footnotes at all.

**Proposed fix:**

Add a new rule 22 to `research.yaml` that (a) names the exact pattern of a non-GAAP methodology footnote, (b) tells the LLM not to create a finding whose `metric_value` is a formula or parenthesized component list, (c) quotes the baseline garble grammar (`"Income (A) before income taxes (B) and PCL (C)"`) verbatim so the LLM has a concrete pattern to match, and (d) explicitly says "do not extract the calculation inputs themselves" to block iter-1's over-decomposition mode. Exception clause preserves the ability to extract a distinct value that is *only* in a footnote and is *not* a calculation input.

Rule is pattern-based rather than judgment-based so the decision is stable across runs — same input → same decision regardless of what other rules (F01/F02/F03) add to the prompt.

**Resolution:**

Implemented rule 22 in `u-retriever/src/retriever/stages/prompts/research.yaml` (inserted between the F03 scope-qualifier rule 21 and the "Reminder" block). Added `test_research_prompt_has_methodology_footnote_rule` in `u-retriever/tests/test_research.py` asserting the key phrases of the rule are present in the flattened prompt body (same YAML-whitespace normalization pattern used by F03 tests).

**Files touched:**
- `u-retriever/src/retriever/stages/prompts/research.yaml` — new rule 22
- `u-retriever/tests/test_research.py` — new `test_research_prompt_has_methodology_footnote_rule`

**Quality gate:**
- `pytest --cov=src` — **224 passed** (was 223), coverage **86%** (unchanged — baseline gaps in `llm_connector.py`, `logging_setup.py`, `oauth_connector.py`, `ssl_setup.py`, `startup.py`, parts of `main.py`)
- `black --check` — clean (38 files unchanged)
- `flake8` — clean (zero warnings)
- `pylint` — **10.00/10** (only the pre-existing R0914 on `consolidate.py:418`, which the session pattern notes as expected)

**Live verification:**
Pending user-triggered re-run. Diff target: the next rts-source trace's page-2 findings and the Metrics table PPPT rows. Success criteria:
1. No finding with `metric_value` containing a parenthesized component list (e.g., `"Income (... m) before income taxes (... m) and PCL (... m)"`)
2. No `PPPT components — …` rows in the Metrics table
3. No invented `Income before income taxes | 6,400 | … | 7,400` row
4. The headline PPPT value (`$8.5B`, `+14% YoY`) from page 1 / investor-slides page 8 is still captured by the other findings
5. Ideally two re-runs showing identical "skip" behavior to validate F30 stability

**Related findings:**
- **F30** — cross-run instability on this same footnote. Rule 22 is designed to be the primary remedy. F30 re-validates after the next re-run; if two consecutive re-runs show identical skip behavior, F30 closes as `Implemented (via F04)`. If drift persists, F30 stays open for a tighter rule or a research.py temperature/sampling investigation.
- **F06 cluster** (F06/F07/F08/F26 — silent coverage drops). Rule 22 creates an *intentional* skip that the upcoming coverage-audit fix must not flag as a silent drop: a methodology footnote whose headline measure is already captured elsewhere is a designed skip, not a lost finding. When F06 lands, its audit logic needs to recognize rule-22 skips as non-droppable.
- **F01 R1 re-fix** — F04's rule 22 had a vocabulary collision with F01's rule 20 ("component values" / "calculation inputs" vs "each visible component"). The collision caused the LLM to skip CET1 waterfall bars in trace `20260407T131653Z_043b7c9f` because it reclassified them as "calculation inputs". F01's R1 re-fix added a narrowing clause to rule 22 scoping its prohibition to the specific "X is calculated as <m1> (<v1>) op <m2> (<v2>)..." footnote pattern, with explicit exclusion for waterfall bars. F04's core functionality (blocking the PPPT footnote garble) is preserved; see F01 Post-validation re-fix section for details.

---

#### Second post-validation re-fix (F04 regression, 2026-04-07)

**Symptom of the second regression:**
The R1 re-fix above added a carve-out clause to rule 22 worded as *"This prohibition applies ONLY to parenthetical component definitions inside non-GAAP calculation footnotes (pattern: 'X is calculated as measure_1 (value_1) operator measure_2 (value_2) ...')"*. In the subsequent cluster-boundary re-run (trace `u-debug/traces/20260407T135735Z_4197c116/`), the Metrics table surfaced three rows that F04 was specifically designed to prevent:

```
| PPPT definition inputs — income        | $MM | 5785 | rts [2] |
| PPPT definition inputs — income taxes  | $MM | 1622 | rts [2] |
| PPPT definition inputs — PCL           | $MM | 1090 | rts [2] |
```

And the Detail paragraph stated *"PPPT is defined as income before income taxes plus PCL; Q1 inputs were income of $5,785MM, income taxes of $1,622MM, and PCL of $1,090MM, consistent with the reported PPPT level"*. This is the exact baseline F04 failure mode, reintroduced through a different path.

**Root cause — two-layer interaction:**

Layer 1 (research stage). The R1 carve-out's `"applies ONLY to parenthetical"` wording was read by the LLM as narrowing rule 22 to strict parenthetical notation only. The rts p2 PPPT footnote is prose-form ("is calculated as income before income taxes and PCL; for Q1 2026 these inputs were: income $5,785MM, income taxes $1,622MM, PCL $1,090MM"), NOT strict parenthetical form. The LLM treated the prose form as outside rule 22's scope and produced a qualitative finding whose `finding` body recites the three inputs verbatim:
```
source_03 finding [1]: metric_name="", metric_value="",
finding: "Pre-provision, pre-tax (PPPT) earnings is calculated as income before
          income taxes and PCL; for Q1 2026 these inputs were: income $5,785MM,
          income taxes $1,622MM, PCL $1,090MM."
```

Layer 2 (consolidation stage). `consolidation.yaml` had NO rule preventing the consolidation LLM from synthesizing Metrics table rows by parsing numeric values out of a qualitative finding's narrative text. The consolidation LLM read the qualitative finding and split it into three structured Metrics rows, bypassing research-stage rule 22's intent entirely.

Neither layer alone is broken; the interaction is. R1's carve-out softened rule 22 enough to admit the qualitative-narrative variant, and consolidation had no backstop.

**Re-fix (two coordinated edits):**

1. **Research-stage rule 22 re-tightening** (`research.yaml`). Replaced the R1 "applies ONLY to parenthetical" carve-out with an "In addition" reinforcement that explicitly names the prose variant and prohibits qualitative-narrative recitation:
   > "In addition, this prohibition extends to prose variants of the same calculation-input pattern. The following formulations are all prohibited and must NOT be extracted as findings: (i) parenthetical form — 'X is calculated as measure_1 (value_1) operator measure_2 (value_2) ...'; (ii) prose form — 'X is calculated as A before B and C; for Q1 these inputs were A of $..., B of $..., C of $...' or any rewording that recites the component values inline as a sentence; (iii) qualitative findings whose finding-text body recites the calculation input values in narrative form, even when metric_name and metric_value are left empty — a qualitative finding is not a loophole for surfacing component values that rule 22 forbids as quantitative findings. Exception (rule 20): waterfall bars, bridge steps, and decomposition line items remain extractable per rule 20 — those are presentation of already-disaggregated data and MUST be extracted per rule 20, regardless of whether individual bar values appear inside parentheses (e.g., '(33) bps' for a negative CET1 movement component is standard accounting notation for a negative number, not a parenthetical calculation input)."

   The new wording reinforces the broad prohibition first (clauses i/ii/iii cover parenthetical, prose, and qualitative-narrative), then provides the rule 20 exception via "Exception (rule 20)" framing rather than narrowing-language. R1's waterfall protection is preserved — the `(33) bps` example and the "MUST be extracted per rule 20" clause stay verbatim.

2. **Consolidation-stage defense (new rule 6 clause (g))** (`consolidation.yaml`). Inserted a new clause (g) between existing clauses (f) and the closing "The only valid reason to omit a finding..." summary, under rule 6 Coverage commitment:
   > "(g) Do NOT synthesize Metrics table rows by parsing numeric values out of a qualitative finding's narrative text. Per clause (b), qualitative findings (those with empty metric_name/metric_value) belong in Detail prose. If a qualitative finding's body happens to recite dollar amounts, basis points, or other figures inline (e.g. a narrative that mentions 'income of $5,785MM, income taxes of $1,622MM, PCL of $1,090MM'), those numbers are NOT independent metrics — they are evidence inside the qualitative narrative and must remain in Detail prose only. Creating a Metrics row from extracted narrative numbers bypasses research-stage rule 22 (which prohibits calculation-input components as findings) and reintroduces forbidden component rows through the consolidation stage. The Metrics table is built ONLY from findings whose metric_name and metric_value fields are populated by the research stage."

   This is belt-and-braces: if rule 22 at research stage ever lets a qualitative-narrative finding through (variance, edge case, etc.), clause (g) prevents the consolidation stage from promoting it into structured Metrics rows.

**Files touched:**
- `u-retriever/src/retriever/stages/prompts/research.yaml` — rule 22 carve-out rewritten
- `u-retriever/src/retriever/stages/prompts/consolidation.yaml` — rule 6 new clause (g) inserted between (f) and closing summary
- `u-retriever/tests/test_research.py`:
  - `test_research_prompt_rule22_has_waterfall_carveout` — updated assertions to `"In addition, this prohibition extends to prose variants"`, `"Exception (rule 20)"`; kept `(33) bps`, `"waterfall bars, bridge steps"`, `"MUST be extracted per rule 20"`
  - New `test_research_prompt_rule22_bans_prose_calculation_inputs` — 5 phrase assertions on the prose/qualitative prohibition
- `u-retriever/tests/test_consolidate.py`:
  - `test_consolidation_prompt_has_coverage_commitment_rule` — extended with 3 new phrase assertions for clause (g)
  - New `test_consolidation_prompt_rule6g_preserves_qualitative_to_detail_routing` — 4 phrase assertions on the Detail-routing intent

**Quality gate:**
- `pytest --cov=src` — **260 passed** (was 257 after R2, +3 new tests), coverage **86%** unchanged
- `black --check` — clean
- `flake8` — clean (zero warnings)
- `pylint` — **10.00/10 with zero warnings**

**Live verification (pending next re-run):**
1. rts p2 research findings contain neither (a) a quantitative `PPPT definition inputs` row, nor (b) a qualitative finding whose body recites `income $5,785MM, income taxes $1,622MM, PCL $1,090MM` (or equivalent recitation)
2. Either a quantitative `metric_value="8,497"` finding exists for the rts p2 PPPT headline, or no p2-footnote finding at all
3. Metrics table has zero `PPPT definition inputs — *` rows
4. If any p2 footnote content appears, it is only in Detail prose with the [REF:2] citation, not in Metrics
5. R1 non-regression check: investor-slides p9 still produces the 5 `CET1 movement — *` findings with correct signs (-33, -26, -13, +79, +8) — the Exception (rule 20) clause must still work

---

### F05 — Capital Markets +$122M QoQ outlier — no analytic note

**Category:** Response analysis
**Severity:** Medium
**Status:** Implemented

**Symptom:**
The response reports Capital Markets impaired PCL grew +$122M QoQ while total impaired PCL grew only +$84M QoQ. Implication: non-CM segments collectively shrank ~$38M QoQ on impaired PCL — a real story (concentrated CM deterioration, releases elsewhere). The response presents both numbers but never makes the inference.

**Evidence:**
`outputs.consolidated_response`, Detail credit-losses paragraph:
> "Impaired loans (Stage 3) PCL was $1,068MM, or 0.40% of average loans, up 1 bp YoY and 2 bps QoQ; **total PCL on impaired loans increased from $984MM in Q4/25 to $1,068MM in Q1/26**, with **Capital Markets impaired PCL of $240MM (+$35MM YoY, +$122MM QoQ)** a notable contributor"

Math: total +$84M QoQ; CM alone +$122M QoQ; therefore other segments collectively −$38M QoQ.

**Hypothesized root cause:**
The consolidation prompt has no instruction to perform light variance attribution or to surface segment-vs-enterprise divergences. The model defaults to reporting numbers without inference. The research stage extracted both pieces correctly; consolidation just didn't connect them.

**Investigation notes:**

Data availability confirmed in baseline research findings (verified by grepping all three `source_0*_…json` traces for `impaired` × `segment != Enterprise`):
- **Enterprise impaired PCL delta is present:** investor-slides p19 finding `metric_name="Total RBC PCL on Impaired Loans"`, `metric_value="1,068 (Q1/26) vs 984 (Q4/25)"` (segment=Enterprise); also surfaces in the baseline Metrics table as `| RBC | Enterprise | PCL on impaired loans ($MM) | 1068 | 84 |  |` (columns are `Q1 2026 | QoQ change | YoY change`).
- **Capital Markets impaired PCL delta is present:** investor-slides p15 finding `metric_name="PCL on Impaired Assets"`, `metric_value="240"` (segment=Capital Markets); surfaces in the baseline Metrics table as `| RBC | Capital Markets | PCL on impaired assets ($MM) | 240 | 122 | 35 |`.
- **No other segment impaired PCL data was extracted.** Personal Banking, Commercial Banking, Wealth Management, Insurance, and Corporate Support all have zero segment-level impaired PCL findings. This is a hard constraint on any inference the consolidator can make: the residual must be expressed as "non-CM segments collectively", not broken down by named segment.

Both deltas ($84M enterprise, $122M CM) are sitting adjacent in the same Detail sentence (baseline `outputs.consolidated_response` line 39), and the arithmetic `$84M − $122M = −$38M` is an algebraic identity — an analyst reading that sentence would immediately note the implied release from non-CM segments. The consolidation LLM had the numbers and the scopes right but never completed the inference.

Existing consolidation rules do not cover this pattern:
- **Rule 7 (reconciliation check)** fires only when the LLM is presenting an *explicitly-labelled* decomposition ("whenever you present a component breakdown, movement walk, waterfall, or decomposition"). F05 is not labelled as a decomposition — it's two deltas in a narrative paragraph.
- **Rule 8 (scope comparability)** fires when scopes differ. F05's values share the same scope (impaired loans, Q1 2026 vs Q4 2025, both from investor-slides).

The gap is: no rule tells the LLM to make an arithmetic implication explicit when a segment delta sits next to an enterprise delta in Detail prose.

**Proposed fix:**

Add a new rule 9 to `consolidation.yaml`, inserted after rule 8, with these properties:
1. **Narrow trigger** — fires only when both enterprise-total and single-segment deltas for the same metric, same period, same scope are directly available in the findings.
2. **Two-branch condition** — triggers when (a) segment change is as large in magnitude as enterprise change (F05 case), OR (b) segment moves opposite to enterprise (covers the inverse case).
3. **Deterministic arithmetic only** — `enterprise Δ − segment Δ = residual`, no new information.
4. **Constrained attribution** — residual must be expressed as "non-[segment] segments collectively"; explicitly forbids attributing to specific other segments unless those segment-level values are in the findings.
5. **Anti-hallucination gating** — "do not invent either value"; only fires when both are directly available.
6. **Citation reuse** — residual clause cites the same `[REF:N]` that supported the two input values.
7. **Worked example with F05's exact numbers** embedded in the rule body, matching the F04 / F01 pattern of quoting the specific failure case verbatim for LLM consistency.

**Resolution:**

Implemented rule 9 "Segment-to-enterprise residual" in `u-retriever/src/retriever/stages/prompts/consolidation.yaml` (appended after rule 8, within the `user_prompt` folded-scalar block). Added `test_consolidation_prompt_has_segment_residual_rule` in `u-retriever/tests/test_consolidate.py` asserting the seven key phrases of the rule are present in the flattened prompt body (same YAML-whitespace normalization pattern used by F03 / F04 / scope-comparability tests). The worked-example numbers (`$84M`, `$122M`, `$38M`) are *not* asserted — those are illustration only, not load-bearing rule text.

**Files touched:**
- `u-retriever/src/retriever/stages/prompts/consolidation.yaml` — new rule 9
- `u-retriever/tests/test_consolidate.py` — new `test_consolidation_prompt_has_segment_residual_rule`

**Quality gate:**
- `pytest --cov=src` — **225 passed** (was 224), coverage **86%** (unchanged; same baseline gaps as prior findings)
- `black --check` — clean (38 files unchanged)
- `flake8` — clean (zero warnings)
- `pylint` — **10.00/10** (only the pre-existing R0914 on `consolidate.py:418`)

**Live verification:**
Pending cluster-boundary re-run (batched with F04 and F06 cluster per advance/pause decision option 3). Success criteria for the re-run's Detail credit-losses paragraph:
1. The paragraph still mentions Capital Markets impaired PCL as a contributor with its `+$122M QoQ` figure
2. The paragraph adds a clause stating the non-CM segments' implied collective change (≈ −$38M QoQ, with some phrasing tolerance like "implying non-Capital Markets segments collectively released ~$38M QoQ")
3. The inferred residual is mathematically `enterprise Δ − CM Δ`
4. The inference cites the same refs as the two input values (no fabricated `[REF:N]`)
5. No new per-segment impaired PCL numbers appear in the response that were not in the research findings (anti-hallucination check — only CM was extracted)
6. Rule does NOT fire inappropriately on segment deltas that are smaller than enterprise deltas in the same direction (e.g., PPPT segment breakdowns where each segment adds to total)

**Related findings:**
- **F06 / F07 / F08 / F26 (silent coverage drops cluster):** the F06 fix (coverage audit) must distinguish "finding present in research, absent in response" (actual silent drop) from "inference derived in response, no original finding" (rule 9 residual clause). The F06 cluster proposal will need to make this distinction explicit so the audit does not flag rule-9 output as a drop.
- **F24 (citation validation):** F24 currently only checks that `[REF:N]` identifiers exist; it does not check factual support. Rule 9 asks the LLM to cite refs that support the *input* values of the inference, not a direct ref for the residual itself. A future semantic citation check would need to recognize this as a valid citation pattern.

---

### F06 — Adjusted PPPT 12% YoY finding dropped from response

**Category:** Coverage / dropped findings
**Severity:** Medium
**Status:** Implemented (cluster lead — fix covers F07, F08, F26 + F03 side-effect)

**Symptom:**
The investor-slides research stage extracted a quantitative finding for Adjusted PPPT (+12% YoY) at REF:1, but the consolidated response only mentions reported PPPT (+14% YoY). The adjusted figure and the 200-bp gap between reported and adjusted are entirely absent from the response.

**Evidence:**
`source_01_…investor-slides_38.json` → REF:1 contains two findings:
```json
{"metric_name": "Pre-Provision, Pre-Tax Earnings(2)", "metric_value": "8,497", ...},
{"metric_name": "Adjusted PPPT(2) up YoY", "metric_value": "12%", ...}
```

`outputs.consolidated_response` Summary + Detail mention `+14% YoY` and `8,497MM` but never mention adjusted PPPT.

**Hypothesized root cause:**
`consolidation.yaml` rule 6 says *"Include every numeric value from the reference index that answers any part of the query. Do not drop values for brevity."* — this rule was violated. The consolidation LLM judged Adjusted PPPT not directly relevant to "earnings strength" and silently dropped it. There is no audit step to catch dropped findings.

**Related findings:** F07, F08, F26 (same root cause: silent finding drops).

**Additional coverage loss logged from F03 re-run (trace `20260407T041015Z_2dc7e8e0`):**

After F03's scope-comparability rule was added, the consolidation LLM started dropping investor-slides REF:9 loans-only ACL data entirely (`$7,752MM` total loans ACL, `$293MM` QoQ, `$5.5B` performing ACL, plus the CNB release qualitative color) rather than presenting it with a scope qualifier. This is the same "silent drop" failure mode as F06/F07/F08/F26 but induced by F03's "be careful about scope" instruction causing LLM retreat from a topic. The F06 cluster fix (coverage audit) MUST catch this case: a finding with a clear scope qualifier in `metric_name` should never be silently dropped. See F03 Resolution → "Known side effect" for full details.

**Investigation notes:**

The cluster has TWO distinct drop modes confirmed by direct inspection of the baseline trace:

| ID | Layer | Ref in baseline | Drop mechanism |
|---|---|---|---|
| F06 — Adjusted PPPT 12% YoY | **Layer-2** | REF:1 cited (reported PPPT) | Adjusted-variant finding within cited ref silently skipped |
| F07 — Net write-offs $634M | **Layer-1** | REF:24 never cited | Whole ref dropped from response |
| F07 — Total impaired exposures $9,294M | **Layer-1** | REF:23 never cited | Whole ref dropped from response |
| F08 — CNB qualitative release color | **Layer-2** | REF:9 cited (for $5.5B ACL claim) | Qualitative sibling finding within cited ref skipped |
| F26 — ACL to loans ratio 73 bps | **Layer-2** | REF:5 cited (for other p4 content) | Borderline-topic finding within cited ref skipped |
| F03 side-effect — $7,752MM loans ACL etc. | **Layer-2 (retreat-induced)** | REF:9 cited (post-F03 trace) | Rule 8 scope-comparability triggered LLM retreat from the ACL topic |

**Verification of cited refs against the baseline:** rendered cite-anchor regex extraction yields cited set `{1..11, 13..21, 25..27}` (23 refs). Uncited set `{12, 22, 23, 24}` (4 refs) — exactly matches the trace header's "gaps at 12, 22, 23, 24". REF:12 is F04's garbled PPPT calc footnote (now blocked at research stage by rule 22). REF:23 / REF:24 are F07. REF:22 is a separate baseline drop unrelated to the cluster.

**Verification of layer-2 token absence in baseline response** (literal substring search on `outputs.consolidated_response`):
```
'12%'           0 hits   (F06 Adjusted PPPT)
'73 bps'        0 hits   (F26 ACL to loans ratio)
'City National' 0 hits   (F08 CNB qualitative)
'Adjusted'      0 hits   (F06 adjusted PPPT, any form)
'634'           0 hits   (F07 net write-offs)
'9,294' / '9294' 0 hits  (F07 impaired exposures)
```

All zero. The drops are unambiguously silent.

**Verification of finding presence in research output** (the data was extracted, just not used downstream):
- `source_01 p8`: `metric_name='Adjusted PPPT(2) up YoY'`, `metric_value='12%'` ✓
- `source_01 p4`: `metric_name='ACL to loans ratio'`, `metric_value='73 bps (+2 bps QoQ)'` ✓
- `source_01 p18`: qualitative `finding='We took $28MM of provisions … release in City National Bank.'` ✓
- `source_02 p16`: `metric_name='Total net write-offs'`, `metric_value='634'` ✓
- `source_02 p17`: `metric_name='Total impaired exposures'`, `metric_value='9294 (Allowance 2238)'` ✓

**Why the existing defences fail:**

1. **Rule 6 is a floor without a check.** Original rule 6 ("Include every numeric value… Do not drop values for brevity") is the LLM grading its own compliance. F06 is direct proof that this self-grading fails — the same model that judged Adjusted PPPT not-worth-including is the one supposed to be enforcing rule 6.

2. **No programmatic audit existed.** `consolidate.py` already had the data needed for layer-1 detection — the streaming `_create_ref_replacer` tracks `used_refs` at line 148 — but the stage never computed `ref_entries − used_refs` or acted on it.

3. **Layer-2 needs finding-level detection.** Layer-1 alone only catches F07. F06/F08/F26 require checking each individual finding inside cited refs against the rendered response text.

4. **F03 ACL retreat is a third pattern**: a "be careful about X" rule (rule 8 scope comparability) caused the LLM to drop the entire ACL topic from REF:9 rather than navigate the scope distinction. Rule 6 needs an explicit anti-retreat clause that points the LLM to the safe escape (rule 4 scope-suffix split).

**Proposed fix (three-mechanism unified cluster fix):**

**Mechanism A — Replace consolidation.yaml rule 6** with an expanded "Coverage commitment" rule with six sub-clauses:
- (a) Quantitative findings: include in Metrics or Detail; conciseness ≠ fewer data points
- (b) Qualitative findings: lift into Detail prose with the CNB-release example quoted verbatim
- (c) Adjusted-vs-reported pairs: both must appear; the gap is itself newsworthy (worked example: reported PPPT +14% vs adjusted PPPT +12%)
- (d) Distinct findings within a single ref: each must be represented independently (worked example: REF:1 holding both reported PPPT and Adjusted PPPT(2))
- (e) Scope-qualified findings: anti-retreat — use scope-qualified rows per rule 4, NOT drop, when rule 8 seems to conflict
- (f) Rule 9 derived inferences carve-out — they don't need their own finding

The rule explicitly tells the LLM "before finalizing, mentally walk each [REF:N] entry and verify EVERY finding listed under that ref is represented somewhere".

**Mechanism B — Programmatic layer-1 audit in `consolidate.py`.** New helper `_compute_coverage_audit(ref_entries, used_refs, response_text)` returns refs that were defined but never cited. Output is rendered into a new `## Coverage audit` section (with `### Uncited refs` subsection) appended to `consolidated_response` after the LLM stream completes, before the References appendix. Streaming is preserved — the audit section is emitted via the same `on_chunk` callback used by the appendix.

**Mechanism C — Programmatic layer-2 audit in `consolidate.py`.** Same `_compute_coverage_audit` helper extracts numeric tokens from each finding's `metric_value` (regex `\d+(?:\.\d+)?(?:%|\s*bps?)?`, with thousands separators stripped from both finding and response text), then substring-searches for any of the finding's tokens in the normalized response text. Findings whose tokens are all absent are surfaced under `### Unincorporated findings (ref cited but value absent)`. Qualitative findings (empty metric_value) are not checked at layer 2 — they rely on mechanism A's clause (b).

**Trace telemetry — three new `NotRequired` fields on `ConsolidatedResult`:**
- `coverage_audit: str` — the rendered audit section text
- `uncited_ref_ids: list[int]` — layer-1 results
- `unincorporated_findings: list[dict]` — layer-2 results (each entry has ref_id, source, page, location_detail, metric_name, metric_value, finding)

The stage logger was extended to surface `uncited_refs` and `unincorporated` counts in the completion message so the debug UI can track silent-drop frequency over time.

**Refactoring done as part of the cluster fix:**

`consolidate_results` was already at pylint R0914 21/20 (pre-existing, expected per session notes). Adding audit code would have pushed it further over. To keep the function clean and to avoid the kind of suppression the project forbids, the post-LLM processing (dedup → audit → appendix → parse → key_findings) was factored into a new helper `_process_consolidation_response`. This dropped `consolidate_results` from 21 locals to ~16, eliminating the pre-existing R0914 entirely. **The consolidate.py module now passes pylint at 10.00/10 with zero warnings of any kind.**

A second incidental fix: `_SECTION_HEADING_RE` was changed from `r"^##\s+(\S+)"` to `r"^##\s+(\S+)[^\n]*\n"` because the old regex captured only the first whitespace-bounded word and did not consume the rest of the heading line. Existing single-word headings (`Summary`, `Metrics`, `Detail`, `Gaps`) were coincidentally safe; the new multi-word `## Coverage audit` heading exposed the bug. Found by a structural test (`test_parse_sections_recognizes_coverage_audit`). This is a latent bug fix that improves robustness for any future multi-word section heading.

**Resolution:**

Implemented all three mechanisms in a single cluster pass.

**Files touched:**
- `u-retriever/src/retriever/stages/prompts/consolidation.yaml` — replaced rule 6 with expanded "Coverage commitment" rule with sub-clauses (a)–(f)
- `u-retriever/src/retriever/stages/consolidate.py`:
  - New helpers: `_extract_metric_value_tokens`, `_finding_token_in_text`, `_compute_coverage_audit`, `_build_coverage_audit_section`, `_process_consolidation_response`
  - Updated `_SECTION_KEYS` to register `Coverage` → `coverage_audit`
  - Updated `_parse_sections` to include `coverage_audit` in result dict
  - Updated `_SECTION_HEADING_RE` to consume the full heading line (latent bug fix for multi-word headings)
  - Refactored `consolidate_results` to call `_process_consolidation_response`, eliminating the pre-existing R0914 and reducing locals from 21 to 16
  - Added new audit field population in the `ConsolidatedResult` return
  - Extended the completion `logger.info` line to surface `uncited_refs` and `unincorporated` counts
- `u-retriever/src/retriever/models.py` — added three new `NotRequired` fields to `ConsolidatedResult`: `coverage_audit`, `uncited_ref_ids`, `unincorporated_findings`
- `u-retriever/tests/test_consolidate.py` — 20 new tests:
  - `test_consolidation_prompt_has_coverage_commitment_rule`
  - `test_extract_metric_value_tokens_basic` / `_strips_separators` / `_empty`
  - `test_finding_token_in_text_qualitative_skipped` / `_present` / `_absent` / `_rounded_match` / `_thousands_separator_match`
  - `test_compute_coverage_audit_layer1_uncited` / `_layer2_unincorporated` / `_qualitative_not_flagged` / `_clean`
  - `test_build_coverage_audit_section_empty` / `_layer1_only` / `_layer2_only`
  - `test_consolidate_audit_surfaces_dropped_finding` / `_streams_to_callback` / `_clean_run_has_no_section`
  - `test_parse_sections_recognizes_coverage_audit`

**Quality gate:**
- `pytest --cov=src` — **245 passed** (was 225, +20 new tests for the cluster), coverage **86%** unchanged at total; `consolidate.py` improved from **97% → 98%** (only pre-existing gaps remain)
- `black --check` — clean (38 files unchanged)
- `flake8` — clean (zero warnings)
- `pylint` — **10.00/10 with ZERO warnings**. The pre-existing R0914 on `consolidate.py:418` has been **eliminated** by the `_process_consolidation_response` refactor. This is the first session in which the project's pylint baseline is fully clean rather than carrying the R0914 caveat.

**Live verification (pending re-run, F04 + F05 + F06 cluster boundary):**

User re-runs the same RBC Q1 2026 query. I diff the new trace against `20260407T025300Z_a2ef222f`:

*Mechanism A validation (prompt rule prevented the drop):*
1. `Adjusted PPPT` or literal `12%` for adjusted PPPT — present in response Summary or Detail
2. `73 bps` (ACL to loans ratio) — present in response
3. `City National` or `CNB` — present in Detail credit-losses paragraph
4. `634` (net write-offs) — present in Metrics or Detail
5. `9,294` (impaired exposures) — present in Metrics or Detail
6. `$7,752MM` loans-only ACL (F03 side-effect retreat) — present with scope qualifier per rule 4

*Mechanism B/C validation (audit deterministic backstop):*
7. New `coverage_audit` and `uncited_ref_ids` fields are populated on the serialized trace
8. If any cluster finding is still missed after mechanism A, it appears in the rendered `## Coverage audit` section
9. False-positive scan: no known-cited finding shows up under `### Unincorporated findings`

*Cross-cluster revalidation (same re-run):*
10. F04: no `Income (... m) before income taxes (... m) and PCL (... m)` formula in rts p2 research findings; no `PPPT components —` rows in the Metrics table
11. F05: Detail credit-losses paragraph contains a residual clause mentioning `non-Capital Markets segments collectively` with the correct ~$38M arithmetic
12. F01 / F02 / F03 stable: CET1 bridge rule 20 still fires; multi-value cells handled per the F02 source-split; scope comparability rule 8 still blocks unsafe juxtaposition
13. F30 stability: PPPT footnote still skipped at research stage (rule 22 unchanged)

**Related findings (cluster members):**
- **F07** — Implemented (via F06). Net write-offs / impaired exposures are caught by mechanism B (layer-1 audit) deterministically; mechanism A (rule 6 sub-clause d about distinct findings within a ref) and the explicit "narrow reading of credit losses" framing should also recover them via the prompt rule.
- **F08** — Implemented (via F06). The CNB qualitative release color is caught by mechanism A clause (b) which quotes the exact CNB-release sentence verbatim as a worked example. Layer-2 audit cannot catch qualitative drops directly; clause (b) is the primary mechanism for F08.
- **F26** — Implemented (via F06). The ACL to loans ratio is caught by mechanism C (layer-2 token check on `73 bps`) and by mechanism A clause (a).
- **F03 side-effect** — Implemented (via F06). The retreat-induced ACL drops are caught by mechanism A clause (e) (anti-retreat scope-qualified rows) and mechanism C (layer-2 token check on `7,752`, `293`, `5.5`).
- **F04** — F04's research rule 22 filters methodology footnotes at research stage so they never enter the ref index. Mechanism B never sees them as audit false positives. Documented as expected behavior.
- **F05** — F05's rule 9 generates derived residual clauses that cite the same refs as their input values. Mechanism B (which checks ref-level citations) never flags rule-9 output. Mechanism C (which checks finding tokens) never flags rule-9 output because rule-9 output doesn't correspond to any source finding. Documented in clause (f) of the new rule 6.

**Notes for future sessions:**
- The audit only flags layer-2 drops on quantitative findings. Qualitative drops are still possible if the prompt rule fails. If the re-run shows qualitative drops that mechanism A didn't catch, a follow-up pass could add LLM-based finding-presence checking — this is intentionally out of scope for this cluster. **Update (R2):** the first cluster-boundary re-run (trace `20260407T131653Z_043b7c9f`) did show a qualitative drop: the F08 CNB-release sentence was dropped at the research stage (not consolidation). The R2 re-fix addresses this upstream by adding research.yaml rule 23 requiring each narrative bullet to become its own finding. See F08 Post-validation re-fix section for the full details. The F06 layer-2 limitation still stands — R2 reduces its impact by making the research-stage drops less likely rather than adding an LLM-based presence check at consolidation.
- The `_SECTION_HEADING_RE` regex change is a latent bug fix. Any future multi-word section heading is now safe.
- The `_process_consolidation_response` helper is reusable for any future post-LLM processing additions, keeping `consolidate_results` below the locals threshold permanently.

---

### F07 — Net write-offs and impaired exposures dropped

**Category:** Coverage / dropped findings
**Severity:** Medium-High
**Status:** Implemented (via F06)

**Symptom:**
Two query-relevant pillar3 findings are in the catalog but never cited in the response:
- REF:24 (catalog) — pillar3 p16 CRB_f_b — net write-offs `$634M` Q1 2026
- REF:23 (catalog) — pillar3 p17 CRB_f_c — total impaired exposures `$9,294M` with allowance `$2,238M`

Both are direct credit-loss data points. Neither appears in the rendered response.

**Evidence:**
`source_02_…pillar3_40.json` accumulator findings:
```json
{"metric_name": "Total net write-offs", "metric_value": "634", "page": 16, "location_detail": "CRB_f_b: ..."},
{"metric_name": "Total impaired exposures", "metric_value": "9294 (Allowance 2238)", "page": 17, "location_detail": "CRB_f_c: ..."}
```

Both refs are missing from the rendered References table — confirming they were never cited.

**Hypothesized root cause:**
Same as F06. Plus a narrow interpretation of "credit losses" as PCL-only rather than the broader credit-quality envelope (PCL + ACL + write-offs + impaired exposures).

**Related findings:** F06, F08, F26.

**Investigation notes:**

This is the **layer-1** member of the cluster: REF:23 and REF:24 are absent from the baseline cited-ref set (`{1..11, 13..21, 25..27}`). The pillar3 source extracted these findings successfully across all three iterations — they're firmly in the research output. They simply never made it into the consolidator's response.

The pattern matches F06's cluster diagnosis: the consolidation LLM made a relevance judgement (write-offs / impaired exposures aren't directly "credit losses" in the LLM's narrow read of the query) and silently dropped both refs without flagging them in Gaps. Of the four cluster findings, F07 is the most clear-cut layer-1 case — the entire ref is dropped, not just a finding within it.

**Resolution:**

Resolved by the F06 cluster fix. F07 specifically benefits from:

- **Mechanism B (layer-1 audit)** in `consolidate.py` — `_compute_coverage_audit` deterministically detects refs in `ref_entries` that are not in `used_refs`. REF:23 and REF:24 will be surfaced in the new `## Coverage audit` section under `### Uncited refs (never appeared in response)` if the LLM still drops them post-fix.
- **Mechanism A (rule 6 replacement)** — the new "Coverage commitment" rule explicitly tells the LLM to walk every [REF:N] entry and verify each finding is represented somewhere. The expanded framing of "credit losses" as a topic should also include write-offs / impaired exposures via the broader query interpretation.

**Live verification (pending re-run):**
- Either: the response now cites REF:23 / REF:24 in Detail or Metrics with `634` and `9,294` mentioned (mechanism A succeeded)
- Or: the `## Coverage audit` section lists `[REF:23] pillar3 p17 — CRB_f_c …` and `[REF:24] pillar3 p16 — CRB_f_b …` (mechanism B caught the residual drop)
- Either outcome closes F07.

**Files touched:** see F06 Resolution. F07 has no F07-specific files — it shares the cluster fix.

**Quality gate:** see F06 Resolution. 245 tests pass, pylint 10.00/10 with zero warnings.

---

### F08 — Qualitative segment color (CNB release) underused

**Category:** Coverage / dropped findings
**Severity:** Medium
**Status:** Implemented (via F06)

**Symptom:**
Useful qualitative breakdown is in the catalog but never lifted into the response prose.

**Evidence:**
`source_01_…investor-slides_38.json` REF:9 second finding (qualitative — empty metric fields):
```json
{
  "finding": "We took $28MM of provisions on performing loans this quarter, mainly in Personal Banking, Capital Markets and Commercial Banking, partially offset by a release in City National Bank.",
  "page": 18, "location_detail": "Allowance for Credit Losses: ..."
}
```

REF:9 IS cited in the response (for the `$5.5B performing ACL down $2MM QoQ` claim), but the segment color about CNB release is never used.

**Hypothesized root cause:**
Same as F06/F07. Plus: the consolidation LLM is biased toward quantitative findings and treats qualitative text as background context only. No rule requires incorporating qualitative findings into Detail.

**Related findings:** F06, F07, F26.

**Investigation notes:**

This is a **layer-2 qualitative drop** — REF:9 is cited (for the `$5.5B` performing ACL claim), but the qualitative sibling finding inside the same ref is never lifted into prose. Verified absent from baseline response: literal substrings `City National`, `CNB`, `Personal Banking, Capital Markets`, and `release` (in this combination) are all zero hits.

Layer-2 quantitative audit (mechanism C in the F06 fix) **cannot catch this** — the finding has empty `metric_value`, so `_extract_metric_value_tokens` returns an empty list and `_finding_token_in_text` returns True (treats as represented). Qualitative findings are intentionally exempt from layer-2 token checking because there's no reliable substring to search for.

F08 therefore relies exclusively on **mechanism A** (the new rule 6 replacement). Specifically clause (b):

> Qualitative findings (empty metric_name/value): lift the factual content into Detail prose. Do NOT treat qualitative findings as background context to be skipped. Example: a segment-color finding like "provisions mainly in Personal Banking, Capital Markets and Commercial Banking, partially offset by a release in City National Bank" must be surfaced in the Detail credit-losses paragraph as written.

The CNB-release sentence is quoted **verbatim** in the rule body as the worked example. Following the F04 / F05 pattern of anchoring rules to concrete failure cases, this should give the LLM a direct match against the exact F08 finding when it processes the consolidation prompt.

**Resolution:**

Resolved by the F06 cluster fix, mechanism A clause (b). No layer-2 audit coverage for qualitative findings — that's an intentional gap, documented in the cluster Resolution as a known limitation.

**Live verification (pending re-run):**
- The Detail credit-losses paragraph should mention `City National Bank` (or `CNB`) and the segment-mix language ("mainly in Personal Banking, Capital Markets and Commercial Banking")
- If the LLM still drops it, mechanism C will NOT catch it (qualitative drops are not layer-2-auditable). This is a known limitation and would justify a future LLM-based finding-presence check if the prompt rule alone is insufficient.

**Files touched:** see F06 Resolution. F08 shares the cluster fix.

**Quality gate:** see F06 Resolution.

---

#### Post-validation re-fix (R2 — 2026-04-07 cluster-boundary re-run)

**Symptom of the regression:**
In the F04/F05/F06/F09/F10 cluster-boundary re-run (trace `20260407T131653Z_043b7c9f`), the CNB-release sentence was entirely absent from the research-stage output. The p18 chunk content still contained the exact sentence verbatim ("We took $28MM of provisions on performing loans this quarter, mainly in Personal Banking, Capital Markets and Commercial Banking, partially offset by a release in City National Bank"), but `source_01 ... investor-slides_38.json → stages.research.findings` contained only a single paraphrased qualitative finding merging the parent bullet and its sub-bullet:

> "PCL on performing loans of $28MM in Q1/2026 was driven by unfavourable credit quality and portfolio growth, partly offset by favourable changes to macroeconomic forecast."

The F06 cluster's mechanism A clause (b) embeds the CNB sentence verbatim in `consolidation.yaml` as the worked example, but that clause can only help if the sentence reaches `ref_entries`. When research drops it at extraction, consolidation never sees it.

**Root cause:**
Hierarchical-bullet collapse. The investor-slides p18 chunk has a main bullet (segment composition, including CNB) and a sub-bullet (driver explanation). The LLM collapsed them into a single "summary" finding that kept the headline `$28MM` and the driver text from the sub-bullet but dropped the segment-specific language from the main bullet (`Personal Banking, Capital Markets and Commercial Banking`, `City National Bank`).

None of the research rules added between baseline and this re-run directly target the multi-bullet pattern. Unlike R1's rule 22 vs rule 20 vocabulary collision, there is no specific rule interaction causing this — the gap is that research.yaml has no explicit rule requiring each bullet in a narrative structure to become its own finding. The closest existing rules are:
- Rule 1 ("one discrete fact per finding") — schema rule, not extraction-completeness
- Rule 10 ("extract it as a qualitative finding" — singular "a" invites the LLM to pick one representative)
- Rule 16 ("extract exhaustively: every relevant value, comparison, and breakdown") — quantitative vocabulary, not narrative

The LLM's behavior was defensible under the existing prompt: it saw two bullets, judged the sub-bullet as canonical, produced one summary finding. The fix must anchor extraction behavior to the concrete bullet pattern instead of hoping LLM judgment navigates narrative-hierarchy ambiguity correctly.

Baseline `20260407T025300Z_a2ef222f` had the CNB sentence extracted, which is why F08 was originally raised at the consolidation layer. Between baseline and this re-run, LLM behavior drifted — likely variance amplified by the absence of an explicit rule. The re-fix stabilizes extraction via pattern-based guidance rather than relying on judgment.

**Re-fix:**

Added new **rule 23** to `u-retriever/src/retriever/stages/prompts/research.yaml`, placed after rule 22 and before the closing "Reminder" block. The rule:

1. Names the exact pattern (multiple narrative bullets including main + sub-bullets conveying distinct factual statements)
2. Requires each distinct statement as its own qualitative finding
3. Explicitly prohibits summarizing, paraphrasing into synthesis, or picking one as representative
4. Distinguishes WHO/WHERE/WHICH-SEGMENTS bullets from WHY/HOW-CAUSED bullets and requires both as separate findings
5. Embeds the exact F08 failure case verbatim (both bullets quoted in full) as a worked example
6. Explicitly says the required output is TWO findings, not one — the main bullet (with CNB + segment names) AND the sub-bullet (with driver analysis)
7. Explicitly says `Do NOT merge them into a synthesized "PCL on performing loans was driven by X, partly offset by Y" finding that drops the segment-specific content` — naming the exact anti-pattern the LLM was exhibiting
8. Explicitly names proper nouns (`"City National Bank"`) and segment names as content that must be preserved verbatim, `because downstream stages may cite them directly`
9. Scopes the rule to bullet structures, numbered lists, parallel prose paragraphs in commentary sections, and callout box bullets adjacent to figures — the presentational patterns where the collapse failure mode occurs

**Files touched:**
- `u-retriever/src/retriever/stages/prompts/research.yaml` — new rule 23 (multi-bullet narrative extraction)
- `u-retriever/tests/test_research.py` — new `test_research_prompt_has_multi_bullet_extraction_rule` asserting 13 key phrases including the verbatim CNB sentence, the prohibitive `Do NOT` clauses, and the anti-synthesis language

**Quality gate:**
- `pytest --cov=src` — **257 passed** (was 256 after R1, +1 new test for R2), coverage **86%** unchanged
- `black --check` — clean after one auto-reformat (test line-length wrap)
- `flake8` — clean (zero warnings)
- `pylint` — **10.00/10 with zero warnings**

**Cross-rule composition verification:**
- **Rule 20** (F01 + R1 re-fix): uses "bar, line item" vocabulary (visual figure patterns); rule 23 uses "bullet" vocabulary (text narrative patterns). No shared vocabulary, no collision.
- **Rule 22** (F04): uses "parenthetical component definitions" vocabulary (non-GAAP footnote patterns); rule 23 uses "narrative bullets" vocabulary. No collision.
- **Rule 21** (F03): uses "scope qualifier" vocabulary (quantitative metric_name fields); rule 23 is about qualitative `finding` text fields. No collision.
- **Rule 10** (management commentary): rule 23 is a more specific extraction-completeness rule for the bulleted sub-case. Rule 10 still applies at the topic level ("commentary relevant to the query topic"); rule 23 adds the structural requirement ("if that commentary is bulleted, extract each bullet"). Clean composition.

**Live verification (pending next re-run):**
1. Investor-slides p18 research findings contain **at least 2 qualitative findings** on the ACL topic
2. At least one of those findings has the literal substring `"City National Bank"` or `"CNB"` in its `finding` field
3. At least one of those findings has the literal substring `"Personal Banking, Capital Markets and Commercial Banking"`
4. Consolidated response's Detail credit-losses paragraph mentions `"City National Bank"` (or `"CNB"`) and the segment-mix language
5. No regression on F01 R1 re-fix: investor-slides p9 still produces the 5 CET1 waterfall bar findings
6. No regression on F04: rts p2 methodology footnote still extracts cleanly as `metric_value="8,497"` with no formula garble

**Related findings:**
- **F06 cluster "Known limitations" note** — the F06 Resolution documented that layer-2 audit cannot catch qualitative drops because `_extract_metric_value_tokens` returns empty lists for qualitative findings. Rule 23 addresses this at the research stage rather than adding LLM-based finding-presence checking at consolidation. The F06 layer-2 limitation remains, but R2 makes it much less likely to bite for narrative-bullet content.
- **R1 (F01 re-fix)** — R1 and R2 share the same anchoring strategy (verbatim worked example + explicit anti-pattern language). R1 handles bar-level component drops, R2 handles narrative-bullet drops. Together they cover the two primary "LLM collapse" failure modes observed in the re-run trace.
- **Hierarchical-bullet collapse broader scope** — bullet 1 of investor-slides p18 (`"Total ACL on loans and acceptances increased $293MM QoQ. ACL on performing loans of $5.5BN was down $2MM"`) also contains quantitative values (`$293MM`, `$5.5BN`, `$2MM`) that are missing from the structured research findings in this re-run. These are a separate quantitative collapse, not addressed by R2. If they persist in future re-runs, a future finding (R3?) could extend rule 23 to require quantitative extraction from bulleted narrative values as well. Scope-disciplined out for now.

---

#### Second post-validation re-fix (R2 structural rewrite, 2026-04-07)

**Symptom of the second regression:**
In the cluster-boundary re-run (trace `u-debug/traces/20260407T135735Z_4197c116/`), investor-slides p18 was retrieved (listed in `final_chunks`) but produced **ZERO findings**. Investor-slides p19 (PCL on Impaired Loans slide) also retrieved but zero findings. Previous run (`20260407T131653Z_043b7c9f`) had at least 1 paraphrased qualitative finding from each page. Global search across all three source traces returned zero hits on `"City National"` or `"CNB"`. The rule 23 addition made things worse than before — rather than collapsing the bullet hierarchy, the LLM now skipped both p18 and p19 entirely.

**Root cause — verbatim-source example confusion:**

The original rule 23 (first R2 attempt) contained a "Worked example" section that quoted the investor-slides p18 CNB bullets verbatim:
```
"We took $28MM of provisions on performing loans this quarter, mainly in
 Personal Banking, Capital Markets and Commercial Banking, partially offset
 by a release in City National Bank"
```
plus:
```
"PCL on performing loans was driven by unfavourable credit quality and
 portfolio growth, partly offset by favourable changes to our
 macroeconomic forecast"
```

When the research LLM processed the prompt, it encountered these exact sentences in TWO places: (i) as instructional content inside rule 23's worked example, and (ii) as actual source content when reading the p18 chunk. The LLM appears to have conflated the two — treating the chunk content as "already covered by the rule's example" and skipping extraction. This is a **verbatim-source example conflation failure mode** that doesn't affect R1 (where rule 20's worked example quotes a figure structure with bar labels, not narrative source content that also appears in chunks).

Contrast with R1 (rule 20): R1's worked example describes a CET1 waterfall FIGURE structure with placeholder metric_names like `"CET1 movement — Net Income"`. The LLM doesn't see these labels in the chunk raw_content as narrative text — the chunk has a figure description with the bar values. So there's no conflation risk.

**Re-fix (placeholder-based structural pattern):**

Rewrote rule 23's Worked example block to mirror R1's shape — strip every literal source token and replace with placeholder-based structural pattern description:

> "Worked example — pattern recognition, not source quotation. Suppose a slide carries a main bullet of the form 'We took <action> in <list of business units>, partially offset by <opposite action> in <specific division>' together with a sub-bullet of the form '<metric> was driven by <cause_1> and <cause_2>, partly offset by <cause_3>'. These are TWO findings, not one: (1) a qualitative finding whose finding field quotes the main bullet verbatim — preserving every proper noun, segment label, and division name exactly as management wrote them; (2) a qualitative finding whose finding field quotes the sub-bullet verbatim — preserving the driver clauses in management's order. Do NOT merge them into a synthesized '<metric> was driven by X, partly offset by Y' finding that drops the segment-specific content from the main bullet. Do NOT rewrite the source language — preserve management's own wording, including proper nouns for specific divisions and segment names, because downstream stages may cite them directly. The placeholders <action>, <list of business units>, <specific division>, <cause_1>, <cause_2>, <cause_3> are pattern slots — at extraction time you fill them from the actual slide text, you do not search for slides that match the placeholders literally. Rule applies to bullet structures, numbered lists, parallel prose paragraphs in commentary sections, and callout box bullets adjacent to figures."

Key changes from the first R2 attempt:
- **Removed** literal strings `"City National Bank"`, `"Personal Banking, Capital Markets and Commercial Banking"`, `"We took $28MM of provisions..."`, `"PCL on performing loans was driven by..."`
- **Added** angle-bracket placeholders (`<action>`, `<list of business units>`, `<specific division>`, `<cause_1>`, `<cause_2>`, `<cause_3>`)
- **Added** explicit "pattern recognition, not source quotation" label at the start
- **Added** explicit "placeholders... are pattern slots — at extraction time you fill them from the actual slide text, you do not search for slides that match the placeholders literally" clause to pre-empt the literal-template-hunt failure mode
- **Kept** all anti-synthesis language from the first attempt (TWO findings not one, Do NOT merge, Do NOT rewrite the source language, preserve management's own wording)

**Files touched:**
- `u-retriever/src/retriever/stages/prompts/research.yaml` — rule 23 worked example rewritten (lines 241-267 of the rule body)
- `u-retriever/tests/test_research.py`:
  - `test_research_prompt_has_multi_bullet_extraction_rule` — removed 3 literal assertions (CNB sentence, segment names, `"City National Bank"` quoted string); added 7 new assertions for the placeholder-based structural pattern (`"pattern recognition, not source quotation"`, `<list of business units>`, `<specific division>`, `<cause_1>`, `placeholders`, `pattern slots`, segment-specific content clause)

The note about the consolidation.yaml CNB verbatim in rule 6(b) (lines 227-231) still applies: that occurrence stays. Consolidation rule 6(b) is a worked example of what the consolidator should do with a qualitative finding when it reaches `ref_entries`, not a research-stage instruction. It does not have the example/source conflation risk because consolidation doesn't re-read chunks.

**Quality gate:**
- `pytest --cov=src` — **260 passed** (same count as F04 re-fix; R2 test assertions changed but the test still exists), coverage **86%** unchanged
- `black --check` — clean
- `flake8` — clean (zero warnings)
- `pylint` — **10.00/10 with zero warnings**

**Live verification (pending next re-run) — primary acceptance criteria:**
1. Investor-slides p18 research findings contain **at least 1 qualitative finding**
2. Investor-slides p19 research findings contain **at least 1 qualitative finding**
3. At least one p18 finding's `finding` field contains the literal string `"City National"` — this verifies the LLM lifted actual source text rather than the placeholder template
4. Consolidated response Detail credit-losses paragraph mentions `"City National Bank"` (or `"CNB"`)
5. No regression on R1: p9 still produces 5 `CET1 movement — *` findings

**User-confirmed fallback (revert-and-defer):** if the primary acceptance criteria (1) or (3) fail in the next re-run — i.e., p18 still produces zero findings, or produces a finding that does not contain the literal CNB text — the user-approved fallback is to:
1. Delete the entire rule 23 block from `research.yaml` (restoring the file to pre-R2 state)
2. Delete `test_research_prompt_has_multi_bullet_extraction_rule` from `test_research.py`
3. Re-run quality gate (expected 259 passed — the 3 F04 defense tests stay, R2 test removed)
4. Flip F08 index status to `Deferred — open, research-stage bullet extraction unresolved`
5. No second re-run required for the revert — it restores a previously-working state
6. Advance to F12

The R2 approach arc (two attempts, both failed at different LLM-interaction layers) suggests that narrative-bullet extraction stability requires a different technique than prompt engineering alone — potentially a research.py code-level chunk pre-processing step that explicitly marks bullet boundaries, or an LLM-based finding-presence check at consolidation. Out of scope for the R2 work window; flagged as deferred work if the fallback triggers.

---

### F09 — Inconsistent units, signs, and formats across Metrics rows

**Category:** Metrics table
**Severity:** Medium
**Status:** Implemented

**Symptom:**
The Metrics table mixes formatting conventions across rows with no consistent standard.

**Evidence:**
Sampling rows from `outputs.consolidated_response` Metrics:
| Row | QoQ | YoY |
|---|---|---|
| Enterprise PPPT | `8%` | `14%` (no `+` sign) |
| CET1 ratio | `+0.20 pp` | `+0.50 pp` (signed, with `pp` unit) |
| CET1 capital ($MM) | `1667` | `—` (bare integer, no `+`, no `$MM`) |
| Total PCL ($MM) | `83` | `40` (bare integers) |
| PCL on performing loans ratio (%) | `flat` | `-0.02 pp` (word vs number) |
| PCL on impaired loans ($MM) | `84` | (blank) |

Three different conventions in one table: bare percentages, bare dollar deltas, signed `pp` for ratios. Some rows include unit hints in the Line Item column (`($MM)`, `(%)`); others don't. The value column header is `Q1 2026` (a period), not a unit indicator.

**Hypothesized root cause:**
`consolidation.yaml` does not specify a value-formatting standard. The LLM picks formats per-row based on whatever the underlying finding text used.

**Related findings:** F10 (units missing because research didn't capture them).

**Investigation notes:**

Cross-run verification against baseline `20260407T025300Z_a2ef222f` and F02 iter-2 re-run `20260407T042647Z_6ca4e478` confirmed four distinct formatting inconsistencies that are **cross-run stable** (same pattern appears in both traces, so they're prompt gaps not random LLM choices):

1. **Unit suffix inconsistency in Line Item.** Baseline: `PPPT` / `PPPT (Pre-provision, pre-tax earnings)` (no unit) vs `CET1 capital ($MM)` / `Total PCL ($MM)` / `CET1 ratio (%)` (with unit). Iter-2 repeats: `Pre-Provision, Pre-Tax Earnings (PPPT)`, `Net income`, `CET1 capital`, `Retained earnings — balance sheet`, `PCL — Total` all missing unit suffixes. Ratio/bps rows get suffixes, dollar rows don't. Reader has to infer scale from context.

2. **Delta sign inconsistency.** Baseline mixes unsigned percent-change (`8%`, `14%` for PPPT), signed `pp` for ratios (`+0.20 pp`, `-0.02 pp`), and unsigned dollar deltas (`1667`, `83`, `84`, `122`). Reader can't tell from `84` whether PCL grew or shrank (it grew).

3. **Word `flat` for zero delta.** Single non-numeric cell in the whole table: `PCL on performing loans ratio (%) | 0.01 | flat | -0.02 pp`. Every other row uses numeric values. Breaks any downstream parser expecting numbers.

4. **Mixed missing-data indicators.** Some cells: em dash `—`. Others: blank between pipes `| 84 |  |`. No consistency.

**Scope decision — consolidation-stage only, not research-stage:** all four issues can be fixed at the consolidation prompt level without touching research. Items 1–4 are *presentation* bugs; the research stage already has the data needed (for items 1 and 4) or doesn't have it regardless (items 2 and 3 are about formats the LLM chooses at consolidation time). F10 addresses a different upstream issue (research finding missing unit context entirely, e.g. `"100415"` with no scale hint) and is tracked separately.

**Proposed fix:**

Add a new `FORMATTING STANDARDS` block to the `## Metrics` section in `consolidation.yaml`, placed between `DUPLICATE ROW CHECK` and the "Omit this section if there are no quantitative findings." closing line. Four sub-clauses matching the four failure modes:

- **(a) Unit suffix in Line Item** — mandatory parenthesized unit suffix on every row: `($MM)`, `($B)`, `(%)`, `(bps)`, `(ratio)`, `(#)`. Worked examples quoting the actual broken baseline rows (`PPPT`, `Net income`, `Retained earnings — balance sheet`). Explicit composition with F02's rule 4 scope-suffix: scope first, unit last (`Retained earnings — regulatory scope ($MM)`).
- **(b) Value columns** — bare numeric only, no currency symbol, no thousands separator, no trailing unit. The Line Item suffix carries the unit. Decimal precision matches source.
- **(c) Delta columns** — SIGN-EXPLICIT numerics in the same unit as the base value. Dollar deltas: `+83` / `-40`. Ratio deltas: `+0.20 pp` / `-0.02 pp`. Percent-change: `+14%` / `-3%`. Explicit prohibition on the words `flat`, `unchanged`, `nil`, `stable`, and the symbol `--` for zero.
- **(d) Missing data** — em dash `—` only. Never blank, never `N/A`, never `TBD`.

Rule follows the same pattern as F01/F04/F05 (worked examples embedded verbatim in rule text for LLM consistency, per the session pattern #1 on pattern-based rules being more stable than judgment-based).

**Resolution:**

Implemented as the new `FORMATTING STANDARDS` block in the `## Metrics` section of `consolidation.yaml`, with four sub-clauses (a)–(d). Worked examples include the specific failing baseline rows (PPPT, Net income, Retained earnings) so the LLM has concrete pattern-match targets. The clause (a) text explicitly handles F02 rule-4 composition (`scope suffix first, unit suffix last`) and F10 upstream gap (`infer the unit from the source context (quarterly bank earnings figures are almost always millions unless stated otherwise)`) so F09 lands cleanly regardless of when F10 is implemented.

Added `test_consolidation_prompt_has_formatting_standards` in `u-retriever/tests/test_consolidate.py` with 11 structural assertions covering `FORMATTING STANDARDS`, `Unit suffix in Line Item`, all six unit suffixes (`($MM)`, `($B)`, `(bps)`), format constraints (`no currency symbol`, `no thousands separator`, `SIGN-EXPLICIT`), the explicit ban on `"flat"` wording, the em-dash requirement, and the no-blank-cells rule.

**Files touched:**
- `u-retriever/src/retriever/stages/prompts/consolidation.yaml` — new `FORMATTING STANDARDS` block in `## Metrics` section
- `u-retriever/tests/test_consolidate.py` — new `test_consolidation_prompt_has_formatting_standards`

**Quality gate:**
- `pytest --cov=src` — **246 passed** (was 245), coverage **86%** unchanged
- `black --check` — clean (38 files unchanged)
- `flake8` — clean (zero warnings)
- `pylint` — **10.00/10 with ZERO warnings** (unchanged from the F06 cluster state — the refactor there permanently eliminated the R0914)

**Live verification (pending re-run):**

At the next cluster-boundary re-run I diff the Metrics table against baseline and confirm:
1. Every row's Line Item ends with a parenthesized unit suffix — no bare `PPPT`, `Net income`, `CET1 capital`, etc.
2. No `flat` / `unchanged` / `nil` / `--` strings anywhere in the table
3. All non-zero delta cells carry an explicit `+` or `-` sign
4. Missing data cells show `—` (em dash), no empty cells between pipes
5. Value columns contain bare numerics (no `$`, no commas, no trailing `%`)
6. F02 multi-source split rows still show their scope suffix AND a unit suffix (e.g., `Retained earnings — balance sheet ($MM)`)
7. No regression on F01 / F02 / F03 / F04 / F05 / F06 cluster behaviors

**Amendment via F10 (restructure):**

F09's clause (a) — the "Unit suffix in Line Item" requirement — has been **replaced** by F10's restructure. The new clause (a) populates a dedicated **Unit column** in the Metrics table instead of appending a parenthesized suffix to Line Item. Line Item now contains the metric name ONLY. F09 clauses (b), (c), (d) — bare numeric value columns, sign-explicit deltas, em-dash for missing — are unchanged and remain in effect.

F09's index status stays `Implemented` because the underlying formatting inconsistency (no unit indicator on dollar rows, `flat` for zero deltas, blank missing cells) is still fixed — only the mechanism changed from "suffix in Line Item" to "dedicated Unit column". The F10 Resolution section has the full details of the restructure and the cross-cluster composition verification.

The fallback "infer the unit from source context" clause was preserved in the new F10 clause (a) for older traces where research didn't populate the `unit` field. With F10 landed, the fallback becomes used primarily for legacy data; new runs will have `unit` populated directly from research.

**Related findings:**
- **F10** — the restructured F10 amended F09 clause (a). F10's research-stage rule now captures units at extraction time via a structured `unit` field. F09's consolidation-stage rule now reads that field and populates the Unit column directly. See F10 Resolution for the full six-change implementation.
- **F29** (period column drift) — F09's rule is period-column-agnostic. Whichever columns the LLM picks, the formatting standards apply.
- **F12** (source-language differences) — separate issue, handled at a different layer.

---

### F10 — Research findings missing unit context from surrounding chunks

**Category:** Research extraction
**Severity:** Medium
**Status:** Implemented (cross-cutting restructure — also amended F09 clause (a))

**Symptom:**
Pillar3 findings frequently have bare numeric `metric_value` without unit indicators (`100415`, `99023`, `7401`, `634`). The actual KM1/CC1/CR1 sheets have `($ in millions)` headers but the unit context isn't preserved in the structured finding.

**Evidence:**
`source_02_…pillar3_40.json` accumulator:
```json
{"metric_name": "Common Equity Tier 1 (CET1)", "metric_value": "100415", ...}
{"metric_name": "Retained earnings", "metric_value": "99023", ...}
{"metric_name": "Allowances/impairments - Loans", "metric_value": "7401", ...}
{"metric_name": "Total net write-offs", "metric_value": "634", ...}
```

By contrast, rts findings include scale indicators because the source prose says it: `"$8.5 billion (up $1.0 billion or 14% YoY)"`, `"$100,415 million"`.

**Hypothesized root cause:**
Two possibilities:
1. The chunk content given to the research LLM doesn't include the table-header unit qualifier (`$ in millions`) — an ingestion/chunking issue.
2. The chunk does include it, but the research prompt doesn't instruct the LLM to capture units.

Need to inspect a chunk passed to the research stage for pillar3 page 3 (KM1) to determine which.

**Related findings:** F09 (downstream symptom in metrics table).

**Investigation notes:**

**Hypothesis 2 confirmed by direct chunk inspection.** The baseline pillar3 KM1 chunk (`source_02_…pillar3_40.json` → `stages.research.final_chunks[0]`) has `raw_content` that *does* contain the unit qualifier `(Millions of Canadian dollars)` at row 4 of the sheet:

```
| 4 |  | (Millions of Canadian dollars) | a | b | c | d | e | f |
| 5 |  |  | January 31 | 2025-10-31 | 2025-07-31 | 2025-04-30 | 2025-01-31 | QoQ Change |
| 6 |  |  | 2026 | 2025 | 2025 | 2025 | 2025 |  |
| 8 | 1 | Common Equity Tier 1 (CET1) | 100415 | 98748 | ... | 1667 |
```

The LLM sees both the row-4 unit header and the row-8 data cells in the same chunk. It extracts the bare cell value `100415` but never performs the row-4 lookup to attach `(Millions of Canadian dollars)` as a unit. The current rule 5 text `"exact value as stated in the source"` effectively discourages the lookup — the LLM interprets it as "don't decorate, just copy".

**Two patterns observed across pillar3 findings:**
1. **Dollar-amount rows** (CET1 capital, RWA, net write-offs, impaired exposures): cells are bare numeric, unit ONLY in sheet-level header. LLM fails to attach → `metric_value="100415"`.
2. **Ratio rows** (CET1 ratio (%), Tier 1 ratio (%)): cells already carry `%` inline. LLM handles these correctly → `metric_value="13.7%"`.

**Contrast with rts findings** (source prose carries units inline):
```json
{"metric_value": "$8.5 billion (up $1.0 billion or 14% YoY)"}
{"metric_value": "$100,415 million"}
```
rts works because the units are in the *same string* as the value in the source prose. Pillar3 doesn't work because units are in a *separate cell* (header row) above the data. The current prompt doesn't ask the LLM to bridge that gap.

**Scope expansion per user preference:** the user requested a structured `unit` field on `ResearchFinding` (not a string suffix in `metric_value`) plus a dedicated `Unit` column in the Metrics table, and a combined Source+Ref column. This turned F10 from a targeted single-rule fix into a cross-cutting restructure that also amends F09's clause (a).

**Proposed fix:**

Six coordinated changes across the stack:

1. **`models.py`** — new `unit: NotRequired[str]` field on `ResearchFinding` TypedDict, placed between `metric_value` and `period`.

2. **`research.yaml` tool schema** — add `unit` property to `produce_research.findings.items.properties` with a description explaining compact tokens (`$MM`, `$B`, `%`, `bps`, `x`, `#`) and header-lookup semantics. Add `unit` to the `required:` list (OpenAI strict tool-calling). Update `metric_value` description to clarify "WITHOUT the unit".

3. **`research.yaml` rule 5** — update the field-population list to include `unit` with explicit header-lookup guidance. Worked example quotes the pillar3 KM1 `(Millions of Canadian dollars)` header verbatim and shows the expected conversion: `metric_value="100,415"` + `unit="$MM"`. Rule 6 (qualitative findings) updated to include `unit` in the "set to empty string" list.

4. **`research.py` parser** — one-line addition: `"unit"` added to the `optional_key` tuple at line 298–306. Existing lenient `.get(key, "")` pattern means mocked test fixtures that don't set `unit` continue to parse cleanly.

5. **`consolidate.py` `_format_finding_summary`** — inline the unit after `metric_value` in the rendered ref-index line. Format: `Common Equity Tier 1 (CET1) = 100,415 $MM (Q1 2026) [Enterprise] — ...`. Backwards compatible: findings without `unit` render as before (`CET1 Ratio = 13.7% (Q1 2026) ...`).

6. **`consolidation.yaml` `## Metrics` section restructure** — four sub-changes:
   - **Column layout**: `Entity | Segment | Line Item | Unit | <period cols> | Source` (was `Entity | Segment | Line Item | Source | <period cols> | Ref`). Unit slot between Line Item and periods. Ref column removed. Source column moved to last position.
   - **Source column new format**: `"source [REF:N]; source [REF:N]"`. Example: `"investor-slides [REF:1]; rts [REF:11]"`. Refs live inline in the Source column — there is NO separate Ref column.
   - **F02 rules 1–4 rewording**: each rule's Source/Ref column references updated to describe the combined format. No logical change to the merge/split decision tree.
   - **F09 clause (a) rewritten**: removes the "unit suffix in Line Item" requirement. Line Item now contains the metric name ONLY. The new Unit column carries the unit. Explicit anti-suffix language: "do NOT append a parenthesized unit suffix to Line Item". Fallback "infer from source context" still present for empty `unit` fields.

**Rationale for the restructure vs. string-suffix:**
- Data model is cleaner: the unit is a property of the *value*, not the *name*. Putting it in `metric_name` as a suffix was a workaround.
- Downstream consumers (Metrics table, future analytics) can group/filter by unit structurally.
- Decouples naming from presentation — Line Item reads clean without parenthesized unit annotation.
- Enables unit-aware features later (currency conversion, scale normalization, sanity checks) without re-parsing strings.

**Resolution:**

Implemented all six changes in a single coordinated pass.

**Files touched:**
- `u-retriever/src/retriever/models.py` — added `unit: NotRequired[str]` to `ResearchFinding` TypedDict
- `u-retriever/src/retriever/stages/prompts/research.yaml`:
  - Rule 5 expanded with `unit` field population and header-lookup guidance (pillar3 KM1 worked example embedded verbatim)
  - Rule 6 updated to include `unit` in the qualitative-empty list
  - Tool schema: new `unit` property with compact-token description, added to `required:` list
  - Tool schema: `metric_value` description updated to clarify unit separation
- `u-retriever/src/retriever/stages/research.py` — added `"unit"` to the `optional_key` tuple in the finding parser (one-line change at line 298–306)
- `u-retriever/src/retriever/stages/consolidate.py` — `_format_finding_summary` now appends ` {unit}` after `= {metric_value}` when unit is present
- `u-retriever/src/retriever/stages/prompts/consolidation.yaml`:
  - `## Metrics` preamble rewritten: new column layout `Entity | Segment | Line Item | Unit | <period cols> | Source`, combined Source+ref format documented
  - F02 rules 1–4 updated to reference the new combined Source column format
  - F09 `FORMATTING STANDARDS` clause (a) rewritten: Unit column instructions replace the old "Unit suffix in Line Item" requirement
- `u-retriever/tests/test_models.py` — new `test_research_finding_with_unit_field`
- `u-retriever/tests/test_research.py` — new `test_research_prompt_has_unit_enrichment_rule`, new `test_research_prompt_qualitative_rule_includes_unit`
- `u-retriever/tests/test_consolidate.py`:
  - Updated `test_consolidation_prompt_metrics_has_source_column` for combined format assertions
  - Updated `test_consolidation_prompt_has_formatting_standards` for Unit column assertions (replaced old "Unit suffix in Line Item" / `($MM)` assertions)
  - New `test_format_finding_summary_includes_unit`
  - New `test_format_finding_summary_without_unit_legacy` (backwards compat)

**F09 amendment:** F09 clause (a) — the "Unit suffix in Line Item" requirement — has been **replaced** by the new "Unit column" instruction. F09's index status stays `Implemented`; the underlying formatting inconsistency is still fixed, the mechanism changed from "suffix in Line Item" to "dedicated Unit column". F09 clauses (b), (c), (d) — bare numeric values, sign-explicit deltas, em-dash for missing — remain unchanged. See the F09 Resolution section for the original clause (a) history and this amendment cross-reference.

**Quality gate:**
- `pytest --cov=src` — **251 passed** (was 246, +5 new tests for F10), coverage **86%** unchanged
- `black --check` — clean (38 files unchanged)
- `flake8` — clean (zero warnings)
- `pylint` — **10.00/10 with zero warnings** (pre-existing R0914 was eliminated by the F06 cluster refactor — the clean state holds)

**Composition verification** (verified preemptively, no runtime check needed):
- **F06 layer-1 coverage audit**: the streaming `_create_ref_replacer` tracks `used_refs` wherever `[REF:N]` appears in the LLM output — including the new Source column cells. Layer-1 detection unchanged.
- **F06 layer-2 coverage audit**: `_extract_metric_value_tokens` regex `\d+(?:\.\d+)?(?:%|\s*bps?)?` operates on `metric_value` which is now bare numeric (no unit suffix in the value itself). Token extraction for `metric_value="100,415"` → `"100415"` (after comma strip) still matches response text containing `"100,415"` (also comma-stripped). No regression.
- **F04 rule 22 (methodology footnote skip)**: orthogonal — rule 22 filters findings at research stage before the `unit` field is ever populated. Skipped findings never enter the reference index regardless of unit.
- **F05 rule 9 (segment residual)**: orthogonal — rule 9 writes derived Detail prose, not Metrics rows. Unit column doesn't exist in Detail.
- **F03 rule 21 (scope qualifier in metric_name)**: composes cleanly with the new Unit column. Scope qualifiers (e.g., `"ACL on loans and acceptances"`) remain in `metric_name` / Line Item; units go in the new column. F04's rule 22 worked example stays intact.

**Live verification criteria (pending re-run):**

*Research stage:*
1. Pillar3 findings' `metric_value` fields are clean numerics (e.g., `"100,415"` not `"100415"`); paired with populated `unit` fields (`"$MM"`, `"%"`, `"bps"`)
2. rts findings — which previously carried units inline in `metric_value` — are extracted with the unit moved to the `unit` field: e.g., `metric_value="8.5", unit="$B"` instead of `metric_value="$8.5 billion"`
3. No double-unit artifacts: no `metric_value="13.7%"` paired with `unit="%"`
4. No invented units on findings where the source has no scale indicator

*Consolidation stage:*
5. Metrics table column layout matches: `| Entity | Segment | Line Item | Unit | Q1 2026 | Source |`
6. Line Item column contains bare metric names — no `($MM)` / `(%)` suffixes anywhere
7. Unit column populated for every row (`$MM`, `%`, `bps`, `x`, `#`)
8. Source column contains combined format (`rts [4]; investor-slides [13]; pillar3 [16]`) with rendered anchor tags
9. No separate Ref column in the table
10. F02 multi-source handling still works: rule 1 merge (same value), rule 2 rounding collapse, rule 3 split (disagreement), rule 4 scope split

*Cross-cluster composition:*
11. F01 CET1 bridge: components extracted with `unit="bps"` populating the Unit column
12. F03 scope qualifiers: still appear in Line Item (e.g., `ACL on loans and acceptances`), with unit in the Unit column
13. F04 methodology footnote skip: unchanged; footnotes still filtered at research stage
14. F05 segment residual: Detail prose unchanged; derived clauses still cite refs
15. F06 cluster: Coverage audit section still appears when drops are detected; layer-1 and layer-2 detection still work on the new data shape
16. F09 clauses (b), (c), (d): bare numeric values, sign-explicit deltas, em-dash for missing — all preserved

*Backwards compatibility note for the debug UI:* older traces (baseline `20260407T025300Z_a2ef222f`, F01/F02/F03 re-runs) have NO `unit` field on any finding. When diffing the next re-run against these older traces, the column structure difference (old `Source | <period> | Ref` → new `Unit | <period> | Source`) is expected and should NOT be flagged as a regression. This is a deliberate format change.

**Related findings:**
- **F09** — F09 clause (a) was rewritten as part of this fix. F09 index status stays `Implemented` with an amendment note. Clauses (b), (c), (d) unchanged.
- **F06 cluster** — composes cleanly. Layer-1 and layer-2 audits both work on the new data shape without code changes (verified via regex analysis preemptively).
- **F29** (period column drift) — unchanged. F10 doesn't affect which period columns the LLM picks. When F29 lands, the new Unit column slots cleanly with whichever period columns are chosen.
- **F11** (deferred — repeated RBC in Entity column) — unrelated. F11's multi-bank restructure will compose with the new column layout naturally.
- **F12** (source-language differences in metric_name) — still open. F12 is orthogonal to units; addresses the case where `metric_name` varies across sources (e.g., `"PCL on impaired loans"` vs `"PCL on impaired assets"`). Next up after F10.

---

### F11 — Repeated `RBC` in Entity column (deferred)

**Category:** Metrics table
**Severity:** Low
**Status:** Deferred

**Symptom:**
Every row in the Metrics table has `Entity = RBC` because this query is single-bank. Visually wasteful but irrelevant once multi-bank queries are supported.

**Resolution:**
Deferred per user — column is needed for multi-bank queries and isn't worth special-casing now.

---

### F12 — Source-language differences (`PCL on impaired loans` vs `… assets`)

**Category:** Metrics table
**Severity:** Medium
**Status:** Open

**Symptom:**
The Metrics table has two rows for what is conceptually the same metric, using different source-language labels:
- Enterprise: `PCL on impaired loans ($MM)` (from page 8 / page 4)
- Capital Markets: `PCL on impaired assets ($MM)` (from page 15)

The terminology mismatch is real — RBC's enterprise disclosures say "loans" while the Capital Markets segment slide says "assets". Both are faithful to the source. The user has stated a preference: **preserve source verbatim and add a footnote noting the difference**, rather than normalizing.

**Evidence:**
`outputs.consolidated_response`, Metrics table rows:
```
| RBC | Enterprise | PCL on impaired loans ($MM) | 1068 | 84 |  | ... |
| RBC | Capital Markets | PCL on impaired assets ($MM) | 240 | 122 | 35 | ... |
```

**Hypothesized root cause:**
`consolidation.yaml` rule says *"Normalize metric names across sources into consistent line items"* — but the LLM (correctly) didn't normalize because it would require domain knowledge to know "loans" and "assets" mean the same thing here. There is no footnoting mechanism in the table format to handle this case.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F13 — Visible numbering gaps in reference table (12, 22, 23, 24)

**Category:** Reference table
**Severity:** Low
**Status:** Open

**Symptom:**
The rendered Reference table shows entries `1–11, 13–21, 25–27`. Refs 12, 22, 23, 24 exist in the catalog but were never cited, and `_build_reference_appendix` filters them out — but the gaps look like a bug to a human reader.

**Evidence:**
`outputs.consolidated_response` References table jumps from `[11]` → `[13]` and from `[21]` → `[25]`.

Catalog reconstruction confirmed: 27 entries built, 4 uncited.

**Hypothesized root cause:**
`consolidate.py:199-225` filters by `used_refs` (working as designed), but the ref IDs are catalog-build-order, not citation-order. The user has confirmed: *"I'm assuming it's just findings that weren't used"* — that's correct.

**Options:**
1. Renumber on display (sequential 1..N over cited refs only); requires rewriting anchors in the response after generation.
2. Show all catalog refs (cited + uncited) with a "(not cited)" tag.
3. Add a small footnote under the table explaining the gaps are normal.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F14 — Page-level "duplication" — investor-slides p8 produces 4 refs

**Category:** Reference index design
**Severity:** Medium
**Status:** Open

**Symptom:**
A single physical page (investor-slides p8) produces four separate refs because the LLM wrote slightly different `location_detail` strings for sub-areas of the same page.

**Evidence:**
| Ref | location_detail string |
|---|---|
| [1] | `Q1/26: Strong results underpinned by solid revenue growth and strong operating leverage(1)` |
| [3] | `... (1) - Segment PPPT table` |
| [7] | `... (1) - Provision for Credit Losses` |
| [8] | `... (1) - Financial Results Table` |

Same page, four refs. Rts page 4 has the same pattern with three refs (`Capital ratios`, `Selected financial highlights table`, `Credit metrics`).

**Hypothesized root cause:**
The dedup key `(filename, page, location_detail)` works correctly — these ARE distinct sub-areas. Whether to collapse them to one page-level ref or keep sub-area precision is a **design call**:
- **Collapse:** shorter reference list, matches reader's mental model ("page 8 of the deck")
- **Keep distinct:** more precise pointing for review/verification, but visually noisy

User confirmed `OK` on this item (acceptance of current behavior), but logging it for the design decision.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F15 — Real dedup failure: pillar3 CR1 split into 2 refs

**Category:** Reference index
**Severity:** Medium
**Status:** Open

**Symptom:**
A single physical CR1 table on pillar3 page 12 became two separate refs because the research LLM wrote two different `location_detail` strings across findings:

| Ref | location_detail | Outcome |
|---|---|---|
| [22] (catalog) | `CR1: Credit quality of assets` | Uncited, hidden from output |
| [26] (rendered) | `CR1: Credit quality of assets and totals` | Cited |

Single trailing word (`and totals`) defeats the literal-tuple dedup.

**Evidence:**
`source_02_…pillar3_40.json` accumulator findings — multiple findings with `page=12` and `location_detail` starting `CR1:` but with different suffixes.

**Hypothesized root cause:**
Dedup key in `consolidate.py:_build_reference_index` uses raw `location_detail` strings. The LLM is not consistent across iterations / extractions about how it labels the same table. No normalization.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F16 — Reference appendix lacks evidence preview

**Category:** Reference table
**Severity:** Low
**Status:** Open

**Symptom:**
The Reference table columns are `# | Source | File | Page | Location` — no indication of what each ref is actually evidence FOR. A reviewer can't quickly verify a citation without opening the source PDF.

**Evidence:**
`outputs.consolidated_response` References table — every row contains only file/page/location. The `findings` array attached to each entry in `_build_reference_index` is discarded by `_build_reference_appendix`.

**Hypothesized root cause:**
`consolidate.py:_build_reference_appendix` (lines 199-225) only emits the location-level columns. The structured findings under each entry could be summarized into a "Key facts" cell.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F17 — Severe within-source duplication in pillar3 (23 → ~14 unique facts)

**Category:** Research stage
**Severity:** High
**Status:** Open

**Symptom:**
The pillar3 research stage's `stages.research.findings` array has 23 entries that boil down to ~14 unique facts. The accumulator across iterations stores findings with slightly varied `metric_name` phrasing as separate entries instead of recognizing them as duplicates.

**Evidence:**
`source_02_…pillar3_40.json` `stages.research.findings`:

| Concept | Duplicate indices |
|---|---|
| `Retained earnings = 99023` (CC1, p10) | `[2]` and `[11]` — identical |
| `Retained earnings = 100247` (CC2, p11) | `[15]` (`Retained earnings - Under regulatory scope of consolidation`) and `[19]` (`Retained earnings`) |
| `CET1 QoQ change = 1667` (KM1, p3) | `[9]` (`QoQ Change (a-b) for CET1`) and `[10]` (`QoQ Change (a-b) - CET1`) |
| `Loans allowances = 7401` (CR1, p12) | `[6]` (`Allowances/impairments`) and `[12]` (`Allowances/impairments - Loans`) |
| `Off-balance allowances = 388` (CR1, p12) | `[14]` (`Off-Balance Sheet exposures - Allowances/impairments`) and `[20]` (`Allowances/impairments - Off-Balance Sheet exposures`) |

**Hypothesized root cause:**
Research stage iterates and **appends** to findings without deduping. Each iteration regenerates many of the same facts with slightly different phrasing. `research.yaml` rule 15 says *"Avoid repeating the same metric from multiple chunks unless the duplication adds reconciliation or source context"* — but this rule isn't enforced *across* iterations because each iteration is a fresh LLM call with no memory of what previous iterations extracted.

**Related findings:** F18 (wasted iterations are why this happened).

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F18 — Pillar3 wasted 2 extra iterations chasing data not in source

**Category:** Research stage
**Severity:** High
**Status:** Open

**Symptom:**
Pillar3's iter 0 and iter 1 set `additional_queries` asking for PPPT and income-statement PCL splits — data that fundamentally lives in MD&A / RTS, not in Pillar 3 regulatory disclosures. The LLM ran 3 iterations on this source vs 1 for investor-slides and 1 for rts.

**Evidence:**
`source_02_…pillar3_40.json` `stages.research.iterations`:
```
iter 0 conf=0.64 additional_queries=[
  'Find PPPT (pre-provision, pre-tax) earnings for RBC Q1 2026 from the Report to Shareholders or MD&A.',
  'Find provision for credit losses (PCL) split between performing and impaired loans...'
]
iter 1 conf=0.62 additional_queries=[same two, rephrased]
iter 2 conf=0.78 additional_queries=[]
```

`research.yaml` rule 16: *"If the current source does not contain data for some topics in the query, that is acceptable — return findings for the topics it does cover."* — violated for two iterations.

**Hypothesized root cause:**
The LLM treats low-confidence (incomplete query coverage) as a reason to retry, even when the missing topics structurally cannot be in the current source. The prompt rule exists but isn't strong enough; confidence scoring conflates "this source doesn't cover the topic" with "I haven't searched enough yet."

**Related findings:** F17 (the duplicate-finding bloat is a direct consequence of these wasted iterations).

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F19 — Empty structured fields where hard metrics exist

**Category:** Research extraction
**Severity:** Medium
**Status:** Open

**Symptom:**
Some findings have hard quantitative facts in the prose `finding` text but leave `metric_name`, `metric_value`, `period`, `segment` empty — treating them as purely qualitative when they aren't.

**Evidence:**
`source_03_…rts_39.json` finding 9 (REF:19):
```json
{
  "finding": "CET1 ratio increased 20 bps quarter-over-quarter to 13.7%, driven by net internal capital generation and fair value OCI adjustments, partially offset by business-driven RWA growth and share repurchases.",
  "page": 39, "location_detail": "Capital management — Continuity of CET1 ratio commentary"
}
```
Contains `13.7%` and `+20 bps QoQ` and Enterprise scope and Q1 2026 period — all four optional fields could be populated.

`source_01_…investor-slides_38.json` REF:9 second finding has the same shape with `$28MM` in the text but empty structured fields.

**Hypothesized root cause:**
`research.yaml` rule 6: *"For qualitative findings (trends, commentary, descriptions), set metric_name, metric_value, period, and segment to empty string"* — the LLM is interpreting "commentary" too broadly, treating any prose-style finding as qualitative even when it embeds hard numbers.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F20 — `location_detail` pinned to slide title, not panel

**Category:** Research extraction
**Severity:** Medium
**Status:** Open

**Symptom:**
REF:2 (investor-slides p15) groups two unrelated metrics under one ref because `location_detail` is the slide title, not the panel name:

```
[REF:2] page=15 location_detail="Capital Markets: Record PPPT(1) earnings underpinned by strong revenue of $4.0 billion"
   - [Capital Markets] Pre-Provision, Pre-Tax Earnings(1) = 1,899
   - [Capital Markets] PCL on Impaired Assets = 240
```

PPPT and PCL on Impaired Assets are physically different panels of the slide. When this ref is later cited for a PCL claim, the location_detail header reads as "Record PPPT earnings…" — a misleading anchor.

**Evidence:**
See above — REF:2 in the catalog reconstruction.

**Hypothesized root cause:**
The research LLM picked the slide-level title for `location_detail` instead of the panel-level subtitle. `research.yaml` rule 3 says *"Set location_detail to the section name, sheet name, or other identifying context from the header"* — too vague about granularity.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F21 — Over-citation by end-of-bullet clustering

**Category:** Citation usage
**Severity:** Medium
**Status:** Open

**Symptom:**
Summary and Detail bullets often cluster 3-4 refs at the end of a multi-clause sentence rather than placing each ref inline next to the specific clause it backs. The reader cannot tell which ref backs which sub-claim.

**Evidence:**
`outputs.consolidated_response` Summary, capital-composition bullet:
> "...with CET1 capital of $100,415MM; retained earnings were $99,265MM and 9% higher YoY, underpinning internal capital generation of +46 bps to CET1 [4][13][15][5]"

Inline-correct version:
> "CET1 capital of $100,415MM [13]; retained earnings were $99,265MM [15] and 9% higher YoY [5], underpinning internal capital generation of +46 bps to CET1 [5]"

Same pattern in the credit-losses summary bullet (`[8][17][18][7]`).

**Hypothesized root cause:**
`consolidation.yaml` requires citations on every factual statement but doesn't require inline (per-clause) placement. The LLM defaults to dropping all refs at the end of the sentence.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F22 — `[REF:4]` filler co-citation in CET1 YoY claim

**Category:** Citation usage
**Severity:** Low
**Status:** Open

**Symptom:**
Detail capital section cites `[REF:4][REF:14][REF:19]` for a sentence about the CET1 ratio, QoQ, YoY, and drivers — but `[REF:4]` (investor-slides p9) only contains the QoQ delta, not the YoY. The YoY is fully covered by `[REF:14]` (rts p4) and the drivers by `[REF:19]` (rts p39). `[REF:4]` is filler.

**Evidence:**
`outputs.consolidated_response` Detail:
> "The CET1 ratio stood at 13.7% as of January 31, 2026, up 20 bps quarter-over-quarter and 50 bps year-over-year, supported by net internal capital generation and fair value OCI, partly offset by business-driven RWA growth and share repurchases [REF:4][REF:14][REF:19]"

REF:4 finding only states `+20 bps QoQ`; no YoY number.

**Hypothesized root cause:**
Same as F21 — clustering. LLM dumps every plausibly-related ref at sentence end without verifying each adds new evidence.

**Related findings:** F21.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F23 — Same physical evidence cited as multiple refs (visual diversity overstating)

**Category:** Citation usage
**Severity:** Low
**Status:** Open

**Symptom:**
The Metrics table cites `[REF:7]` for five separate rows (PCL ratio, PCL performing $MM, PCL performing ratio, PCL impaired $MM, PCL impaired ratio). REF:7 is investor-slides p8 PCL panel containing all three Stage 1&2 / Stage 3 / Total findings. The Summary credit-losses bullet cites `[8][17][18][7]` — four different refs for the same physical PCL panel (just different sub-locations on page 8 + the rts equivalents).

**Evidence:**
See Metrics table in `outputs.consolidated_response`. REF:7, REF:8, REF:17, REF:18 all describe PCL on the same Q1 2026 RBC enterprise.

**Hypothesized root cause:**
Combination of F14 (page sub-area dedup creates multiple refs for one physical area) + F21 (clustering at sentence end picks up every ref tangentially relevant). Visually inflates evidence breadth.

**Related findings:** F14, F21.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F24 — `citation_validation` only checks ref existence, not factual support

**Category:** Pipeline validation
**Severity:** Medium
**Status:** Open

**Symptom:**
The pipeline's `citation_validation` block reports `27 citations checked, 0 warnings, 0 unreplaced refs` — yet the response contains real arithmetic errors (F01), scope conflations (F03), filler citations (F22), and dropped findings (F06–F08). Validation passes on a response with multiple substantive issues.

**Evidence:**
`run_trace.json` lines 47-82:
```json
"citation_validation": {
  "catalog_sources": 3,
  "citations_checked": 27,
  "skipped": false,
  "unreplaced_ref_count": 0,
  "warning_count": 0
}
```

**Hypothesized root cause:**
Validation only verifies ref ID existence and HTML replacement. There is no semantic check (does cited ref actually back the claim?), no arithmetic check (do bridges sum?), no coverage check (were query-relevant findings cited?), no comparability check (do juxtaposed values share scope?).

**Related findings:** F01, F03, F06, F07, F08, F22.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F25 — `key_findings` is naive regex split of summary bullets

**Category:** Pipeline output
**Severity:** Low
**Status:** Open

**Symptom:**
`outputs.key_findings` contains the four raw summary bullets verbatim, including embedded HTML anchors. Not a clean fact-per-line list.

**Evidence:**
`run_trace.json` `outputs.key_findings[0]`:
```
- Earnings strength: Enterprise PPPT was $8.5 billion in Q1 2026, up 14% year-over-year, reflecting solid revenue growth and operating leverage <a href="...">[1]</a><a href="...">[11]</a>.
```

`consolidate.py:_extract_cited_sentences` (lines 282-293) splits the summary on sentence boundaries and keeps the first 5 with a `[`. With long compound sentences, each "key finding" is essentially a whole bullet.

**Hypothesized root cause:**
Naive sentence splitter on a section that's structured as bullets, not sentences. Plus HTML anchors aren't stripped.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F26 — Investor-slides "ACL to loans ratio = 73 bps" finding dropped

**Category:** Coverage / dropped findings
**Severity:** Low
**Status:** Implemented (via F06)

**Symptom:**
Investor-slides REF:5 contains a finding for `ACL to loans ratio = 73 bps (+2 bps QoQ)` from page 4 Key Messages. The response never mentions this ratio.

**Evidence:**
`source_01_…investor-slides_38.json` REF:5 findings include:
```json
{"metric_name": "ACL to loans ratio", "metric_value": "73 bps (+2 bps QoQ)", "page": 4}
```

Absent from `outputs.consolidated_response`.

**Hypothesized root cause:**
Same as F06/F07/F08 — silent drop. ACL coverage ratio is borderline-relevant to "credit losses" but the consolidator chose not to surface it.

**Related findings:** F06, F07, F08.

**Investigation notes:**

This is a **layer-2 quantitative drop** — REF:5 is cited in the baseline response (for other p4 Key Messages content), but the ACL-to-loans ratio finding within the same ref is silently dropped. Literal substring `73 bps` is zero hits in the baseline response.

`_extract_metric_value_tokens("73 bps (+2 bps QoQ)")` returns `["73 bps", "2 bps"]`, so the F06 cluster's mechanism C will substring-search the rendered response for `73 bps` (the more distinctive token) AND `2 bps`. If neither appears, the audit flags the finding under `### Unincorporated findings (ref cited but value absent)`.

**Resolution:**

Resolved by the F06 cluster fix. F26 is caught at two layers:

- **Mechanism A (rule 6 replacement)** — the new "Coverage commitment" rule sub-clause (a) explicitly tells the LLM that quantitative findings must appear in Metrics or Detail. ACL coverage ratios are directly relevant to a "credit losses" query and should not be borderline-dropped.
- **Mechanism C (layer-2 token audit)** — if the LLM still drops it, `_compute_coverage_audit` will catch the missing `73 bps` token and surface the finding under the Unincorporated subsection of `## Coverage audit`.

**Live verification (pending re-run):**
- Either: the response now includes `73 bps` somewhere (Metrics row or Detail prose) — mechanism A succeeded
- Or: the `## Coverage audit` section's Unincorporated subsection lists `[REF:5] investor-slides p4 — ACL to loans ratio = 73 bps (+2 bps QoQ)` — mechanism C caught the residual drop

**Files touched:** see F06 Resolution. F26 shares the cluster fix.

**Quality gate:** see F06 Resolution.

---

### F27 — Footnote markers `(1)(2)` polluting `metric_name`

**Category:** Research extraction
**Severity:** Low
**Status:** Open

**Symptom:**
Investor-slides findings consistently include footnote anchors in `metric_name`: `Pre-Provision, Pre-Tax Earnings(2)`, `PCL on loans ratio(1)`, `CET1 ratio(1)`. These come from the source slide's footnote markers and prevent string-equality dedup against other-source findings that don't carry the markers (e.g., rts uses `Pre-provision, pre-tax earnings`).

**Evidence:**
`source_01_…investor-slides_38.json` findings — pervasive across all investor-slides extractions.

**Hypothesized root cause:**
Research stage rule 5 says *"metric_name: the metric label verbatim from the source (do not normalize or rename)"* — verbatim is correct in spirit but verbatim-with-footnote-marker is noisier than verbatim-without. The marker is layout chrome, not part of the metric label.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F28 — HTML anchor `href` not URL-encoded

**Category:** Future fragility
**Severity:** Low
**Status:** Open

**Symptom:**
The placeholder `href` builder in `consolidate.py:161` does no URL escaping:
```python
href = f"https://docs.local/files/{filename}#page={page}"
return f'<a href="{href}">[{ref_id}]</a>'
```
Current filenames are alphanumeric+underscore so it works. When wired to real S3 paths with spaces, parens, `&`, or non-ASCII characters, this will produce broken anchors.

**Evidence:**
`u-retriever/src/retriever/stages/consolidate.py:159-162`.

**Hypothesized root cause:**
Placeholder code waiting for the real S3 URL wiring step. Easy to miss when that wiring lands.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

---

### F29 — Period columns in Metrics table drift across runs

**Category:** Metrics table
**Severity:** Low-Medium
**Status:** Open
**Discovered:** During F02 iteration 1 and iteration 2 re-runs

**Symptom:**
Three consecutive runs of the same query produced three different period-column configurations in the Metrics table. The LLM has no consistent guidance on which periods to display, so the column structure drifts from run to run. A user trying to compare outputs across runs sees the historical context appear and disappear.

**Evidence:**

| Run | Period/value column configuration |
|---|---|
| Baseline `20260407T025300Z_a2ef222f` | `\| ... \| Q1 2026 \| QoQ change \| YoY change \| Ref \|` — current value + two delta columns the LLM invented (not "periods") |
| F02 iter 1 re-run `20260407T041015Z_2dc7e8e0` | `\| ... \| Q1 2025 \| Q4 2025 \| Q1 2026 \| Ref \|` — three explicit period columns showing full comparatives |
| F02 iter 2 re-run `20260407T042647Z_6ca4e478` | `\| ... \| Q1 2026 \| Ref \|` — current period only, no comparatives |

The same query against the same data produced three different table shapes. For a financial research answer, the loss of YoY/QoQ context in iter 2 is a real usability regression.

**Hypothesized root cause:**
`consolidation.yaml` `## Metrics` rule says *"one column per period found in the data"* which is ambiguous about:
1. Whether to include comparative periods that only some findings report
2. Whether to use delta columns (`QoQ change`, `YoY change`) vs. absolute period columns (`Q4 2025`, `Q1 2025`)
3. Whether to omit prior periods entirely when comparatives are thin
4. How to handle the mixed case (some findings have prior-period values, some don't)

There is no explicit instruction to always include a baseline comparative set, so the LLM makes a different judgment call each run.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

**Related findings:**
- F09 (formatting inconsistencies) — F29 is a specific instance of "the Metrics table has no enforced schema". F09's fix should define a consistent period-column convention that covers F29.
- F02 — F02's rules are orthogonal to period selection and work correctly regardless of which columns the LLM chooses.

---

### F30 — Research stage instability on methodology footnotes (PPPT calc footnote)

**Category:** Research extraction
**Severity:** Low
**Status:** Open
**Discovered:** Cross-run comparison during F01/F02/F03 cluster validation

**Symptom:**
The same source (rts page 2 MD&A footnote explaining PPPT methodology) has been extracted with three structurally different outputs across three runs, with **no extraction being fully clean**. The research LLM is unstable when processing methodology prose that describes a calculation rather than reporting a directly-stated metric.

**Evidence:**

| Run | Extracted as |
|---|---|
| Baseline `20260407T025300Z_a2ef222f` (REF:12) | Single finding with garbled math: `metric_name = "Pre-provision, pre-tax (PPPT) earnings"`, `metric_value = "Income ($5,785m) before income taxes ($1,622m) and PCL ($1,090m)"`. The finding text mis-labels `$1,622M` as "income before income taxes" — it's the tax provision. Original F04 diagnosis. |
| F02 iter 1 re-run `20260407T041015Z_2dc7e8e0` (REF:2) | Broken into four Metrics rows — `PPPT components — Income (for PPPT calc) = 5,785`, `— Income taxes = 1,622`, `— PCL = 1,090`, plus a derived `Income before income taxes \| 6,400 \|  \| 7,400 \|`. The `6,400` Q1 2025 value looks invented/speculative. |
| F02 iter 2 re-run `20260407T042647Z_6ca4e478` (REF:2) | Reduced to one unambiguous row: `Net income \| rts \| 5,785`. The PPPT methodology description and the component values are dropped entirely. |

Three different extraction patterns from identical source content and identical research prompt. The behavior is drifting in response to unrelated prompt changes (F01, F02, F03 rule additions).

**Hypothesized root cause:**
The research prompt has no rule-level guidance for methodology footnotes that describe a *calculation* rather than report a *value*. The LLM has three failure modes when it encounters such text:

1. **Garble** (baseline): attempt to extract the formula as a `metric_value`, producing text that doesn't compute
2. **Over-decompose** (iter 1): break the formula into component rows, including derived numbers that aren't directly stated
3. **Retreat** (iter 2): extract only the single value closest to an unambiguous metric, dropping the methodology context entirely

All three are symptoms of the same underlying gap: no explicit rule for "how to handle a methodology footnote". F04's proposed fix (*"research stops extracting derived/computed metrics"*) partially addresses this — mode 1 and mode 2 would be blocked by that rule — but the "retreat" mode 3 is an avoidance response the F04 rule doesn't directly prevent.

**Investigation notes:**
_(empty)_

**Proposed fix:**
_(empty)_

**Resolution:**
_(empty)_

**Related findings:**
- F04 — F30 is the cross-run behavior pattern for the same underlying issue F04 flagged. F04's fix should be the primary remedy; F30 should be **re-validated after F04's fix** across at least two runs to confirm the behavior is stable, not just clean in one run.
- F06 / F07 / F08 / F26 (silent coverage drops cluster) — iter 2's "retreat" mode is a coverage drop triggered by the LLM's uncertainty handling. F06's coverage-audit fix should surface this as "methodology context was available in REF:2 but not incorporated".
