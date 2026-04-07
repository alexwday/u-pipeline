/* U-Pipeline Debug Interface — vanilla JS application */

"use strict";

/* ────────────────────────────── state ─────────────────────────────── */

var testCases = [];
var currentResult = null;
var currentTraceId = "";
var streamBuffer = "";
var lastQueryParams = null;
var stopwatchStart = 0;
var stopwatchInterval = null;

/* ────────────────────────────── init ──────────────────────────────── */

document.addEventListener("DOMContentLoaded", function () {
    loadTestQueries();
    loadBanksPeriods();
});

function loadTestQueries() {
    fetch("/api/test-queries")
        .then(function (r) { return r.json(); })
        .then(function (cases) {
            testCases = cases;
            var sel = document.getElementById("test-query-select");
            cases.forEach(function (tc, i) {
                var opt = document.createElement("option");
                opt.value = i;
                opt.textContent = tc.name;
                sel.appendChild(opt);
            });
            sel.addEventListener("change", onTestQuerySelect);
        })
        .catch(function (err) {
            console.error("Failed to load test queries:", err);
        });
}

function loadBanksPeriods() {
    fetch("/api/banks-periods")
        .then(function (r) { return r.json(); })
        .then(function (data) {
            renderSelectOptions("bank-select", data.banks);
            renderSelectOptions("period-select", data.periods);
            renderSelectOptions("source-select", data.sources);
        })
        .catch(function (err) {
            console.error("Failed to load banks/periods:", err);
        });
}

function renderSelectOptions(selectId, items) {
    var sel = document.getElementById(selectId);
    sel.innerHTML = "";
    items.forEach(function (item) {
        var opt = document.createElement("option");
        opt.value = item;
        opt.textContent = item;
        sel.appendChild(opt);
    });
    makeToggleSelect(sel);
}

function makeToggleSelect(sel) {
    sel.addEventListener("mousedown", function (e) {
        if (e.target.tagName !== "OPTION") return;
        e.preventDefault();
        e.target.selected = !e.target.selected;
        sel.dispatchEvent(new Event("change"));
    });
}

function renderCheckboxes(containerId, items, name) {
    var container = document.getElementById(containerId);
    container.innerHTML = "";
    items.forEach(function (item) {
        var lbl = document.createElement("label");
        var cb = document.createElement("input");
        cb.type = "checkbox";
        cb.name = name;
        cb.value = item;
        var span = document.createElement("span");
        span.textContent = item;
        lbl.appendChild(cb);
        lbl.appendChild(span);
        container.appendChild(lbl);
    });
}

/* ──────────────────────── test query select ───────────────────────── */

function onTestQuerySelect() {
    var sel = document.getElementById("test-query-select");
    if (sel.value === "") return;

    var tc = testCases[parseInt(sel.value, 10)];
    document.getElementById("query-input").value = tc.query;

    clearSelect("bank-select");
    clearSelect("period-select");
    clearSelect("source-select");

    if (tc.combos) {
        tc.combos.forEach(function (c) {
            selectValue("bank-select", c.bank);
            selectValue("period-select", c.period);
        });
    }
    if (tc.sources) {
        tc.sources.forEach(function (s) {
            selectValue("source-select", s);
        });
    }
}

function clearSelect(selectId) {
    var sel = document.getElementById(selectId);
    Array.from(sel.options).forEach(function (opt) {
        opt.selected = false;
    });
}

function selectValue(selectId, value) {
    var sel = document.getElementById(selectId);
    Array.from(sel.options).forEach(function (opt) {
        if (opt.value === value) opt.selected = true;
    });
}

function getSelectedValues(selectId) {
    var sel = document.getElementById(selectId);
    var values = [];
    Array.from(sel.selectedOptions).forEach(function (opt) {
        values.push(opt.value);
    });
    return values;
}

function uncheckAll(name) {
    var cbs = document.querySelectorAll(
        "input[name=\"" + name + "\"]"
    );
    cbs.forEach(function (cb) { cb.checked = false; });
}

function checkValue(name, value) {
    var cb = document.querySelector(
        "input[name=\"" + name + "\"][value=\"" + value + "\"]"
    );
    if (cb) cb.checked = true;
}

function getCheckedValues(name) {
    var checked = [];
    document.querySelectorAll(
        "input[name=\"" + name + "\"]:checked"
    ).forEach(function (cb) { checked.push(cb.value); });
    return checked;
}

/* ─────────────────────────── run query ────────────────────────────── */

function runQuery() {
    var query = document.getElementById("query-input").value.trim();
    var banks = getSelectedValues("bank-select");
    var periods = getSelectedValues("period-select");
    var sources = getSelectedValues("source-select");

    if (!query) { alert("Enter a query"); return; }
    if (banks.length === 0) { alert("Select at least one bank"); return; }
    if (periods.length === 0) {
        alert("Select at least one period");
        return;
    }

    var combos = [];
    banks.forEach(function (b) {
        periods.forEach(function (p) {
            combos.push({ bank: b, period: p });
        });
    });

    var body = { query: query, combos: combos };
    if (sources.length > 0) body.sources = sources;

    lastQueryParams = {
        query: query,
        banks: banks,
        periods: periods,
        sources: sources,
    };

    resetUI();
    startStopwatch();
    document.getElementById("run-btn").disabled = true;
    document.getElementById("export-btn").disabled = true;

    fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
    }).then(function (response) {
        var reader = response.body.getReader();
        var decoder = new TextDecoder();
        var buffer = "";

        function processChunk(readResult) {
            if (readResult.done) return;
            buffer += decoder.decode(readResult.value, { stream: true });

            var parts = buffer.split("\n\n");
            buffer = parts.pop();

            parts.forEach(function (part) {
                var eventType = "";
                var data = "";
                part.split("\n").forEach(function (line) {
                    if (line.indexOf("event: ") === 0) {
                        eventType = line.slice(7);
                    }
                    if (line.indexOf("data: ") === 0) {
                        data = line.slice(6);
                    }
                });
                if (eventType && data) {
                    handleSSE(eventType, data);
                }
            });

            return reader.read().then(processChunk);
        }

        return reader.read().then(processChunk);
    }).catch(function (err) {
        stopStopwatch();
        setStatus("error", "Request failed: " + err.message);
        document.getElementById("run-btn").disabled = false;
    });
}

/* ─────────────────────────── stopwatch ────────────────────────────── */

function startStopwatch() {
    stopwatchStart = Date.now();
    setStatus("running", "Running 0:00.0");
    if (stopwatchInterval) clearInterval(stopwatchInterval);
    stopwatchInterval = setInterval(function () {
        var elapsed = (Date.now() - stopwatchStart) / 1000;
        setStatus("running", "Running " + formatStopwatch(elapsed));
    }, 100);
}

function stopStopwatch() {
    if (stopwatchInterval) {
        clearInterval(stopwatchInterval);
        stopwatchInterval = null;
    }
    return (Date.now() - stopwatchStart) / 1000;
}

function formatStopwatch(seconds) {
    var minutes = Math.floor(seconds / 60);
    var rest = seconds - minutes * 60;
    return minutes + ":" + rest.toFixed(1).padStart(4, "0");
}

function handleSSE(eventType, data) {
    if (eventType === "chunk") {
        var text = JSON.parse(data);
        appendResponse(text);
    } else if (eventType === "result") {
        var elapsed = stopStopwatch();
        var result = JSON.parse(data);
        currentResult = result;
        currentTraceId = result.trace_id || "";
        renderResult(result);
        setStatus("done", "Complete in " + formatStopwatch(elapsed));
        document.getElementById("run-btn").disabled = false;
        document.getElementById("export-btn").disabled = false;
    } else if (eventType === "error") {
        stopStopwatch();
        var msg = JSON.parse(data);
        setStatus("error", "Error: " + msg);
        document.getElementById("run-btn").disabled = false;
    }
}

/* ─────────────────────────── UI helpers ───────────────────────────── */

function setStatus(cls, text) {
    var bar = document.getElementById("status-bar");
    bar.className = "status-bar " + cls;
    bar.textContent = text;
}

function resetUI() {
    streamBuffer = "";
    var el = document.getElementById("response-text");
    el.innerHTML = "";
    el.classList.remove("empty");
    document.getElementById("debug-empty").style.display = "none";
    document.getElementById("debug-content").style.display = "block";
    document.getElementById("metrics-grid").innerHTML = "";
    document.getElementById("token-grid").innerHTML = "";
    document.getElementById("prepared-body").innerHTML = "";
    document.getElementById("sources-body").innerHTML = "";
    document.getElementById("trace-content").innerHTML = "";
    document.getElementById("source-traces-body").innerHTML = "";
    currentResult = null;
    currentTraceId = "";
}

function appendResponse(text) {
    streamBuffer += text;
    var el = document.getElementById("response-text");
    el.innerHTML = marked.parse(streamBuffer);
    var section = el.parentElement;
    section.scrollTop = section.scrollHeight;
}

/* ──────────────────────── render final result ─────────────────────── */

function renderResult(result) {
    renderMetrics(result.metrics || {});
    renderTokens(result.metrics || {});
    renderSources(result.combo_results || []);
    addCopyButtonsToTables();
    if (currentTraceId) {
        loadTrace();
    }
}

/* ─────────────────────────── export ───────────────────────────────── */

function exportResult() {
    if (!currentResult || !lastQueryParams) return;
    var html = buildExportHtml(currentResult, lastQueryParams);
    var blob = new Blob([html], { type: "text/html;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "query_export_" + exportTimestamp() + ".html";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function exportTimestamp() {
    var d = new Date();
    var pad = function (n) { return n.toString().padStart(2, "0"); };
    return d.getFullYear() +
        pad(d.getMonth() + 1) + pad(d.getDate()) + "_" +
        pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds());
}

function escapeHtml(text) {
    var div = document.createElement("div");
    div.textContent = text == null ? "" : String(text);
    return div.innerHTML;
}

function buildExportHtml(result, params) {
    var responseHtml = marked.parse(
        result.consolidated_response || ""
    );
    var sourceCardsHtml = buildSourceCardsHtml(
        result.combo_results || []
    );
    var metaList = buildMetaList(params, result);
    var title = "Query Export — " + exportTimestamp();

    return [
        "<!DOCTYPE html>",
        '<html lang="en"><head>',
        '<meta charset="UTF-8">',
        "<title>" + escapeHtml(title) + "</title>",
        "<style>" + EXPORT_CSS + "</style>",
        "</head><body>",
        '<div class="wrap">',
        "<header>",
        "<h1>U-Pipeline Query Export</h1>",
        '<div class="timestamp">' + escapeHtml(new Date().toString()) + "</div>",
        "</header>",
        '<section class="query-info">',
        "<h2>Query</h2>",
        '<p class="query-text">' + escapeHtml(params.query) + "</p>",
        '<dl class="meta-list">',
        metaList,
        "</dl>",
        "</section>",
        '<section class="response">',
        "<h2>Response</h2>",
        '<div class="response-body">',
        responseHtml,
        "</div>",
        "</section>",
        '<section class="sources">',
        "<h2>Source Findings</h2>",
        sourceCardsHtml,
        "</section>",
        "</div>",
        "</body></html>",
    ].join("\n");
}

function buildMetaList(params, result) {
    var rows = [];
    rows.push(metaRow("Banks", params.banks.join(", ")));
    rows.push(metaRow("Periods", params.periods.join(", ")));
    rows.push(metaRow(
        "Sources",
        params.sources.length > 0 ? params.sources.join(", ") : "All"
    ));
    var metrics = result.metrics || {};
    if (metrics.wall_time_seconds) {
        rows.push(metaRow(
            "Wall Time",
            metrics.wall_time_seconds.toFixed(2) + "s"
        ));
    }
    if (result.trace_id) {
        rows.push(metaRow("Trace ID", result.trace_id));
    }
    return rows.join("");
}

function metaRow(label, value) {
    return "<dt>" + escapeHtml(label) + "</dt>" +
        "<dd>" + escapeHtml(value) + "</dd>";
}

function buildSourceCardsHtml(comboResults) {
    if (comboResults.length === 0) {
        return "<p>No source findings.</p>";
    }
    var parts = [];
    comboResults.forEach(function (cr) {
        var combo = cr.combo || {};
        var source = cr.source || {};
        var title = (combo.bank || "?") + " " +
            (combo.period || "?") + " — " +
            (source.data_source || "?") +
            " (" + (source.filename || "") + ")";
        var findings = cr.findings || [];
        var findingsHtml = buildFindingsHtml(findings);
        parts.push(
            "<details>" +
            "<summary>" + escapeHtml(title) +
            " <span class=\"count\">(" +
            findings.length + " findings)</span></summary>" +
            '<div class="findings">' + findingsHtml + "</div>" +
            "</details>"
        );
    });
    return parts.join("\n");
}

function buildFindingsHtml(findings) {
    if (findings.length === 0) {
        return '<p class="empty">No findings.</p>';
    }
    return findings.map(function (f) {
        var meta = [];
        if (f.metric_name) {
            var m = f.metric_name;
            if (f.metric_value) m += " = " + f.metric_value;
            meta.push(m);
        }
        if (f.period) meta.push("Period: " + f.period);
        if (f.segment) meta.push("Segment: " + f.segment);
        var location = "Page " + f.page;
        if (f.location_detail) location += " — " + f.location_detail;
        return '<div class="finding">' +
            '<div class="finding-text">' +
            escapeHtml(f.finding) + "</div>" +
            '<div class="finding-meta">' +
            escapeHtml(location) + "</div>" +
            (meta.length > 0
                ? '<div class="finding-meta">' +
                    escapeHtml(meta.join(" | ")) + "</div>"
                : "") +
            "</div>";
    }).join("");
}

var EXPORT_CSS = [
    "* { box-sizing: border-box; margin: 0; padding: 0; }",
    "body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;",
    "  background: #fafafa; color: #222; line-height: 1.6;",
    "  padding: 40px 20px; }",
    ".wrap { max-width: 900px; margin: 0 auto; background: #fff;",
    "  padding: 40px 50px; border-radius: 8px;",
    "  box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
    "header { border-bottom: 2px solid #2a5d9f; padding-bottom: 16px;",
    "  margin-bottom: 24px; }",
    "header h1 { font-size: 22px; color: #2a5d9f; font-weight: 600; }",
    ".timestamp { font-size: 12px; color: #888; margin-top: 4px; }",
    "section { margin-bottom: 32px; }",
    "h2 { font-size: 16px; color: #2a5d9f; margin-bottom: 12px;",
    "  text-transform: uppercase; letter-spacing: 0.5px;",
    "  font-weight: 600; padding-bottom: 6px;",
    "  border-bottom: 1px solid #e0e0e0; }",
    ".query-text { font-size: 15px; padding: 12px 16px;",
    "  background: #f5f5f5; border-left: 3px solid #2a5d9f;",
    "  border-radius: 4px; margin-bottom: 16px; }",
    ".meta-list { display: grid; grid-template-columns: 120px 1fr;",
    "  gap: 6px 16px; font-size: 13px; }",
    ".meta-list dt { color: #666; font-weight: 600; }",
    ".meta-list dd { color: #222; }",
    ".response-body { font-size: 14px; }",
    ".response-body h1, .response-body h2, .response-body h3 {",
    "  color: #2a5d9f; margin: 16px 0 8px; font-weight: 600;",
    "  border: none; padding: 0; text-transform: none;",
    "  letter-spacing: 0; }",
    ".response-body h2 { font-size: 17px; }",
    ".response-body h3 { font-size: 14px; }",
    ".response-body p { margin: 8px 0; }",
    ".response-body ul, .response-body ol {",
    "  margin: 8px 0; padding-left: 24px; }",
    ".response-body table { border-collapse: collapse;",
    "  width: 100%; margin: 12px 0; font-size: 13px; }",
    ".response-body th, .response-body td {",
    "  border: 1px solid #d0d0d0; padding: 8px 12px;",
    "  text-align: left; }",
    ".response-body th { background: #f0f4fa; font-weight: 600;",
    "  color: #2a5d9f; }",
    ".response-body a { color: #2a5d9f; text-decoration: none; }",
    ".response-body a:hover { text-decoration: underline; }",
    ".response-body code { background: #f0f0f0; padding: 1px 5px;",
    "  border-radius: 3px; font-family: 'SF Mono', monospace;",
    "  font-size: 12px; }",
    "details { margin-bottom: 8px; border: 1px solid #e0e0e0;",
    "  border-radius: 4px; }",
    "summary { padding: 10px 14px; cursor: pointer;",
    "  font-size: 14px; background: #f8f9fb; user-select: none;",
    "  font-weight: 500; }",
    "summary .count { color: #888; font-weight: 400;",
    "  font-size: 12px; margin-left: 6px; }",
    "details[open] summary { border-bottom: 1px solid #e0e0e0; }",
    ".findings { padding: 12px 16px; }",
    ".finding { padding: 10px 0; border-bottom: 1px solid #f0f0f0; }",
    ".finding:last-child { border-bottom: none; }",
    ".finding-text { font-size: 13px; color: #222;",
    "  margin-bottom: 4px; }",
    ".finding-meta { font-size: 11px; color: #888;",
    "  font-family: 'SF Mono', monospace; }",
    ".empty { color: #999; font-style: italic; font-size: 13px; }",
].join("\n");

function addCopyButtonsToTables() {
    var responseEl = document.getElementById("response-text");
    var tables = responseEl.querySelectorAll("table");
    tables.forEach(function (table) {
        var bodyRows = table.querySelectorAll("tbody tr").length;
        if (bodyRows < 2) return;
        var next = table.nextElementSibling;
        if (next && next.classList.contains("table-actions")) return;
        var bar = document.createElement("div");
        bar.className = "table-actions";
        var btn = document.createElement("button");
        btn.className = "copy-table-btn";
        btn.type = "button";
        btn.textContent = "Copy table";
        btn.onclick = function () {
            copyTableToClipboard(table, btn);
        };
        bar.appendChild(btn);
        table.parentNode.insertBefore(bar, table.nextSibling);
    });
}

function copyTableToClipboard(table, btn) {
    var rows = [];
    table.querySelectorAll("tr").forEach(function (tr) {
        var cells = [];
        tr.querySelectorAll("th, td").forEach(function (cell) {
            var text = cell.textContent.trim()
                .replace(/\t/g, " ")
                .replace(/\n/g, " ");
            cells.push(text);
        });
        if (cells.length > 0) rows.push(cells.join("\t"));
    });
    var tsv = rows.join("\n");
    navigator.clipboard.writeText(tsv).then(function () {
        var original = btn.textContent;
        btn.textContent = "Copied!";
        btn.classList.add("copied");
        setTimeout(function () {
            btn.textContent = original;
            btn.classList.remove("copied");
        }, 1500);
    }).catch(function (err) {
        console.error("Copy failed:", err);
        btn.textContent = "Failed";
    });
}

/* ──────────────────────────── metrics ─────────────────────────────── */

function renderMetrics(metrics) {
    var grid = document.getElementById("metrics-grid");
    grid.innerHTML = "";

    var rows = [
        ["Wall Time", fmtSeconds(metrics.wall_time_seconds)],
        ["Query Prep", fmtSeconds(
            metrics.query_prep
                ? metrics.query_prep.wall_time_seconds
                : null
        )],
        ["Doc Resolution", fmtSeconds(
            metrics.prep_resolution_wall_seconds
        )],
        ["Research (parallel)", fmtSeconds(
            metrics.parallel_research_wall_seconds
        )],
        ["Research (sequential)", fmtSeconds(
            metrics.sequential_unit_seconds
        )],
        ["Parallel Savings", fmtSeconds(
            metrics.parallel_savings_seconds
        )],
        ["Consolidation", fmtSeconds(
            metrics.consolidation
                ? metrics.consolidation.wall_time_seconds
                : null
        )],
        ["Research Units", metrics.research_units || "-"],
        ["Max Workers", metrics.max_workers || "-"]
    ];

    rows.forEach(function (row) {
        var lbl = document.createElement("div");
        lbl.className = "metric-label";
        lbl.textContent = row[0];
        var val = document.createElement("div");
        val.className = "metric-value";
        val.textContent = row[1];
        grid.appendChild(lbl);
        grid.appendChild(val);
    });
}

function renderTokens(metrics) {
    var grid = document.getElementById("token-grid");
    grid.innerHTML = "";

    var con = metrics.consolidation || {};
    var qp = metrics.query_prep || {};

    var rows = [
        ["Consolidation Prompt", fmtNumber(con.prompt_tokens)],
        ["Consolidation Completion", fmtNumber(con.completion_tokens)],
        ["Query Prep Prompt", fmtNumber(qp.prompt_tokens)],
        ["Query Prep Completion", fmtNumber(qp.completion_tokens)],
        ["Key Findings", con.key_findings || "-"],
        ["Data Gaps", con.data_gaps || "-"]
    ];

    rows.forEach(function (row) {
        var lbl = document.createElement("div");
        lbl.className = "metric-label";
        lbl.textContent = row[0];
        var val = document.createElement("div");
        val.className = "metric-value";
        val.textContent = row[1];
        grid.appendChild(lbl);
        grid.appendChild(val);
    });
}

/* ────────────────────────── source cards ──────────────────────────── */

function renderSources(comboResults) {
    var body = document.getElementById("sources-body");
    body.innerHTML = "";

    if (comboResults.length === 0) {
        body.textContent = "No source results";
        return;
    }

    comboResults.forEach(function (cr, i) {
        var card = document.createElement("div");
        card.className = "source-card";
        card.id = "source-card-" + i;

        var combo = cr.combo || {};
        var source = cr.source || {};
        var title = (combo.bank || "?") + " " +
            (combo.period || "?") + " — " +
            (source.data_source || "?");

        var header = document.createElement("div");
        header.className = "source-card-header";
        header.textContent = title;
        header.onclick = function () {
            card.classList.toggle("open");
        };

        var cardBody = document.createElement("div");
        cardBody.className = "source-card-body";

        var stats = document.createElement("div");
        stats.innerHTML =
            "<span class='source-stat'>Chunks: <strong>" +
            (cr.chunk_count || 0) + "</strong></span>" +
            "<span class='source-stat'>Tokens: <strong>" +
            fmtNumber(cr.total_tokens) + "</strong></span>" +
            "<span class='source-stat'>Iterations: <strong>" +
            (cr.research_iterations
                ? cr.research_iterations.length : 0) +
            "</strong></span>" +
            "<span class='source-stat'>Findings: <strong>" +
            (cr.findings ? cr.findings.length : 0) +
            "</strong></span>";
        cardBody.appendChild(stats);

        if (cr.findings && cr.findings.length > 0) {
            var findingsDiv = document.createElement("div");
            findingsDiv.style.marginTop = "8px";
            cr.findings.forEach(function (f) {
                var item = document.createElement("div");
                item.className = "finding-item key-finding";
                item.textContent = f.finding +
                    " (p." + f.page + ")";
                findingsDiv.appendChild(item);
            });
            cardBody.appendChild(findingsDiv);
        }

        card.appendChild(header);
        card.appendChild(cardBody);
        body.appendChild(card);
    });
}

/* ──────────────────────────── traces ──────────────────────────────── */

function loadTrace() {
    if (!currentTraceId) {
        alert("No trace available");
        return;
    }

    fetch("/api/trace/" + currentTraceId)
        .then(function (r) { return r.json(); })
        .then(function (trace) {
            renderRunTrace(trace);
            loadSourceTraces();
        })
        .catch(function (err) {
            document.getElementById("trace-content").textContent =
                "Failed to load trace: " + err.message;
        });
}

function renderRunTrace(trace) {
    var el = document.getElementById("trace-content");
    el.innerHTML = "";

    if (trace.prepared_query) {
        renderPreparedQuery(trace.prepared_query);
    }

    var block = document.createElement("div");
    block.className = "json-block";
    block.style.marginTop = "8px";

    var summary = {
        trace_id: trace.trace_id,
        created_at: trace.created_at,
        combos: trace.combos,
        sources: trace.sources,
        document_resolution: trace.document_resolution,
        metrics: trace.metrics
    };
    block.textContent = JSON.stringify(summary, null, 2);
    el.appendChild(block);
}

function renderPreparedQuery(pq) {
    var body = document.getElementById("prepared-body");
    body.innerHTML = "";

    addLabelValue(body, "Rewritten Query", pq.rewritten_query || "");
    addLabelValue(body, "HyDE Answer", pq.hyde_answer || "");
    addTagList(body, "Sub-Queries", pq.sub_queries || []);
    addTagList(body, "Keywords", pq.keywords || []);
    addTagList(body, "Entities", pq.entities || []);
}

function loadSourceTraces() {
    if (!currentTraceId) return;

    fetch("/api/trace/" + currentTraceId + "/sources")
        .then(function (r) { return r.json(); })
        .then(function (filenames) {
            renderSourceTraceList(filenames);
        })
        .catch(function () {});
}

function renderSourceTraceList(filenames) {
    var body = document.getElementById("source-traces-body");
    body.innerHTML = "";

    if (filenames.length === 0) {
        body.textContent = "No source traces";
        return;
    }

    filenames.forEach(function (fn) {
        var card = document.createElement("div");
        card.className = "source-card";

        var header = document.createElement("div");
        header.className = "source-card-header";
        header.textContent = fn.replace("source_", "")
            .replace(".json", "").replace(/_/g, " ");

        var cardBody = document.createElement("div");
        cardBody.className = "source-card-body";
        cardBody.textContent = "Click to load...";

        var loaded = false;
        header.onclick = function () {
            card.classList.toggle("open");
            if (!loaded) {
                loaded = true;
                loadSourceTrace(fn, cardBody);
            }
        };

        card.appendChild(header);
        card.appendChild(cardBody);
        body.appendChild(card);
    });
}

function loadSourceTrace(filename, container) {
    fetch("/api/trace/" + currentTraceId + "/source/" + filename)
        .then(function (r) { return r.json(); })
        .then(function (trace) {
            container.innerHTML = "";
            renderSourceTraceDetail(trace, container);
        })
        .catch(function (err) {
            container.textContent = "Failed: " + err.message;
        });
}

function renderSourceTraceDetail(trace, container) {
    var stages = trace.stages || {};

    if (stages.search) {
        addStageSection(container, "Search", stages.search);
    }
    if (stages.rerank) {
        addStageSection(container, "Rerank", stages.rerank);
    }
    if (stages.expand) {
        addStageSection(container, "Expand", stages.expand);
    }
    if (stages.research) {
        addStageSection(container, "Research", stages.research);
    }

    if (trace.outputs) {
        var outBlock = document.createElement("div");
        outBlock.style.marginTop = "8px";
        outBlock.innerHTML =
            "<div class='metric-label'>Outputs</div>";
        var json = document.createElement("div");
        json.className = "json-block";
        json.textContent = JSON.stringify(trace.outputs, null, 2);
        outBlock.appendChild(json);
        container.appendChild(outBlock);
    }
}

function addStageSection(container, name, data) {
    var section = document.createElement("div");
    section.style.marginBottom = "8px";

    var header = document.createElement("div");
    header.className = "metric-label";
    header.style.marginBottom = "4px";
    header.textContent = name;

    var block = document.createElement("div");
    block.className = "json-block";
    block.textContent = JSON.stringify(data, null, 2);

    section.appendChild(header);
    section.appendChild(block);
    container.appendChild(section);
}

/* ──────────────────────── debug section toggle ────────────────────── */

function toggleSection(id) {
    document.getElementById(id).classList.toggle("open");
}

/* ────────────────────────── helper fns ────────────────────────────── */

function fmtSeconds(val) {
    if (val === null || val === undefined) return "-";
    return val.toFixed(2) + "s";
}

function fmtNumber(val) {
    if (val === null || val === undefined) return "-";
    return val.toLocaleString();
}

function addLabelValue(parent, label, value) {
    var div = document.createElement("div");
    div.style.marginBottom = "6px";
    var lbl = document.createElement("div");
    lbl.className = "metric-label";
    lbl.textContent = label;
    var val = document.createElement("div");
    val.style.fontSize = "12px";
    val.style.lineHeight = "1.4";
    val.textContent = value;
    div.appendChild(lbl);
    div.appendChild(val);
    parent.appendChild(div);
}

function addTagList(parent, label, items) {
    if (items.length === 0) return;
    var div = document.createElement("div");
    div.style.marginBottom = "6px";
    var lbl = document.createElement("div");
    lbl.className = "metric-label";
    lbl.textContent = label;
    div.appendChild(lbl);
    var list = document.createElement("div");
    list.className = "tag-list";
    items.forEach(function (item) {
        var tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = item;
        list.appendChild(tag);
    });
    div.appendChild(list);
    parent.appendChild(div);
}
