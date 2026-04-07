# U-Pipeline Prompt Standard

> Standardized prompt template for all LLM interactions in the u-pipeline project.
> Based on research from OpenAI (GPT-5.4 guidance, March 2026) and Anthropic (context engineering, 2025-2026).

---

## 1. Research Summary

### Model Context

- **Target model**: GPT-5.4 mini (or GPT-5 mini) via OpenAI SDK
- **Interaction mode**: Tool calling only (no freeform text), structured output via function schemas
- **Key parameter**: `reasoning_effort` — controls how much the model deliberates before responding
- **API**: Responses API recommended for new projects (better caching, reasoning preservation)

### Key Findings from OpenAI (GPT-5.4 Prompting Guidance)

1. **GPT-5.4 mini is more literal than larger models.** It makes fewer assumptions. Use numbered steps, explicit action definitions, and structural scaffolding. Don't rely on "you MUST" alone.
2. **XML tags + Markdown headers together** are the recommended formatting approach. Markdown for hierarchy/sections, XML for content boundaries and metadata.
3. **Over-thoroughness backfires.** Saying "Be THOROUGH" causes over-calling tools. Use softer phrasing: "If you've performed an edit that partially fulfills the query but you're not confident, gather more information."
4. **Remove maximize prefixes.** Tags like `<maximize_context_understanding>` are counterproductive — GPT-5 naturally gathers context.
5. **Frontload key rules in tool descriptions.** Placing usage criteria before general descriptions showed a 6% accuracy improvement.
6. **Flat schemas outperform nested.** Easier for the model to reason about.
7. **Strict mode always.** `strict: true` + `additionalProperties: false` on every tool.
8. **Prompt caching**: Static content first (system prompt, tool definitions), dynamic content last (user input). Never put timestamps or request IDs in system prompts.
9. **Developer role**: For reasoning models, system messages are auto-converted to developer messages. Use `developer` role explicitly for clarity.
10. **Reasoning effort**: Default for GPT-5.4 is `none`. At lower effort, prompt clarity matters MORE. At higher effort, examples matter MORE.

### Key Findings from Anthropic (Context Engineering)

1. **XML tags remain the primary recommendation** for structuring prompts (model-trained to respect XML boundaries).
2. **Detailed tool descriptions are the single most important factor** in tool performance. Aim for 3-4+ sentences.
3. **Tell the model what to do, not what not to do.** Positive framing outperforms negative constraints.
4. **Provide motivation for rules** — explain *why*, not just *what*.
5. **Long documents at the top**, queries/instructions at the end improves quality by up to 30%.
6. **Newer models are more responsive to system prompts** — dial back aggressive language.

### Changes from Older Prompting (CO-STAR, GPT-4 era)

| Old Pattern | New Pattern | Why |
|-------------|-------------|-----|
| CO-STAR framework | Role-Instruction-Context-Tools | CO-STAR is for generative text, not tool-calling workflows |
| "You MUST always..." | "Use X when Y." | Newer models overtrigger on aggressive language |
| `<maximize_X>` wrapper tags | Flat, descriptive sections | GPT-5+ naturally gathers context; maximize tags cause over-thoroughness |
| Minimal tool descriptions | 3-4 sentence descriptions with usage criteria | Description quality is the #1 factor in tool accuracy |
| Optional `strict` mode | Always `strict: true` | Eliminates malformed JSON and schema drift |
| Deeply nested JSON schemas | Flat schemas | Models reason better about flat structures |
| "Think step-by-step" | Remove for tool-calling models | Degrades tool-calling performance on reasoning models |

---

## 2. YAML Prompt Template

Every prompt YAML file in this project follows this structure:

```yaml
# ─── Metadata ───────────────────────────────────────────────
stage: "<pipeline_stage_name>"
version: "<semver>"
description: >
  One-line purpose of this prompt. What decision or extraction
  does it perform?

# ─── System Prompt ──────────────────────────────────────────
#
# Structure:
#   1. Role       — one sentence, who the model is
#   2. Task       — one sentence, what it must accomplish
#   3. Context    — domain knowledge the model needs (brief)
#   4. Tool rule  — always end with "Always use the provided tool."
#
# Guidelines:
#   - Keep under 100 words. System prompt sets identity, not logic.
#   - No "you MUST" or "CRITICAL" — use declarative statements.
#   - Do not duplicate rules that belong in user_prompt.
#   - Static content only (no variable interpolation).
#
system_prompt: >
  You are a [specific role] for [specific document type].
  [One sentence: what the model must accomplish].
  [One sentence: key domain context if needed].
  Always use the provided tool.

# ─── User Prompt ────────────────────────────────────────────
#
# Structure (in order):
#   1. Task statement    — what to do with the input
#   2. Input description — what the model will receive (if not obvious)
#   3. Decision criteria — when/how to choose between outputs
#   4. Rules             — numbered list, ordered by importance
#   5. Examples          — concrete input/output pairs (optional)
#
# Guidelines:
#   - Use numbered rules when order matters. Use bullets when it doesn't.
#   - Frontload the most important rules (mini models read top-down).
#   - Tell the model what to do, not what not to do.
#   - Explain WHY behind non-obvious rules.
#   - Keep examples short — 2-3 input/output pairs max.
#   - Dynamic content (page images, context from prior stages) is
#     injected at call time by the pipeline, not written here.
#
user_prompt: >
  [Task statement — what to do with the input provided.]

  [Decision criteria — when applicable, describe the choices and
  what drives each decision.]

  Rules:
  1. [Most important rule first.]
  2. [Second most important rule.]
  3. [Continue in priority order.]

  Examples:
  - [Input description] -> [Expected output/decision]
  - [Input description] -> [Expected output/decision]

# ─── Tool Choice ────────────────────────────────────────────
#
# Options:
#   "required" — model must call exactly one tool (default for this project)
#   "auto"     — model decides whether to call a tool
#   "none"     — tools are provided for schema reference but model must not call them
#   {type: "function", function: {name: "X"}} — force a specific tool
#
tool_choice: required

# ─── Tool Definitions ──────────────────────────────────────
#
# Each tool follows this structure. Key rules:
#
#   1. DESCRIPTION (most important field):
#      - First sentence: WHEN to use this tool (usage criteria).
#      - Second sentence: WHAT it does.
#      - Third sentence: WHAT it returns / side effects.
#      - Add negative guidance if there's a common misuse case.
#      - Minimum 2 sentences, target 3-4 for complex tools.
#
#   2. PARAMETERS:
#      - Use flat schemas (avoid nesting objects inside objects).
#      - Every property must be in the `required` list.
#      - Use `enum` whenever the value set is known and closed.
#      - Use descriptive names: `handling_mode` not `mode`.
#      - Include format hints in descriptions: "ISO 8601 date", "0.0 to 1.0".
#      - For optional values, use nullable types: type: ["string", "null"]
#
#   3. NAMING:
#      - verb_noun format: classify_sheet, extract_content, detect_sections
#      - Match the pipeline stage name where possible.
#
#   4. STRICT MODE:
#      - Always set strict: true at the API call level.
#      - Always include additionalProperties: false in the schema.
#      - These are enforced by the LLM connector, not in the YAML.
#
tools:
  - type: function
    function:
      name: verb_noun_action
      description: >
        Call this tool when [specific trigger condition].
        [What the tool does in one sentence].
        Returns [what the caller receives].
      parameters:
        type: object
        properties:
          # ── Classification/decision fields ──
          decision_field:
            type: string
            enum:
              - option_a
              - option_b
            description: >
              [What this field represents]. Choose option_a when
              [criteria]. Choose option_b when [criteria].

          # ── Extracted content fields ──
          content_field:
            type: string
            description: >
              [What to put here]. [Format guidance if needed].

          # ── Numeric score fields ──
          confidence:
            type: number
            description: >
              Confidence in the decision, from 0.0 (uncertain)
              to 1.0 (certain).

          # ── Reasoning/traceability field ──
          rationale:
            type: string
            description: >
              Brief explanation of why this decision was made.
              Used for pipeline traceability, not shown to end users.

        required:
          - decision_field
          - content_field
          - confidence
          - rationale
        additionalProperties: false
```

---

## 3. Reasoning Effort Guide

Set `reasoning_effort` per-call based on the task type:

| Level | Use For | Examples in This Pipeline |
|-------|---------|--------------------------|
| `none` | Simple extraction, routing, formatting | Page furniture detection, metadata field extraction |
| `low` | Classification with clear criteria | Sheet classification (page_like vs dense_table), structure classification |
| `medium` | Extraction requiring judgment | Page content extraction, section detection, chunk summarization |
| `high` | Complex synthesis, ambiguous inputs | Document rollup, cross-page continuation decisions |

**Rule of thumb**: If the tool schema has an `enum` with clear criteria in the user prompt, `low` is usually sufficient. If the model must generate free-text content or make judgment calls, use `medium` or `high`.

---

## 4. API Call Parameters

The LLM connector should enforce these defaults on every call:

```python
# Standard parameters for all tool-calling requests
{
    "model": "gpt-5-mini",          # or gpt-5.4-mini when available
    "tools": tools,                  # from YAML
    "tool_choice": tool_choice,      # from YAML (default: "required")
    "reasoning_effort": "medium",    # override per-stage as needed
    "strict": True,                  # enforced on all tool schemas
    "temperature": None,             # not supported when reasoning_effort != "none"
}
```

**Caching optimization**: The system prompt + tool definitions are static per prompt template. The user message (containing the actual document content) is dynamic. This ordering maximizes prompt cache hits.

---

## 5. Prompt Quality Checklist

Before merging a new or modified prompt YAML, verify:

- [ ] **System prompt** is under 100 words and contains: role, task, tool rule
- [ ] **User prompt** leads with the task statement, not background context
- [ ] **Rules are numbered** and ordered by importance (most important first)
- [ ] **Tool description** has 2+ sentences starting with WHEN to use
- [ ] **All parameters** are in the `required` list
- [ ] **Enum fields** are used wherever the value set is closed
- [ ] **`additionalProperties: false`** is set on the parameters object
- [ ] **`confidence` field** is present for classification/decision tools
- [ ] **`rationale` field** is present for traceability
- [ ] **No aggressive language** ("MUST", "CRITICAL", "ALWAYS" in caps) — use declarative statements
- [ ] **No "think step-by-step"** or chain-of-thought instructions
- [ ] **Examples** are provided for non-obvious classification decisions
- [ ] **Reasoning effort** level is documented or justified for the stage

---

## 6. Migration Notes

When updating existing prompts to this standard:

1. **System prompt**: Trim to role + task + tool rule. Move all logic to user prompt.
2. **User prompt**: Reorder to task-first, then criteria, then numbered rules, then examples.
3. **Tool descriptions**: Expand to 2+ sentences with usage criteria first.
4. **Add `additionalProperties: false`** to all parameter schemas.
5. **Add `rationale` field** to all tools that make decisions/classifications.
6. **Remove aggressive language**: Replace "Always do X" with "Do X." Replace "NEVER do Y" with "Do Z instead of Y."
7. **Set reasoning effort**: Add a comment or config mapping for each prompt's recommended reasoning effort.

---

## Sources

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [GPT-5 Prompting Guide (Cookbook)](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide)
- [GPT-4.1 Prompting Guide (Cookbook)](https://developers.openai.com/cookbook/examples/gpt4-1_prompting_guide)
- [Prompt Guidance for GPT-5.4](https://developers.openai.com/api/docs/guides/prompt-guidance)
- [o3/o4-mini Function Calling Guide](https://developers.openai.com/cookbook/examples/o-series/o3o4-mini_prompting_guide)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI Structured Outputs Guide](https://developers.openai.com/api/docs/guides/structured-outputs)
- [OpenAI Prompt Caching Guide](https://platform.openai.com/docs/guides/prompt-caching)
- [Anthropic Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Anthropic Writing Tools for Agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [Anthropic Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
- [Anthropic XML Tags Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags)
