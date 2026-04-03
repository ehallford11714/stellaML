# OpenClaw-Style Integration Review (stellaML)

## Current infrastructure overview

stellaML infrastructure is split into core subsystems:

1. **Provider + model API layer**
   - OpenAI-compatible and Hugging Face adapters via a shared provider interface.
2. **Orchestration layer**
   - `AgentRunner` for ReAct-like loops.
   - `OpenClawStyleHarness` as high-level coordinator.
3. **Data intelligence layer**
   - Structured analytics (CSV/Excel clean/EDA/chart).
   - Unstructured extraction (URL/HTML/XML/PPTX + NLP primitives).
4. **Experiment planning layer**
   - Hardware feasibility + local SOTA recommendation + API/local switching prompts.

## OpenClaw-style integration touchpoints

`OpenClawStyleHarness` currently mirrors OpenClaw-like orchestration principles:

- **Problem intake**: classifies user objective and routes flow.
- **Tool routing**: invokes structured or unstructured pipelines.
- **Capability checks**: evaluates local hardware before expensive operations.
- **Decision prompts**: asks API vs local model strategy.
- **Skill acquisition**: pulls web results and infers actionable tool skills.

## Request lifecycle (current)

1. User sends request.
2. Harness runs `evaluate_problem(...)`.
3. Depending on request:
   - `run_data_flow(...)` for tabular analysis,
   - `run_unstructured_data_flow(...)` for web/docs text extraction,
   - `run_web_search_skill_acquisition(...)` for capability discovery,
   - `propose_api_vs_local_switch(...)` for deployment decision.
4. Optional key management via `ensure_hf_key(...)`.

## Gaps and recommended next steps

1. Add explicit state machine + durable run context.
2. Add async tool execution and retry policy.
3. Add policy layer for safe web extraction and PII-aware output filtering.
4. Add end-to-end run report object with artifacts, logs, and decisions.
5. Add multi-agent role separation (planner, retriever, analyst, critic).
