# stellaML Framework Summary and Expansion Roadmap

## 1) Current framework summary

stellaML currently includes six core layers:

1. **Model + provider layer**
   - Unified request/response datamodels (`InferenceRequest`, `InferenceResponse`, `Message`, `Role`).
   - Provider abstraction (`BaseModelProvider`) and registry (`ProviderRegistry`).
   - Adapters for OpenAI-compatible and HuggingFace inference APIs.

2. **Agent orchestration layer**
   - `AgentRunner`: lightweight ReAct-style loop (`TOOL:` / `FINAL:` protocol).
   - `OpenClawStyleHarness`: higher-level orchestration for problem assessment, structured data flow, unstructured extraction, and ecosystem setup.

3. **Structured data intelligence layer**
   - CSV/Excel loading, type inference, casting, missing-value imputation.
   - Cleaning toolbox (dedupe, fill missing, binarize, discretize, normalization, standardization, one-hot).
   - EDA summary + recommendation generation.
   - Chart generation for multiple chart types.

4. **Unstructured data intelligence layer**
   - URL pulling (`requests`, `selenium`), HTML/XML/PPTX parsing.
   - Text auto-EDA, tokenization, n-grams, optional NLTK n-grams and spaCy NER.
   - Orchestrated pipeline for unstructured extraction + NLP analysis.

5. **Feasibility and planning layer**
   - Local hardware detection (CPU/RAM/GPU info).
   - Hypothesis-driven auto-imputation of experiment specs.
   - Feasibility scoring and mitigation suggestions.

6. **ML ecosystem integration layer**
   - Optional ecosystem introspection/installation for NLP/DL/probabilistic stack.
   - CUDA readiness checks and CPU low-bit recommendation hints.
   - Helper architecture builders for TensorFlow/Keras and PyTorch MLP scaffolds.

---

## 2) Current functionality matrix

### ✅ Already supported

- Tabular ingest (CSV + Excel)
- Unstructured ingest (URL, HTML, XML, PPTX)
- Auto-EDA for table and text
- Basic feature cleaning and transformations
- Multi-chart exploration
- Hardware feasibility scoring
- Provider-based model invocation
- Agentic tool orchestration

### ⚠️ Partially supported / starter-level

- Deep-learning training loops (scaffolded model builders, no trainer abstraction yet)
- AutoML (mode-selection and planning helpers, no complete model search engine yet)
- Unstructured web crawling (single URL pull supported; no crawler/pagination manager)
- Selenium runtime management (assumes Chrome/WebDriver availability)

---

## 3) Recommended expansion plan

## Phase A — reliability and contracts

1. **Typed orchestration states**
   - Introduce a finite-state execution graph with explicit step contracts and retries.
2. **Unified error envelope**
   - Standardize errors across providers, extractors, and pipeline stages.
3. **Telemetry and run metadata**
   - Add run IDs, stage latency, token/compute counters, and artifact tracking.

## Phase B — data extraction at scale

1. **Crawler module**
   - Depth-limited crawling, robots policy handling, domain allow/deny lists.
2. **Document loaders**
   - Add PDF/DOCX/JSONL/parquet connectors.
3. **Content normalization schema**
   - Structured output with source URL, timestamp, mime-type, language, and chunk IDs.

## Phase C — analytics and modeling depth

1. **AutoML execution engine**
   - Search spaces for classification/regression/time series with early stopping.
2. **Feature store abstraction**
   - Reusable feature definitions and transformation lineage.
3. **Probabilistic and causal modules**
   - Expand PyMC workflows, posterior diagnostics, and intervention analysis templates.

## Phase D — agentic productionization

1. **Multi-agent roles**
   - Planner, extractor, analyst, critic, and reporter with handoff protocol.
2. **Policy guardrails**
   - Safe tool invocation constraints and sensitive-domain policy checks.
3. **Human-in-the-loop review**
   - Approval checkpoints before expensive or destructive actions.

---

## 4) Architecture recommendations

1. **Plugin registry pattern**
   - Convert data connectors, chart engines, and model providers into plugin interfaces.
2. **Execution graph abstraction**
   - Implement DAG-based orchestration (node inputs/outputs + state persistence).
3. **Artifact layer**
   - Persist outputs (tables, reports, charts, logs) under a run directory with manifest.
4. **Config profiles**
   - Extend per-agent configs to include default extraction pipelines and cost limits.

---

## 5) Suggested near-term backlog (high impact)

1. Add `PipelineRun` class with event log + status transitions.
2. Add `ExtractorRegistry` with URL/PDF/PPTX/HTML/XML plugins.
3. Add `ModelTrainer` abstraction with sklearn baseline trainers.
4. Add `ReportComposer` to generate markdown/html analysis reports from artifacts.
5. Add integration tests for full scenario: `URL -> NLP -> hypothesis -> feasibility -> EDA -> chart -> report`.

---

## 6) Success metrics to track

- End-to-end run success rate
- Median pipeline runtime
- % runs requiring manual retries
- Extraction coverage (content parsed / content fetched)
- Model experiment throughput on local hardware
- Cost and latency per run

