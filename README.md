# stellaML

`stellaML` is a starter framework for model APIs + agentic orchestration + data intelligence workflows.

## Current schema (high-level)

- **Model layer**
  - `InferenceRequest`, `InferenceResponse`, `Message`, `Role` for normalized provider payloads.
  - Providers: `OpenAICompatibleProvider`, `HuggingFaceProvider`.
- **Agent config layer**
  - `StellaConfig` / `AgentProfile`: per-agent model, provider default, API keys, and base URLs.
  - `build_registry_for_agent()` builds live provider registry from stored keys.
- **Orchestration layer**
  - `AgentRunner`: ReAct-style single-agent loop (`TOOL:` / `FINAL:` protocol).
  - `OpenClawStyleHarness`: higher-level problem assessment + data flow orchestration.
- **Data intelligence layer**
  - `load_tabular_file()` supports CSV and Excel.
  - `infer_structure()`, `impute_missing()`, `auto_eda()`.
  - `apply_cleaning_operations()` toolbox (binarize, discretize, normalize, standardize, one-hot, fill missing, dedupe).
  - `explore_chart()` generates multiple chart types.
  - `explore_data()` emits insight summary and next-step recommendations.

## Install

```bash
pip install -e .
pip install -e '.[analytics]'
```

`analytics` extra installs `matplotlib` (for charting) and `openpyxl` (for Excel support).

## Per-agent API key config

Create `~/.stella_ml/config.json`:

```json
{
  "agents": {
    "analyst": {
      "model": "gpt-4.1-mini",
      "default_provider": "openai",
      "api_keys": {
        "openai": "<OPENAI_OR_COMPATIBLE_KEY>",
        "huggingface": "<HF_TOKEN>"
      },
      "base_urls": {
        "openai": "https://api.openai.com/v1",
        "huggingface": "https://api-inference.huggingface.co/models"
      }
    }
  }
}
```

## Orchestration harness usage

```python
from stella_ml import OpenClawStyleHarness, choose_analysis_mode

harness = OpenClawStyleHarness()
assessment = harness.evaluate_problem("Forecast monthly demand from this CSV with AutoML")
plan = choose_analysis_mode(user_preference="ask", objective=assessment.objective)
print(assessment)
print(plan.prompt_user)
```

## Data flow example (CSV/Excel → EDA → charts)

```python
from stella_ml import OpenClawStyleHarness

harness = OpenClawStyleHarness()
result = harness.solve(
    user_query="Please explore this dataset and recommend next analysis steps",
    file_path="examples/sales.csv",
    cleaning_ops=[
        {"op": "fill_missing", "column": "sales", "strategy": "mean"},
        {"op": "binarize", "column": "sales", "threshold": 1050, "new_column": "high_sales"},
        {"op": "discretize", "column": "sales", "bins": 3}
    ],
    charts=[
        {"chart_type": "bar", "x": "region", "output_path": "artifacts/region_bar.png"},
        {"chart_type": "hist", "y": "sales", "output_path": "artifacts/sales_hist.png"}
    ],
)

print(result.assessment)
print(result.eda_report)
print(result.insight_summary)
print(result.chart_paths)
```

## SOTA-inspired extension roadmap

Framework patterns to continue borrowing:
- **LangGraph-style graph states**: explicit state machine nodes for plan/act/review loops.
- **AutoGen-style multi-agent teams**: planner, analyst, critic, and tool-executor roles.
- **CrewAI-style role memories**: role goals + context windows + delegated subtasks.
- **OpenClaw-style tool rigor**: strict tool contracts, retries, and guarded execution.

Potential next additions:
1. DAG-based orchestrator with branch/merge policies.
2. Judge/critic model pass for quality scoring.
3. Retrieval layer for long context (vector + SQL hybrid).
4. AutoML executor plugins (classification/regression/time-series).
5. Cost + latency budgets per run.
6. Human-in-the-loop checkpoints for irreversible actions.
