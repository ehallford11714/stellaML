# stellaML

`stellaML` is a starter framework for model APIs + agentic orchestration + data intelligence workflows.

## Current schema (high-level)

- **Model layer**: normalized request/response schemas + OpenAI-compatible and Hugging Face providers.
- **Config/runtime layer**: per-agent API keys and provider registry construction.
- **Orchestration layer**:
  - `AgentRunner` for ReAct loop,
  - `OpenClawStyleHarness` for problem evaluation, data workflow execution, and hardware-feasibility planning.
- **Data layer**: CSV/Excel load, auto-impute, cleaning toolbox, auto-EDA, `explore_chart`, and `explore_data` recommendations.
- **Feasibility layer**:
  - `detect_local_hardware()` to infer local specs,
  - `is_hardware_feasible()` to score experiments,
  - `autoimpute_experiment_specs()` to generate hypothesis-driven experiments,
  - `generate_feasibility_chain()` to evaluate all candidate experiments on local hardware.

## Install

```bash
pip install -e .
pip install -e '.[analytics]'
```

## Hardware feasibility chain

```python
from stella_ml import OpenClawStyleHarness, detect_local_hardware

harness = OpenClawStyleHarness()
hardware = detect_local_hardware()

# isHardwareFeasible chain: auto-impute experiments from hypothesis and score feasibility
feasibility = harness.isHardwareFeasible(
    hypothesis="Hypothesis: gradient-boosted forecasting will beat linear baselines for demand planning",
    hardware=hardware,
)

for exp_name, report in feasibility:
    print(exp_name, report.feasible, report.score, report.reasons)
```

This provides local experiment candidates (baseline + stronger models) and whether your hardware is sufficient.

## Data flow example (CSV/Excel → clean → EDA → charts)

```python
from stella_ml import OpenClawStyleHarness

harness = OpenClawStyleHarness()
result = harness.solve(
    user_query="Please explore this dataset and recommend next analysis steps",
    file_path="examples/sales.csv",
    cleaning_ops=[
        {"op": "fill_missing", "column": "sales", "strategy": "mean"},
        {"op": "binarize", "column": "sales", "threshold": 1050, "new_column": "high_sales"},
        {"op": "discretize", "column": "sales", "bins": 3},
        {"op": "one_hot_encode", "column": "region"}
    ],
    charts=[
        {"chart_type": "bar", "x": "region", "output_path": "artifacts/region_bar.png"},
        {"chart_type": "hist", "y": "sales", "output_path": "artifacts/sales_hist.png"},
        {"chart_type": "pie", "x": "channel", "output_path": "artifacts/channel_pie.png"}
    ],
)

print(result.eda_report)
print(result.insight_summary)
print(result.chart_paths)
```

## Notes

- `explore_chart` supports: `auto`, `bar`, `count`, `hist`, `line`, `scatter`, `box`, `pie`, `area`, `heatmap`.
- `apply_cleaning_operations` supports: dedupe, binarize, discretize, normalize, standardize, one-hot, fill-missing.
- Use `choose_analysis_mode("automl" | "manual" | "ask", objective)` to enable AutoML mode or request user choice.
