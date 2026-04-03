# stellaML

`stellaML` is a starter framework for model APIs + agentic orchestration + data intelligence workflows.

## Install

```bash
pip install -e .
pip install -e '.[analytics]'
# Full ecosystem (NLP + DL + probabilistic)
pip install -e '.[ecosystem]'
```

## ML ecosystem integration (NLTK, spaCy, TensorFlow, Keras, PyTorch, sklearn, PyMC)

```python
from stella_ml import OpenClawStyleHarness

harness = OpenClawStyleHarness()

# Detect available backends and optionally install missing ones.
ecosystem = harness.setup_ml_ecosystem(install_missing=False)
print(ecosystem["missing"])
print(ecosystem["cuda"])
print(len(ecosystem["sklearn_estimators"]))

# Build custom architectures.
pt_model = harness.create_custom_architecture("pytorch", input_dim=32, output_dim=1)
tf_model = harness.create_custom_architecture("tensorflow", input_dim=32, output_dim=1)
```

### CUDA demo + CPU fallback using 1-bit/ultra-quantized recommendations

If CUDA is unavailable, `setup_ml_ecosystem()` exposes `cpu_1bit_recommendations` for BitNet-style / GGUF low-bit CPU workflows.

## Hardware feasibility chain

```python
from stella_ml import OpenClawStyleHarness, detect_local_hardware

harness = OpenClawStyleHarness()
hardware = detect_local_hardware()

feasibility = harness.isHardwareFeasible(
    hypothesis="Hypothesis: gradient-boosted forecasting will beat linear baselines",
    hardware=hardware,
)
for exp_name, report in feasibility:
    print(exp_name, report.feasible, report.score, report.reasons)
```

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
- `list_sklearn_estimators()` exposes all available sklearn estimator names in the current environment.
- `run_pymc_linear_regression()` provides a lightweight Bayesian regression utility when PyMC is installed.
