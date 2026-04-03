# stellaML

`stellaML` is a starter framework for model APIs + agentic orchestration + structured/unstructured data intelligence.

## Install

```bash
pip install -e .
pip install -e '.[analytics]'
pip install -e '.[ecosystem]'
pip install -e '.[unstructured]'
```

## Unstructured data extraction core (requests + BeautifulSoup + Selenium + HTML/XML/PPTX)

```python
from stella_ml import OpenClawStyleHarness

harness = OpenClawStyleHarness()

# Pull from URL (requests mode) and run NLP auto-EDA + n-grams + entities.
out = harness.run_unstructured_data_flow(
    source="https://example.com",
    source_type="url",
    fetch_mode="requests",
    ngram_n=2,
)
print(out["text_eda"])
print(out["ngrams"][:10])
print(out["entities"][:10])

# You can also parse inline HTML/XML or local PPTX:
# harness.run_unstructured_data_flow(source="<html>...</html>", source_type="html")
# harness.run_unstructured_data_flow(source="<root>...</root>", source_type="xml")
# harness.run_unstructured_data_flow(source="slides.pptx", source_type="pptx")
```

## ML ecosystem integration

```python
from stella_ml import OpenClawStyleHarness

harness = OpenClawStyleHarness()
ecosystem = harness.setup_ml_ecosystem(install_missing=False)
print(ecosystem["missing"])
print(ecosystem["cuda"])
print(len(ecosystem["sklearn_estimators"]))
```

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
    print(exp_name, report.feasible, report.score)
```

## Tabular data flow (CSV/Excel → clean → EDA → charts)

```python
from stella_ml import OpenClawStyleHarness

harness = OpenClawStyleHarness()
result = harness.solve(
    user_query="Please explore this dataset and recommend next analysis steps",
    file_path="examples/sales.csv",
    cleaning_ops=[
        {"op": "fill_missing", "column": "sales", "strategy": "mean"},
        {"op": "one_hot_encode", "column": "region"},
    ],
    charts=[{"chart_type": "bar", "x": "region", "output_path": "artifacts/region_bar.png"}],
)
print(result.eda_report)
```

## Notes

- Unstructured library core includes: URL pulling, HTML/XML/PPTX parsing, text auto-EDA, n-grams, NLTK n-grams, spaCy NER.
- `explore_chart` supports: `auto`, `bar`, `count`, `hist`, `line`, `scatter`, `box`, `pie`, `area`, `heatmap`.
- `list_sklearn_estimators()` exposes installed sklearn estimator names.
