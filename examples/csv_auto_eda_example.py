from stella_ml import OpenClawStyleHarness

harness = OpenClawStyleHarness()
result = harness.solve(
    user_query="Auto-EDA this dataset and provide recommended next steps.",
    file_path="examples/sales.csv",
    cleaning_ops=[
        {"op": "fill_missing", "column": "sales", "strategy": "mean"},
        {"op": "one_hot_encode", "column": "region"},
        {"op": "discretize", "column": "sales", "bins": 3},
    ],
    charts=[
        {"chart_type": "bar", "x": "region", "output_path": "artifacts/sales_region_bar.png"},
        {"chart_type": "hist", "y": "sales", "output_path": "artifacts/sales_hist.png"},
        {"chart_type": "pie", "x": "channel", "output_path": "artifacts/channel_pie.png"},
    ],
)

print(result.assessment)
print(result.eda_report)
print(result.insight_summary)
print(result.chart_paths)
