from stella_ml import OpenClawStyleHarness, detect_local_hardware

harness = OpenClawStyleHarness()

# 0) Unstructured extraction demo (inline HTML so it runs offline).
unstructured = harness.run_unstructured_data_flow(
    source="<html><body><h1>Sales Report</h1><p>West region grew rapidly in Q4.</p></body></html>",
    source_type="html",
    ngram_n=2,
)
print("Unstructured EDA:", unstructured["text_eda"])
print("Top ngrams:", unstructured["ngrams"][:5])

# 1) Integrate ML ecosystem and show CUDA readiness + CPU fallback recommendations.
ecosystem = harness.setup_ml_ecosystem(install_missing=False)
print("Missing packages:", ecosystem["missing"])
print("CUDA:", ecosystem["cuda"])

# 2) Evaluate local feasibility for hypothesis-driven experiments.
hardware = detect_local_hardware()
feasibility = harness.isHardwareFeasible(
    "Hypothesis: forecasting with tree-based models improves demand accuracy over linear baseline",
    hardware=hardware,
)
for name, report in feasibility:
    print(name, report.feasible, report.score)

# 3) Run tabular orchestration flow.
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
    ],
)

print(result.assessment)
print(result.eda_report)
print(result.insight_summary)
print(result.chart_paths)
