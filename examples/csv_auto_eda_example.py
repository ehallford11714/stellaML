from stella_ml import auto_eda, generate_bar_chart, load_and_impute_csv

rows, inferred_types = load_and_impute_csv("examples/sales.csv")
report = auto_eda(rows, inferred_types)
print(report)

chart_path = generate_bar_chart(rows, inferred_types, "artifacts/sales_bar.png")
print(f"Chart saved: {chart_path}")
