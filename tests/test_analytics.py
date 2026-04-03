from pathlib import Path

from stella_ml.analytics import apply_cleaning_operations, auto_eda, load_and_impute_csv


def test_load_impute_clean_and_auto_eda(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("cat,val\nA,1\nB,\nA,3\n")

    rows, inferred = load_and_impute_csv(csv_path)
    rows = apply_cleaning_operations(
        rows,
        inferred,
        [
            {"op": "binarize", "column": "val", "threshold": 2, "new_column": "val_bin"},
            {"op": "discretize", "column": "val", "bins": 2},
        ],
    )
    report = auto_eda(rows, inferred)

    assert inferred["val"] == "numeric"
    assert report.rows == 3
    assert report.numeric_summary["val"]["mean"] > 0
    assert report.categorical_summary["cat"]["A"] == 2
    assert "val_bin" in report.columns
    assert report.recommendations
