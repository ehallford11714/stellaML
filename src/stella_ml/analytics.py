"""CSV auto-structure, imputation, and lightweight auto-EDA helpers."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(slots=True)
class EDAReport:
    rows: int
    columns: list[str]
    inferred_types: dict[str, str]
    missing_counts: dict[str, int]
    numeric_summary: dict[str, dict[str, float]]
    categorical_summary: dict[str, dict[str, int]]


def load_and_impute_csv(csv_path: str | Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Load CSV, infer column types, and impute missing values."""
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return [], {}

    columns = list(rows[0].keys())
    inferred_types: dict[str, str] = {}

    for col in columns:
        non_empty = [r[col].strip() for r in rows if r[col] is not None and str(r[col]).strip() != ""]
        numeric_ratio = 0.0
        if non_empty:
            numeric_ratio = sum(_is_float(v) for v in non_empty) / len(non_empty)
        inferred_types[col] = "numeric" if numeric_ratio >= 0.7 else "categorical"

    typed_rows: list[dict[str, object]] = []
    for row in rows:
        typed: dict[str, object] = {}
        for col in columns:
            value = (row[col] or "").strip()
            if inferred_types[col] == "numeric":
                typed[col] = float(value) if _is_float(value) else None
            else:
                typed[col] = value if value else None
        typed_rows.append(typed)

    for col in columns:
        values = [r[col] for r in typed_rows if r[col] is not None]
        if not values:
            continue
        if inferred_types[col] == "numeric":
            fill_value = mean(float(v) for v in values)
        else:
            fill_value = Counter(str(v) for v in values).most_common(1)[0][0]

        for row in typed_rows:
            if row[col] is None:
                row[col] = fill_value

    return typed_rows, inferred_types


def auto_eda(rows: list[dict[str, object]], inferred_types: dict[str, str]) -> EDAReport:
    if not rows:
        return EDAReport(0, [], {}, {}, {}, {})

    columns = list(rows[0].keys())
    missing_counts: dict[str, int] = {}
    numeric_summary: dict[str, dict[str, float]] = {}
    categorical_summary: dict[str, dict[str, int]] = {}

    for col in columns:
        col_values = [row[col] for row in rows]
        missing_counts[col] = sum(v is None for v in col_values)
        if inferred_types.get(col) == "numeric":
            numeric_values = [float(v) for v in col_values if v is not None]
            numeric_summary[col] = {
                "min": min(numeric_values),
                "max": max(numeric_values),
                "mean": mean(numeric_values),
            }
        else:
            counts = Counter(str(v) for v in col_values if v is not None)
            categorical_summary[col] = dict(counts.most_common(10))

    return EDAReport(
        rows=len(rows),
        columns=columns,
        inferred_types=inferred_types,
        missing_counts=missing_counts,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
    )


def generate_bar_chart(rows: list[dict[str, object]], inferred_types: dict[str, str], output_path: str | Path) -> Path:
    """Generate a bar chart from the first categorical column."""
    categorical_column = next((c for c, t in inferred_types.items() if t == "categorical"), None)
    if categorical_column is None:
        raise ValueError("No categorical column available for bar chart generation.")

    counts = Counter(str(row[categorical_column]) for row in rows)

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("matplotlib is required for chart generation") from exc

    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.title(f"Distribution of {categorical_column}")
    plt.xlabel(categorical_column)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    return out


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
