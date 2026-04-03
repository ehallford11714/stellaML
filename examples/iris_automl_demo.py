"""Iris AutoML demo for stellaML."""

from stella_ml import run_iris_automl_demo

try:
    result = run_iris_automl_demo()
except ModuleNotFoundError as exc:
    print("scikit-learn is required for this demo. Install with: pip install -e '.[ecosystem]'")
    print(f"Details: {exc}")
    raise SystemExit(1)

print("Best model:", result.best_model)
print(f"Best {result.metric_name}: {result.metric_value:.4f}")
print("Leaderboard:")
for name, score in result.leaderboard:
    print(f"  - {name}: {score:.4f}")
