"""AutoML planning helpers and optional baseline execution hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AutoMLPlan:
    enabled: bool
    mode: str
    rationale: str
    prompt_user: str | None = None


@dataclass(slots=True)
class AutoMLResult:
    best_model: str
    metric_name: str
    metric_value: float
    leaderboard: list[tuple[str, float]]


def choose_analysis_mode(user_preference: str | None, objective: str) -> AutoMLPlan:
    preference = (user_preference or "ask").lower()

    if preference == "automl":
        return AutoMLPlan(True, objective, "User explicitly requested AutoML.")

    if preference == "manual":
        return AutoMLPlan(False, objective, "User explicitly requested manual analysis.")

    return AutoMLPlan(
        enabled=False,
        mode=objective,
        rationale="User choice required before expensive AutoML runs.",
        prompt_user="Choose analysis mode: 'automl' for model search or 'manual' for guided EDA/statistics.",
    )


def run_iris_automl_demo(random_state: int = 42) -> AutoMLResult:
    """Run a compact AutoML-style model sweep on the Iris dataset."""
    from sklearn.datasets import load_iris
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=random_state,
        stratify=data.target,
    )

    candidates: dict[str, Any] = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=False)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }

    leaderboard: list[tuple[str, float]] = []
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        score = float(accuracy_score(y_test, preds))
        leaderboard.append((name, score))

    leaderboard.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = leaderboard[0]

    return AutoMLResult(
        best_model=best_name,
        metric_name="accuracy",
        metric_value=best_score,
        leaderboard=leaderboard,
    )
