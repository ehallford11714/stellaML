import importlib.util

from stella_ml.automl import choose_analysis_mode, run_iris_automl_demo


def test_choose_analysis_mode_asks_by_default() -> None:
    plan = choose_analysis_mode(None, "regression")
    assert plan.enabled is False
    assert plan.prompt_user is not None


def test_choose_analysis_mode_automl() -> None:
    plan = choose_analysis_mode("automl", "classification")
    assert plan.enabled is True
    assert plan.mode == "classification"


def test_run_iris_automl_demo_if_sklearn_installed() -> None:
    if importlib.util.find_spec("sklearn") is None:
        return
    result = run_iris_automl_demo()
    assert result.best_model
    assert result.metric_name == "accuracy"
    assert 0.0 <= result.metric_value <= 1.0
    assert result.leaderboard
