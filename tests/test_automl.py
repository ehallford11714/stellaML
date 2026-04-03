from stella_ml.automl import choose_analysis_mode


def test_choose_analysis_mode_asks_by_default() -> None:
    plan = choose_analysis_mode(None, "regression")
    assert plan.enabled is False
    assert plan.prompt_user is not None


def test_choose_analysis_mode_automl() -> None:
    plan = choose_analysis_mode("automl", "classification")
    assert plan.enabled is True
    assert plan.mode == "classification"
