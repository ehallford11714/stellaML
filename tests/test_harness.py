from stella_ml.harness import OpenClawStyleHarness


def test_evaluate_problem_detects_regression_and_automl() -> None:
    harness = OpenClawStyleHarness()
    assessment = harness.evaluate_problem("Please predict monthly revenue from this csv using automl")

    assert assessment.objective == "regression"
    assert assessment.requires_data is True
    assert assessment.automl_recommended is True
    assert assessment.follow_up_question is not None
