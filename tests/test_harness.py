from stella_ml.harness import OpenClawStyleHarness


def test_evaluate_problem_detects_regression_and_automl() -> None:
    harness = OpenClawStyleHarness()
    assessment = harness.evaluate_problem("Please predict monthly revenue from this csv using automl")

    assert assessment.objective == "regression"
    assert assessment.requires_data is True
    assert assessment.automl_recommended is True
    assert assessment.follow_up_question is not None


def test_propose_api_vs_local_switch_uses_recommendation(monkeypatch) -> None:
    harness = OpenClawStyleHarness()

    class FakeRecommendation:
        prompt_user = "api or local?"
        should_switch_to_local = True

    monkeypatch.setattr("stella_ml.harness.recommend_max_local_model", lambda task="text-generation": FakeRecommendation())
    result = harness.propose_api_vs_local_switch()
    assert result["prompt_user"] == "api or local?"
