from stella_ml.harness import OpenClawStyleHarness
from stella_ml.ml_backends import detect_backend_availability, recommend_cpu_sota_1bit_models


def test_detect_backend_availability_shape() -> None:
    status = detect_backend_availability()
    for key in ["nltk", "spacy", "tensorflow", "keras", "torch", "sklearn", "pymc"]:
        assert key in status


def test_cpu_1bit_recommendations_non_empty() -> None:
    recs = recommend_cpu_sota_1bit_models()
    assert recs
    assert any("BitNet" in r["family"] for r in recs)


def test_harness_setup_ml_ecosystem_returns_expected_keys() -> None:
    harness = OpenClawStyleHarness()
    ecosystem = harness.setup_ml_ecosystem(install_missing=False)
    assert "status" in ecosystem
    assert "cuda" in ecosystem
    assert "sklearn_estimators" in ecosystem
