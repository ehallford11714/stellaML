from pathlib import Path

from stella_ml.config import AgentProfile, StellaConfig
from stella_ml.feasibility import HardwareProfile
from stella_ml.local_sota import ensure_huggingface_api_key, recommend_max_local_model


def test_recommend_max_local_model_with_injected_search(monkeypatch) -> None:
    from stella_ml import local_sota

    def fake_search(task: str = "text-generation", limit: int = 30):
        return [
            local_sota.ModelCandidate("model-1b", downloads=100, likes=10, estimated_params_b=1.0),
            local_sota.ModelCandidate("model-7b", downloads=90, likes=9, estimated_params_b=7.0),
        ]

    monkeypatch.setattr(local_sota, "web_search_hf_sota", fake_search)
    rec = recommend_max_local_model(HardwareProfile("Linux", 8, 32, None, None))
    assert rec.model_id == "model-1b"


def test_ensure_huggingface_api_key_writes_config(tmp_path: Path) -> None:
    cfg = StellaConfig(
        agents={
            "analyst": AgentProfile(
                name="analyst",
                model="x",
                default_provider="huggingface",
            )
        }
    )
    path = tmp_path / "cfg.json"
    cfg.save(path)

    msg = ensure_huggingface_api_key("analyst", "hf_abc", str(path))
    loaded = StellaConfig.load(path)

    assert "successfully" in msg.lower()
    assert loaded.agents["analyst"].api_keys["huggingface"] == "hf_abc"
