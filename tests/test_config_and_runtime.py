from pathlib import Path

from stella_ml.config import AgentProfile, StellaConfig
from stella_ml.runtime import build_registry_for_agent


def test_config_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    cfg = StellaConfig(
        agents={
            "analyst": AgentProfile(
                name="analyst",
                model="gpt-4.1-mini",
                default_provider="openai",
                api_keys={"openai": "k1", "huggingface": "k2"},
                base_urls={"openai": "https://example.com/v1"},
            )
        }
    )

    cfg.save(config_path)
    loaded = StellaConfig.load(config_path)

    assert loaded.agents["analyst"].api_keys["openai"] == "k1"
    assert loaded.agents["analyst"].default_provider == "openai"


def test_runtime_builds_providers() -> None:
    profile = AgentProfile(
        name="agent",
        model="model",
        default_provider="huggingface",
        api_keys={"openai": "okey", "huggingface": "hkey"},
    )

    registry = build_registry_for_agent(profile)

    assert registry.get("openai")
    assert registry.get("huggingface")
