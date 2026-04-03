"""Configuration handling for per-agent model credentials and defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_CONFIG_PATH = Path.home() / ".stella_ml" / "config.json"


@dataclass(slots=True)
class AgentProfile:
    """Configuration bundle for one agent."""

    name: str
    model: str
    default_provider: str
    api_keys: dict[str, str] = field(default_factory=dict)
    base_urls: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class StellaConfig:
    """Top-level framework config."""

    agents: dict[str, AgentProfile] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "StellaConfig":
        if not path.exists():
            return cls()

        raw = json.loads(path.read_text())
        agents: dict[str, AgentProfile] = {}
        for name, payload in raw.get("agents", {}).items():
            agents[name] = AgentProfile(
                name=name,
                model=payload["model"],
                default_provider=payload["default_provider"],
                api_keys=payload.get("api_keys", {}),
                base_urls=payload.get("base_urls", {}),
            )
        return cls(agents=agents)

    def save(self, path: Path = DEFAULT_CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = {
            "agents": {
                name: {
                    "model": profile.model,
                    "default_provider": profile.default_provider,
                    "api_keys": profile.api_keys,
                    "base_urls": profile.base_urls,
                }
                for name, profile in self.agents.items()
            }
        }
        path.write_text(json.dumps(serialized, indent=2))
