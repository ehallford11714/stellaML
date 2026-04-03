"""Helpers to build provider registries from stored agent config."""

from __future__ import annotations

from stella_ml.config import AgentProfile
from stella_ml.providers.base import ProviderRegistry
from stella_ml.providers.huggingface import HuggingFaceProvider
from stella_ml.providers.openai_compatible import OpenAICompatibleProvider


def build_registry_for_agent(profile: AgentProfile) -> ProviderRegistry:
    """Create provider registry using API keys/base URLs stored in agent profile."""
    registry = ProviderRegistry()

    if "openai" in profile.api_keys:
        registry.register(
            "openai",
            OpenAICompatibleProvider(
                api_key=profile.api_keys["openai"],
                base_url=profile.base_urls.get("openai", OpenAICompatibleProvider.base_url),
            ),
        )

    if "huggingface" in profile.api_keys:
        registry.register(
            "huggingface",
            HuggingFaceProvider(
                api_key=profile.api_keys["huggingface"],
                base_url=profile.base_urls.get("huggingface", HuggingFaceProvider.base_url),
            ),
        )

    return registry
