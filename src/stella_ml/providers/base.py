"""Provider abstractions and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from stella_ml.models import InferenceRequest, InferenceResponse


class BaseModelProvider(ABC):
    """Interface implemented by every model provider adapter."""

    @abstractmethod
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run a single inference request."""


class ProviderRegistry:
    """Simple provider lookup by name."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseModelProvider] = {}

    def register(self, name: str, provider: BaseModelProvider) -> None:
        self._providers[name] = provider

    def get(self, name: str) -> BaseModelProvider:
        if name not in self._providers:
            registered = ", ".join(sorted(self._providers)) or "<none>"
            raise KeyError(f"Provider '{name}' is not registered. Known providers: {registered}")
        return self._providers[name]
