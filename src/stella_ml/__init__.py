"""stellaML: model inference, agent orchestration, and data analysis helpers."""

from stella_ml.analytics import EDAReport, auto_eda, generate_bar_chart, load_and_impute_csv
from stella_ml.config import AgentProfile, StellaConfig
from stella_ml.models import InferenceRequest, InferenceResponse, Message, Role
from stella_ml.orchestrator import AgentConfig, AgentRunResult, AgentRunner
from stella_ml.providers.base import BaseModelProvider, ProviderRegistry
from stella_ml.providers.huggingface import HuggingFaceProvider
from stella_ml.providers.openai_compatible import OpenAICompatibleProvider
from stella_ml.runtime import build_registry_for_agent

__all__ = [
    "AgentConfig",
    "AgentProfile",
    "AgentRunResult",
    "AgentRunner",
    "BaseModelProvider",
    "EDAReport",
    "HuggingFaceProvider",
    "InferenceRequest",
    "InferenceResponse",
    "Message",
    "OpenAICompatibleProvider",
    "ProviderRegistry",
    "Role",
    "StellaConfig",
    "auto_eda",
    "build_registry_for_agent",
    "generate_bar_chart",
    "load_and_impute_csv",
]
