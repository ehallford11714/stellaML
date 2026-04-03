"""stellaML: model inference, orchestration, and data intelligence helpers."""

from stella_ml.analytics import (
    EDAReport,
    apply_cleaning_operations,
    auto_eda,
    explore_chart,
    explore_data,
    generate_bar_chart,
    infer_structure,
    load_and_impute_csv,
    load_tabular_file,
)
from stella_ml.automl import AutoMLPlan, choose_analysis_mode
from stella_ml.config import AgentProfile, StellaConfig
from stella_ml.harness import HarnessResult, OpenClawStyleHarness, ProblemAssessment
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
    "AutoMLPlan",
    "BaseModelProvider",
    "EDAReport",
    "HarnessResult",
    "HuggingFaceProvider",
    "InferenceRequest",
    "InferenceResponse",
    "Message",
    "OpenAICompatibleProvider",
    "OpenClawStyleHarness",
    "ProblemAssessment",
    "ProviderRegistry",
    "Role",
    "StellaConfig",
    "apply_cleaning_operations",
    "auto_eda",
    "build_registry_for_agent",
    "choose_analysis_mode",
    "explore_chart",
    "explore_data",
    "generate_bar_chart",
    "infer_structure",
    "load_and_impute_csv",
    "load_tabular_file",
]
