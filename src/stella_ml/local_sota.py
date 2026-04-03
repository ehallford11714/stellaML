"""Local SOTA model discovery and hardware-aware Hugging Face selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


from stella_ml.config import DEFAULT_CONFIG_PATH, StellaConfig
from stella_ml.feasibility import HardwareProfile, detect_local_hardware


@dataclass(slots=True)
class ModelCandidate:
    model_id: str
    downloads: int
    likes: int
    estimated_params_b: float | None


@dataclass(slots=True)
class LocalModelRecommendation:
    model_id: str | None
    reason: str
    candidates_considered: int
    should_switch_to_local: bool
    prompt_user: str


def web_search_hf_sota(task: str = "text-generation", limit: int = 30) -> list[ModelCandidate]:
    """Discover top Hugging Face models by download trend for a task."""
    url = "https://huggingface.co/api/models"
    params = {
        "pipeline_tag": task,
        "sort": "downloads",
        "direction": -1,
        "limit": limit,
        "full": "true",
    }
    import requests

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    results: list[ModelCandidate] = []
    for entry in payload:
        model_id = entry.get("id", "")
        results.append(
            ModelCandidate(
                model_id=model_id,
                downloads=int(entry.get("downloads", 0) or 0),
                likes=int(entry.get("likes", 0) or 0),
                estimated_params_b=_estimate_params_b(model_id),
            )
        )
    return results


def recommend_max_local_model(
    hardware: HardwareProfile | None = None,
    task: str = "text-generation",
) -> LocalModelRecommendation:
    hw = hardware or detect_local_hardware()
    candidates = web_search_hf_sota(task=task)

    vram = hw.vram_gb or 0.0
    ram = hw.ram_gb

    # heuristic capacity in billions for quantized local serving
    capacity_b = max(vram * 1.1, ram * 0.35)

    fit = [
        c
        for c in candidates
        if c.estimated_params_b is None or c.estimated_params_b <= capacity_b
    ]

    chosen = fit[0] if fit else None
    switch_local = chosen is not None and (hw.gpu_name is not None or hw.ram_gb >= 16)

    if chosen is None:
        return LocalModelRecommendation(
            model_id=None,
            reason=f"No candidate confidently fits estimated local capacity ({capacity_b:.1f}B).",
            candidates_considered=len(candidates),
            should_switch_to_local=False,
            prompt_user="Continue with API hosting or provide stronger local hardware/profile override.",
        )

    return LocalModelRecommendation(
        model_id=chosen.model_id,
        reason=(
            f"Selected highest-ranked likely-fit model for {task}. "
            f"Estimated capacity: {capacity_b:.1f}B params."
        ),
        candidates_considered=len(candidates),
        should_switch_to_local=switch_local,
        prompt_user=(
            f"Use API or switch to local model '{chosen.model_id}' based on your hardware?"
        ),
    )


def ensure_huggingface_api_key(
    agent_name: str,
    api_key: str | None,
    config_path: str | None = None,
) -> str:
    """Persist HF API key in config if provided, otherwise request it."""
    if not api_key:
        return (
            "Missing Hugging Face API key. Please provide one to store in config "
            "or proceed with local model mode."
        )

    from pathlib import Path

    resolved_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    cfg = StellaConfig.load(resolved_path)
    profile = cfg.agents.get(agent_name)
    if profile is None:
        return f"Agent '{agent_name}' not found in config. Create the agent profile first."

    profile.api_keys["huggingface"] = api_key
    cfg.agents[agent_name] = profile
    cfg.save(resolved_path)
    return "Hugging Face API key saved to config successfully."


def _estimate_params_b(model_id: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_id)
    if match:
        return float(match.group(1))
    return None
