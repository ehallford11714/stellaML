"""AutoML planning helpers and optional baseline execution hooks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AutoMLPlan:
    enabled: bool
    mode: str
    rationale: str
    prompt_user: str | None = None


def choose_analysis_mode(user_preference: str | None, objective: str) -> AutoMLPlan:
    preference = (user_preference or "ask").lower()

    if preference == "automl":
        return AutoMLPlan(True, objective, "User explicitly requested AutoML.")

    if preference == "manual":
        return AutoMLPlan(False, objective, "User explicitly requested manual analysis.")

    return AutoMLPlan(
        enabled=False,
        mode=objective,
        rationale="User choice required before expensive AutoML runs.",
        prompt_user="Choose analysis mode: 'automl' for model search or 'manual' for guided EDA/statistics.",
    )
