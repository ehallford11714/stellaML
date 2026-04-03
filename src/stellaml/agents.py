"""Agent interfaces for STELLA orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .lattice import StellaNode


@dataclass
class AgentContext:
    objective: str
    current: StellaNode
    target: StellaNode
    memory: dict[str, str] = field(default_factory=dict)


@dataclass
class AgentOutcome:
    agent: str
    summary: str
    patch: dict[str, str]
    confidence: float


class StellaAgent(Protocol):
    """Contract implemented by orchestration agents."""

    name: str

    def run(self, context: AgentContext) -> AgentOutcome:
        ...


@dataclass
class PlannerAgent:
    """Plans a phased approach to close lattice deltas."""

    name: str = "planner"

    def run(self, context: AgentContext) -> AgentOutcome:
        gap = context.current.distance(context.target)
        confidence = max(0.4, 1.0 - (gap / 5.0))
        summary = (
            f"Plan objective '{context.objective}' with delta={gap:.2f}; "
            "prioritize strategy/tasking calibration."
        )
        return AgentOutcome(
            agent=self.name,
            summary=summary,
            patch={"plan": summary},
            confidence=confidence,
        )


@dataclass
class ExecutorAgent:
    """Executes planned actions and produces implementation directives."""

    name: str = "executor"

    def run(self, context: AgentContext) -> AgentOutcome:
        summary = (
            "Execute prioritized tasks, emit artifacts, and capture telemetry for learning loop."
        )
        return AgentOutcome(
            agent=self.name,
            summary=summary,
            patch={"execution": "run:queued", "telemetry": "enabled"},
            confidence=0.78,
        )


@dataclass
class CriticAgent:
    """Evaluates outcomes for alignment and learning quality."""

    name: str = "critic"

    def run(self, context: AgentContext) -> AgentOutcome:
        summary = "Review outputs for alignment drift; attach corrective prompts if needed."
        return AgentOutcome(
            agent=self.name,
            summary=summary,
            patch={"review": "alignment-check"},
            confidence=0.81,
        )
