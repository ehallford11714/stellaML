"""High-level harness for quickly bootstrapping stellaML orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from .agents import CriticAgent, ExecutorAgent, PlannerAgent
from .lattice import StellaLattice, StellaNode
from .orchestrator import OrchestrationResult, StellaOrchestrator


@dataclass
class HarnessConfig:
    objective: str
    start: StellaNode
    target: StellaNode
    steps: int = 5


def build_default_orchestrator() -> StellaOrchestrator:
    lattice = StellaLattice(
        [
            StellaNode(0.2, 0.2, 0.2, 0.2, 0.8),
            StellaNode(0.6, 0.4, 0.5, 0.3, 0.9),
            StellaNode(0.9, 0.8, 0.8, 0.7, 1.0),
        ]
    )
    agents = [PlannerAgent(), ExecutorAgent(), CriticAgent()]
    return StellaOrchestrator(lattice=lattice, agents=agents)


def run_harness(config: HarnessConfig) -> OrchestrationResult:
    orchestrator = build_default_orchestrator()
    return orchestrator.run(
        objective=config.objective,
        start=config.start,
        target=config.target,
        steps=config.steps,
    )
