"""Agentic orchestration harness on top of the STELLA lattice."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .agents import AgentContext, AgentOutcome, StellaAgent
from .lattice import StellaLattice, StellaNode


@dataclass
class OrchestrationResult:
    objective: str
    start: StellaNode
    target: StellaNode
    path: list[StellaNode]
    outcomes: list[AgentOutcome]
    state: dict[str, str] = field(default_factory=dict)


class StellaOrchestrator:
    """Coordinates agents to move work from one lattice node to another."""

    def __init__(self, lattice: StellaLattice, agents: Iterable[StellaAgent]) -> None:
        self.lattice = lattice
        self.agents = list(agents)
        if not self.agents:
            raise ValueError("At least one agent must be provided")

    def run(self, objective: str, start: StellaNode, target: StellaNode, steps: int = 5) -> OrchestrationResult:
        path = self.lattice.trajectory(start, target, steps=steps)
        memory: dict[str, str] = {}
        outcomes: list[AgentOutcome] = []

        for checkpoint in path:
            context = AgentContext(objective=objective, current=checkpoint, target=target, memory=memory)
            for agent in self.agents:
                outcome = agent.run(context)
                outcomes.append(outcome)
                memory.update(outcome.patch)

        return OrchestrationResult(
            objective=objective,
            start=start,
            target=target,
            path=path,
            outcomes=outcomes,
            state=memory,
        )
