"""STELLA lattice primitives.

STELLA dimensions are represented as bounded scalar values in [0, 1].
The lattice enables policy evaluation and transition planning across nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class StellaAxis(str, Enum):
    """Canonical STELLA axes for orchestration state."""

    STRATEGY = "strategy"
    TASKING = "tasking"
    EXECUTION = "execution"
    LEARNING = "learning"
    ALIGNMENT = "alignment"


@dataclass(frozen=True)
class StellaNode:
    """A point in the STELLA lattice."""

    strategy: float
    tasking: float
    execution: float
    learning: float
    alignment: float

    def clamp(self) -> "StellaNode":
        """Return a new node with all axis values clamped to [0, 1]."""

        return StellaNode(*(max(0.0, min(1.0, value)) for value in self.as_tuple()))

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (
            self.strategy,
            self.tasking,
            self.execution,
            self.learning,
            self.alignment,
        )

    def distance(self, other: "StellaNode") -> float:
        """Simple L1 distance in the lattice."""

        return sum(abs(a - b) for a, b in zip(self.as_tuple(), other.as_tuple()))

    def blend(self, other: "StellaNode", weight: float = 0.5) -> "StellaNode":
        """Interpolate between two nodes."""

        w = max(0.0, min(1.0, weight))
        return StellaNode(
            *(a * (1.0 - w) + b * w for a, b in zip(self.as_tuple(), other.as_tuple()))
        )


class StellaLattice:
    """A lightweight index over known STELLA states."""

    def __init__(self, nodes: Iterable[StellaNode] | None = None) -> None:
        self._nodes = list(nodes or [])

    @property
    def nodes(self) -> list[StellaNode]:
        return list(self._nodes)

    def register(self, node: StellaNode) -> None:
        self._nodes.append(node.clamp())

    def nearest(self, query: StellaNode) -> StellaNode:
        if not self._nodes:
            raise ValueError("Lattice is empty; register at least one node")
        query = query.clamp()
        return min(self._nodes, key=lambda node: node.distance(query))

    def trajectory(self, start: StellaNode, target: StellaNode, steps: int = 5) -> list[StellaNode]:
        """Generate a deterministic interpolation path from start to target."""

        if steps < 2:
            raise ValueError("steps must be >= 2")
        return [start.blend(target, idx / (steps - 1)).clamp() for idx in range(steps)]
