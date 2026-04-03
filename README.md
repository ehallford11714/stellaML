# stellaML

`stellaML` is an agentic machine-learning orchestration framework built around a **STELLA lattice** state model.

## What this framework provides

- **STELLA lattice core** to represent orchestration state across Strategy, Tasking, Execution, Learning, and Alignment.
- **Agent contract + built-in agents** (`Planner`, `Executor`, `Critic`).
- **Orchestration harness** that progresses along a lattice trajectory and runs agents at each checkpoint.
- **Extensible package structure** for custom agents, policies, and runtime integrations.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python examples/run_demo.py
```

## Core concepts

### 1) STELLA lattice

The framework models runtime intent and control as a 5D node:

- `strategy`
- `tasking`
- `execution`
- `learning`
- `alignment`

Each axis is a normalized value in `[0, 1]`.

### 2) Agentic orchestration harness

The orchestrator:

1. builds a trajectory from a start node to a target node,
2. runs each registered agent at each trajectory checkpoint,
3. merges agent patches into a shared state memory,
4. returns complete orchestration results (path, outcomes, final state).

## Minimal usage

```python
from stellaml import HarnessConfig, StellaNode, run_harness

config = HarnessConfig(
    objective="Train robust forecasting model",
    start=StellaNode(0.2, 0.2, 0.2, 0.2, 0.8),
    target=StellaNode(0.9, 0.8, 0.9, 0.8, 1.0),
    steps=5,
)

result = run_harness(config)
print(result.state)
```

## Project layout

- `src/stellaml/lattice.py` — STELLA lattice model.
- `src/stellaml/agents.py` — agent interfaces and reference agents.
- `src/stellaml/orchestrator.py` — multi-agent orchestration engine.
- `src/stellaml/harness.py` — default framework bootstrap.
- `tests/test_harness.py` — smoke test for end-to-end harness behavior.

## Next extensions

- Add policy engine for dynamic agent selection.
- Integrate experiment tracker and model registry.
- Add async runner and distributed execution backend.
