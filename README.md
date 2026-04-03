# stellaML

`stellaML` is a starter framework for model APIs + agentic orchestration + structured/unstructured data intelligence.

## Install via pip

```bash
pip install -e .
# or build/install wheel in your own pipeline:
# python -m build && pip install dist/stella_ml-*.whl
```

Optional extras:

```bash
pip install -e '.[analytics]'
pip install -e '.[ecosystem]'
pip install -e '.[unstructured]'
```

## Current infrastructure (short summary)

- **Provider layer**: OpenAI-compatible and Hugging Face model adapters.
- **OpenClaw-style orchestration**: high-level `OpenClawStyleHarness` and low-level `AgentRunner`.
- **Structured data intelligence**: CSV/Excel cleaning, auto-EDA, and chart generation.
- **Unstructured intelligence**: URL/HTML/XML/PPTX extraction + NLP helpers.
- **Local deployment planning**: hardware feasibility + local SOTA recommendation + API/local switch prompt.

Detailed review: `docs/openclaw_integration_review.md`.
Roadmap: `docs/framework_roadmap.md`.

## API vs Local decision + HF key management

```python
from stella_ml import OpenClawStyleHarness

harness = OpenClawStyleHarness()
recommendation = harness.propose_api_vs_local_switch(task="text-generation")
print(recommendation["prompt_user"])

# If missing HF key, request from user and save into config:
print(harness.ensure_hf_key(agent_name="analyst", api_key=None))
# Later, save provided key:
# print(harness.ensure_hf_key(agent_name="analyst", api_key="hf_xxx"))
```

## Iris AutoML demo

```bash
python examples/iris_automl_demo.py
```

This runs a compact model sweep (logistic regression, random forest, gradient boosting, SVM, KNN) and prints a leaderboard.

## Web-search orchestration + skill acquisition

```python
from stella_ml import OpenClawStyleHarness

harness = OpenClawStyleHarness()
out = harness.run_web_search_skill_acquisition(
    query="best methods for extracting article text and entity extraction",
    max_results=5,
)
print(out["skills"])
```

## Merge conflict resolution helper

```bash
python scripts/check_merge_conflicts.py
```
