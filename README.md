# stellaML

`stellaML` provides a practical starter framework for:

1. Connecting to multiple model APIs (OpenAI-compatible + Hugging Face).
2. Storing provider API keys per agent profile in config.
3. Running agentic orchestration loops with tool execution.
4. Loading CSV data, auto-imputing structure, running auto-EDA, and generating bar charts.

## Install

```bash
pip install -e .
```

## Agent config with all API keys

Create `~/.stella_ml/config.json`:

```json
{
  "agents": {
    "analyst": {
      "model": "gpt-4.1-mini",
      "default_provider": "openai",
      "api_keys": {
        "openai": "<OPENAI_OR_COMPATIBLE_KEY>",
        "huggingface": "<HF_TOKEN>"
      },
      "base_urls": {
        "openai": "https://api.openai.com/v1",
        "huggingface": "https://api-inference.huggingface.co/models"
      }
    }
  }
}
```

Then load it and construct provider registry for the agent:

```python
from stella_ml import StellaConfig, build_registry_for_agent

cfg = StellaConfig.load()
profile = cfg.agents["analyst"]
registry = build_registry_for_agent(profile)
provider = registry.get(profile.default_provider)
```

## API-key model inference

```python
from stella_ml import InferenceRequest, Message, Role

response = provider.infer(
    InferenceRequest(
        model=profile.model,
        messages=[
            Message(role=Role.SYSTEM, content="You are concise."),
            Message(role=Role.USER, content="Summarize auto-EDA in one sentence."),
        ],
    )
)
print(response.content)
```

## Agentic orchestration harness

```python
from stella_ml import AgentConfig, AgentRunner

runner = AgentRunner(
    provider=provider,
    config=AgentConfig(
        provider_name=profile.default_provider,
        model=profile.model,
        max_steps=4,
    ),
    tools={
        "lookup": lambda query: f"Lookup result for: {query}",
    },
)

result = runner.run("Use lookup for Seattle weather, then answer.")
print(result.final_answer)
```

## CSV auto-impute + auto-EDA + bar chart example

```python
from stella_ml import auto_eda, generate_bar_chart, load_and_impute_csv

rows, inferred_types = load_and_impute_csv("examples/sales.csv")
report = auto_eda(rows, inferred_types)
print(report)

chart_path = generate_bar_chart(rows, inferred_types, "artifacts/sales_bar.png")
print("Chart saved to", chart_path)
```

## Package layout

- `stella_ml.models`: request/response/message schemas.
- `stella_ml.providers.base`: provider abstraction + registry.
- `stella_ml.providers.openai_compatible`: OpenAI-compatible adapter.
- `stella_ml.providers.huggingface`: Hugging Face Inference API adapter.
- `stella_ml.config`: per-agent API key/base URL configuration.
- `stella_ml.runtime`: build provider registry from agent profile.
- `stella_ml.orchestrator`: ReAct-style multi-step tool loop.
- `stella_ml.analytics`: CSV structure inference, imputation, EDA, charting.
