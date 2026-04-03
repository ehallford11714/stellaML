# stellaML

`stellaML` now includes a basic framework for:

1. Connecting to any OpenAI-compatible model endpoint with an API key.
2. Running standardized inference requests.
3. Orchestrating simple agentic loops with tool calls.

## Install

```bash
pip install -e .
```

## Quick start: API-key model inference

```python
from stella_ml import InferenceRequest, Message, OpenAICompatibleProvider, Role

provider = OpenAICompatibleProvider.from_env(
    api_key_env="MODEL_API_KEY",     # required
    base_url_env="MODEL_BASE_URL",   # optional, defaults to https://api.openai.com/v1
)

response = provider.infer(
    InferenceRequest(
        model="gpt-4.1-mini",
        messages=[
            Message(role=Role.SYSTEM, content="You are concise."),
            Message(role=Role.USER, content="Summarize agentic orchestration in one sentence."),
        ],
    )
)

print(response.content)
```

## Quick start: agentic orchestration harness

```python
from stella_ml import AgentConfig, AgentRunner, OpenAICompatibleProvider

provider = OpenAICompatibleProvider.from_env()

runner = AgentRunner(
    provider=provider,
    config=AgentConfig(
        provider_name="openai-compatible",
        model="gpt-4.1-mini",
        max_steps=4,
    ),
    tools={
        "lookup": lambda query: f"Lookup result for: {query}",
    },
)

result = runner.run("Use lookup for weather in Seattle, then answer.")
print(result.final_answer)
```

## Package layout

- `stella_ml.models`: request/response/message schemas.
- `stella_ml.providers.base`: provider abstraction + registry.
- `stella_ml.providers.openai_compatible`: API-key driven provider adapter.
- `stella_ml.orchestrator`: basic ReAct-style multi-step tool loop.
