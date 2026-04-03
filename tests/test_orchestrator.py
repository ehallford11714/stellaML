from stella_ml.models import InferenceRequest, InferenceResponse
from stella_ml.orchestrator import AgentConfig, AgentRunner
from stella_ml.providers.base import BaseModelProvider


class StubProvider(BaseModelProvider):
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[InferenceRequest] = []

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.calls.append(request)
        content = self.outputs.pop(0)
        return InferenceResponse(content=content, raw={}, model=request.model)


def test_agent_uses_tool_and_returns_final() -> None:
    provider = StubProvider(outputs=["TOOL:echo:hello", "FINAL:done"])
    runner = AgentRunner(
        provider=provider,
        config=AgentConfig(provider_name="stub", model="fake-model", max_steps=3),
        tools={"echo": lambda text: f"echoed({text})"},
    )

    result = runner.run("say hi")

    assert result.final_answer == "done"
    assert any(msg.content == "echoed(hello)" for msg in result.transcript)


def test_agent_times_out_without_final() -> None:
    provider = StubProvider(outputs=["TOOL:missing:input", "UNKNOWN"])
    runner = AgentRunner(
        provider=provider,
        config=AgentConfig(provider_name="stub", model="fake-model", max_steps=2),
        tools={},
    )

    result = runner.run("run")

    assert "Unable to produce FINAL" in result.final_answer
