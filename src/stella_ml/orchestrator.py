"""Lightweight agentic orchestration harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from stella_ml.models import InferenceRequest, Message, Role
from stella_ml.providers.base import BaseModelProvider


class Tool(Protocol):
    """Tool function contract."""

    def __call__(self, tool_input: str) -> str:  # pragma: no cover - protocol only
        ...


@dataclass(slots=True)
class AgentConfig:
    provider_name: str
    model: str
    max_steps: int = 5
    temperature: float = 0.2
    system_prompt: str = (
        "You are an orchestration agent. If a tool is needed, respond exactly as "
        "TOOL:<tool_name>:<tool_input>. Otherwise give FINAL:<answer>."
    )


@dataclass(slots=True)
class AgentRunResult:
    final_answer: str
    transcript: list[Message] = field(default_factory=list)


class AgentRunner:
    """Simple ReAct-like harness for tool-enabled orchestration."""

    def __init__(
        self,
        provider: BaseModelProvider,
        config: AgentConfig,
        tools: dict[str, Tool] | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.tools = tools or {}

    def run(self, user_prompt: str) -> AgentRunResult:
        transcript: list[Message] = [
            Message(role=Role.SYSTEM, content=self.config.system_prompt),
            Message(role=Role.USER, content=user_prompt),
        ]

        for _ in range(self.config.max_steps):
            response = self.provider.infer(
                InferenceRequest(
                    model=self.config.model,
                    messages=transcript,
                    temperature=self.config.temperature,
                )
            )
            assistant_message = Message(role=Role.ASSISTANT, content=response.content)
            transcript.append(assistant_message)

            if response.content.startswith("FINAL:"):
                return AgentRunResult(
                    final_answer=response.content.removeprefix("FINAL:").strip(),
                    transcript=transcript,
                )

            if response.content.startswith("TOOL:"):
                _, tool_name, tool_input = response.content.split(":", maxsplit=2)
                if tool_name not in self.tools:
                    transcript.append(
                        Message(
                            role=Role.TOOL,
                            name=tool_name,
                            content=f"Tool '{tool_name}' not available.",
                        )
                    )
                    continue

                tool_output = self.tools[tool_name](tool_input)
                transcript.append(Message(role=Role.TOOL, name=tool_name, content=tool_output))
                continue

            transcript.append(
                Message(
                    role=Role.TOOL,
                    name="orchestrator",
                    content="Invalid response format. Use TOOL:<name>:<input> or FINAL:<answer>.",
                )
            )

        return AgentRunResult(
            final_answer="Unable to produce FINAL response within step budget.",
            transcript=transcript,
        )
