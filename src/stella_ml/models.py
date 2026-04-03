"""Core data models for model requests and agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Chat role in a model conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class Message:
    """Single message in a chat-like interaction."""

    role: Role
    content: str
    name: str | None = None


@dataclass(slots=True)
class InferenceRequest:
    """Model inference payload."""

    model: str
    messages: list[Message]
    temperature: float = 0.2
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InferenceResponse:
    """Normalized inference output from any provider."""

    content: str
    raw: dict[str, Any]
    model: str
    usage: dict[str, int] = field(default_factory=dict)
