"""OpenAI-compatible provider implementation using API keys."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request

from stella_ml.models import InferenceRequest, InferenceResponse
from stella_ml.providers.base import BaseModelProvider


@dataclass(slots=True)
class OpenAICompatibleProvider(BaseModelProvider):
    """Provider for any OpenAI-compatible chat completion endpoint."""

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(
        cls,
        api_key_env: str = "MODEL_API_KEY",
        base_url_env: str = "MODEL_BASE_URL",
    ) -> "OpenAICompatibleProvider":
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in environment variable {api_key_env}")

        base_url = os.getenv(base_url_env, cls.base_url)
        return cls(api_key=api_key, base_url=base_url)

    def infer(self, request_payload: InferenceRequest) -> InferenceResponse:
        payload: dict[str, Any] = {
            "model": request_payload.model,
            "messages": [
                {
                    "role": message.role.value,
                    "content": message.content,
                    **({"name": message.name} if message.name else {}),
                }
                for message in request_payload.messages
            ],
            "temperature": request_payload.temperature,
        }
        if request_payload.max_tokens is not None:
            payload["max_tokens"] = request_payload.max_tokens

        req = request.Request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        with request.urlopen(req, timeout=self.timeout_seconds) as response:  # nosec: B310
            body = json.loads(response.read().decode("utf-8"))

        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        model = body.get("model", request_payload.model)

        return InferenceResponse(content=content, raw=body, usage=usage, model=model)
