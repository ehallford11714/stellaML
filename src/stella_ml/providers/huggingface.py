"""Hugging Face Inference API provider."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import request

from stella_ml.models import InferenceRequest, InferenceResponse
from stella_ml.providers.base import BaseModelProvider


@dataclass(slots=True)
class HuggingFaceProvider(BaseModelProvider):
    """Provider for Hugging Face hosted inference models."""

    api_key: str
    base_url: str = "https://api-inference.huggingface.co/models"
    timeout_seconds: float = 60.0

    def infer(self, request_payload: InferenceRequest) -> InferenceResponse:
        prompt = "\n".join(f"{msg.role.value}: {msg.content}" for msg in request_payload.messages)
        payload: dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "temperature": request_payload.temperature,
            },
        }
        if request_payload.max_tokens is not None:
            payload["parameters"]["max_new_tokens"] = request_payload.max_tokens

        req = request.Request(
            url=f"{self.base_url.rstrip('/')}/{request_payload.model}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        with request.urlopen(req, timeout=self.timeout_seconds) as response:  # nosec: B310
            body = json.loads(response.read().decode("utf-8"))

        content = self._extract_content(body)
        return InferenceResponse(content=content, raw=body, model=request_payload.model)

    @staticmethod
    def _extract_content(body: Any) -> str:
        if isinstance(body, list) and body:
            first = body[0]
            if isinstance(first, dict) and "generated_text" in first:
                return str(first["generated_text"])
        if isinstance(body, dict):
            if "generated_text" in body:
                return str(body["generated_text"])
            if "error" in body:
                return f"HF_ERROR: {body['error']}"
        return str(body)
