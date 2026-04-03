"""Agentic harness demo with an open-source Hugging Face model."""

from __future__ import annotations

import os

from stella_ml import AgentConfig, AgentRunner, HuggingFaceProvider


def main() -> int:
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        print("HF_API_KEY is not set; skipping live HF inference demo.")
        print("Set HF_API_KEY then re-run this script to test with open-source HF models.")
        return 0

    provider = HuggingFaceProvider(api_key=api_key)
    runner = AgentRunner(
        provider=provider,
        config=AgentConfig(
            provider_name="huggingface",
            model="HuggingFaceH4/zephyr-7b-beta",
            max_steps=1,
            system_prompt="Respond with FINAL:<short answer>.",
        ),
        tools={},
    )

    result = runner.run("Explain what agentic orchestration means in one sentence.")
    print(result.final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
