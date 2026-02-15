"""Async latent reasoning step for ambiguity handling."""

import asyncio
import json

from nanobot.agent.memory_types import SuperpositionalState
from nanobot.providers.base import LLMProvider


class LatentReasoner:
    """Performs a short hidden reasoning pass before tool execution."""

    def __init__(self, provider: LLMProvider, model: str, timeout_seconds: int = 10):
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def reason(self, user_message: str, context_summary: str) -> SuperpositionalState:
        fallback_state = SuperpositionalState(
            hypotheses=[],
            entropy=0.0,
            strategic_direction="Proceed with standard processing due to reasoning timeout/error.",
        )

        system_prompt = (
            "You are the subconscious reasoning engine of an AI agent. "
            "Analyze the user input and context. Generate 2-3 intent hypotheses, "
            "their confidence, entropy of ambiguity, and a strategic direction. "
            "Return only JSON for SuperpositionalState."
        )
        prompt = (
            f"Context: {context_summary}\n"
            f"User Input: {user_message}\n\n"
            "Tasks:\n"
            "1. Identify 2-3 potential distinct intents.\n"
            "2. Assign confidence to each.\n"
            "3. If confidence is split evenly, set high entropy near 1.0.\n"
            "4. If one hypothesis dominates, set low entropy near 0.0.\n"
        )

        try:
            response = await asyncio.wait_for(
                self.provider.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                    temperature=0.1,
                ),
                timeout=self.timeout_seconds,
            )
            payload = (response.content or "").strip()
            if payload.startswith("```"):
                payload = payload.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return SuperpositionalState.model_validate(json.loads(payload))
        except Exception:
            return fallback_state
