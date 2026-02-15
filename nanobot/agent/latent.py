"""Async latent reasoning step for ambiguity handling."""

import asyncio
import json
import math
from json import JSONDecodeError
from typing import Any

from loguru import logger
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
from nanobot.providers.base import LLMProvider


class LatentReasoner:
    """Performs a short hidden reasoning pass before tool execution."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        timeout_seconds: int = 10,
        memory_config: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.memory_config = memory_config or {}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call_llm_with_backoff(self, system_prompt: str, prompt: str) -> str:
        response = await self.provider.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0.1,
        )
        return (response.content or "").strip()

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
            hypotheses = await asyncio.wait_for(
                self._generate_initial_hypotheses(system_prompt, prompt),
                timeout=self.timeout_seconds,
            )
            if not hypotheses:
                return fallback_state

            max_depth = max(int(self.memory_config.get("latent_max_depth", 1)), 1)
            beam_width = max(int(self.memory_config.get("beam_width", 3)), 1)
            clarify_threshold = float(self.memory_config.get("clarify_entropy_threshold", 0.8))
            entropy = self._calculate_entropy(hypotheses)
            for current_depth in range(max_depth):
                logger.info(f"Depth {current_depth}: Entropy={entropy:.3f}")
                if entropy < clarify_threshold:
                    break
                if int(self.memory_config.get("monte_carlo_samples", 0)) > 0:
                    hypotheses.extend(
                        await self._monte_carlo_expand(hypotheses, user_message, context_summary)
                    )
                hypotheses = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:beam_width]
                entropy = self._calculate_entropy(hypotheses)

            best_hypothesis = max(hypotheses, key=lambda h: h.confidence)
            return SuperpositionalState(
                hypotheses=hypotheses,
                entropy=entropy,
                strategic_direction=best_hypothesis.reasoning
                or "Proceed with standard processing.",
            )
        except (asyncio.TimeoutError, JSONDecodeError, ValidationError) as exc:
            logger.debug(f"Latent reasoning fallback triggered: {exc}")
            return fallback_state
        except Exception as exc:
            logger.warning(f"Unexpected latent reasoning error: {exc}")
            return fallback_state

    async def _generate_initial_hypotheses(
        self, system_prompt: str, prompt: str
    ) -> list[Hypothesis]:
        payload = await self._call_llm_with_backoff(system_prompt, prompt)
        if payload.startswith("```"):
            payload = payload.removeprefix("```json").removeprefix("```").strip()
            if payload.endswith("```"):
                payload = payload[:-3].strip()

        parsed = json.loads(payload)
        if "hypotheses" in parsed:
            hypotheses_data = parsed["hypotheses"]
        elif all(k in parsed for k in ("intent", "confidence")):
            hypotheses_data = [parsed]
        else:
            hypotheses_data = []

        hypotheses = []
        for item in hypotheses_data:
            if not isinstance(item, dict):
                continue
            confidence = float(item.get("confidence", 0.0))
            hypotheses.append(
                Hypothesis(
                    intent=str(item.get("intent", "unknown")),
                    confidence=max(confidence, 0.0),
                    reasoning=str(item.get("reasoning", "latent inference")),
                )
            )
        return hypotheses

    def _calculate_entropy(self, hypotheses: list[Hypothesis]) -> float:
        total_confidence = sum(max(h.confidence, 0.0) for h in hypotheses) or 1.0
        probabilities = [max(h.confidence, 0.0) / total_confidence for h in hypotheses]
        return -sum(p * math.log2(p) for p in probabilities if p > 0)

    async def _monte_carlo_expand(
        self,
        hypotheses: list[Hypothesis],
        user_message: str,
        context_summary: str,
    ) -> list[Hypothesis]:
        _ = (hypotheses, user_message, context_summary)
        return []
