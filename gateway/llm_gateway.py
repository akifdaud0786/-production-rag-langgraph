"""
Unified LLM Gateway.

Mirrors the "LLM Gateway" box in the architecture diagram (Portkey-style
unified gateway sitting in front of Llama 3.3 70B primary / Llama 3.1 8B
fallback, both served by Groq). Responsibilities:

  1. Present one call() interface regardless of which underlying model answers.
  2. Retry transient failures with backoff.
  3. Automatically fall back from the primary to a smaller/faster model if the
     primary errors out or times out.
  4. Emit structured latency/token/cost logs for the observability layer.

Swap in Portkey's hosted gateway by pointing GROQ base_url at Portkey's proxy
and adding the `x-portkey-*` headers — the call() interface below stays the
same either way.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

logger = logging.getLogger("gateway.llm")

PRIMARY_MODEL = os.getenv("GROQ_PRIMARY_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "20"))

# Rough per-1M-token pricing for cost logging (USD). Update to current Groq pricing as needed.
_COST_PER_MILLION_TOKENS = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
}


@dataclass
class LLMResponse:
    text: str
    model_used: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    used_fallback: bool


class LLMGatewayError(Exception):
    pass


class LLMGateway:
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMGatewayError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=api_key, timeout=TIMEOUT_SECONDS)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_model(self, model: str, messages: list[dict], temperature: float, max_tokens: int):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def call(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Call the gateway. Tries `model` (or PRIMARY_MODEL) first, falls back on failure."""
        primary = model or PRIMARY_MODEL
        start = time.perf_counter()
        used_fallback = False
        model_used = primary

        try:
            completion = self._call_model(primary, messages, temperature, max_tokens)
        except Exception as primary_exc:  # noqa: BLE001
            logger.warning("Primary model '%s' failed (%s); falling back to '%s'", primary, primary_exc, FALLBACK_MODEL)
            used_fallback = True
            model_used = FALLBACK_MODEL
            try:
                completion = self._call_model(FALLBACK_MODEL, messages, temperature, max_tokens)
            except Exception as fallback_exc:  # noqa: BLE001
                logger.error("Fallback model also failed: %s", fallback_exc)
                raise LLMGatewayError(f"Both primary and fallback models failed: {fallback_exc}") from fallback_exc

        latency_ms = (time.perf_counter() - start) * 1000
        usage = getattr(completion, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        pricing = _COST_PER_MILLION_TOKENS.get(model_used, {"input": 0.0, "output": 0.0})
        cost = (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]

        logger.info(
            "llm_call model=%s fallback=%s latency_ms=%.1f in_tok=%d out_tok=%d cost_usd=%.6f",
            model_used, used_fallback, latency_ms, input_tokens, output_tokens, cost,
        )

        return LLMResponse(
            text=completion.choices[0].message.content or "",
            model_used=model_used,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            used_fallback=used_fallback,
        )


_gateway_singleton: Optional[LLMGateway] = None


def get_gateway() -> LLMGateway:
    global _gateway_singleton
    if _gateway_singleton is None:
        _gateway_singleton = LLMGateway()
    return _gateway_singleton
