"""Cloud LLM clients via litellm (strictly external APIs — no local inference).

litellm gives us one call signature for DeepSeek / Gemini plus
built-in retries and per-call cost tracking. On top of that we keep Aethel's
guarantees:

- JSON requested natively where the API supports it,
- response validated against a Pydantic schema,
- exactly one schema-retry on invalid output,
- LLMUnavailable raised on anything else.

Callers treat LLMUnavailable as fail-closed: NO TRADE. Deliberately there is
NO cross-provider fallback for decisions — Athena must be Gemini 3.1 Pro; silently
substituting a different reviewer model would change the risk profile.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from aethel.config import get_settings
from aethel.observability.metrics import metrics

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(Exception):
    """Provider error, timeout, or persistently invalid output. Fail closed."""


def _extract_json(text: str) -> dict:
    """Tolerate code fences but nothing fancier — schemas are requested
    natively, this is just a seatbelt."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


class LLMClient:
    def __init__(self) -> None:
        self.s = get_settings()
        # provider -> (litellm model string, api key, supports json_object)
        self._providers = {
            "deepseek": (f"deepseek/{self.s.deepseek_model}", self.s.deepseek_api_key, True),
            "athena": (f"gemini/{self.s.athena_model}", self.s.gemini_api_key, True),
            "mnemosyne": (f"gemini/{self.s.mnemosyne_model}", self.s.gemini_api_key, True),
            "apollo": (f"gemini/{self.s.apollo_model}", self.s.gemini_api_key, True),
            "themis": (f"gemini/{self.s.themis_model}", self.s.gemini_api_key, True),
        }

    async def _call_provider(self, provider: str, system: str, user: str) -> str:
        import litellm

        model, api_key, json_mode = self._providers[provider]
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        else:
            system += "\nRespond with ONLY a raw JSON object — no prose, no code fences."
        resp = await litellm.acompletion(
            model=model,
            api_key=api_key,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=2048,
            timeout=self.s.llm_timeout_seconds,
            num_retries=1,  # transport-level retries; schema retry handled below
            **kwargs,
        )
        try:
            cost = litellm.completion_cost(completion_response=resp)
            metrics.counters[f"llm_cost_usd_x1e6_{provider}"] += int(cost * 1_000_000)
        except Exception:
            pass  # cost tracking is best-effort, never blocks
        return resp.choices[0].message.content or ""

    async def structured(self, provider: str, system: str, user: str, schema: type[T]) -> T:
        """Call the provider and parse into `schema`. One retry on bad JSON,
        then LLMUnavailable."""
        last_err: Exception | None = None
        for _attempt in range(2):
            try:
                raw = await self._call_provider(provider, system, user)
                return schema.model_validate(_extract_json(raw))
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                user = (user + "\n\nYour previous response was invalid: "
                        f"{e}\nRespond with ONLY valid JSON matching the schema.")
            except Exception as e:  # litellm raises provider-specific exceptions
                raise LLMUnavailable(f"{provider}: {e}") from e
        raise LLMUnavailable(f"{provider}: invalid output after retry: {last_err}")
