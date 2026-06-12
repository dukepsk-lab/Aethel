"""Cloud LLM clients (strictly external APIs — no local inference).

All three providers are wrapped behind one interface that:
- requests JSON output natively where the API supports it,
- validates the response against a Pydantic schema,
- retries exactly once on invalid JSON,
- raises LLMUnavailable on any other failure.

Callers treat LLMUnavailable as fail-closed: NO TRADE. There is no
default-approve path anywhere.
"""

from __future__ import annotations

import json
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from aethel.config import get_settings

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
        self._http = httpx.AsyncClient(timeout=self.s.llm_timeout_seconds)

    async def _call_provider(self, provider: str, system: str, user: str) -> str:
        if provider == "deepseek":
            r = await self._http.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {self.s.deepseek_api_key}"},
                json={
                    "model": self.s.deepseek_model,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        if provider == "anthropic":
            r = await self._http.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.s.anthropic_api_key,
                         "anthropic-version": "2023-06-01"},
                json={
                    "model": self.s.anthropic_model,
                    "max_tokens": 2048,
                    "system": system,
                    "messages": [{"role": "user", "content": user},
                                 {"role": "assistant", "content": "{"}],
                    "temperature": 0.2,
                },
            )
            r.raise_for_status()
            return "{" + r.json()["content"][0]["text"]

        if provider == "gemini":
            r = await self._http.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.s.gemini_model}:generateContent",
                headers={"x-goog-api-key": self.s.gemini_api_key},
                json={
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"parts": [{"text": user}]}],
                    "generationConfig": {"responseMimeType": "application/json",
                                         "temperature": 0.2},
                },
            )
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]

        raise ValueError(f"unknown provider {provider}")

    async def structured(self, provider: str, system: str, user: str, schema: type[T]) -> T:
        """Call the provider and parse into `schema`. One retry on bad JSON,
        then LLMUnavailable."""
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                raw = await self._call_provider(provider, system, user)
                return schema.model_validate(_extract_json(raw))
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                user = (user + "\n\nYour previous response was invalid: "
                        f"{e}\nRespond with ONLY valid JSON matching the schema.")
            except (httpx.HTTPError, KeyError) as e:
                raise LLMUnavailable(f"{provider}: {e}") from e
        raise LLMUnavailable(f"{provider}: invalid output after retry: {last_err}")
