"""Thin OpenAI-compatible client: strict JSON out, one retry, token and cost tracking."""

import json
import re
from typing import Any, Protocol, TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from agent.config import Settings

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class Usage(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class LlmError(Exception):
    """The LLM call itself failed (network, auth, provider error). Nothing was decided."""


class LlmInvalidOutput(Exception):
    """The model returned invalid JSON (or JSON that fails the schema) twice."""

    def __init__(self, message: str, usage: Usage) -> None:
        super().__init__(message)
        self.usage = usage


class JsonLlm(Protocol):
    def complete_json(self, system: str, user: str, schema: type[T]) -> tuple[T, Usage]: ...


class LlmClient:
    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client or OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=60.0,
            max_retries=2,
        )
        self.total = Usage()

    def complete_json(self, system: str, user: str, schema: type[T]) -> tuple[T, Usage]:
        """Ask for one JSON object matching `schema`. Retry once with the error, then give up."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        usage = Usage()
        problem = ""
        for _attempt in range(2):
            text, call_usage = self._chat(messages)
            usage += call_usage
            try:
                return schema.model_validate(json.loads(_strip_fences(text))), usage
            except (ValueError, ValidationError) as exc:
                problem = _short_error(exc)
                messages += [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": "That was not valid. Problem: "
                        f"{problem}\nReturn only the corrected JSON object, nothing else.",
                    },
                ]
        raise LlmInvalidOutput(f"Model output invalid twice: {problem}", usage)

    def _chat(self, messages: list[dict[str, str]]) -> tuple[str, Usage]:
        try:
            response = self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except OpenAIError as exc:
            raise LlmError(f"LLM call failed: {type(exc).__name__}") from exc
        text = response.choices[0].message.content or ""
        usage = self._usage(response)
        self.total += usage
        return text, usage

    def _usage(self, response: Any) -> Usage:
        raw = getattr(response, "usage", None)
        tokens_in = int(getattr(raw, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(raw, "completion_tokens", 0) or 0)
        cost = (
            tokens_in * self._settings.llm_price_per_million_input
            + tokens_out * self._settings.llm_price_per_million_output
        ) / 1_000_000
        return Usage(tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost)


def _strip_fences(text: str) -> str:
    """Some models wrap JSON in ```json fences even when asked not to."""
    return _FENCE_RE.sub("", text.strip())


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'root'}: {err['msg']}" for err in exc.errors()
        )[:500]
    return f"not valid JSON ({exc})"[:500]
