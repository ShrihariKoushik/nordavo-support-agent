"""LlmClient JSON handling with a fake OpenAI client. No network."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from agent.config import Settings
from agent.llm import LlmClient, LlmInvalidOutput

SETTINGS = Settings.model_validate(
    {
        "helpdesk_url": "https://helpdesk.test",
        "helpdesk_key": "k",
        "candidate_email": "dev@example.com",
        "llm_base_url": "https://llm.test/v1",
        "llm_api_key": "k",
        "llm_model": "test-model",
        "llm_price_per_million_input": 1.0,
        "llm_price_per_million_output": 2.0,
    }
)


class Answer(BaseModel):
    value: int


class FakeOpenAI:
    """Mimics `client.chat.completions.create` and replays canned texts."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        text = self.texts.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100),
        )


def test_valid_json_first_time() -> None:
    fake = FakeOpenAI(['{"value": 7}'])
    llm = LlmClient(SETTINGS, client=fake)

    parsed, usage = llm.complete_json("sys", "user", Answer)

    assert parsed.value == 7
    assert fake.requests[0]["temperature"] == 0
    assert fake.requests[0]["response_format"] == {"type": "json_object"}
    assert (usage.tokens_in, usage.tokens_out) == (1000, 100)
    assert usage.cost_usd == pytest.approx((1000 * 1.0 + 100 * 2.0) / 1_000_000)


def test_fenced_json_is_accepted() -> None:
    llm = LlmClient(SETTINGS, client=FakeOpenAI(['```json\n{"value": 3}\n```']))
    assert llm.complete_json("s", "u", Answer)[0].value == 3


def test_retries_once_with_the_error_then_succeeds() -> None:
    fake = FakeOpenAI(["not json", '{"value": 1}'])
    llm = LlmClient(SETTINGS, client=fake)

    parsed, usage = llm.complete_json("s", "u", Answer)

    assert parsed.value == 1
    assert len(fake.requests) == 2
    assert "not valid" in fake.requests[1]["messages"][-1]["content"]
    assert usage.tokens_in == 2000
    assert llm.total.tokens_in == 2000


def test_invalid_twice_raises_with_usage() -> None:
    fake = FakeOpenAI(['{"value": "x"}', "still not json"])
    llm = LlmClient(SETTINGS, client=fake)

    with pytest.raises(LlmInvalidOutput) as caught:
        llm.complete_json("s", "u", Answer)

    assert len(fake.requests) == 2
    assert caught.value.usage.tokens_in == 2000


def test_seed_and_provider_pin_are_sent() -> None:
    settings = SETTINGS.model_copy(update={"llm_provider": "deepinfra/fp8", "llm_seed": 7})
    fake = FakeOpenAI(['{"value": 1}', '{"value": 2}'])
    llm = LlmClient(settings, client=fake)

    llm.complete_json("s", "u", Answer)
    llm.complete_json("s", "u", Answer, model="other-model")

    main, other = fake.requests
    assert main["seed"] == 7
    assert main["extra_body"] == {
        "provider": {"order": ["deepinfra/fp8"], "allow_fallbacks": False}
    }
    # A different (verifier) model is not forced onto the drafter's provider.
    assert other["seed"] == 7
    assert "extra_body" not in other


def test_no_provider_means_no_extra_body() -> None:
    fake = FakeOpenAI(['{"value": 1}'])
    LlmClient(SETTINGS, client=fake).complete_json("s", "u", Answer)

    assert "extra_body" not in fake.requests[0]
    assert fake.requests[0]["seed"] == 42  # default seed
