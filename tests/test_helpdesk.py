"""Helpdesk client: retries, typed errors, pagination. No real network."""

import json
from collections.abc import Callable

import httpx
import pytest

from agent.config import Settings
from agent.helpdesk import (
    AlreadyHandled,
    AlreadyHandledAfterUncertainPost,
    HelpdeskClient,
    HelpdeskError,
    NotFound,
    OutcomeUnknown,
    ServiceUnavailable,
    Unauthorized,
    ValidationFailed,
)
from agent.models import Decision

STATUS = {
    "mode": "practice",
    "candidate": "Test",
    "endpoints": ["GET  /status"],
    "live_ends_at": None,
    "open_tickets": 30,
    "handled_tickets": 0,
}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def settings(**overrides: object) -> Settings:
    data: dict[str, object] = {
        "helpdesk_url": "https://helpdesk.test/functions/v1/helpdesk",
        "helpdesk_key": "test-key",
        "candidate_email": "dev@example.com",
        "candidate_name": "Test Candidate",
        "llm_base_url": "https://llm.test/v1",
        "llm_api_key": "llm-key",
        "llm_model": "test-model",
        "llm_price_per_million_input": 0.1,
        "llm_price_per_million_output": 0.2,
    }
    data.update(overrides)
    return Settings.model_validate(data)


def client_for(
    handler: Callable[[httpx.Request], httpx.Response],
    **setting_overrides: object,
) -> tuple[HelpdeskClient, FakeClock]:
    clock = FakeClock()
    client = HelpdeskClient(
        settings(**setting_overrides),
        transport=httpx.MockTransport(handler),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    return client, clock


def reply_decision() -> Decision:
    return Decision(
        language="nl",
        category="question",
        priority="P4",
        reason="How-to answered by the FAQ.",
        action="reply",
        draft_reply="Hoi, dit staat in de FAQ.",
        confidence=0.9,
    )


def ticket(ticket_id: str, **extra: object) -> dict[str, object]:
    body: dict[str, object] = {
        "id": ticket_id,
        "subject": "Onderwerp",
        "body": "Vraag",
        "channel": "email",
        "customer_plan": "Basis",
        "received_at": "2026-10-05T06:12:00+00:00",
    }
    body.update(extra)
    return body


def test_503_then_200_retries_with_backoff() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": "service_unavailable", "message": "down"})
        return httpx.Response(200, json=STATUS)

    client, clock = client_for(handler)
    with client:
        status = client.get_status()

    assert status.open_tickets == 30
    assert calls == 2
    assert clock.slept
    assert all(delay >= 0 for delay in clock.slept)


def test_503_stops_after_five_attempts() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": "service_unavailable", "message": "down"})

    client, _clock = client_for(handler)
    with client, pytest.raises(ServiceUnavailable) as caught:
        client.get_status()

    assert calls == 5
    assert caught.value.error == "service_unavailable"


def test_429_waits_for_retry_after() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "3"},
                json={"error": "rate_limited", "message": "slow down"},
            )
        return httpx.Response(200, json=STATUS)

    client, clock = client_for(handler)
    with client:
        status = client.get_status()

    assert status.mode == "practice"
    assert calls == 2
    assert 3.0 in clock.slept


def test_409_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            409,
            json={"error": "already_handled", "message": "This ticket was already handled."},
        )

    client, clock = client_for(handler)
    with client, pytest.raises(AlreadyHandled) as caught:
        client.reply("T001", reply_decision())

    assert calls == 1
    assert clock.slept == []
    assert caught.value.error == "already_handled"
    assert caught.value.status_code == 409


def test_422_surfaces_problems_and_is_not_retried() -> None:
    problems = [
        "category must be one of: bug, billing, feature_request, question, other",
        "draft_reply is required for a reply",
    ]
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            422,
            json={
                "error": "validation_failed",
                "message": "The request body is not valid.",
                "problems": problems,
            },
        )

    client, _clock = client_for(handler)
    with client, pytest.raises(ValidationFailed) as caught:
        client.reply("T001", reply_decision())

    assert calls == 1
    assert caught.value.problems == problems
    assert caught.value.error == "validation_failed"


def test_pagination_follows_next_cursor_until_null() -> None:
    requested: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["limit"] == "10"
        assert request.url.params["status"] == "open"
        cursor = request.url.params.get("cursor")
        requested.append(cursor)
        if cursor is None:
            return httpx.Response(
                200,
                json={"tickets": [ticket("T001"), ticket("T002")], "next_cursor": "T002"},
            )
        if cursor == "T002":
            return httpx.Response(200, json={"tickets": [ticket("T003")], "next_cursor": None})
        raise AssertionError(f"unexpected cursor {cursor}")

    client, _clock = client_for(handler)
    with client:
        found = list(client.iter_open_tickets())

    assert [item.id for item in found] == ["T001", "T002", "T003"]
    assert requested == [None, "T002"]
    assert found[0].is_open


def test_requests_are_spaced_under_five_per_second() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=STATUS)

    client, clock = client_for(handler)
    with client:
        client.get_status()
        client.get_status()

    assert clock.slept == [0.25]
    assert 1 / clock.slept[0] < 5


def test_sends_auth_headers_and_omits_blank_name() -> None:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json=STATUS)

    named, _clock = client_for(handler)
    with named:
        named.get_status()
    assert seen[0]["Authorization"] == "Bearer test-key"
    assert seen[0]["X-Candidate-Email"] == "dev@example.com"
    assert seen[0]["X-Candidate-Name"] == "Test Candidate"

    unnamed, _clock = client_for(handler, candidate_name="")
    with unnamed:
        unnamed.get_status()
    assert "x-candidate-name" not in seen[1]


def test_timeout_on_post_reposts_only_when_the_ticket_is_still_open() -> None:
    posts = 0
    sent: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "POST":
            posts += 1
            if posts == 1:
                raise httpx.ConnectError("connection dropped")
            sent.update(json.loads(request.content))
            return httpx.Response(200, json={"ok": True})
        assert request.url.path.endswith("/tickets/T001")
        return httpx.Response(200, json=ticket("T001", handled=None))

    client, _clock = client_for(handler)
    with client:
        result = client.reply("T001", reply_decision())

    assert result == {"ok": True}
    assert posts == 2
    assert sent["draft_reply"] == "Hoi, dit staat in de FAQ."
    assert "handoff_note" not in sent
    assert "action" not in sent


def test_timeout_on_post_does_not_repost_when_the_ticket_was_handled() -> None:
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "POST":
            posts += 1
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json=ticket("T001", handled={"action": "reply"}))

    client, _clock = client_for(handler)
    with client, pytest.raises(AlreadyHandled):
        client.reply("T001", reply_decision())

    assert posts == 1


def test_second_timeout_does_not_post_a_third_time() -> None:
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "POST":
            posts += 1
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json=ticket("T001", handled=None))

    client, _clock = client_for(handler)
    with client, pytest.raises(OutcomeUnknown):
        client.reply("T001", reply_decision())

    assert posts == 2


def test_reset_posts_to_practice_reset() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith("/practice/reset")
        return httpx.Response(200, json={"open_tickets": 30})

    client, _clock = client_for(handler)
    with client:
        body = client.reset()
    assert body["open_tickets"] == 30


def test_401_and_404_are_typed() -> None:
    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized", "message": "bad key"})

    def missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not_found", "message": "no ticket"})

    client, _clock = client_for(unauthorized)
    with client, pytest.raises(Unauthorized):
        client.get_status()

    client, _clock = client_for(missing)
    with client, pytest.raises(NotFound):
        client.get_ticket("T999")


def test_get_timeout_then_200_retries_with_backoff() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json=STATUS)

    client, clock = client_for(handler)
    with client:
        status = client.get_status()

    assert status.open_tickets == 30
    assert calls == 2
    assert clock.slept
    assert all(delay >= 0 for delay in clock.slept)


def test_get_connection_errors_stop_after_five_attempts() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection refused")

    client, _clock = client_for(handler)
    with client, pytest.raises(HelpdeskError) as caught:
        client.get_status()

    assert calls == 5
    assert isinstance(caught.value.__cause__, httpx.ConnectError)


def test_uncertain_post_then_handled_raises_the_after_uncertain_post_subclass() -> None:
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "POST":
            posts += 1
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json=ticket("T001", handled={"action": "reply"}))

    client, _clock = client_for(handler)
    with client, pytest.raises(AlreadyHandledAfterUncertainPost) as caught:
        client.reply("T001", reply_decision())

    assert posts == 1
    assert isinstance(caught.value, AlreadyHandled)


def test_real_409_is_not_the_after_uncertain_post_subclass() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={"error": "already_handled", "message": "This ticket was already handled."},
        )

    client, _clock = client_for(handler)
    with client, pytest.raises(AlreadyHandled) as caught:
        client.reply("T001", reply_decision())

    assert not isinstance(caught.value, AlreadyHandledAfterUncertainPost)
