"""The run loop against a small in-memory helpdesk. No real network."""

import json
from collections.abc import Callable
from pathlib import Path

import httpx

from agent.__main__ import run_tickets
from agent.config import Settings
from agent.decide import DecisionResult
from agent.helpdesk import HelpdeskClient
from agent.logging_ import Outcome, RunLog
from agent.models import Decision, Ticket
from agent.state import Event, Journal, Step


def handoff_decide(ticket: Ticket) -> DecisionResult:
    """Stands in for decide(): always a handoff, no model."""
    decision = Decision(
        language="en",
        category="other",
        priority="P3",
        reason="Test decision.",
        action="handoff",
        handoff_note="test",
    )
    return DecisionResult(decision=decision, rule_hits=["legal_threat"], tokens_in=10, cost_usd=0.001)


class FakeHelpdesk:
    """Remembers which tickets are handled and counts every POST."""

    def __init__(self, ids: list[str], *, list_all: bool = False) -> None:
        self.handled: dict[str, object] = {ticket_id: None for ticket_id in ids}
        self.list_all = list_all  # list handled tickets as open too (stale listing)
        self.posts: list[str] = []
        self.ticket_gets: list[str] = []
        self.fail_post: Callable[[str], None] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        tail = request.url.path.split("/helpdesk", 1)[1]
        parts = tail.strip("/").split("/")
        if request.method == "GET" and parts == ["tickets"]:
            ids = [i for i, h in self.handled.items() if self.list_all or h is None]
            return httpx.Response(
                200, json={"tickets": [_ticket(i, None) for i in ids], "next_cursor": None}
            )
        if request.method == "GET" and len(parts) == 2:
            self.ticket_gets.append(parts[1])
            return httpx.Response(200, json=_ticket(parts[1], self.handled[parts[1]]))
        if request.method == "POST" and len(parts) == 3:
            ticket_id, action = parts[1], parts[2]
            self.posts.append(ticket_id)
            if self.fail_post:
                self.fail_post(ticket_id)
            if self.handled[ticket_id] is not None:
                return httpx.Response(409, json={"error": "already_handled", "message": "done"})
            self.handled[ticket_id] = {"action": action, **json.loads(request.content)}
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(f"unexpected {request.method} {tail}")


def _ticket(ticket_id: str, handled: object) -> dict[str, object]:
    return {"id": ticket_id, "subject": "Onderwerp", "body": "Vraag", "handled": handled}


def _client(fake: FakeHelpdesk) -> HelpdeskClient:
    settings = Settings.model_validate(
        {
            "helpdesk_url": "https://helpdesk.test/functions/v1/helpdesk",
            "helpdesk_key": "test-key",
            "candidate_email": "dev@example.com",
            "llm_base_url": "https://llm.test/v1",
            "llm_api_key": "llm-key",
            "llm_model": "test-model",
        }
    )
    return HelpdeskClient(
        settings,
        transport=httpx.MockTransport(fake),
        sleep=lambda _seconds: None,
        monotonic=lambda: 0.0,
    )


def _run(
    fake: FakeHelpdesk,
    tmp_path: Path,
    *,
    decide: Callable[[Ticket], DecisionResult] = handoff_decide,
    dry_run: bool = False,
) -> tuple[dict[Outcome, int], list[dict[str, object]]]:
    run_log = RunLog(tmp_path / "logs")
    with _client(fake) as client:
        counts = run_tickets(
            client, Journal(tmp_path / "journal.jsonl"), run_log, decide, dry_run=dry_run
        )
    lines = []
    if run_log.path.exists():
        lines = [json.loads(x) for x in run_log.path.read_text(encoding="utf-8").splitlines()]
    return dict(counts), lines


def test_fresh_tickets_are_journaled_and_posted_once(tmp_path: Path) -> None:
    fake = FakeHelpdesk(["T1", "T2"])

    counts, lines = _run(fake, tmp_path)

    assert counts == {Outcome.posted: 2}
    assert fake.posts == ["T1", "T2"]
    events = [e["event"] for e in Journal(tmp_path / "journal.jsonl").events("T1")]
    assert events == ["decided", "posted"]
    assert lines[0]["action"] == "handoff" and lines[0]["outcome"] == "posted"
    assert lines[0]["reason"] == "Test decision."
    assert lines[0]["rule_hits"] == ["legal_threat"]
    assert lines[0]["tokens_in"] == 10 and lines[0]["cost_usd"] == 0.001
    assert lines[0]["ts"]
    assert "body" not in lines[0] and "subject" not in lines[0]


def test_crash_after_decided_rechecks_and_does_not_post_again(tmp_path: Path) -> None:
    # Last run wrote `decided`, the POST landed, then the process died before `posted`.
    fake = FakeHelpdesk(["T1"], list_all=True)
    fake.handled["T1"] = {"action": "handoff"}
    journal = Journal(tmp_path / "journal.jsonl")
    journal.record("T1", Event.decided, action="handoff")

    counts, _lines = _run(fake, tmp_path)

    assert counts == {Outcome.skipped: 1}
    assert fake.ticket_gets == ["T1"]
    assert fake.posts == []
    assert journal.next_step("T1") is Step.skip


def test_crash_after_decided_posts_when_ticket_is_still_open(tmp_path: Path) -> None:
    # Last run wrote `decided` and died before the POST left the machine.
    fake = FakeHelpdesk(["T1"])
    Journal(tmp_path / "journal.jsonl").record("T1", Event.decided, action="handoff")

    counts, _lines = _run(fake, tmp_path)

    assert counts == {Outcome.posted: 1}
    assert fake.ticket_gets == ["T1"]
    assert fake.posts == ["T1"]


def test_posted_tickets_are_skipped_on_rerun(tmp_path: Path) -> None:
    fake = FakeHelpdesk(["T1", "T2"], list_all=True)
    _run(fake, tmp_path)
    assert fake.posts == ["T1", "T2"]

    counts, lines = _run(fake, tmp_path)

    assert counts == {Outcome.skipped: 2}
    assert fake.posts == ["T1", "T2"]
    assert fake.ticket_gets == []
    assert {line["outcome"] for line in lines} == {"skipped"}


def test_uncertain_post_then_handled_is_recorded_as_posted_probably(tmp_path: Path) -> None:
    fake = FakeHelpdesk(["T1"])

    def drop_response(ticket_id: str) -> None:
        fake.handled[ticket_id] = {"action": "handoff"}  # the write lands...
        raise httpx.ReadTimeout("timed out")  # ...but the response is lost

    fake.fail_post = drop_response

    counts, _lines = _run(fake, tmp_path)

    assert counts == {Outcome.posted_probably: 1}
    assert fake.posts == ["T1"]
    events = [e["event"] for e in Journal(tmp_path / "journal.jsonl").events("T1")]
    assert events == ["decided", "posted_probably"]


def test_real_409_is_recorded_as_already_handled(tmp_path: Path) -> None:
    fake = FakeHelpdesk(["T1"])
    fake.handled["T1"] = {"action": "reply"}
    fake.list_all = True

    counts, _lines = _run(fake, tmp_path)

    assert counts == {Outcome.already_handled: 1}
    assert Journal(tmp_path / "journal.jsonl").next_step("T1") is Step.skip


def test_one_failing_ticket_does_not_stop_the_run(tmp_path: Path) -> None:
    fake = FakeHelpdesk(["T1", "T2"])

    def decide(ticket: Ticket) -> DecisionResult:
        if ticket.id == "T1":
            raise RuntimeError("model fell over")
        return handoff_decide(ticket)

    counts, lines = _run(fake, tmp_path, decide=decide)

    assert counts == {Outcome.error: 1, Outcome.posted: 1}
    assert fake.posts == ["T2"]
    assert lines[0]["outcome"] == "error" and "model fell over" in lines[0]["error"]


def test_dry_run_never_posts_or_writes_the_journal(tmp_path: Path) -> None:
    fake = FakeHelpdesk(["T1", "T2"])

    counts, _lines = _run(fake, tmp_path, dry_run=True)

    assert counts == {Outcome.dry_run: 2}
    assert fake.posts == []
    assert not (tmp_path / "journal.jsonl").exists()


def test_only_restricts_the_run_to_given_ticket_ids(tmp_path: Path) -> None:
    fake = FakeHelpdesk(["T1", "T2", "T3"])
    run_log = RunLog(tmp_path / "logs")
    with _client(fake) as client:
        counts = run_tickets(
            client, Journal(tmp_path / "journal.jsonl"), run_log, handoff_decide,
            dry_run=True, only={"T2", "T3"},
        )
    assert dict(counts) == {Outcome.dry_run: 2}
    logged = [json.loads(x)["ticket_id"] for x in run_log.path.read_text(encoding="utf-8").splitlines()]
    assert logged == ["T2", "T3"]
