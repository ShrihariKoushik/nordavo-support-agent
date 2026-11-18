"""Decision validation: a reply needs a draft, a handoff needs a note."""

import pytest
from pydantic import ValidationError

from agent.models import Decision, Ticket


def test_reply_requires_draft_reply() -> None:
    with pytest.raises(ValidationError, match="draft_reply"):
        Decision(
            language="nl",
            category="question",
            priority="P4",
            reason="FAQ.",
            action="reply",
        )


def test_blank_draft_reply_is_rejected() -> None:
    with pytest.raises(ValidationError, match="draft_reply"):
        Decision(
            language="en",
            category="bug",
            priority="P3",
            reason="Old iOS.",
            action="reply",
            draft_reply="   ",
        )


def test_handoff_requires_handoff_note() -> None:
    with pytest.raises(ValidationError, match="handoff_note"):
        Decision(
            language="nl",
            category="billing",
            priority="P2",
            reason="Refund.",
            action="handoff",
        )


def test_close_needs_neither_text_field() -> None:
    decision = Decision(
        language="nl",
        category="other",
        priority="P4",
        reason="Spam.",
        action="close",
    )
    assert decision.api_body() == {
        "language": "nl",
        "category": "other",
        "priority": "P4",
        "reason": "Spam.",
    }


def test_handoff_body_carries_the_note() -> None:
    decision = Decision(
        language="nl",
        category="billing",
        priority="P2",
        reason="Refund request.",
        action="handoff",
        handoff_note="Customer asked for August back.",
        confidence=0.4,
    )
    body = decision.api_body()
    assert body["handoff_note"] == "Customer asked for August back."
    assert body["confidence"] == 0.4
    assert "draft_reply" not in body


def test_confidence_must_be_between_zero_and_one() -> None:
    with pytest.raises(ValidationError):
        Decision(
            language="de",
            category="question",
            priority="P4",
            reason="Preis.",
            action="reply",
            draft_reply="Siehe FAQ.",
            confidence=1.2,
        )


def test_ticket_is_open_only_while_handled_is_null() -> None:
    open_ticket = Ticket(id="T001", subject="S", body="B", handled=None)
    done = Ticket(id="T001", subject="S", body="B", handled={"action": "reply"})
    assert open_ticket.is_open
    assert not done.is_open
