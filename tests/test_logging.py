"""The per-ticket log line. Redaction itself is tested in test_rules.py."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.logging_ import Outcome, RunLog, TicketLog
from agent.rules import CARD_MASK, IBAN_MASK


def test_run_log_writes_one_redacted_line(tmp_path: Path) -> None:
    run_log = RunLog(tmp_path)
    run_log.write(
        TicketLog(
            ticket_id="T001",
            outcome=Outcome.error,
            action="handoff",
            error="POST failed for card 4111 1111 1111 1111 and NL91 ABNA 0417 1643 00",
            rule_hits=["refund_or_dispute"],
        )
    )

    lines = run_log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["error"] == f"POST failed for card {CARD_MASK} and {IBAN_MASK}"
    assert entry["rule_hits"] == ["refund_or_dispute"]
    assert run_log.path.name.startswith("run-")


def test_log_line_refuses_ticket_text() -> None:
    with pytest.raises(ValidationError):
        TicketLog(ticket_id="T001", outcome=Outcome.posted, body="secret")  # type: ignore[call-arg]
