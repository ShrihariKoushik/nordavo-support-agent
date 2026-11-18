"""Redaction and the per-ticket log line."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.logging_ import CARD_MASK, IBAN_MASK, Outcome, RunLog, TicketLog, redact


@pytest.mark.parametrize(
    "card",
    ["4111111111111111", "4111 1111 1111 1111", "4111-1111-1111-1111", "5500 0000 0000 0004"],
)
def test_card_numbers_are_redacted(card: str) -> None:
    text = f"Mijn kaart is {card}, graag terugboeken."
    assert redact(text) == f"Mijn kaart is {CARD_MASK}, graag terugboeken."


@pytest.mark.parametrize("iban", ["NL91 ABNA 0417 1643 00", "NL91ABNA0417164300", "de89370400440532013000"])
def test_ibans_are_redacted(iban: str) -> None:
    assert redact(f"IBAN: {iban} graag.") == f"IBAN: {IBAN_MASK} graag."


@pytest.mark.parametrize(
    "text",
    [
        "We hebben 25 medewerkers en 3 vestigingen.",
        "Bel me op +31 6 12345678.",
        "Factuur 2026-10-05, ticket T005, bedrag 49,95.",
        "Kaart 4111 1111 1111 1112 klopt niet",  # 16 digits, fails Luhn
    ],
)
def test_normal_numbers_are_left_alone(text: str) -> None:
    assert redact(text) == text


def test_run_log_writes_one_redacted_line(tmp_path: Path) -> None:
    run_log = RunLog(tmp_path)
    run_log.write(
        TicketLog(
            ticket_id="T001",
            outcome=Outcome.error,
            action="handoff",
            error="POST failed for card 4111 1111 1111 1111",
        )
    )

    lines = run_log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["error"] == f"POST failed for card {CARD_MASK}"
    assert run_log.path.name.startswith("run-")


def test_log_line_refuses_ticket_text() -> None:
    with pytest.raises(ValidationError):
        TicketLog(ticket_id="T001", outcome=Outcome.posted, body="secret")  # type: ignore[call-arg]
