"""One redacted JSON line per ticket in `logs/run-<timestamp>.jsonl`.

The log line model has no subject or body field, so ticket text cannot be logged
by accident. Every string is redacted for card numbers and IBANs before writing.
"""

import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.config import PROJECT_ROOT

DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

CARD_MASK = "[REDACTED_CARD]"
IBAN_MASK = "[REDACTED_IBAN]"

# 13-19 digits, single spaces or dashes allowed between digits.
_CARD_RE = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
# Country code, 2 check digits, then 11-30 letters/digits, optionally in groups of 4.
_IBAN_RE = re.compile(
    r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b",
    re.IGNORECASE,
)


class Outcome(StrEnum):
    posted = "posted"
    posted_probably = "posted_probably"
    skipped = "skipped"
    already_handled = "already_handled"
    outcome_unknown = "outcome_unknown"
    dry_run = "dry_run"
    error = "error"


class TicketLog(BaseModel):
    """What we log for one ticket. No ticket subject, body, reply or note text."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    outcome: Outcome
    language: str | None = None
    category: str | None = None
    priority: str | None = None
    action: str | None = None
    confidence: float | None = None
    faq_ids: list[str] = Field(default_factory=list)
    rule_hits: list[str] = Field(default_factory=list)
    verifier: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: str | None = None


class RunLog:
    def __init__(self, log_dir: Path = DEFAULT_LOG_DIR) -> None:
        # Microseconds keep two quick runs from sharing one file.
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        self.path = log_dir / f"run-{stamp}.jsonl"

    def write(self, entry: TicketLog) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = redact_value(entry.model_dump(mode="json"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def redact(text: str) -> str:
    """Mask IBANs and Luhn-valid card numbers. Other numbers stay as they are."""
    text = _IBAN_RE.sub(_mask_iban, text)
    return _CARD_RE.sub(_mask_card, text)


def redact_value(value: Any) -> Any:
    """Redact every string inside dicts and lists."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


def _mask_iban(match: re.Match[str]) -> str:
    compact = match.group().replace(" ", "")
    # Real IBANs are 15-34 characters. Shorter matches are something else.
    return IBAN_MASK if 15 <= len(compact) <= 34 else match.group()


def _mask_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group())
    return CARD_MASK if _luhn_ok(digits) else match.group()


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        number = int(char)
        if index % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0
