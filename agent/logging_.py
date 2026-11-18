"""One redacted JSON line per ticket in `logs/run-<timestamp>.jsonl`.

The log line model has no subject or body field, so ticket text cannot be logged
by accident. Every string is redacted for card numbers and IBANs before writing.
"""

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.config import PROJECT_ROOT
from agent.rules import redact_payment_details

DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


class Outcome(StrEnum):
    posted = "posted"
    posted_probably = "posted_probably"
    skipped = "skipped"
    already_handled = "already_handled"
    outcome_unknown = "outcome_unknown"
    dry_run = "dry_run"
    error = "error"


class TicketLog(BaseModel):
    """What we log for one ticket. No ticket subject, body, reply or note text.

    `reason` is the model's short classification reason. `back_translation` is the English
    back-translation of our own draft reply, so a non-Dutch reader can audit what was sent.
    Both are redacted like every string.
    """

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    outcome: Outcome
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    language: str | None = None
    category: str | None = None
    priority: str | None = None
    action: str | None = None
    confidence: float | None = None
    reason: str | None = None
    faq_ids: list[str] = Field(default_factory=list)
    rule_hits: list[str] = Field(default_factory=list)
    policy_notes: list[str] = Field(default_factory=list)
    language_source: str | None = None  # code | model | default
    verifier_verdict: str | None = None  # pass | fail | None when not a reply
    verifier_claims: int | None = None
    verifier_unsupported: int | None = None
    verifier_failures: list[str] = Field(default_factory=list)
    language_issues: list[str] = Field(default_factory=list)
    back_translation: str | None = None
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


def redact_value(value: Any) -> Any:
    """Redact every string inside dicts and lists."""
    if isinstance(value, str):
        return redact_payment_details(value)
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value
