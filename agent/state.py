"""Append-only journal so a crash or a re-run never answers a customer twice.

One JSON object per line in `state/journal.jsonl`. We write `decided` before a POST
and `posted` after it, so a crash between the two is visible on the next run.
"""

import json
import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from agent.config import PROJECT_ROOT

DEFAULT_JOURNAL_PATH = PROJECT_ROOT / "state" / "journal.jsonl"


class Event(StrEnum):
    decided = "decided"  # about to POST; written before the request
    posted = "posted"  # the POST returned 2xx
    already_handled = "already_handled"  # a real 409
    posted_probably = "posted_probably"  # POST lost its response, re-check showed it handled
    outcome_unknown = "outcome_unknown"  # POST lost its response twice, ticket still open


class Step(StrEnum):
    skip = "skip"  # the ticket is done; never POST again
    recheck = "recheck"  # we may have posted; GET the ticket before doing anything
    fresh = "fresh"  # no trace of this ticket


_DONE = {Event.posted, Event.posted_probably, Event.already_handled}
_MAYBE_SENT = {Event.decided, Event.outcome_unknown}


class Journal:
    def __init__(self, path: Path = DEFAULT_JOURNAL_PATH) -> None:
        self.path = path

    def record(self, ticket_id: str, event: Event, **fields: Any) -> None:
        """Append one event and fsync it, so it survives a crash right after."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "ts": datetime.now(UTC).isoformat(),
            "ticket_id": ticket_id,
            "event": event.value,
            **fields,
        }
        prefix = "" if self._ends_with_newline() else "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            # A crash can leave a half line with no newline. Start a new line so this
            # event is not glued onto the broken one.
            handle.write(prefix + json.dumps(line, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def events(self, ticket_id: str | None = None) -> list[dict[str, Any]]:
        """Every readable event in file order. Broken or unknown lines are ignored."""
        if not self.path.exists():
            return []
        found: list[dict[str, Any]] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(raw)
            except ValueError:
                continue  # half-written line after a crash
            if not isinstance(item, dict) or item.get("event") not in Event.__members__:
                continue
            if ticket_id is None or item.get("ticket_id") == ticket_id:
                found.append(item)
        return found

    def next_step(self, ticket_id: str) -> Step:
        return next_step_from(self.events(ticket_id))

    def archive(self) -> Path | None:
        """Move the journal aside, for use after `POST /practice/reset` re-opens every ticket."""
        if not self.path.exists():
            return None
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = self.path.with_name(f"{self.path.stem}-{stamp}{self.path.suffix}")
        self.path.rename(target)
        return target

    def _ends_with_newline(self) -> bool:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return True
        with self.path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) == b"\n"


def next_step_from(events: list[dict[str, Any]]) -> Step:
    """Decide what to do with one ticket from its events, oldest first."""
    kinds = {item["event"] for item in events}
    if kinds & _DONE:
        return Step.skip
    if kinds & _MAYBE_SENT:
        return Step.recheck
    return Step.fresh
