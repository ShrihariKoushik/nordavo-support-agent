"""CLI: `python -m agent run [--dry-run] [--limit N]`, `reset`, `status`."""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable
from itertools import islice

from agent.config import ConfigError, load_settings
from agent.helpdesk import (
    AlreadyHandled,
    AlreadyHandledAfterUncertainPost,
    HelpdeskClient,
    HelpdeskError,
    OutcomeUnknown,
)
from agent.logging_ import Outcome, RunLog, TicketLog
from agent.models import Action, Category, Decision, Language, Priority, Ticket
from agent.state import Event, Journal, Step

DecideFn = Callable[[Ticket], Decision]


def stub_decide(ticket: Ticket) -> Decision:
    """Placeholder until decide.py exists: hand everything to a human."""
    return Decision(
        language=Language.en,
        category=Category.other,
        priority=Priority.P3,
        reason="Stub decision while the pipeline is being built.",
        action=Action.handoff,
        handoff_note="stub",
    )


def run_tickets(
    client: HelpdeskClient,
    journal: Journal,
    run_log: RunLog,
    decide: DecideFn,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> Counter[Outcome]:
    """Handle each open ticket once. One failing ticket never stops the run."""
    counts: Counter[Outcome] = Counter()
    for ticket in islice(client.iter_open_tickets(), limit):
        started = time.monotonic()
        decision: Decision | None = None
        error: str | None = None
        try:
            outcome, decision = _handle(client, journal, decide, ticket, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001 - logged per ticket, then the run continues
            outcome, error = Outcome.error, f"{type(exc).__name__}: {exc}"
        entry = _log_entry(ticket.id, outcome, decision, error, started)
        run_log.write(entry)
        counts[outcome] += 1
        print(f"{ticket.id}  {entry.action or '-':8}  {outcome.value}")
    return counts


def _handle(
    client: HelpdeskClient,
    journal: Journal,
    decide: DecideFn,
    ticket: Ticket,
    *,
    dry_run: bool,
) -> tuple[Outcome, Decision | None]:
    step = journal.next_step(ticket.id)
    if step is Step.skip:
        return Outcome.skipped, None
    if step is Step.recheck:
        # An earlier run may have posted. Trust the helpdesk, not our memory.
        ticket = client.get_ticket(ticket.id)
        if not ticket.is_open:
            if not dry_run:
                journal.record(ticket.id, Event.posted_probably)
            return Outcome.skipped, None

    decision = decide(ticket)
    if dry_run:
        return Outcome.dry_run, decision

    journal.record(ticket.id, Event.decided, action=decision.action.value)
    try:
        _post(client, ticket.id, decision)
    except AlreadyHandledAfterUncertainPost:  # must come before its parent AlreadyHandled
        journal.record(ticket.id, Event.posted_probably, action=decision.action.value)
        return Outcome.posted_probably, decision
    except AlreadyHandled:
        journal.record(ticket.id, Event.already_handled, action=decision.action.value)
        return Outcome.already_handled, decision
    except OutcomeUnknown:
        journal.record(ticket.id, Event.outcome_unknown, action=decision.action.value)
        return Outcome.outcome_unknown, decision
    journal.record(ticket.id, Event.posted, action=decision.action.value)
    return Outcome.posted, decision


def _post(client: HelpdeskClient, ticket_id: str, decision: Decision) -> None:
    senders = {
        Action.reply: client.reply,
        Action.handoff: client.handoff,
        Action.close: client.close,
    }
    senders[decision.action](ticket_id, decision)


def _log_entry(
    ticket_id: str,
    outcome: Outcome,
    decision: Decision | None,
    error: str | None,
    started: float,
) -> TicketLog:
    entry = TicketLog(
        ticket_id=ticket_id,
        outcome=outcome,
        error=error,
        latency_ms=round((time.monotonic() - started) * 1000),
    )
    if decision is not None:
        entry.language = decision.language.value
        entry.category = decision.category.value
        entry.priority = decision.priority.value
        entry.action = decision.action.value
        entry.confidence = decision.confidence
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent")
    commands = parser.add_subparsers(dest="command", required=True)
    run_cmd = commands.add_parser("run", help="decide and post every open ticket")
    run_cmd.add_argument("--dry-run", action="store_true", help="decide only; no POST, no journal")
    run_cmd.add_argument("--limit", type=int, default=None, help="handle at most N tickets")
    commands.add_parser("reset", help="re-open all practice tickets and archive the journal")
    commands.add_parser("status", help="show mode and ticket counts")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    journal = Journal()
    try:
        with HelpdeskClient(settings) as client:
            if args.command == "status":
                status = client.get_status()
                print(f"mode={status.mode} open={status.open_tickets} handled={status.handled_tickets}")
                return 0
            if args.command == "reset":
                client.reset()
                archived = journal.archive()
                print("Practice tickets re-opened.")
                if archived:
                    print(f"Journal archived to {archived}")
                return 0
            run_log = RunLog()
            counts = run_tickets(
                client, journal, run_log, stub_decide, dry_run=args.dry_run, limit=args.limit
            )
    except HelpdeskError as exc:
        print(f"Helpdesk error: {exc}", file=sys.stderr)
        return 1

    summary = ", ".join(f"{name.value}={count}" for name, count in sorted(counts.items()))
    print(f"Done: {summary or 'no open tickets'}. Log: {run_log.path}")
    failed = counts[Outcome.error] + counts[Outcome.outcome_unknown]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
