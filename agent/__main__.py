"""CLI: `python -m agent run [--dry-run] [--limit N]`, `reset`, `status`."""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable
from functools import partial
from itertools import islice

from agent.config import ConfigError, load_settings
from agent.decide import DecisionResult, decide
from agent.faq import FaqError, load_faq
from agent.helpdesk import (
    AlreadyHandled,
    AlreadyHandledAfterUncertainPost,
    HelpdeskClient,
    HelpdeskError,
    OutcomeUnknown,
)
from agent.llm import LlmClient
from agent.logging_ import Outcome, RunLog, TicketLog
from agent.models import Action, Decision, Ticket
from agent.rules import redact_payment_details
from agent.state import Event, Journal, Step

DecideFn = Callable[[Ticket], DecisionResult]


def run_tickets(
    client: HelpdeskClient,
    journal: Journal,
    run_log: RunLog,
    decide_fn: DecideFn,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> Counter[Outcome]:
    """Handle each open ticket once. One failing ticket never stops the run."""
    counts: Counter[Outcome] = Counter()
    for ticket in islice(client.iter_open_tickets(), limit):
        started = time.monotonic()
        result: DecisionResult | None = None
        error: str | None = None
        try:
            outcome, result = _handle(client, journal, decide_fn, ticket, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001 - logged per ticket, then the run continues
            outcome, error = Outcome.error, f"{type(exc).__name__}: {exc}"
        entry = _log_entry(ticket.id, outcome, result, error, started)
        run_log.write(entry)
        counts[outcome] += 1
        print(f"{ticket.id}  {entry.action or '-':8}  {outcome.value}")
        if dry_run and result is not None:
            _print_preview(result)
    return counts


def _handle(
    client: HelpdeskClient,
    journal: Journal,
    decide_fn: DecideFn,
    ticket: Ticket,
    *,
    dry_run: bool,
) -> tuple[Outcome, DecisionResult | None]:
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

    result = decide_fn(ticket)
    decision = result.decision
    if dry_run:
        return Outcome.dry_run, result

    journal.record(ticket.id, Event.decided, action=decision.action.value)
    try:
        _post(client, ticket.id, decision)
    except AlreadyHandledAfterUncertainPost:  # must come before its parent AlreadyHandled
        journal.record(ticket.id, Event.posted_probably, action=decision.action.value)
        return Outcome.posted_probably, result
    except AlreadyHandled:
        journal.record(ticket.id, Event.already_handled, action=decision.action.value)
        return Outcome.already_handled, result
    except OutcomeUnknown:
        journal.record(ticket.id, Event.outcome_unknown, action=decision.action.value)
        return Outcome.outcome_unknown, result
    journal.record(ticket.id, Event.posted, action=decision.action.value)
    return Outcome.posted, result


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
    result: DecisionResult | None,
    error: str | None,
    started: float,
) -> TicketLog:
    latency_ms = round((time.monotonic() - started) * 1000)
    if result is None:
        return TicketLog(ticket_id=ticket_id, outcome=outcome, error=error, latency_ms=latency_ms)
    decision = result.decision
    verifier = result.verifier
    verifier_fields: dict[str, object] = {}
    if verifier is not None:
        verifier_fields = {
            "verifier_verdict": "pass" if verifier.passed else "fail",
            "verifier_claims": verifier.claims,
            "verifier_unsupported": verifier.unsupported,
            "verifier_failures": verifier.failures,
            "language_issues": verifier.language_issues,
            "back_translation": verifier.back_translation,
        }
    return TicketLog(
        **verifier_fields,
        ticket_id=ticket_id,
        outcome=outcome,
        language=decision.language.value,
        category=decision.category.value,
        priority=decision.priority.value,
        action=decision.action.value,
        confidence=decision.confidence,
        reason=decision.reason,
        faq_ids=result.faq_ids,
        rule_hits=result.rule_hits,
        policy_notes=result.policy_notes,
        language_source=result.language_source,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_usd=result.cost_usd,
        latency_ms=latency_ms,
        error=error,
    )


def _print_preview(result: DecisionResult) -> None:
    """Dry run only: show the text that would be sent, so a person can review it."""
    decision = result.decision
    print(
        f"    {decision.language.value} {decision.category.value} {decision.priority.value}"
        f" conf={decision.confidence} faq={result.faq_ids} rules={result.rule_hits}"
    )
    print(f"    reason: {decision.reason}")
    if result.policy_notes:
        print(f"    policy: {'; '.join(result.policy_notes)}")
    verifier = result.verifier
    if verifier is not None:
        verdict = "pass" if verifier.passed else "fail"
        print(
            f"    verifier: {verdict}, claims={verifier.claims}, unsupported={verifier.unsupported}"
        )
        for failure in verifier.failures:
            print(f"      fail: {failure}")
        for issue in verifier.language_issues:
            print(f"      language issue: {issue}")
        if verifier.back_translation:
            back = redact_payment_details(verifier.back_translation).replace("\n", "\n      ")
            print(f"    back_translation:\n      {back}")
    text = decision.draft_reply or decision.handoff_note
    if text:
        label = "draft_reply" if decision.draft_reply else "handoff_note"
        body = redact_payment_details(text).replace("\n", "\n      ")
        print(f"    {label}:\n      {body}")
    print()


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
            try:
                faq = load_faq()
            except FaqError as exc:
                print(f"FAQ error: {exc}", file=sys.stderr)
                return 2
            llm = LlmClient(settings)
            decide_fn = partial(decide, faq=faq, llm=llm, settings=settings)
            run_log = RunLog()
            counts = run_tickets(
                client, journal, run_log, decide_fn, dry_run=args.dry_run, limit=args.limit
            )
    except HelpdeskError as exc:
        print(f"Helpdesk error: {exc}", file=sys.stderr)
        return 1

    summary = ", ".join(f"{name.value}={count}" for name, count in sorted(counts.items()))
    print(f"Done: {summary or 'no open tickets'}. Log: {run_log.path}")
    print(
        f"LLM: {llm.total.tokens_in} tokens in, {llm.total.tokens_out} out, "
        f"${llm.total.cost_usd:.4f}"
    )
    failed = counts[Outcome.error] + counts[Outcome.outcome_unknown]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
