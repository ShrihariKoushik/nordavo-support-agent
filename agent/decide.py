"""One ticket in, one decision out: rules -> LLM -> policy in code -> verify replies -> Decision.

The model proposes; this module decides. Every time code overrides the model, the
reason is added to `policy_notes` so the log and the eval can show it.
"""

import time

from pydantic import BaseModel, Field

from agent.config import Settings
from agent.faq import Faq
from agent.llm import JsonLlm, LlmInvalidOutput, Usage
from agent.models import Action, Category, Decision, Language, Priority, Ticket
from agent.prompts import LlmDecision, decide_system, decide_user
from agent.rules import forced_handoff_hits, guess_language, is_empty, redact_payment_details
from agent.verify import VerifyOutcome, verify_reply


class DecisionResult(BaseModel):
    decision: Decision
    faq_ids: list[str] = Field(default_factory=list)
    rule_hits: list[str] = Field(default_factory=list)
    policy_notes: list[str] = Field(default_factory=list)
    verifier: VerifyOutcome | None = None  # None when the decision was not a reply
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


def decide(ticket: Ticket, *, faq: Faq, llm: JsonLlm, settings: Settings) -> DecisionResult:
    started = time.monotonic()
    result = _decide(ticket, faq=faq, llm=llm, settings=settings)
    result.latency_ms = round((time.monotonic() - started) * 1000)
    return result


def _decide(ticket: Ticket, *, faq: Faq, llm: JsonLlm, settings: Settings) -> DecisionResult:
    # a) Nothing to answer: close without spending a model call.
    if is_empty(ticket.subject, ticket.body):
        decision = Decision(
            language=Language(guess_language(ticket.subject, ticket.body)),
            category=Category.other,
            priority=Priority.P4,
            reason="Empty ticket: no question or content to act on.",
            action=Action.close,
            handoff_note="Empty ticket (only greetings or a signature); closed.",
        )
        return DecisionResult(decision=decision, policy_notes=["empty ticket: closed without LLM"])

    hits = forced_handoff_hits(ticket.subject, ticket.body)
    rule_names = [hit.rule.value for hit in hits]

    # b) Ask the model.
    try:
        proposal, usage = llm.complete_json(decide_system(faq), decide_user(ticket), LlmDecision)
    except LlmInvalidOutput as exc:
        return _fallback_handoff(ticket, rule_names, exc.usage)

    # c) Policy in code.
    action, priority, notes = _apply_policy(proposal, rule_names, faq, settings)

    # d) Final decision. Payment details never leave in any text field.
    reason = proposal.reason
    if notes:
        reason = f"{reason} Policy: {'; '.join(notes)}."
    handoff_note = proposal.handoff_note
    if action is Action.handoff and notes:
        handoff_note = f"[{'; '.join(notes)}] {handoff_note}"
    decision = Decision(
        language=proposal.language,
        category=proposal.category,
        priority=priority,
        reason=redact_payment_details(reason),
        action=action,
        draft_reply=redact_payment_details(proposal.draft_reply) if action is Action.reply else None,
        handoff_note=redact_payment_details(handoff_note) if action is Action.handoff else None,
        confidence=proposal.confidence,
    )

    # e) Only replies reach a customer, so only replies are checked.
    verified: VerifyOutcome | None = None
    if decision.action is Action.reply:
        cited = [entry for entry in faq.entries if entry.id in proposal.faq_ids]
        verified = verify_reply(ticket, decision, cited, llm=llm, settings=settings)
        usage += verified.usage
        if not verified.passed:
            failed = "Draft failed verification: " + "; ".join(verified.failures) + "."
            notes.append("reply failed verification, handed off")
            decision = decision.model_copy(
                update={
                    "action": Action.handoff,
                    "draft_reply": None,
                    "handoff_note": redact_payment_details(f"{failed} {proposal.handoff_note}"),
                    "reason": redact_payment_details(
                        f"{proposal.reason} Policy: reply failed verification, handed off."
                    ),
                }
            )

    return DecisionResult(
        decision=decision,
        faq_ids=proposal.faq_ids,
        rule_hits=rule_names,
        policy_notes=notes,
        verifier=verified,
        tokens_in=usage.tokens_in,
        tokens_out=usage.tokens_out,
        cost_usd=usage.cost_usd,
    )


def _apply_policy(
    proposal: LlmDecision,
    rule_names: list[str],
    faq: Faq,
    settings: Settings,
) -> tuple[Action, Priority, list[str]]:
    """Return the action and priority code allows, plus a note for each override."""
    threshold = settings.reply_confidence_min
    action = proposal.proposed_action
    priority = proposal.priority
    notes: list[str] = []

    forced = sorted(set(rule_names) | set(proposal.needs_human_reasons))
    if forced:
        notes.append(f"forced handoff ({', '.join(forced)})")
        action = Action.handoff
        if priority in (Priority.P3, Priority.P4):
            priority = Priority.P2
        return action, priority, notes

    if action is Action.reply:
        unknown = [item for item in proposal.faq_ids if item not in faq.ids]
        if not proposal.faq_ids:
            notes.append("reply cites no FAQ entry, handed off")
        elif unknown:
            notes.append(f"reply cites unknown FAQ id {', '.join(unknown)}, handed off")
        elif not proposal.draft_reply or not proposal.draft_reply.strip():
            notes.append("reply has no draft text, handed off")
        elif proposal.is_spam:
            notes.append("model proposed a reply to spam, handed off")
        elif proposal.confidence < threshold:
            notes.append(f"reply confidence {proposal.confidence:.2f} < {threshold:.2f}, handed off")
        if notes:
            action = Action.handoff

    elif action is Action.close:
        if not proposal.is_spam:
            notes.append("close proposed but not marked spam, handed off")
            action = Action.handoff
        elif proposal.confidence < threshold:
            notes.append(f"close confidence {proposal.confidence:.2f} < {threshold:.2f}, handed off")
            action = Action.handoff

    return action, priority, notes


def _fallback_handoff(ticket: Ticket, rule_names: list[str], usage: Usage) -> DecisionResult:
    """The model gave unusable output twice. A human reads it; we guess only what we must."""
    note = "model output invalid twice, handed off"
    if rule_names:
        note += f"; forced handoff ({', '.join(rule_names)})"
    decision = Decision(
        language=Language(guess_language(ticket.subject, ticket.body)),
        category=Category.other,
        priority=Priority.P2 if rule_names else Priority.P3,
        reason=f"Automatic triage failed. Policy: {note}.",
        action=Action.handoff,
        handoff_note=(
            f"[{note}] Automatic triage could not classify this ticket. "
            "Please read it and handle it manually."
        ),
    )
    return DecisionResult(
        decision=decision,
        rule_hits=rule_names,
        policy_notes=[note],
        tokens_in=usage.tokens_in,
        tokens_out=usage.tokens_out,
        cost_usd=usage.cost_usd,
    )
