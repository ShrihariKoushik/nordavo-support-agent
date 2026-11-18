"""One ticket in, one decision out: rules -> LLM -> policy in code -> verify replies -> Decision.

The model proposes; this module decides. Every time code overrides the model, the
reason is added to `policy_notes` so the log and the eval can show it.
"""

import time
from functools import partial

from pydantic import BaseModel, Field

from agent.config import Settings
from agent.faq import Faq
from agent.lang import detect_language
from agent.llm import JsonLlm, LlmInvalidOutput, Usage
from agent.models import Action, Category, Decision, Language, Priority, Ticket
from agent.prompts import (
    ENGLISH_FIELDS_CORRECTION,
    INSTRUCTIONS_NOTE,
    PROMISE_CORRECTION,
    SAFE_NOTE,
    LlmDecision,
    decide_system,
    decide_user,
)
from agent.rules import (
    Rule,
    commitment_phrases,
    forced_handoff_hits,
    is_empty,
    promise_phrases,
    redact_payment_details,
    remove_invented_name,
)
from agent.verify import VerifyOutcome, verify_reply


class DecisionResult(BaseModel):
    decision: Decision
    faq_ids: list[str] = Field(default_factory=list)
    rule_hits: list[str] = Field(default_factory=list)
    policy_notes: list[str] = Field(default_factory=list)
    # Where decision.language came from: "code" (detected), "model", or "default" (nl).
    language_source: str = "model"
    # Detected language of the model's reason (should be "en"; None when too short to tell).
    reason_language: str | None = None
    # True when code removed a greeting name that does not appear in the ticket.
    name_removed: bool = False
    # The promise-to-the-customer phrase code found in a draft, if any (see rules.promise_phrases).
    promise_phrase: str | None = None
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
    # The language comes from code when the text is clear; the model only fills gaps.
    detected = detect_language(f"{ticket.subject}\n{ticket.body}")
    code_language = detected[0] if detected else None

    # a) Nothing to answer: close without spending a model call.
    if is_empty(ticket.subject, ticket.body):
        decision = Decision(
            language=code_language or Language.nl,
            category=Category.other,
            priority=Priority.P4,
            reason="Empty ticket: no question or content to act on.",
            action=Action.close,
            handoff_note="Empty ticket (only greetings or a signature); closed.",
        )
        return DecisionResult(
            decision=decision,
            policy_notes=["empty ticket: closed without LLM"],
            language_source="code" if code_language else "default",
            reason_language="en",
        )

    hits = forced_handoff_hits(ticket.subject, ticket.body)
    rule_names = [hit.rule.value for hit in hits]

    # b) Ask the model. reason and handoff_note are our English audit trail: one retry if not.
    system, user = decide_system(faq), decide_user(ticket, language=code_language)
    usage = Usage()
    try:
        proposal, call_usage = llm.complete_json(system, user, LlmDecision)
        usage += call_usage
    except LlmInvalidOutput as exc:
        return _fallback_handoff(ticket, rule_names, usage + exc.usage, code_language)
    if not _english_fields(proposal):
        try:
            retried, call_usage = llm.complete_json(
                system, f"{user}\n\n{ENGLISH_FIELDS_CORRECTION}", LlmDecision
            )
            usage += call_usage
            proposal = retried
        except LlmInvalidOutput as exc:
            usage += exc.usage  # keep the first, valid proposal
    # c-e) Policy in code, the final decision, and no invented greeting names.
    build = partial(
        _build, ticket=ticket, rule_names=rule_names, faq=faq, settings=settings,
        code_language=code_language,
    )
    decision, notes, name_removed = build(proposal)

    # f) No promises to the customer ("your request is on the roadmap"). Code decides this,
    # not the checker: one redraft with a correction, then a handoff.
    promise_phrase: str | None = None
    found = promise_phrases(decision.draft_reply or "") if decision.action is Action.reply else []
    if found:
        promise_phrase = found[0]
        redraft: LlmDecision | None = None
        try:
            redraft, call_usage = llm.complete_json(
                system, f"{user}\n\n{PROMISE_CORRECTION}", LlmDecision
            )
            usage += call_usage
        except LlmInvalidOutput as exc:
            usage += exc.usage
        if redraft is not None:
            new_decision, new_notes, new_name_removed = build(redraft)
            still = (
                promise_phrases(new_decision.draft_reply or "")
                if new_decision.action is Action.reply
                else []
            )
            if not still:
                # Use the redraft as it is: a clean reply, or a handoff the model now prefers.
                proposal, decision, name_removed = redraft, new_decision, new_name_removed
                notes = [f"draft promised the customer something ('{promise_phrase}'), redrafted"]
                notes += new_notes
                found = []
            else:
                proposal, decision, notes = redraft, new_decision, new_notes
                promise_phrase = still[0]
        if found:
            notes.append(f"draft promised the customer something ('{promise_phrase}'), handed off")
            not_sent = (
                "A draft reply was written but NOT sent: it told the customer something was done "
                f"or decided for them ('{promise_phrase}')."
            )
            decision = decision.model_copy(
                update={
                    "action": Action.handoff,
                    "draft_reply": None,
                    "handoff_note": redact_payment_details(
                        f"{not_sent} {_honest_note(proposal, decision.priority, notes)}"
                    ),
                    "reason": redact_payment_details(
                        f"{proposal.reason} Policy: reply promised the customer something, handed off."
                    ),
                }
            )
    reason_detected = detect_language(proposal.reason)
    reason_language = reason_detected[0].value if reason_detected else None

    # g) Only replies reach a customer, so only replies are checked.
    verified: VerifyOutcome | None = None
    if decision.action is Action.reply:
        cited = [entry for entry in faq.entries if entry.id in proposal.faq_ids]
        verified = verify_reply(
            ticket, decision, cited, llm=llm, settings=settings, expected_language=code_language
        )
        usage += verified.usage
        if not verified.passed:
            notes.append("reply failed verification, handed off")
            # Say plainly that the draft was not sent, so its content is not mistaken for an answer.
            failed = (
                "A draft reply was written but NOT sent: it failed verification ("
                + "; ".join(verified.failures)
                + ")."
            )
            decision = decision.model_copy(
                update={
                    "action": Action.handoff,
                    "draft_reply": None,
                    "handoff_note": redact_payment_details(
                        f"{failed} {_honest_note(proposal, decision.priority, notes)}"
                    ),
                    "reason": redact_payment_details(
                        f"{proposal.reason} Policy: reply failed verification, handed off."
                    ),
                }
            )

    if code_language and proposal.language is not code_language:
        notes.append(
            f"language set to {code_language.value} by code (model said {proposal.language.value})"
        )
    return DecisionResult(
        decision=decision,
        faq_ids=proposal.faq_ids,
        rule_hits=rule_names,
        policy_notes=notes,
        language_source="code" if code_language else "model",
        reason_language=reason_language,
        name_removed=name_removed,
        promise_phrase=promise_phrase,
        verifier=verified,
        tokens_in=usage.tokens_in,
        tokens_out=usage.tokens_out,
        cost_usd=usage.cost_usd,
    )


def _build(
    proposal: LlmDecision,
    *,
    ticket: Ticket,
    rule_names: list[str],
    faq: Faq,
    settings: Settings,
    code_language: Language | None,
) -> tuple[Decision, list[str], bool]:
    """Policy in code, then the Decision, then the greeting-name check.

    Returns the decision, the policy notes and whether an invented name was removed.
    Used for the first proposal and, if needed, for one redraft.
    """
    action, priority, notes = _apply_policy(proposal, rule_names, faq, settings)
    injected = Rule.instructions_in_ticket.value in set(rule_names) | set(
        proposal.needs_human_reasons
    )

    # Payment details never leave in any text field.
    if injected:
        # The model read the injected text; none of its prose about this ticket is trusted.
        reason = "Ticket contains instructions aimed at the support agent; treated as untrusted."
    else:
        reason = proposal.reason
    if notes:
        reason = f"{reason} Policy: {'; '.join(notes)}."
    handoff_note: str | None = None
    if action is Action.handoff:
        if injected:
            handoff_note = INSTRUCTIONS_NOTE.format(summary=_question_summary(proposal))
        else:
            # Prefix with the overrides so far; _honest_note's own note is for the log only.
            prefix = f"[{'; '.join(notes)}] " if notes else ""
            handoff_note = prefix + _honest_note(proposal, priority, notes)
    decision = Decision(
        language=code_language or proposal.language,
        category=proposal.category,
        priority=priority,
        reason=redact_payment_details(reason),
        action=action,
        draft_reply=redact_payment_details(proposal.draft_reply) if action is Action.reply else None,
        handoff_note=redact_payment_details(handoff_note) if handoff_note else None,
        confidence=proposal.confidence,
    )

    # Never greet a customer by a name they did not give us.
    name_removed = False
    if decision.action is Action.reply and decision.draft_reply:
        fixed, removed = remove_invented_name(
            decision.draft_reply, f"{ticket.subject}\n{ticket.body}", decision.language.value
        )
        if removed:
            name_removed = True
            notes.append("greeting name not found in ticket, removed")
            decision = decision.model_copy(update={"draft_reply": fixed})
    return decision, notes, name_removed


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
        if Rule.instructions_in_ticket.value in forced:
            # The ticket may have asked for its own priority; code sets it instead.
            priority = Priority.P2
        elif priority in (Priority.P3, Priority.P4):
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


def _english_fields(proposal: LlmDecision) -> bool:
    """reason and handoff_note are English, or too short to tell."""
    for text in (proposal.reason, proposal.handoff_note):
        detected = detect_language(text)
        if detected and detected[0] is not Language.en:
            return False
    return True


def _honest_note(proposal: LlmDecision, priority: Priority, notes: list[str]) -> str:
    """The model's note, unless it claims we did or promised something. Then a safe template."""
    found = commitment_phrases(proposal.handoff_note)
    if not found:
        return proposal.handoff_note
    phrases = sorted({phrase.lower() for phrase in found})
    notes.append(f"handoff note claimed an action ({', '.join(phrases)}), replaced")
    reason = proposal.reason if not commitment_phrases(proposal.reason) else "see the ticket text."
    return SAFE_NOTE.format(category=proposal.category.value, priority=priority.value, reason=reason)


def _question_summary(proposal: LlmDecision) -> str:
    """One clean English line, or a pointer to the ticket when the summary is not trustworthy."""
    summary = " ".join(proposal.question_summary.split())[:200]
    detected = detect_language(summary)
    echoes_instructions = any(
        hit.rule is Rule.instructions_in_ticket for hit in forced_handoff_hits("", summary)
    )
    if (
        not summary
        or commitment_phrases(summary)
        or echoes_instructions
        or (detected and detected[0] is not Language.en)
    ):
        return "see the ticket text."
    return summary


def _fallback_handoff(
    ticket: Ticket, rule_names: list[str], usage: Usage, code_language: Language | None
) -> DecisionResult:
    """The model gave unusable output twice. A human reads it; we guess only what we must."""
    note = "model output invalid twice, handed off"
    if rule_names:
        note += f"; forced handoff ({', '.join(rule_names)})"
    decision = Decision(
        language=code_language or Language.nl,
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
        language_source="code" if code_language else "default",
        reason_language="en",
        tokens_in=usage.tokens_in,
        tokens_out=usage.tokens_out,
        cost_usd=usage.cost_usd,
    )
