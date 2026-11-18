"""Independent check of a draft reply, for a team that cannot read Dutch.

One separate LLM call (optionally a different model) sees only the ticket, the draft and
the cited FAQ entries. It back-translates the draft to English and checks each claim
against the FAQ text. Pass or fail is decided here in code from those facts, not from the
checker's own verdict.
"""

from pydantic import BaseModel, Field

from agent.config import Settings
from agent.faq import FaqEntry
from agent.llm import JsonLlm, LlmInvalidOutput, Usage
from agent.lang import detect_language
from agent.models import Decision, Language, Ticket
from agent.prompts import BACK_TRANSLATION_CORRECTION, VERIFY_SYSTEM, VerifyResult, verify_user


class VerifyOutcome(BaseModel):
    """What decide() and the log need from one verification."""

    passed: bool
    failures: list[str] = Field(default_factory=list)
    claims: int = 0
    unsupported: int = 0
    language_issues: list[str] = Field(default_factory=list)
    back_translation: str | None = None
    usage: Usage = Field(default_factory=Usage)


def verify_reply(
    ticket: Ticket,
    decision: Decision,
    cited_faq_entries: list[FaqEntry],
    *,
    llm: JsonLlm,
    settings: Settings,
    expected_language: Language | None = None,
) -> VerifyOutcome:
    """`expected_language` is the ticket language detected by code, when it was confident."""
    draft = decision.draft_reply or ""
    user = verify_user(ticket, draft, cited_faq_entries, expected_language)
    usage = Usage()
    try:
        result, call_usage = llm.complete_json(
            VERIFY_SYSTEM, user, VerifyResult, model=settings.verify_model
        )
        usage += call_usage
        # The back-translation is our English audit trail. A copied draft is useless, so ask once more.
        if not _is_english(result.back_translation):
            result, call_usage = llm.complete_json(
                VERIFY_SYSTEM,
                f"{user}\n\n{BACK_TRANSLATION_CORRECTION}",
                VerifyResult,
                model=settings.verify_model,
            )
            usage += call_usage
    except LlmInvalidOutput as exc:
        # Fail safe: a reply nobody could check is not sent.
        return VerifyOutcome(
            passed=False, failures=["verifier output invalid twice"], usage=usage + exc.usage
        )

    cited_ids = {entry.id for entry in cited_faq_entries}
    failures = verification_failures(result, cited_ids)
    if not _is_english(result.back_translation):
        failures.append("back-translation not in English")
    # Code check on top of the checker: a clearly-detected draft language must match the ticket.
    draft_language = detect_language(draft)
    if expected_language and draft_language and draft_language[0] is not expected_language:
        failures.append(
            f"reply is in {draft_language[0].value}, ticket is in {expected_language.value}"
        )
    unsupported = sum(1 for claim in result.claims if not _grounded(claim.supported, claim.faq_id, cited_ids))
    return VerifyOutcome(
        passed=not failures,
        failures=failures,
        claims=len(result.claims),
        unsupported=unsupported,
        language_issues=result.language_issues,
        back_translation=result.back_translation,
        usage=usage,
    )


def verification_failures(result: VerifyResult, cited_ids: set[str]) -> list[str]:
    """Reasons to block the reply. Language issues alone are logged, not blocking."""
    failures: list[str] = []
    bad = [c.claim for c in result.claims if not _grounded(c.supported, c.faq_id, cited_ids)]
    if bad:
        failures.append(f"{len(bad)} claim(s) not supported by the cited FAQ: " + " | ".join(bad))
    if not result.claims:
        failures.append("checker found no claims to ground in the FAQ")
    if not result.language_matches_ticket:
        failures.append("reply language does not match the ticket")
    if not result.answers_the_question:
        failures.append("reply does not answer the customer's question")
    return failures


def _is_english(text: str) -> bool:
    """English, or too short to tell. Only a clear nl/de detection counts as not English."""
    detected = detect_language(text)
    return detected is None or detected[0] is Language.en


def _grounded(supported: bool, faq_id: str | None, cited_ids: set[str]) -> bool:
    """Supported only counts if it points at an entry the reply actually cited."""
    return supported and faq_id in cited_ids
