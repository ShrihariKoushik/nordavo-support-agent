"""Eval harness: labels never reach the model, and the scoring is right. No network."""

import json
from pathlib import Path
from typing import Any

import pytest

from agent.config import Settings
from agent.decide import DecisionResult, decide
from agent.faq import Faq, FaqEntry
from agent.llm import Usage
from agent.models import Decision, Example, Ticket
from agent.prompts import LlmDecision
from eval.run_eval import (
    evaluate,
    expected_faq_ids,
    render_markdown,
    strip_label,
    summarize,
    write_results,
)

SECRET = "LABEL-ONLY-MARKER-7f3a"

SETTINGS = Settings.model_validate(
    {
        "helpdesk_url": "https://helpdesk.test",
        "helpdesk_key": "k",
        "candidate_email": "dev@example.com",
        "llm_base_url": "https://llm.test/v1",
        "llm_api_key": "k",
        "llm_model": "m",
    }
)
FAQ = Faq(
    entries=[FaqEntry(id="F02", title="Adding an employee", text="Team → Medewerker toevoegen.")],
    client_rules=["Only answer with information from the help-centre texts above."],
)


def example(example_id: str, action: str, reason: str = "Answered by FAQ F02.", **label: Any) -> Example:
    return Example.model_validate(
        {
            "id": example_id,
            "subject": "Nieuwe medewerker",
            "body": "Hoe voeg ik iemand toe?",
            "channel": "email",
            "customer_plan": "Basis",
            "label": {
                "language": label.get("language", "nl"),
                "category": label.get("category", "question"),
                "priority": label.get("priority", "P4"),
                "action": action,
                "reason": reason,
            },
        }
    )


class RecordingLlm:
    """Records every prompt; answers like a well-behaved model."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_json(
        self, system: str, user: str, schema: type[Any], *, model: str | None = None
    ) -> tuple[Any, Usage]:
        self.prompts += [system, user]
        if schema is LlmDecision:
            payload: dict[str, Any] = {
                "language": "nl", "category": "question", "priority": "P4",
                "reason": "How-to.", "proposed_action": "reply", "faq_ids": ["F02"],
                "needs_human_reasons": [], "is_spam": False, "confidence": 0.9,
                "draft_reply": "Hoi! Ga naar Team → Medewerker toevoegen. Team Roosterly",
                "handoff_note": "How-to question.",
            }
        else:
            payload = {
                "back_translation": "Hi! Go to Team → Medewerker toevoegen.",
                "claims": [{"claim": "Add via Team menu", "supported": True, "faq_id": "F02"}],
                "language_matches_ticket": True, "answers_the_question": True,
                "language_issues": [], "verdict": "pass",
            }
        return schema.model_validate(payload), Usage(tokens_in=100, tokens_out=10, cost_usd=0.001)


def test_strip_label_returns_a_plain_ticket() -> None:
    ticket = strip_label(example("E01", "reply", reason=SECRET))

    assert type(ticket) is Ticket
    assert not hasattr(ticket, "label")
    assert SECRET not in ticket.model_dump_json()


def test_labels_never_reach_the_prompt() -> None:
    labelled = example("E01", "handoff", reason=f"{SECRET} refund rule", category="billing", priority="P2")
    llm = RecordingLlm()

    evaluate([labelled], lambda t: decide(t, faq=FAQ, llm=llm, settings=SETTINGS), repeats=2)

    assert len(llm.prompts) == 8  # 2 runs × (decide + verify) × (system + user)
    joined = "\n".join(llm.prompts)
    assert SECRET not in joined
    assert "label" not in joined.lower()
    assert "Hoe voeg ik iemand toe?" in joined  # the ticket itself does get through


def _result(action: str, faq_ids: list[str] | None = None, **fields: Any) -> DecisionResult:
    decision = Decision(
        language=fields.get("language", "nl"),
        category=fields.get("category", "question"),
        priority=fields.get("priority", "P4"),
        reason="r",
        action=action,
        draft_reply="Hoi" if action == "reply" else None,
        handoff_note="note" if action == "handoff" else None,
        confidence=0.9,
    )
    return DecisionResult(decision=decision, faq_ids=faq_ids or [], tokens_in=100, tokens_out=10, cost_usd=0.002)


def test_scoring_safety_flips_and_cost(tmp_path: Path) -> None:
    examples = [
        example("E01", "reply"),  # we always reply correctly
        example("E02", "handoff", reason="Refund.", category="billing", priority="P2"),  # we reply: unsafe
        example("E03", "reply"),  # we flip between reply and handoff
    ]
    scripted = {
        "E01": [_result("reply", ["F02"])] * 2,
        "E02": [_result("reply", ["F02"], category="billing", priority="P2")] * 2,
        "E03": [_result("reply", ["F02"]), _result("handoff")],
    }
    calls: dict[str, int] = {}

    def fake_decide(ticket: Ticket) -> DecisionResult:
        index = calls.get(ticket.id, 0)
        calls[ticket.id] = index + 1
        return scripted[ticket.id][index]

    reports = evaluate(examples, fake_decide, repeats=2)
    summary = summarize(reports)

    assert summary["unsafe_replies"] == 2
    assert summary["unsafe_runs"] == ["E02#1", "E02#2"]
    assert summary["needless_handoffs"] == 1
    assert summary["flips"] == ["E03"]
    assert summary["accuracy"]["action"]["correct"] == 3
    assert summary["accuracy"]["faq"] == {"correct": 3, "scored": 4, "rate": 0.75}
    assert summary["cost_per_100_tickets_usd"] == pytest.approx(0.2)

    meta = {"date": "d", "model": "m", "verify_model": "m", "provider": "", "seed": 42,
            "repeats": 2, "price_in": 0.36, "price_out": 0.4}
    md = render_markdown("t", meta, reports, summary)
    assert "**Unsafe replies: 2**" in md
    assert "| E03 | reply → reply×1, handoff×1 (1/2) |" in md

    json_path, md_path = write_results("t", meta, reports, summary, results_dir=tmp_path)
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["summary"]["unsafe_replies"] == 2
    assert md_path.read_text(encoding="utf-8") == md


def test_errors_are_recorded_not_raised() -> None:
    def broken(ticket: Ticket) -> DecisionResult:
        raise RuntimeError("provider down")

    summary = summarize(evaluate([example("E01", "reply")], broken, repeats=1))

    assert summary["errors"] == 1
    assert summary["unsafe_replies"] == 0


def test_expected_faq_ids_come_from_the_label_reason() -> None:
    assert expected_faq_ids(example("E01", "reply", reason="FAQ F06 covers it, see also F02.")) == ["F02", "F06"]
    assert expected_faq_ids(example("E02", "handoff", reason="Refund request.")) == []


def test_compare_reports_before_and_after(tmp_path: Path) -> None:
    from eval.compare import load, render

    meta = {"date": "d", "model": "m", "verify_model": "m", "provider": "p", "seed": 42,
            "repeats": 1, "price_in": 0.36, "price_out": 0.4}
    examples = [example("E01", "reply", language="en")]
    before_reports = evaluate(examples, lambda t: _result("handoff", language="nl"), repeats=1)
    after_reports = evaluate(examples, lambda t: _result("reply", ["F02"], language="en"), repeats=1)
    write_results("before", meta, before_reports, summarize(before_reports), results_dir=tmp_path)
    write_results("after", meta, after_reports, summarize(after_reports), results_dir=tmp_path)

    md = render(load("before", tmp_path), load("after", tmp_path))

    assert "# Eval comparison: before → after" in md
    assert "| **Unsafe replies** (must be 0) | 0 | 0 |" in md
    assert "| Needless handoffs | 1/1 | 0/1 |" in md
    assert "| language | 0% (0/1) | 100% (1/1) | +100 pts |" in md
    assert "| E01 | 0/1 | 1/1 | 0/1 | 1/1 |" in md
