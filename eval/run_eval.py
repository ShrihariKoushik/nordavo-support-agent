"""Run the real decide() pipeline on Nordavo's 10 labelled examples and score it.

    python -m eval.run_eval [--repeats N] [--tag NAME]

- Labels come only from GET /examples (cached in state/examples.json). They are stripped
  before the pipeline sees a ticket, and only used afterwards for scoring.
- Never POSTs and never touches the journal: decide() is called directly.
- Safety first: an "unsafe reply" is a reply where the label says handoff or close.
- Writes eval/results/<tag>.json (full detail) and eval/results/<tag>.md (summary table).
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent.config import PROJECT_ROOT, ConfigError, Settings, load_settings
from agent.decide import DecisionResult, decide
from agent.faq import load_faq
from agent.helpdesk import HelpdeskClient
from agent.llm import LlmClient
from agent.models import Action, Example, Ticket

EXAMPLES_CACHE = PROJECT_ROOT / "state" / "examples.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
FIELDS = ("action", "category", "priority", "language")
# The only ticket fields the pipeline may see. Everything else (label, handled) is dropped.
TICKET_FIELDS = ("id", "subject", "body", "channel", "customer_plan", "received_at")

_FAQ_ID_RE = re.compile(r"\bF\d{2}\b")

DecideFn = Callable[[Ticket], DecisionResult]


class RunRecord(BaseModel):
    """One pipeline run on one example."""

    run: int
    action: str | None = None
    category: str | None = None
    priority: str | None = None
    language: str | None = None
    confidence: float | None = None
    faq_ids: list[str] = Field(default_factory=list)
    rule_hits: list[str] = Field(default_factory=list)
    policy_notes: list[str] = Field(default_factory=list)
    language_source: str | None = None
    reason_language: str | None = None
    name_removed: bool = False
    promise_phrase: str | None = None
    verifier: str | None = None  # pass | fail | None when not a reply
    language_issues: list[str] = Field(default_factory=list)
    language_override: bool = False
    back_translation: str | None = None
    draft_reply: str | None = None
    handoff_note: str | None = None
    reason: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: str | None = None
    match: dict[str, bool | None] = Field(default_factory=dict)


class ExampleReport(BaseModel):
    id: str
    expected: dict[str, Any]
    runs: list[RunRecord]

    @property
    def flipped(self) -> bool:
        return len({run.action for run in self.runs}) > 1


# --- Loading -----------------------------------------------------------------------------------


def load_examples(settings: Settings, cache: Path = EXAMPLES_CACHE) -> list[Example]:
    """Read the cached examples, or fetch them once with GET /examples and cache them."""
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        items = payload["examples"] if isinstance(payload, dict) else payload
        return [Example.model_validate(item) for item in items]
    with HelpdeskClient(settings) as client:
        examples = client.get_examples()
    cache.parent.mkdir(parents=True, exist_ok=True)
    data = {"examples": [example.model_dump(mode="json") for example in examples]}
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return examples


def strip_label(example: Example) -> Ticket:
    """A plain Ticket with only the customer-visible fields. The label cannot leak through."""
    return Ticket.model_validate({name: getattr(example, name) for name in TICKET_FIELDS})


def expected_faq_ids(example: Example) -> list[str]:
    """Labels have no faq field; the ids are named in the label's reason ("FAQ F06 ...")."""
    return sorted(set(_FAQ_ID_RE.findall(example.label.reason)))


# --- Running and scoring -----------------------------------------------------------------------


def evaluate(examples: list[Example], decide_fn: DecideFn, repeats: int) -> list[ExampleReport]:
    reports: list[ExampleReport] = []
    for example in examples:
        label = example.label
        expected = {
            "action": label.action.value,
            "category": label.category.value,
            "priority": label.priority.value,
            "language": label.language.value,
            "faq_ids": expected_faq_ids(example),
        }
        runs = [_run_once(example, decide_fn, run, expected) for run in range(1, repeats + 1)]
        reports.append(ExampleReport(id=example.id, expected=expected, runs=runs))
        actions = ", ".join(run.action or "error" for run in runs)
        print(f"{example.id}: expected {expected['action']}, got {actions}", file=sys.stderr)
    return reports


def _run_once(example: Example, decide_fn: DecideFn, run: int, expected: dict[str, Any]) -> RunRecord:
    started = time.monotonic()
    try:
        result = decide_fn(strip_label(example))
    except Exception as exc:  # noqa: BLE001 - one broken run is recorded, the eval goes on
        record = RunRecord(
            run=run,
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=round((time.monotonic() - started) * 1000),
        )
    else:
        decision, verifier = result.decision, result.verifier
        record = RunRecord(
            run=run,
            action=decision.action.value,
            category=decision.category.value,
            priority=decision.priority.value,
            language=decision.language.value,
            confidence=decision.confidence,
            faq_ids=result.faq_ids,
            rule_hits=result.rule_hits,
            policy_notes=result.policy_notes,
            language_source=result.language_source,
            reason_language=result.reason_language,
            name_removed=result.name_removed,
            promise_phrase=result.promise_phrase,
            verifier=None if verifier is None else ("pass" if verifier.passed else "fail"),
            language_issues=[] if verifier is None else verifier.language_issues,
            language_override=False if verifier is None else verifier.language_override,
            back_translation=None if verifier is None else verifier.back_translation,
            draft_reply=decision.draft_reply,
            handoff_note=decision.handoff_note,
            reason=decision.reason,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
        )
    record.match = _match(record, expected)
    return record


def _match(record: RunRecord, expected: dict[str, Any]) -> dict[str, bool | None]:
    match: dict[str, bool | None] = {
        field: getattr(record, field) == expected[field] for field in FIELDS
    }
    # FAQ only counts for examples labelled reply that name an entry.
    if expected["action"] == Action.reply.value and expected["faq_ids"]:
        match["faq"] = record.action == Action.reply.value and bool(
            set(record.faq_ids) & set(expected["faq_ids"])
        )
    else:
        match["faq"] = None
    return match


def summarize(reports: list[ExampleReport]) -> dict[str, Any]:
    runs = [(report, run) for report in reports for run in report.runs]
    total = len(runs)
    unsafe = [
        f"{report.id}#{run.run}"
        for report, run in runs
        if run.action == Action.reply.value and report.expected["action"] != Action.reply.value
    ]
    needless = [
        f"{report.id}#{run.run}"
        for report, run in runs
        if run.action == Action.handoff.value and report.expected["action"] == Action.reply.value
    ]
    accuracy: dict[str, dict[str, Any]] = {}
    for field in (*FIELDS, "faq"):
        scored = [run.match[field] for _report, run in runs if run.match.get(field) is not None]
        hits = sum(1 for value in scored if value)
        accuracy[field] = {"correct": hits, "scored": len(scored), "rate": _rate(hits, len(scored))}
    verifier = Counter(run.verifier for _report, run in runs if run.verifier)
    tokens_in = sum(run.tokens_in for _report, run in runs)
    tokens_out = sum(run.tokens_out for _report, run in runs)
    cost = sum(run.cost_usd for _report, run in runs)
    return {
        "runs": total,
        "examples": len(reports),
        "unsafe_replies": len(unsafe),
        "unsafe_runs": unsafe,
        "needless_handoffs": len(needless),
        "needless_handoff_runs": needless,
        "replies_expected": sum(
            len(report.runs) for report in reports if report.expected["action"] == Action.reply.value
        ),
        "flips": [report.id for report in reports if report.flipped],
        "errors": sum(1 for _report, run in runs if run.error),
        "accuracy": accuracy,
        "verifier": dict(verifier),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost,
        "cost_per_100_tickets_usd": cost / total * 100 if total else 0.0,
        "avg_latency_ms": round(sum(run.latency_ms for _r, run in runs) / total) if total else 0,
    }


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 3) if total else None


# --- Output ------------------------------------------------------------------------------------


def render_markdown(tag: str, meta: dict[str, Any], reports: list[ExampleReport], summary: dict[str, Any]) -> str:
    lines = [
        f"# Eval: {tag}",
        "",
        f"{meta['date']} · model `{meta['model']}` · verifier `{meta['verify_model']}` · "
        f"provider `{meta['provider'] or 'auto'}` · seed {meta['seed']} · "
        f"{summary['examples']} examples × {meta['repeats']} runs",
        "",
        "## Safety",
        "",
        f"- **Unsafe replies: {summary['unsafe_replies']}** (replied where the label is handoff or close; must be 0)"
        + (f" → {', '.join(summary['unsafe_runs'])}" if summary["unsafe_runs"] else ""),
        f"- Needless handoffs: {summary['needless_handoffs']} of {summary['replies_expected']} "
        "runs labelled reply"
        + (f" → {', '.join(summary['needless_handoff_runs'])}" if summary["needless_handoff_runs"] else ""),
        f"- Action flips between runs: {len(summary['flips'])}"
        + (f" ({', '.join(summary['flips'])})" if summary["flips"] else ""),
        f"- Errors: {summary['errors']}",
        "",
        "## Per example",
        "",
        "Cells are `expected → got (correct/runs)`. FAQ is scored only for examples labelled reply.",
        "",
        "| id | action | category | priority | language | faq | verifier |",
        "|---|---|---|---|---|---|---|",
    ]
    for report in reports:
        cells = [_cell(report, field) for field in FIELDS]
        cells.append(_faq_cell(report))
        verdicts = Counter(run.verifier or "–" for run in report.runs)
        cells.append(", ".join(f"{name}×{count}" for name, count in verdicts.items()))
        lines.append(f"| {report.id} | " + " | ".join(cells) + " |")

    acc = summary["accuracy"]
    lines += ["", "## Totals", "", "| field | correct | rate |", "|---|---|---|"]
    for field in (*FIELDS, "faq"):
        rate = acc[field]["rate"]
        shown = "n/a" if rate is None else f"{rate:.0%}"
        lines.append(f"| {field} | {acc[field]['correct']}/{acc[field]['scored']} | {shown} |")
    verifier = ", ".join(f"{k}={v}" for k, v in summary["verifier"].items()) or "no replies checked"
    lines += [
        "",
        f"- Verifier verdicts: {verifier}",
        f"- Tokens: {summary['tokens_in']:,} in / {summary['tokens_out']:,} out over {summary['runs']} runs",
        f"- Cost: ${summary['cost_usd']:.4f} total, **${summary['cost_per_100_tickets_usd']:.4f} per 100 tickets** "
        f"(prices per 1M tokens: ${meta['price_in']} in / ${meta['price_out']} out)",
        f"- Average latency per ticket: {summary['avg_latency_ms']} ms",
        "",
    ]
    return "\n".join(lines)


def _cell(report: ExampleReport, field: str) -> str:
    expected = report.expected[field]
    got = Counter(getattr(run, field) or "error" for run in report.runs)
    correct = sum(1 for run in report.runs if run.match[field])
    shown = ", ".join(value if count == len(report.runs) else f"{value}×{count}" for value, count in got.items())
    return f"{expected} → {shown} ({correct}/{len(report.runs)})"


def _faq_cell(report: ExampleReport) -> str:
    expected = report.expected["faq_ids"]
    if report.runs[0].match["faq"] is None:
        return "–"
    got = Counter(",".join(run.faq_ids) or "none" for run in report.runs)
    correct = sum(1 for run in report.runs if run.match["faq"])
    shown = ", ".join(value if count == len(report.runs) else f"{value}×{count}" for value, count in got.items())
    return f"{','.join(expected)} → {shown} ({correct}/{len(report.runs)})"


def write_results(
    tag: str,
    meta: dict[str, Any],
    reports: list[ExampleReport],
    summary: dict[str, Any],
    results_dir: Path = RESULTS_DIR,
) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{tag}.json"
    md_path = results_dir / f"{tag}.md"
    payload = {
        "tag": tag,
        "meta": meta,
        "summary": summary,
        "examples": [report.model_dump(mode="json") for report in reports],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(tag, meta, reports, summary), encoding="utf-8")
    return json_path, md_path


# --- CLI ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval")
    parser.add_argument("--repeats", type=int, default=3, help="runs per example (default 3)")
    parser.add_argument("--tag", default=None, help="results name (default: UTC timestamp)")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    tag = args.tag or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", tag):
        parser.error("--tag may only contain letters, digits, '.', '_' and '-'")

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    examples = load_examples(settings)
    llm = LlmClient(settings)
    decide_fn = partial(decide, faq=load_faq(), llm=llm, settings=settings)
    reports = evaluate(examples, decide_fn, args.repeats)
    summary = summarize(reports)
    meta = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": settings.llm_model,
        "verify_model": settings.verify_model,
        "provider": settings.llm_provider,
        "seed": settings.llm_seed,
        "repeats": args.repeats,
        "reply_confidence_min": settings.reply_confidence_min,
        "price_in": settings.llm_price_per_million_input,
        "price_out": settings.llm_price_per_million_output,
    }
    json_path, md_path = write_results(tag, meta, reports, summary)
    print(md_path.read_text(encoding="utf-8"))
    print(f"Wrote {json_path} and {md_path}", file=sys.stderr)
    return 0 if summary["unsafe_replies"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
