"""Compare two eval runs: python -m eval.compare <before-tag> <after-tag>

Writes eval/results/compare.md: field accuracy, safety counts and cost, before -> after.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.run_eval import FIELDS, RESULTS_DIR


def load(tag: str, results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    path = results_dir / f"{tag}.json"
    if not path.exists():
        raise FileNotFoundError(f"No results for tag '{tag}' at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def render(before: dict[str, Any], after: dict[str, Any]) -> str:
    b, a = before["summary"], after["summary"]
    lines = [
        f"# Eval comparison: {before['tag']} → {after['tag']}",
        "",
        f"Both: {b['examples']} labelled examples × {before['meta']['repeats']} / "
        f"{after['meta']['repeats']} runs, model `{after['meta']['model']}`, "
        f"provider `{after['meta']['provider'] or 'auto'}`, seed {after['meta']['seed']}.",
        "",
        "## Safety",
        "",
        "| metric | before | after |",
        "|---|---|---|",
        f"| **Unsafe replies** (must be 0) | {b['unsafe_replies']} | {a['unsafe_replies']} |",
        f"| Needless handoffs | {b['needless_handoffs']}/{b['replies_expected']} | "
        f"{a['needless_handoffs']}/{a['replies_expected']} |",
        f"| Examples with action flips | {_list(b['flips'])} | {_list(a['flips'])} |",
        f"| Errors | {b['errors']} | {a['errors']} |",
        "",
        "## Accuracy per field",
        "",
        "| field | before | after | change |",
        "|---|---|---|---|",
    ]
    for field in (*FIELDS, "faq"):
        fb, fa = b["accuracy"][field], a["accuracy"][field]
        lines.append(
            f"| {field} | {_rate(fb)} | {_rate(fa)} | {_delta(fb['rate'], fa['rate'])} |"
        )

    lines += [
        "",
        "## Per example (correct runs: action / language)",
        "",
        "| id | action before | action after | language before | language after |",
        "|---|---|---|---|---|",
    ]
    after_by_id = {item["id"]: item for item in after["examples"]}
    for item in before["examples"]:
        other = after_by_id.get(item["id"])
        if other is None:
            continue
        lines.append(
            f"| {item['id']} | {_correct(item, 'action')} | {_correct(other, 'action')} | "
            f"{_correct(item, 'language')} | {_correct(other, 'language')} |"
        )

    lines += [
        "",
        "## Cost",
        "",
        "| metric | before | after |",
        "|---|---|---|",
        f"| Cost per 100 tickets | ${b['cost_per_100_tickets_usd']:.4f} | ${a['cost_per_100_tickets_usd']:.4f} |",
        f"| Tokens in per ticket | {b['tokens_in'] / b['runs']:.0f} | {a['tokens_in'] / a['runs']:.0f} |",
        f"| Tokens out per ticket | {b['tokens_out'] / b['runs']:.0f} | {a['tokens_out'] / a['runs']:.0f} |",
        f"| Avg latency per ticket | {b['avg_latency_ms']} ms | {a['avg_latency_ms']} ms |",
        f"| Verifier verdicts | {_verdicts(b)} | {_verdicts(a)} |",
        "",
        f"Prices per 1M tokens: ${after['meta']['price_in']} in / ${after['meta']['price_out']} out.",
        "",
    ]
    return "\n".join(lines)


def _rate(entry: dict[str, Any]) -> str:
    if entry["rate"] is None:
        return "n/a"
    return f"{entry['rate']:.0%} ({entry['correct']}/{entry['scored']})"


def _delta(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "–"
    points = round((after - before) * 100)
    return "=" if points == 0 else f"{points:+d} pts"


def _correct(item: dict[str, Any], field: str) -> str:
    runs = item["runs"]
    return f"{sum(1 for run in runs if run['match'][field])}/{len(runs)}"


def _list(items: list[str]) -> str:
    return f"{len(items)} ({', '.join(items)})" if items else "0"


def _verdicts(summary: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(summary["verifier"].items())) or "–"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.compare")
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args(argv)
    try:
        before, after = load(args.before), load(args.after)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    markdown = render(before, after)
    out = RESULTS_DIR / "compare.md"
    out.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
