"""FAQ parsing: the real docs/faq.md plus a small made-up file."""

from pathlib import Path

import pytest

from agent.faq import FaqError, load_faq, parse_faq, render_faq

SAMPLE = """# Help centre

Intro text that is not an entry.

## F01 · First thing
Line one of the answer.
Line two, wrapped.

## F02 - Second thing
Go to Instellingen → Abonnement (Settings → Subscription).

---

## The client's rules for the support team

- Rule one.
- Rule two is long and
  wraps onto a second line.
"""


def test_parses_entries_and_client_rules() -> None:
    faq = parse_faq(SAMPLE)

    assert [entry.id for entry in faq.entries] == ["F01", "F02"]
    assert faq.entries[0].title == "First thing"
    assert faq.entries[0].text == "Line one of the answer. Line two, wrapped."
    assert faq.entries[1].text == "Go to Instellingen → Abonnement (Settings → Subscription)."
    assert faq.client_rules == ["Rule one.", "Rule two is long and wraps onto a second line."]


def test_render_is_one_line_per_entry() -> None:
    block = render_faq(parse_faq(SAMPLE).entries)

    assert block.splitlines() == [
        "[F01] First thing: Line one of the answer. Line two, wrapped.",
        "[F02] Second thing: Go to Instellingen → Abonnement (Settings → Subscription).",
    ]


def test_real_faq_loads_with_rules() -> None:
    faq = load_faq()

    assert len(faq.entries) >= 10
    assert all(entry.id.startswith("F") and entry.title and entry.text for entry in faq.entries)
    assert len(faq.ids) == len(faq.entries)
    assert faq.client_rules
    assert faq.get(faq.entries[0].id) == faq.entries[0]
    assert faq.get("F99") is None
    assert "→" in render_faq(faq.entries)  # menu paths survive exactly


def test_missing_or_empty_faq_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FaqError):
        load_faq(tmp_path / "missing.md")
    empty = tmp_path / "empty.md"
    empty.write_text("# Nothing here\n", encoding="utf-8")
    with pytest.raises(FaqError):
        load_faq(empty)
