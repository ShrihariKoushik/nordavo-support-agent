"""Journal: resume decisions and crash tolerance."""

from pathlib import Path

from agent.state import Event, Journal, Step


def test_next_step_per_event(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.jsonl")
    journal.record("T1", Event.decided, action="reply")
    journal.record("T1", Event.posted, action="reply")
    journal.record("T2", Event.decided, action="handoff")
    journal.record("T3", Event.already_handled)
    journal.record("T4", Event.posted_probably)
    journal.record("T5", Event.decided, action="reply")
    journal.record("T5", Event.outcome_unknown, action="reply")

    assert journal.next_step("T1") is Step.skip
    assert journal.next_step("T2") is Step.recheck
    assert journal.next_step("T3") is Step.skip
    assert journal.next_step("T4") is Step.skip
    assert journal.next_step("T5") is Step.recheck
    assert journal.next_step("T6") is Step.fresh


def test_missing_journal_means_fresh(tmp_path: Path) -> None:
    assert Journal(tmp_path / "nope.jsonl").next_step("T1") is Step.fresh


def test_truncated_last_line_is_ignored_and_next_write_is_readable(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = Journal(path)
    journal.record("T1", Event.posted, action="reply")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"ts": "2026-11-18T10:00:00+00:00", "ticket_id": "T2", "ev')  # crash here

    assert journal.next_step("T1") is Step.skip
    assert journal.next_step("T2") is Step.fresh

    journal.record("T3", Event.decided, action="close")
    assert journal.next_step("T3") is Step.recheck
    assert [item["ticket_id"] for item in journal.events()] == ["T1", "T3"]


def test_archive_moves_the_journal_aside(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.jsonl")
    journal.record("T1", Event.posted)

    archived = journal.archive()

    assert archived is not None and archived.exists()
    assert journal.next_step("T1") is Step.fresh
