"""Parse `docs/faq.md` at runtime into FAQ entries and the client's rules.

Nothing from the FAQ is copied into code: change the file and the agent follows.
"""

import re
from pathlib import Path

from pydantic import BaseModel

from agent.config import PROJECT_ROOT

DEFAULT_FAQ_PATH = PROJECT_ROOT / "docs" / "faq.md"

# "## F03 · Swapping shifts" (also accepts -, – or : as the separator)
_ENTRY_RE = re.compile(r"^##\s+(F\d+)\s*[·\-–:]\s*(.+?)\s*$")
_HEADING_RE = re.compile(r"^#{1,6}\s")


class FaqError(Exception):
    """The FAQ file is missing or has no entries."""


class FaqEntry(BaseModel):
    id: str
    title: str
    text: str


class Faq(BaseModel):
    entries: list[FaqEntry]
    client_rules: list[str]

    def get(self, entry_id: str) -> FaqEntry | None:
        return next((entry for entry in self.entries if entry.id == entry_id), None)

    @property
    def ids(self) -> set[str]:
        return {entry.id for entry in self.entries}


def load_faq(path: Path = DEFAULT_FAQ_PATH) -> Faq:
    try:
        markdown = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FaqError(f"Cannot read FAQ at {path}") from exc
    faq = parse_faq(markdown)
    if not faq.entries:
        raise FaqError(f"No FAQ entries found in {path}")
    return faq


def parse_faq(markdown: str) -> Faq:
    entries: list[FaqEntry] = []
    rules: list[str] = []
    current: tuple[str, str] | None = None  # (id, title) of the entry being read
    lines: list[str] = []
    in_rules = False

    def finish_entry() -> None:
        if current is not None:
            entries.append(FaqEntry(id=current[0], title=current[1], text=_one_line(lines)))

    for line in markdown.splitlines():
        stripped = line.strip()
        match = _ENTRY_RE.match(stripped)
        if match:
            finish_entry()
            current, lines, in_rules = (match.group(1), match.group(2)), [], False
            continue
        if _HEADING_RE.match(stripped) or stripped == "---":
            finish_entry()
            current, lines = None, []
            in_rules = _HEADING_RE.match(stripped) is not None and "rules" in stripped.lower()
            continue
        if current is not None:
            lines.append(stripped)
        elif in_rules:
            if stripped.startswith(("- ", "* ")):
                rules.append(stripped[2:].strip())
            elif stripped and rules:
                rules[-1] = f"{rules[-1]} {stripped}"  # wrapped bullet line
    finish_entry()
    return Faq(entries=entries, client_rules=rules)


def render_faq(entries: list[FaqEntry]) -> str:
    """Compact prompt block, one entry per line: `[F03] Swapping shifts: ...`."""
    return "\n".join(f"[{entry.id}] {entry.title}: {entry.text}" for entry in entries)


def _one_line(lines: list[str]) -> str:
    return " ".join(" ".join(lines).split())
