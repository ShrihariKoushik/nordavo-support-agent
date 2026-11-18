"""API and decision models. Wire values match `docs/api_reference.md`."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Language(StrEnum):
    nl = "nl"
    en = "en"
    de = "de"


class Category(StrEnum):
    bug = "bug"
    billing = "billing"
    feature_request = "feature_request"
    question = "question"
    other = "other"


class Priority(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Action(StrEnum):
    reply = "reply"
    handoff = "handoff"
    close = "close"


class Ticket(BaseModel):
    """One helpdesk ticket. `handled` is null until a decision is stored."""

    model_config = ConfigDict(extra="ignore")

    id: str
    subject: str
    body: str
    channel: str | None = None
    customer_plan: str | None = None
    received_at: datetime | None = None
    handled: Any = None

    @property
    def is_open(self) -> bool:
        """True when GET /tickets/{id} still has `handled: null`."""
        return self.handled is None


class ExampleLabel(BaseModel):
    """Nordavo's label on a GET /examples ticket. The model under test does not write this."""

    model_config = ConfigDict(extra="ignore")

    language: Language
    category: Category
    priority: Priority
    action: Action
    reason: str


class Example(Ticket):
    label: ExampleLabel


class TicketPage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tickets: list[Ticket] = Field(default_factory=list)
    next_cursor: str | None = None


class HelpdeskStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str
    open_tickets: int
    handled_tickets: int
    candidate: str | None = None
    live_ends_at: datetime | None = None
    endpoints: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    """One decision for one ticket. The action chooses which text field is required."""

    model_config = ConfigDict(extra="ignore")

    language: Language
    category: Category
    priority: Priority
    reason: str
    action: Action
    draft_reply: str | None = None
    handoff_note: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)

    @field_validator("reason")
    @classmethod
    def _reason_not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("reason is required")
        return text

    @field_validator("draft_reply", "handoff_note")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None

    @model_validator(mode="after")
    def _action_needs_its_text(self) -> "Decision":
        if self.action is Action.reply and not self.draft_reply:
            raise ValueError("reply requires draft_reply")
        if self.action is Action.handoff and not self.handoff_note:
            raise ValueError("handoff requires handoff_note")
        return self

    def api_body(self) -> dict[str, object]:
        """JSON body for POST /tickets/{id}/reply|handoff|close. The path carries the action."""
        body: dict[str, object] = {
            "language": self.language.value,
            "category": self.category.value,
            "priority": self.priority.value,
            "reason": self.reason,
        }
        if self.action is Action.reply:
            body["draft_reply"] = self.draft_reply
        elif self.action is Action.handoff:
            body["handoff_note"] = self.handoff_note
        if self.confidence is not None:
            body["confidence"] = self.confidence
        return body
