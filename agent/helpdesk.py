"""HTTP client for the Roosterly helpdesk.

Every call is paced to stay under 5 requests/second. 503 and 429 are retried.
409 and 422 are not. A GET that times out or loses its connection is retried like
a 503. A POST that does is re-checked with GET and sent again only when the
ticket is still open, so a customer is never answered twice.
"""

import random
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import quote

import httpx

from agent.config import Settings
from agent.models import Action, Decision, Example, HelpdeskStatus, Ticket, TicketPage

# 0.25s apart is 4 requests/second, under the server limit of 5.
_MIN_INTERVAL_SECONDS = 0.25
_MAX_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 0.25
_BACKOFF_CAP_SECONDS = 8.0


class HelpdeskError(Exception):
    """The helpdesk rejected a call or a call failed before we could trust the outcome."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error: str | None = None,
        problems: list[Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.problems = list(problems or [])
        self.body = body or {}


class BadRequest(HelpdeskError):
    """400: the request itself is wrong. Do not retry."""


class Unauthorized(HelpdeskError):
    """401: missing or wrong key. Do not retry."""


class NotFound(HelpdeskError):
    """404: unknown endpoint or ticket. Do not retry."""


class AlreadyHandled(HelpdeskError):
    """409: this ticket already has a decision. Do not retry."""


class AlreadyHandledAfterUncertainPost(AlreadyHandled):
    """Our POST lost its response, and the re-check shows the ticket handled.

    Most likely our own POST landed, so this is not the same as a real 409.
    """


class ValidationFailed(HelpdeskError):
    """422: the body is invalid. `problems` lists what the API rejected. Do not retry."""


class RateLimited(HelpdeskError):
    """429: still limited after Retry-After waits."""


class ServiceUnavailable(HelpdeskError):
    """503: still unavailable after the backoff attempts. Nothing was saved."""


class OutcomeUnknown(HelpdeskError):
    """A POST may or may not have landed, and the ticket is still open. Do not post again here."""


_ERRORS: dict[int, type[HelpdeskError]] = {
    400: BadRequest,
    401: Unauthorized,
    404: NotFound,
    409: AlreadyHandled,
    422: ValidationFailed,
    429: RateLimited,
    503: ServiceUnavailable,
}


class HelpdeskClient:
    """Synchronous helpdesk client. Use it as a context manager so the pool closes."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._settings = settings
        self._sleep = sleep
        self._monotonic = monotonic
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._not_before = 0.0
        headers = {
            "Authorization": f"Bearer {settings.helpdesk_key.get_secret_value()}",
            "X-Candidate-Email": settings.candidate_email,
        }
        if settings.candidate_name:
            headers["X-Candidate-Name"] = settings.candidate_name
        self._http = httpx.Client(
            base_url=settings.helpdesk_url,
            headers=headers,
            transport=transport,
            timeout=timeout,
        )

    def __enter__(self) -> "HelpdeskClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self._http.close()

    def get_status(self) -> HelpdeskStatus:
        return HelpdeskStatus.model_validate(self._request("GET", "/status"))

    def get_examples(self) -> list[Example]:
        payload = self._request("GET", "/examples")
        examples = payload.get("examples")
        if not isinstance(examples, list):
            raise HelpdeskError("GET /examples did not return an examples list", body=payload)
        return [Example.model_validate(item) for item in examples]

    def iter_open_tickets(self) -> Iterator[Ticket]:
        """Yield every open ticket, following `next_cursor` until it is null."""
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            params: dict[str, Any] = {"limit": 10, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            page = TicketPage.model_validate(self._request("GET", "/tickets", params=params))
            yield from page.tickets
            nxt = page.next_cursor
            if not nxt or nxt in seen:
                return
            seen.add(nxt)
            cursor = nxt

    def get_ticket(self, ticket_id: str) -> Ticket:
        return Ticket.model_validate(self._request("GET", _ticket_path(ticket_id)))

    def reply(self, ticket_id: str, decision: Decision) -> dict[str, Any]:
        return self._post(ticket_id, Action.reply, decision)

    def handoff(self, ticket_id: str, decision: Decision) -> dict[str, Any]:
        return self._post(ticket_id, Action.handoff, decision)

    def close(self, ticket_id: str, decision: Decision) -> dict[str, Any]:
        return self._post(ticket_id, Action.close, decision)

    def reset(self) -> dict[str, Any]:
        """Re-open every practice ticket. POST /practice/reset."""
        return self._request("POST", "/practice/reset")

    def _post(self, ticket_id: str, action: Action, decision: Decision) -> dict[str, Any]:
        if decision.action is not action:
            raise ValueError(
                f"decision.action is {decision.action.value}, this call posts {action.value}"
            )
        return self._request(
            "POST",
            _ticket_path(ticket_id, action.value),
            json_body=decision.api_body(),
            ticket_id=ticket_id,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        ticket_id: str | None = None,
        reposted: bool = False,
    ) -> dict[str, Any]:
        for attempt in range(_MAX_ATTEMPTS):
            self._pace()
            try:
                response = self._send(method, path, json_body=json_body, params=params)
            except httpx.TransportError as exc:
                if method.upper() == "GET":
                    # A GET changes nothing, so a timeout or dropped connection is safe to retry.
                    if attempt + 1 < _MAX_ATTEMPTS:
                        self._sleep(self._backoff(attempt))
                        continue
                    raise HelpdeskError(
                        f"{method} {path} failed before a response after {_MAX_ATTEMPTS} attempts"
                    ) from exc
                if ticket_id is None:
                    raise HelpdeskError(f"{method} {path} failed before a response") from exc
                return self._after_uncertain_post(
                    path, json_body, ticket_id, exc, reposted=reposted
                )
            # 503 saved nothing. 429 asks us to wait. Anything else, including 409, stops here.
            if response.status_code == 503 and attempt + 1 < _MAX_ATTEMPTS:
                self._sleep(self._backoff(attempt))
                continue
            if response.status_code == 429 and attempt + 1 < _MAX_ATTEMPTS:
                self._sleep(self._retry_after(response))
                continue
            return self._ok(response)
        raise HelpdeskError(f"{method} {path} failed after {_MAX_ATTEMPTS} attempts")

    def _after_uncertain_post(
        self,
        path: str,
        json_body: dict[str, Any] | None,
        ticket_id: str,
        exc: BaseException,
        *,
        reposted: bool,
    ) -> dict[str, Any]:
        """A lost POST response is not proof the write failed. Re-read, then maybe post once."""
        try:
            ticket = self.get_ticket(ticket_id)
        except HelpdeskError as check_error:
            raise check_error from exc
        if not ticket.is_open:
            raise AlreadyHandledAfterUncertainPost(
                f"Ticket {ticket_id} is no longer open after an uncertain POST",
                status_code=409,
                error="already_handled",
            ) from exc
        if reposted:
            raise OutcomeUnknown(
                f"POST for ticket {ticket_id} outcome unknown and the ticket is still open"
            ) from exc
        return self._request(
            "POST",
            path,
            json_body=json_body,
            ticket_id=ticket_id,
            reposted=True,
        )

    def _send(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None,
        params: dict[str, Any] | None,
    ) -> httpx.Response:
        kwargs: dict[str, Any] = {}
        if json_body is not None:
            kwargs["json"] = json_body
        if params is not None:
            kwargs["params"] = params
        return self._http.request(method, path, **kwargs)

    def _pace(self) -> None:
        """Reserve the next send slot, then sleep outside the lock if we are early."""
        with self._lock:
            now = self._monotonic()
            wait = self._not_before - now
            slot = now if wait <= 0 else self._not_before
            self._not_before = slot + _MIN_INTERVAL_SECONDS
        if wait > 0:
            self._sleep(wait)

    def _backoff(self, attempt: int) -> float:
        """Full jitter: uniform between 0 and a capped exponential delay."""
        cap = min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * (2**attempt))
        return self._rng.uniform(0, cap)

    def _retry_after(self, response: httpx.Response) -> float:
        raw = response.headers.get("Retry-After", "").strip()
        if not raw:
            return 1.0
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 1.0

    def _ok(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise self._error(response)
        if not response.content:
            return {}
        try:
            data = response.json()
        except ValueError as exc:
            raise HelpdeskError(
                "Helpdesk returned invalid JSON",
                status_code=response.status_code,
            ) from exc
        if not isinstance(data, dict):
            raise HelpdeskError(
                "Helpdesk returned a body that is not a JSON object",
                status_code=response.status_code,
            )
        return data

    def _error(self, response: httpx.Response) -> HelpdeskError:
        body = _read_body(response)
        error = body.get("error")
        error_code = error if isinstance(error, str) else None
        message = body.get("message")
        text = message if isinstance(message, str) and message else (
            error_code or f"HTTP {response.status_code}"
        )
        raw_problems = body.get("problems") or []
        problems = raw_problems if isinstance(raw_problems, list) else [raw_problems]
        cls = _ERRORS.get(response.status_code, HelpdeskError)
        return cls(
            text,
            status_code=response.status_code,
            error=error_code,
            problems=problems,
            body=body,
        )


def _ticket_path(ticket_id: str, action: str | None = None) -> str:
    safe_id = quote(ticket_id, safe="")
    if action is None:
        return f"/tickets/{safe_id}"
    return f"/tickets/{safe_id}/{action}"


def _read_body(response: httpx.Response) -> dict[str, Any]:
    if not response.content:
        return {}
    try:
        data = response.json()
    except ValueError:
        return {}
    if isinstance(data, dict):
        return data
    return {}
