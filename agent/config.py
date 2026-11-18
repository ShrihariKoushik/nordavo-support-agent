"""Settings loaded from the environment and `.env`.

This is the only module that reads `os.environ`.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_REQUIRED: dict[str, str] = {
    "helpdesk_url": "HELPDESK_URL",
    "helpdesk_key": "HELPDESK_KEY",
    "candidate_email": "CANDIDATE_EMAIL",
    "llm_base_url": "LLM_BASE_URL",
    "llm_api_key": "LLM_API_KEY",
    "llm_model": "LLM_MODEL",
}

_OPTIONAL_FLOATS: dict[str, str] = {
    "llm_price_per_million_input": "LLM_PRICE_PER_MILLION_INPUT",
    "llm_price_per_million_output": "LLM_PRICE_PER_MILLION_OUTPUT",
    "reply_confidence_min": "REPLY_CONFIDENCE_MIN",
}


class ConfigError(Exception):
    """A required setting is missing or a setting is not valid."""


class Settings(BaseModel):
    """Typed configuration for the helpdesk and the LLM."""

    helpdesk_url: str
    helpdesk_key: SecretStr = Field(repr=False)
    candidate_email: str
    candidate_name: str = ""
    llm_base_url: str
    llm_api_key: SecretStr = Field(repr=False)
    llm_model: str
    llm_price_per_million_input: float = Field(default=0.0, ge=0)
    llm_price_per_million_output: float = Field(default=0.0, ge=0)
    reply_confidence_min: float = Field(default=0.75, ge=0, le=1)


def load_settings(env_file: Path | None = None) -> Settings:
    """Load `.env` then build `Settings`. Existing environment variables win."""
    path = PROJECT_ROOT / ".env" if env_file is None else env_file
    load_dotenv(path, override=False)

    missing = [env_name for env_name in _REQUIRED.values() if not os.environ.get(env_name, "").strip()]
    if missing:
        raise ConfigError("Missing environment variables: " + ", ".join(missing))

    data: dict[str, str] = {
        field: os.environ[env_name].strip() for field, env_name in _REQUIRED.items()
    }
    name = os.environ.get("CANDIDATE_NAME", "").strip()
    if name:
        data["candidate_name"] = name
    for field, env_name in _OPTIONAL_FLOATS.items():
        raw = os.environ.get(env_name, "").strip()
        if raw:
            data[field] = raw

    try:
        settings = Settings.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_validation_message(exc)) from exc
    return settings.model_copy(
        update={
            "helpdesk_url": settings.helpdesk_url.rstrip("/"),
            "llm_base_url": settings.llm_base_url.rstrip("/"),
        }
    )


def _validation_message(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        field = ".".join(str(item) for item in error["loc"])
        parts.append(f"{field}: {error['msg']}")
    return "Invalid settings: " + "; ".join(parts)
