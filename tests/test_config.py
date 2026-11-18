"""Settings come from the environment and `.env`, with the documented defaults."""

import pytest

from agent.config import ConfigError, load_settings

_KEYS = [
    "HELPDESK_URL",
    "HELPDESK_KEY",
    "CANDIDATE_EMAIL",
    "CANDIDATE_NAME",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "VERIFY_MODEL",
    "LLM_PRICE_PER_MILLION_INPUT",
    "LLM_PRICE_PER_MILLION_OUTPUT",
    "REPLY_CONFIDENCE_MIN",
]


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)


def test_load_settings_reads_env_file_and_defaults(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "HELPDESK_URL=https://example.test/helpdesk/",
                "HELPDESK_KEY=nrd_test",
                "CANDIDATE_EMAIL=dev@example.com",
                "CANDIDATE_NAME=Ada",
                "LLM_BASE_URL=https://llm.test/v1/",
                "LLM_API_KEY=sk_test",
                "LLM_MODEL=test-model",
                "LLM_PRICE_PER_MILLION_INPUT=0.4",
                "LLM_PRICE_PER_MILLION_OUTPUT=1.2",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file)

    assert settings.helpdesk_url == "https://example.test/helpdesk"
    assert settings.llm_base_url == "https://llm.test/v1"
    assert settings.candidate_email == "dev@example.com"
    assert settings.candidate_name == "Ada"
    assert settings.llm_model == "test-model"
    assert settings.verify_model == "test-model"  # defaults to LLM_MODEL
    assert settings.helpdesk_key.get_secret_value() == "nrd_test"
    assert settings.llm_price_per_million_input == 0.4
    assert settings.llm_price_per_million_output == 1.2
    assert settings.reply_confidence_min == 0.75
    assert "nrd_test" not in repr(settings)


def test_missing_required_variable_raises(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("HELPDESK_URL=https://example.test/helpdesk\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="HELPDESK_KEY"):
        load_settings(env_file)


def test_bad_price_raises(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "HELPDESK_URL=https://example.test/helpdesk",
                "HELPDESK_KEY=nrd_test",
                "CANDIDATE_EMAIL=dev@example.com",
                "LLM_BASE_URL=https://llm.test/v1",
                "LLM_API_KEY=sk_test",
                "LLM_MODEL=test-model",
                "LLM_PRICE_PER_MILLION_INPUT=abc",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="llm_price_per_million_input"):
        load_settings(env_file)


def test_verify_model_can_differ(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "HELPDESK_URL=https://example.test/helpdesk",
                "HELPDESK_KEY=k",
                "CANDIDATE_EMAIL=dev@example.com",
                "LLM_BASE_URL=https://llm.test/v1",
                "LLM_API_KEY=k",
                "LLM_MODEL=drafter",
                "VERIFY_MODEL=checker",
            ]
        ),
        encoding="utf-8",
    )
    settings = load_settings(env_file)
    assert (settings.llm_model, settings.verify_model) == ("drafter", "checker")
