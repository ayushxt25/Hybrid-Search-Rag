import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_hybrid_weight_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "2.0")
    monkeypatch.setenv("HYBRID_SPARSE_WEIGHT", "0.75")
    monkeypatch.setenv("HYBRID_RRF_K", "40")

    settings = Settings(_env_file=None)

    assert settings.hybrid_dense_weight == 2.0
    assert settings.hybrid_sparse_weight == 0.75
    assert settings.hybrid_rrf_k == 40


def test_observability_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "warning")
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")

    settings = Settings(_env_file=None)

    assert settings.log_level == "WARNING"
    assert settings.observability_enabled is False


def test_context_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_MAX_CHARACTERS", "5000")
    monkeypatch.setenv("CONTEXT_MAX_SOURCES", "4")
    monkeypatch.setenv("CONTEXT_INCLUDE_METADATA_HEADERS", "false")

    settings = Settings(_env_file=None)

    assert settings.context_max_characters == 5000
    assert settings.context_max_sources == 4
    assert settings.context_include_metadata_headers is False


def test_prompt_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("PROMPT_MAX_QUESTION_CHARACTERS", "1500")
    monkeypatch.setenv("PROMPT_REQUIRE_CITATIONS", "false")
    monkeypatch.setenv("PROMPT_ALLOW_GENERAL_KNOWLEDGE", "true")

    settings = Settings(_env_file=None)

    assert settings.prompt_max_question_characters == 1500
    assert settings.prompt_require_citations is False
    assert settings.prompt_allow_general_knowledge is True


def test_generation_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("GENERATION_REQUIRE_ANSWER_CITATIONS", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_GENERATION_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_GENERATION_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OPENAI_GENERATION_MAX_RETRIES", "4")

    settings = Settings(_env_file=None)

    assert settings.generation_require_answer_citations is False
    assert settings.openai_api_key == "test-key"
    assert settings.openai_base_url == "https://example.test/v1"
    assert settings.openai_generation_model == "gpt-test"
    assert settings.openai_generation_timeout_seconds == 12.5
    assert settings.openai_generation_max_retries == 4


def test_grounded_answer_rate_limit_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("GROUNDED_ANSWER_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("GROUNDED_ANSWER_RATE_LIMIT_REQUESTS", "7")
    monkeypatch.setenv("GROUNDED_ANSWER_RATE_LIMIT_WINDOW_SECONDS", "30")

    settings = Settings(_env_file=None)

    assert settings.grounded_answer_rate_limit_enabled is False
    assert settings.grounded_answer_rate_limit_requests == 7
    assert settings.grounded_answer_rate_limit_window_seconds == 30


def test_blank_openai_base_url_is_treated_as_unset(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "")

    settings = Settings(_env_file=None)

    assert settings.openai_base_url is None


def test_zero_openai_generation_max_retries_is_allowed(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_GENERATION_MAX_RETRIES", "0")

    settings = Settings(_env_file=None)

    assert settings.openai_generation_max_retries == 0


@pytest.mark.parametrize(
    ("env_name", "env_value", "message"),
    [
        ("HYBRID_DENSE_WEIGHT", "0", "hybrid weights must be greater than zero"),
        ("HYBRID_DENSE_WEIGHT", "-1", "hybrid weights must be greater than zero"),
        ("HYBRID_DENSE_WEIGHT", "nan", "hybrid weights must be finite"),
        ("HYBRID_DENSE_WEIGHT", "inf", "hybrid weights must be finite"),
        ("HYBRID_SPARSE_WEIGHT", "0", "hybrid weights must be greater than zero"),
        ("HYBRID_SPARSE_WEIGHT", "-1", "hybrid weights must be greater than zero"),
        ("HYBRID_SPARSE_WEIGHT", "nan", "hybrid weights must be finite"),
        ("HYBRID_SPARSE_WEIGHT", "inf", "hybrid weights must be finite"),
        ("HYBRID_RRF_K", "0", "greater than 0"),
        ("CONTEXT_MAX_CHARACTERS", "0", "greater than 0"),
        ("CONTEXT_MAX_CHARACTERS", "-1", "greater than 0"),
        ("CONTEXT_MAX_SOURCES", "0", "greater than 0"),
        ("CONTEXT_MAX_SOURCES", "-1", "greater than 0"),
        ("PROMPT_MAX_QUESTION_CHARACTERS", "0", "greater than 0"),
        ("PROMPT_MAX_QUESTION_CHARACTERS", "-1", "greater than 0"),
        ("OPENAI_GENERATION_TIMEOUT_SECONDS", "0", "greater than 0"),
        ("OPENAI_GENERATION_TIMEOUT_SECONDS", "nan", "openai timeout must be finite"),
        ("OPENAI_GENERATION_MAX_RETRIES", "-1", "greater than or equal to 0"),
        ("GROUNDED_ANSWER_RATE_LIMIT_REQUESTS", "0", "greater than 0"),
        ("GROUNDED_ANSWER_RATE_LIMIT_REQUESTS", "-1", "greater than 0"),
        ("GROUNDED_ANSWER_RATE_LIMIT_WINDOW_SECONDS", "0", "greater than 0"),
        ("GROUNDED_ANSWER_RATE_LIMIT_WINDOW_SECONDS", "-1", "greater than 0"),
        ("LOG_LEVEL", "TRACE", "log_level must be one of"),
    ],
)
def test_hybrid_setting_validation(
    monkeypatch,
    env_name: str,
    env_value: str,
    message: str,
) -> None:
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None)
