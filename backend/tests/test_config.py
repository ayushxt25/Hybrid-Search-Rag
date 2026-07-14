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

    settings = Settings(_env_file=None)

    assert settings.generation_require_answer_citations is False


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
