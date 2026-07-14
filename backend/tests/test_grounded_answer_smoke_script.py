from unittest.mock import patch

import pytest

from app.api import dependencies
from app.main import app
from scripts import grounded_answer_smoke_test as smoke


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()

    yield

    app.dependency_overrides.clear()


def create_valid_response(document_id: str = "a" * 64) -> dict:
    return {
        "question": smoke.QUESTION,
        "answer": smoke.STUB_ANSWER,
        "model_name": smoke.STUB_MODEL_NAME,
        "finish_reason": "stop",
        "insufficient_context": False,
        "retrieved_result_count": 1,
        "context_source_count": 1,
        "citations": [
            {
                "source_number": 1,
                "chunk_id": "b" * 64,
                "document_id": document_id,
            }
        ],
        "citation_markers": [1],
        "output_characters": len(smoke.STUB_ANSWER),
        "input_characters": 10,
    }


def test_generation_dependency_is_overridden() -> None:
    stub = smoke.StubGenerationProvider()
    original_factory = dependencies.get_generation_provider

    with smoke.generation_provider_override(stub):
        assert dependencies.get_generation_provider() is stub
        assert app.dependency_overrides[original_factory]() is stub


def test_real_openai_provider_is_never_constructed() -> None:
    stub = smoke.StubGenerationProvider()

    with patch("app.api.dependencies.OpenAIGenerationProvider") as provider_class:
        with smoke.generation_provider_override(stub):
            assert dependencies.get_generation_provider() is stub

    provider_class.assert_not_called()


def test_valid_response_assertions_pass() -> None:
    document_id = "a" * 64

    smoke.validate_grounded_answer_response(
        create_valid_response(document_id),
        document_id=document_id,
    )


def test_invalid_citation_marker_fails() -> None:
    data = create_valid_response()
    data["citation_markers"] = [2]

    with pytest.raises(RuntimeError, match="citation_markers"):
        smoke.validate_grounded_answer_response(data, document_id="a" * 64)


def test_mismatched_document_id_fails() -> None:
    with pytest.raises(RuntimeError, match="document_id"):
        smoke.validate_grounded_answer_response(
            create_valid_response("b" * 64),
            document_id="a" * 64,
        )


def test_stub_character_counts_are_exact() -> None:
    provider = smoke.StubGenerationProvider()

    output = provider.generate(system_prompt="system", user_prompt="user")

    assert output.input_characters == len("system") + len("user")
    assert output.output_characters == len(smoke.STUB_ANSWER)


def test_dependency_override_is_cleared_after_success() -> None:
    stub = smoke.StubGenerationProvider()
    original_factory = dependencies.get_generation_provider

    with smoke.generation_provider_override(stub):
        pass

    assert dependencies.get_generation_provider is original_factory
    assert original_factory not in app.dependency_overrides


def test_dependency_override_is_cleared_after_failure() -> None:
    stub = smoke.StubGenerationProvider()
    original_factory = dependencies.get_generation_provider

    with pytest.raises(RuntimeError, match="boom"):
        with smoke.generation_provider_override(stub):
            raise RuntimeError("boom")

    assert dependencies.get_generation_provider is original_factory
    assert original_factory not in app.dependency_overrides


def test_insufficient_context_validation() -> None:
    smoke.validate_insufficient_context_response(
        {
            "model_name": "not-invoked",
            "finish_reason": "insufficient_context",
            "citation_markers": [],
            "citations": [],
        }
    )
