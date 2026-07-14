import pytest
from pydantic import ValidationError

from app.generation.models import (
    AnswerCitation,
    GenerationOutput,
    GroundedAnswerRequest,
    GroundedAnswerResult,
)

CHUNK_ID = "a" * 64
DOCUMENT_ID = "d" * 64


def create_citation() -> AnswerCitation:
    return AnswerCitation(
        source_number=1,
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        file_name="policy.txt",
        heading="Policy",
        page_number=2,
    )


def test_generation_output_valid() -> None:
    output = GenerationOutput(
        text="Grounded answer.",
        model_name="test-model",
        input_characters=120,
        output_characters=len("Grounded answer."),
        finish_reason="stop",
    )

    assert output.text == "Grounded answer."
    assert output.model_name == "test-model"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"text": "   "}, "value cannot be blank"),
        ({"model_name": "   "}, "value cannot be blank"),
        ({"finish_reason": "   "}, "finish_reason cannot be blank"),
        ({"output_characters": 999}, "output_characters must equal len"),
    ],
)
def test_generation_output_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values = {
        "text": "Grounded answer.",
        "model_name": "test-model",
        "input_characters": 120,
        "output_characters": len("Grounded answer."),
        "finish_reason": "stop",
    }
    values.update(kwargs)

    with pytest.raises(ValidationError, match=message):
        GenerationOutput(**values)


def test_grounded_answer_request_validates_candidate_limit() -> None:
    with pytest.raises(ValidationError, match="candidate_limit"):
        GroundedAnswerRequest(
            question="What is the policy?",
            limit=5,
            candidate_limit=4,
        )


def test_grounded_answer_request_validates_document_id() -> None:
    with pytest.raises(ValidationError, match="at least 64"):
        GroundedAnswerRequest(
            question="What is the policy?",
            document_id="short",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_number": 0}, "greater than 0"),
        ({"chunk_id": "short"}, "at least 64"),
        ({"document_id": "short"}, "at least 64"),
        ({"file_name": "   "}, "file_name cannot be blank"),
        ({"page_number": 0}, "greater than 0"),
    ],
)
def test_answer_citation_rejects_invalid_metadata(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values = {
        "source_number": 1,
        "chunk_id": CHUNK_ID,
        "document_id": DOCUMENT_ID,
        "file_name": "policy.txt",
        "heading": "Policy",
        "page_number": 2,
    }
    values.update(kwargs)

    with pytest.raises(ValidationError, match=message):
        AnswerCitation(**values)


def test_grounded_answer_result_valid() -> None:
    result = GroundedAnswerResult(
        question="What is the policy?",
        answer="Use the policy.",
        model_name="test-model",
        citations=[create_citation()],
        retrieved_result_count=2,
        context_source_count=1,
        context_truncated=False,
        insufficient_context=False,
        input_characters=100,
        output_characters=len("Use the policy."),
        finish_reason="stop",
    )

    assert result.context_source_count == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"question": "   "}, "value cannot be blank"),
        ({"answer": "   "}, "value cannot be blank"),
        ({"model_name": "   "}, "value cannot be blank"),
        ({"context_source_count": 2}, "context_source_count must equal"),
        ({"output_characters": 999}, "output_characters must equal len"),
        (
            {
                "context_source_count": 0,
                "citations": [],
                "insufficient_context": False,
            },
            "insufficient_context",
        ),
    ],
)
def test_grounded_answer_result_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values = {
        "question": "What is the policy?",
        "answer": "Use the policy.",
        "model_name": "test-model",
        "citations": [create_citation()],
        "retrieved_result_count": 2,
        "context_source_count": 1,
        "context_truncated": False,
        "insufficient_context": False,
        "input_characters": 100,
        "output_characters": len("Use the policy."),
        "finish_reason": "stop",
    }
    values.update(kwargs)

    with pytest.raises(ValidationError, match=message):
        GroundedAnswerResult(**values)
