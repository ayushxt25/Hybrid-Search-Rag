from unittest.mock import Mock

import pytest

from app.context.models import AssembledContext, ContextSource
from app.generation.models import GenerationOutput, GroundedAnswerRequest
from app.generation.service import (
    INSUFFICIENT_CONTEXT_ANSWER,
    GroundedAnswerService,
)
from app.prompting.models import GroundedPromptPackage
from app.schemas.search import DenseSearchResult
from app.schemas.search_request import HybridSearchResponse

DOCUMENT_ID = "d" * 64


class StubGenerationProvider:
    def __init__(self, output: GenerationOutput | None = None) -> None:
        self.output = output
        self.calls: list[dict[str, str]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> GenerationOutput:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

        if self.output is None:
            total_input = len(system_prompt) + len(user_prompt)
            answer = "Employees may work remotely. [Source 1]"
            return GenerationOutput(
                text=answer,
                model_name="stub-model",
                input_characters=total_input,
                output_characters=len(answer),
                finish_reason="stop",
            )

        return self.output


def create_result(chunk_id: str = "a" * 64) -> DenseSearchResult:
    return DenseSearchResult(
        point_id=f"point-{chunk_id}",
        chunk_id=chunk_id,
        document_id=DOCUMENT_ID,
        score=0.9,
        file_name="policy.txt",
        file_extension=".txt",
        chunk_index=0,
        section_index=0,
        page_number=2,
        heading="Policy",
        text="Employees may work remotely.",
        start_word=0,
        end_word=4,
        word_count=4,
    )


def create_source(source_number: int = 1) -> ContextSource:
    return ContextSource(
        source_number=source_number,
        chunk_id=chr(96 + source_number) * 64,
        document_id=DOCUMENT_ID,
        file_name="policy.txt",
        text=f"Policy text {source_number}.",
        score=0.9,
        chunk_index=source_number - 1,
        section_index=0,
        heading=f"Heading {source_number}",
        page_number=source_number,
    )


def create_context(
    *,
    source_count: int = 1,
    truncated: bool = False,
) -> AssembledContext:
    sources = [create_source(index + 1) for index in range(source_count)]
    context_text = "\n\n".join(
        f"[Source {source.source_number}]\n{source.text}" for source in sources
    )
    return AssembledContext(
        context_text=context_text,
        sources=sources,
        source_count=len(sources),
        total_characters=len(context_text),
        truncated=truncated,
        omitted_result_count=1 if truncated else 0,
    )


def create_prompt_package(
    *,
    question: str = "What is the policy?",
    context: AssembledContext | None = None,
) -> GroundedPromptPackage:
    assembled_context = context if context is not None else create_context()
    system_prompt = "system"
    user_prompt = "user"
    return GroundedPromptPackage(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        question=question,
        source_count=assembled_context.source_count,
        context_characters=assembled_context.total_characters,
        context_truncated=assembled_context.truncated,
        insufficient_context=assembled_context.source_count == 0,
        total_prompt_characters=len(system_prompt) + len(user_prompt),
    )


def create_service(
    *,
    hybrid_response: HybridSearchResponse | None = None,
    assembled_context: AssembledContext | None = None,
    prompt_package: GroundedPromptPackage | None = None,
    provider: StubGenerationProvider | None = None,
    require_answer_citations: bool = True,
    timing_callback=None,
    observability_enabled: bool = True,
) -> tuple[GroundedAnswerService, Mock, Mock, Mock, StubGenerationProvider]:
    resolved_context = (
        assembled_context if assembled_context is not None else create_context()
    )
    resolved_prompt = (
        prompt_package
        if prompt_package is not None
        else create_prompt_package(context=resolved_context)
    )
    search_service = Mock()
    search_service.search.return_value = (
        hybrid_response
        if hybrid_response is not None
        else HybridSearchResponse(
            query="What is the policy?",
            result_count=2,
            results=[create_result("a" * 64), create_result("b" * 64)],
        )
    )
    context_assembler = Mock()
    context_assembler.assemble.return_value = resolved_context
    prompt_builder = Mock()
    prompt_builder.build.return_value = resolved_prompt
    generation_provider = provider if provider is not None else StubGenerationProvider()

    return (
        GroundedAnswerService(
            hybrid_search_service=search_service,
            context_assembler=context_assembler,
            prompt_builder=prompt_builder,
            generation_provider=generation_provider,
            require_answer_citations=require_answer_citations,
            timing_callback=timing_callback,
            observability_enabled=observability_enabled,
        ),
        search_service,
        context_assembler,
        prompt_builder,
        generation_provider,
    )


def test_answer_orchestrates_non_empty_context_flow() -> None:
    service, search_service, context_assembler, prompt_builder, provider = (
        create_service()
    )
    request = GroundedAnswerRequest(
        question="  What is the policy?  ",
        limit=3,
        candidate_limit=10,
        document_id=DOCUMENT_ID,
    )

    result = service.answer(request)

    search_request = search_service.search.call_args.args[0]
    assert search_request.query == "What is the policy?"
    assert search_request.limit == 3
    assert search_request.candidate_limit == 10
    assert search_request.document_id == DOCUMENT_ID
    context_assembler.assemble.assert_called_once_with(
        search_service.search.return_value.results
    )
    prompt_builder.build.assert_called_once_with(
        question="What is the policy?",
        context=context_assembler.assemble.return_value,
    )
    assert provider.calls == [{"system_prompt": "system", "user_prompt": "user"}]
    assert result.question == "What is the policy?"
    assert result.answer == "Employees may work remotely. [Source 1]"
    assert result.model_name == "stub-model"
    assert result.retrieved_result_count == 2
    assert result.context_source_count == 1
    assert result.context_truncated is False
    assert result.insufficient_context is False
    assert result.input_characters == len("system") + len("user")
    assert result.output_characters == len(result.answer)
    assert result.finish_reason == "stop"
    assert result.citation_markers == [1]


def test_citations_are_copied_in_source_order() -> None:
    context = AssembledContext(
        context_text="[Source 2]\nText\n\n[Source 1]\nText",
        sources=[create_source(2), create_source(1)],
        source_count=2,
        total_characters=len("[Source 2]\nText\n\n[Source 1]\nText"),
        truncated=True,
        omitted_result_count=1,
    )
    service, _, _, _, _ = create_service(assembled_context=context)

    result = service.answer(GroundedAnswerRequest(question="What is the policy?"))

    assert [citation.source_number for citation in result.citations] == [1, 2]
    assert result.citations[0].chunk_id == "a" * 64
    assert result.citations[0].document_id == DOCUMENT_ID
    assert result.citations[0].file_name == "policy.txt"
    assert result.citations[0].heading == "Heading 1"
    assert result.citations[0].page_number == 1
    assert result.context_truncated is True


def test_valid_citation_markers_are_returned() -> None:
    answer = "Use policy A. [Source 1] Use policy B. [Source 2]"
    provider = StubGenerationProvider(
        GenerationOutput(
            text=answer,
            model_name="stub-model",
            input_characters=len("system") + len("user"),
            output_characters=len(answer),
        )
    )
    service, _, _, _, _ = create_service(
        assembled_context=create_context(source_count=2),
        provider=provider,
    )

    result = service.answer(GroundedAnswerRequest(question="What is the policy?"))

    assert result.citation_markers == [1, 2]


def test_repeated_citation_markers_are_preserved() -> None:
    answer = "Use the policy. [Source 1] It applies. [Source 1]"
    provider = StubGenerationProvider(
        GenerationOutput(
            text=answer,
            model_name="stub-model",
            input_characters=len("system") + len("user"),
            output_characters=len(answer),
        )
    )
    service, _, _, _, _ = create_service(provider=provider)

    result = service.answer(GroundedAnswerRequest(question="What is the policy?"))

    assert result.citation_markers == [1, 1]


def test_unknown_citation_marker_causes_failure() -> None:
    answer = "Use the policy. [Source 2]"
    provider = StubGenerationProvider(
        GenerationOutput(
            text=answer,
            model_name="stub-model",
            input_characters=len("system") + len("user"),
            output_characters=len(answer),
        )
    )
    service, _, _, _, _ = create_service(provider=provider)

    with pytest.raises(ValueError, match="unavailable source"):
        service.answer(GroundedAnswerRequest(question="What is the policy?"))


def test_missing_citation_causes_failure_when_required() -> None:
    answer = "Use the policy."
    provider = StubGenerationProvider(
        GenerationOutput(
            text=answer,
            model_name="stub-model",
            input_characters=len("system") + len("user"),
            output_characters=len(answer),
        )
    )
    service, _, _, _, _ = create_service(provider=provider)

    with pytest.raises(ValueError, match="at least one citation"):
        service.answer(GroundedAnswerRequest(question="What is the policy?"))


def test_missing_citation_allowed_when_requirement_disabled() -> None:
    answer = "Use the policy."
    provider = StubGenerationProvider(
        GenerationOutput(
            text=answer,
            model_name="stub-model",
            input_characters=len("system") + len("user"),
            output_characters=len(answer),
        )
    )
    service, _, _, _, _ = create_service(
        provider=provider,
        require_answer_citations=False,
    )

    result = service.answer(GroundedAnswerRequest(question="What is the policy?"))

    assert result.citation_markers == []


def test_empty_retrieval_skips_generation_provider() -> None:
    empty_context = create_context(source_count=0)
    prompt_package = create_prompt_package(context=empty_context)
    provider = StubGenerationProvider()
    service, search_service, _, _, provider = create_service(
        hybrid_response=HybridSearchResponse(
            query="What is the policy?",
            result_count=0,
            results=[],
        ),
        assembled_context=empty_context,
        prompt_package=prompt_package,
        provider=provider,
    )

    result = service.answer(GroundedAnswerRequest(question="What is the policy?"))

    assert search_service.search.return_value.result_count == 0
    assert provider.calls == []
    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.model_name == "not-invoked"
    assert result.input_characters == prompt_package.total_prompt_characters
    assert result.output_characters == len(INSUFFICIENT_CONTEXT_ANSWER)
    assert result.finish_reason == "insufficient_context"
    assert result.insufficient_context is True
    assert result.context_source_count == 0
    assert result.citations == []
    assert result.citation_markers == []


def test_provider_output_is_not_silently_modified() -> None:
    answer = "Use the policy. [Source 1]"
    provider = StubGenerationProvider(
        GenerationOutput(
            text=answer,
            model_name="stub-model",
            input_characters=len("system") + len("user"),
            output_characters=len(answer),
        )
    )
    service, _, _, _, _ = create_service(provider=provider)

    result = service.answer(GroundedAnswerRequest(question="What is the policy?"))

    assert result.answer == answer


def test_provider_input_character_mismatch_is_rejected() -> None:
    provider = StubGenerationProvider(
        GenerationOutput(
            text="Answer.",
            model_name="stub-model",
            input_characters=999,
            output_characters=len("Answer."),
        )
    )
    service, _, _, _, _ = create_service(provider=provider)

    with pytest.raises(ValueError, match="input character count"):
        service.answer(GroundedAnswerRequest(question="What is the policy?"))


def test_provider_output_character_mismatch_is_rejected() -> None:
    bad_output = GenerationOutput.model_construct(
        text="Answer.",
        model_name="stub-model",
        input_characters=len("system") + len("user"),
        output_characters=999,
        finish_reason=None,
    )
    service, _, _, _, _ = create_service(provider=StubGenerationProvider(bad_output))

    with pytest.raises(ValueError, match="output character count"):
        service.answer(GroundedAnswerRequest(question="What is the policy?"))


def test_blank_provider_output_is_rejected() -> None:
    bad_output = GenerationOutput.model_construct(
        text="   ",
        model_name="stub-model",
        input_characters=len("system") + len("user"),
        output_characters=3,
        finish_reason=None,
    )
    service, _, _, _, _ = create_service(provider=StubGenerationProvider(bad_output))

    with pytest.raises(ValueError, match="output text cannot be blank"):
        service.answer(GroundedAnswerRequest(question="What is the policy?"))


def test_provider_exceptions_propagate() -> None:
    provider = Mock()
    provider.generate.side_effect = RuntimeError("provider failed")
    service, _, _, _, _ = create_service(provider=provider)

    with pytest.raises(RuntimeError, match="provider failed"):
        service.answer(GroundedAnswerRequest(question="What is the policy?"))


def test_retrieval_exceptions_propagate() -> None:
    service, search_service, _, _, _ = create_service()
    search_service.search.side_effect = RuntimeError("retrieval failed")

    with pytest.raises(RuntimeError, match="retrieval failed"):
        service.answer(GroundedAnswerRequest(question="What is the policy?"))


def test_context_assembler_exceptions_propagate() -> None:
    service, _, context_assembler, _, _ = create_service()
    context_assembler.assemble.side_effect = ValueError("context failed")

    with pytest.raises(ValueError, match="context failed"):
        service.answer(GroundedAnswerRequest(question="What is the policy?"))


def test_prompt_builder_exceptions_propagate() -> None:
    service, _, _, prompt_builder, _ = create_service()
    prompt_builder.build.side_effect = ValueError("prompt failed")

    with pytest.raises(ValueError, match="prompt failed"):
        service.answer(GroundedAnswerRequest(question="What is the policy?"))


def test_input_objects_are_not_mutated() -> None:
    request = GroundedAnswerRequest(
        question="  What is the policy?  ",
        limit=3,
        candidate_limit=10,
        document_id=DOCUMENT_ID,
    )
    original_request = request.model_copy(deep=True)
    service, _, _, _, _ = create_service()

    service.answer(request)

    assert request == original_request


def test_timing_callback_called_once_on_success() -> None:
    timings = []
    service, _, _, _, _ = create_service(timing_callback=timings.append)

    service.answer(GroundedAnswerRequest(question="What is the policy?"))

    assert len(timings) == 1
    timing = timings[0]
    assert timing.retrieval_ms >= 0
    assert timing.context_assembly_ms >= 0
    assert timing.prompt_construction_ms >= 0
    assert timing.generation_ms >= 0
    assert timing.total_ms >= 0
    stage_sum = (
        timing.retrieval_ms
        + timing.context_assembly_ms
        + timing.prompt_construction_ms
        + timing.generation_ms
    )
    assert timing.total_ms + 5 >= stage_sum


def test_empty_context_timing_has_zero_generation_and_skips_provider() -> None:
    timings = []
    empty_context = create_context(source_count=0)
    provider = StubGenerationProvider()
    service, _, _, _, provider = create_service(
        assembled_context=empty_context,
        prompt_package=create_prompt_package(context=empty_context),
        provider=provider,
        timing_callback=timings.append,
    )

    service.answer(GroundedAnswerRequest(question="What is the policy?"))

    assert provider.calls == []
    assert timings[0].generation_ms == 0.0


def test_completion_log_contains_safe_metadata(caplog) -> None:
    service, _, _, _, _ = create_service()

    with caplog.at_level("INFO", logger="app.grounded_answer"):
        service.answer(GroundedAnswerRequest(question="SECRET question?"))

    record = next(
        record
        for record in caplog.records
        if record.event == "grounded_answer_completed"
    )
    assert record.success is True
    assert record.retrieved_result_count == 2
    assert record.context_source_count == 1
    assert record.context_truncated is False
    assert record.insufficient_context is False
    assert record.model_name == "stub-model"
    assert record.finish_reason == "stop"
    assert record.citation_marker_count == 1
    assert record.retrieval_ms >= 0
    assert record.context_assembly_ms >= 0
    assert record.prompt_construction_ms >= 0
    assert record.generation_ms >= 0
    assert record.total_ms >= 0
    log_text = caplog.text
    assert "SECRET question" not in log_text
    assert "Employees may work remotely" not in log_text
    assert "system" not in log_text
    assert "user" not in log_text
    assert DOCUMENT_ID not in log_text
    assert "policy.txt" not in log_text


def test_failure_log_contains_exception_type_not_raw_message(caplog) -> None:
    provider = Mock()
    provider.generate.side_effect = RuntimeError("raw provider secret")
    service, _, _, _, _ = create_service(provider=provider)

    with caplog.at_level("WARNING", logger="app.grounded_answer"):
        with pytest.raises(RuntimeError):
            service.answer(GroundedAnswerRequest(question="What is the policy?"))

    record = next(
        record for record in caplog.records if record.event == "grounded_answer_failed"
    )
    assert record.exception_type == "RuntimeError"
    assert record.stage == "generation"
    assert record.elapsed_ms >= 0
    assert "raw provider secret" not in caplog.text


def test_observability_disabled_suppresses_logs(caplog) -> None:
    service, _, _, _, _ = create_service(observability_enabled=False)

    with caplog.at_level("INFO", logger="app.grounded_answer"):
        service.answer(GroundedAnswerRequest(question="What is the policy?"))

    assert "grounded_answer_completed" not in caplog.text

    provider = Mock()
    provider.generate.side_effect = RuntimeError("raw provider secret")
    service, _, _, _, _ = create_service(
        provider=provider,
        observability_enabled=False,
    )
    with caplog.at_level("WARNING", logger="app.grounded_answer"):
        with pytest.raises(RuntimeError):
            service.answer(GroundedAnswerRequest(question="What is the policy?"))

    assert "grounded_answer_failed" not in caplog.text
