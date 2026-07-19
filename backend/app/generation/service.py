import logging
from collections.abc import Callable
from time import perf_counter

from app.context.assembler import ContextAssembler
from app.context.models import AssembledContext, ContextSource
from app.core.config import get_settings
from app.generation.base import GenerationProvider
from app.generation.citations import validate_citation_markers
from app.generation.models import (
    AnswerCitation,
    GenerationOutput,
    GroundedAnswerRequest,
    GroundedAnswerResult,
)
from app.observability.models import GroundedAnswerTimings
from app.observability.request_context import get_request_id
from app.prompting.builder import GroundedPromptBuilder
from app.prompting.models import GroundedPromptPackage
from app.retrieval.filters import RetrievalFilters
from app.schemas.search_request import HybridSearchRequest
from app.services.hybrid_search import HybridSearchService

logger = logging.getLogger("app.grounded_answer")

INSUFFICIENT_CONTEXT_ANSWER = (
    "The provided documents do not contain enough information to answer this question."
)


class GroundedAnswerService:
    """Orchestrate retrieval, context assembly, prompting, and generation."""

    def __init__(
        self,
        *,
        hybrid_search_service: HybridSearchService,
        context_assembler: ContextAssembler,
        prompt_builder: GroundedPromptBuilder,
        generation_provider: GenerationProvider,
        require_answer_citations: bool = True,
        timing_callback: Callable[[GroundedAnswerTimings], None] | None = None,
        observability_enabled: bool | None = None,
    ) -> None:
        if not isinstance(require_answer_citations, bool):
            raise ValueError("require_answer_citations must be a boolean.")

        self.hybrid_search_service = hybrid_search_service
        self.context_assembler = context_assembler
        self.prompt_builder = prompt_builder
        self.generation_provider = generation_provider
        self.require_answer_citations = require_answer_citations
        self.timing_callback = timing_callback
        self.observability_enabled = (
            get_settings().observability_enabled
            if observability_enabled is None
            else observability_enabled
        )

    def answer(self, request: GroundedAnswerRequest) -> GroundedAnswerResult:
        """Answer a grounded question using the configured generation provider."""
        total_start = perf_counter()
        stage = "validation"

        try:
            normalized_question = request.question.strip()

            if not normalized_question:
                raise ValueError("question cannot be blank.")

            stage = "retrieval"
            stage_start = perf_counter()
            hybrid_response = self.hybrid_search_service.search(
                HybridSearchRequest(
                    query=normalized_question,
                    limit=request.limit,
                    candidate_limit=request.candidate_limit,
                    document_id=request.document_id,
                    document_ids=request.document_ids,
                    content_types=request.content_types,
                )
            )
            retrieval_ms = self._elapsed_ms(stage_start)

            stage = "context_assembly"
            stage_start = perf_counter()
            assembled_context = self.context_assembler.assemble(hybrid_response.results)
            context_assembly_ms = self._elapsed_ms(stage_start)

            stage = "prompt_construction"
            stage_start = perf_counter()
            prompt_package = self.prompt_builder.build(
                question=normalized_question,
                context=assembled_context,
            )
            prompt_construction_ms = self._elapsed_ms(stage_start)
            citations = self._create_citations(assembled_context.sources)

            generation_ms = 0.0
            if assembled_context.source_count == 0:
                result = self._create_insufficient_context_result(
                    question=normalized_question,
                    retrieved_result_count=hybrid_response.result_count,
                    assembled_context=assembled_context,
                    prompt_package=prompt_package,
                    citations=citations,
                )
            else:
                stage = "generation"
                stage_start = perf_counter()
                generation_output = self.generation_provider.generate(
                    system_prompt=prompt_package.system_prompt,
                    user_prompt=prompt_package.user_prompt,
                )
                generation_ms = self._elapsed_ms(stage_start)
                self._validate_generation_output(
                    output=generation_output,
                    prompt_package=prompt_package,
                )
                provider_reported_insufficient_context = (
                    generation_output.finish_reason == "insufficient_context"
                    or generation_output.text == INSUFFICIENT_CONTEXT_ANSWER
                )

                if provider_reported_insufficient_context:
                    citation_markers = []
                    citations = []
                    context_source_count = 0
                    insufficient_context = True
                else:
                    citation_markers = validate_citation_markers(
                        text=generation_output.text,
                        available_source_numbers=[
                            citation.source_number for citation in citations
                        ],
                        require_citations=self.require_answer_citations,
                    )
                    context_source_count = assembled_context.source_count
                    insufficient_context = prompt_package.insufficient_context

                result = GroundedAnswerResult(
                    question=normalized_question,
                    answer=generation_output.text,
                    model_name=generation_output.model_name,
                    citations=citations,
                    citation_markers=citation_markers,
                    retrieved_result_count=hybrid_response.result_count,
                    context_source_count=context_source_count,
                    context_truncated=assembled_context.truncated,
                    insufficient_context=insufficient_context,
                    input_characters=generation_output.input_characters,
                    output_characters=generation_output.output_characters,
                    finish_reason=generation_output.finish_reason,
                )

            timings = GroundedAnswerTimings(
                retrieval_ms=retrieval_ms,
                context_assembly_ms=context_assembly_ms,
                prompt_construction_ms=prompt_construction_ms,
                generation_ms=generation_ms,
                total_ms=self._elapsed_ms(total_start),
            )
            if self.timing_callback is not None:
                self.timing_callback(timings)
            self._log_success(result=result, timings=timings, request=request)
            return result
        except Exception as error:
            self._log_failure(
                exception_type=type(error).__name__,
                stage=stage,
                elapsed_ms=self._elapsed_ms(total_start),
            )
            raise

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        return max((perf_counter() - start_time) * 1000, 0.0)

    def _log_success(
        self,
        *,
        result: GroundedAnswerResult,
        timings: GroundedAnswerTimings,
        request: GroundedAnswerRequest,
    ) -> None:
        if not self.observability_enabled:
            return

        filters = RetrievalFilters.from_legacy(
            document_id=request.document_id,
            document_ids=request.document_ids,
            content_types=request.content_types,
        )

        logger.info(
            "grounded_answer_completed",
            extra={
                "event": "grounded_answer_completed",
                "request_id": get_request_id(),
                "success": True,
                "retrieved_result_count": result.retrieved_result_count,
                "context_source_count": result.context_source_count,
                "context_truncated": result.context_truncated,
                "insufficient_context": result.insufficient_context,
                "model_name": result.model_name,
                "finish_reason": result.finish_reason,
                "citation_marker_count": len(result.citation_markers),
                "filter_document_count": len(filters.document_ids),
                "filter_content_type_count": len(filters.content_types),
                "filtered": bool(filters.document_ids or filters.content_types),
                "retrieval_ms": timings.retrieval_ms,
                "context_assembly_ms": timings.context_assembly_ms,
                "prompt_construction_ms": timings.prompt_construction_ms,
                "generation_ms": timings.generation_ms,
                "total_ms": timings.total_ms,
            },
        )

    def _log_failure(
        self,
        *,
        exception_type: str,
        stage: str,
        elapsed_ms: float,
    ) -> None:
        if not self.observability_enabled:
            return

        logger.warning(
            "grounded_answer_failed",
            extra={
                "event": "grounded_answer_failed",
                "request_id": get_request_id(),
                "exception_type": exception_type,
                "stage": stage,
                "elapsed_ms": elapsed_ms,
            },
        )

    def _create_insufficient_context_result(
        self,
        *,
        question: str,
        retrieved_result_count: int,
        assembled_context: AssembledContext,
        prompt_package: GroundedPromptPackage,
        citations: list[AnswerCitation],
    ) -> GroundedAnswerResult:
        return GroundedAnswerResult(
            question=question,
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            model_name="not-invoked",
            citations=citations,
            citation_markers=[],
            retrieved_result_count=retrieved_result_count,
            context_source_count=assembled_context.source_count,
            context_truncated=assembled_context.truncated,
            insufficient_context=True,
            input_characters=prompt_package.total_prompt_characters,
            output_characters=len(INSUFFICIENT_CONTEXT_ANSWER),
            finish_reason="insufficient_context",
        )

    def _validate_generation_output(
        self,
        *,
        output: GenerationOutput,
        prompt_package: GroundedPromptPackage,
    ) -> None:
        expected_input_characters = prompt_package.total_prompt_characters

        if not output.text.strip():
            raise ValueError("generation output text cannot be blank.")

        if output.output_characters != len(output.text):
            raise ValueError("generation output character count is inconsistent.")

        if output.input_characters != expected_input_characters:
            raise ValueError("generation input character count is inconsistent.")

    def _create_citations(
        self,
        sources: list[ContextSource],
    ) -> list[AnswerCitation]:
        return [
            AnswerCitation(
                source_number=source.source_number,
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                file_name=source.file_name,
                heading=source.heading,
                page_number=source.page_number,
            )
            for source in sorted(sources, key=lambda source: source.source_number)
        ]
