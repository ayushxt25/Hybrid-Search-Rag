from app.context.assembler import ContextAssembler
from app.context.models import AssembledContext, ContextSource
from app.generation.base import GenerationProvider
from app.generation.citations import validate_citation_markers
from app.generation.models import (
    AnswerCitation,
    GenerationOutput,
    GroundedAnswerRequest,
    GroundedAnswerResult,
)
from app.prompting.builder import GroundedPromptBuilder
from app.prompting.models import GroundedPromptPackage
from app.schemas.search_request import HybridSearchRequest
from app.services.hybrid_search import HybridSearchService

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
    ) -> None:
        if not isinstance(require_answer_citations, bool):
            raise ValueError("require_answer_citations must be a boolean.")

        self.hybrid_search_service = hybrid_search_service
        self.context_assembler = context_assembler
        self.prompt_builder = prompt_builder
        self.generation_provider = generation_provider
        self.require_answer_citations = require_answer_citations

    def answer(self, request: GroundedAnswerRequest) -> GroundedAnswerResult:
        """Answer a grounded question using the configured generation provider."""
        normalized_question = request.question.strip()

        if not normalized_question:
            raise ValueError("question cannot be blank.")

        hybrid_response = self.hybrid_search_service.search(
            HybridSearchRequest(
                query=normalized_question,
                limit=request.limit,
                candidate_limit=request.candidate_limit,
                document_id=request.document_id,
            )
        )
        assembled_context = self.context_assembler.assemble(hybrid_response.results)
        prompt_package = self.prompt_builder.build(
            question=normalized_question,
            context=assembled_context,
        )
        citations = self._create_citations(assembled_context.sources)

        if assembled_context.source_count == 0:
            return self._create_insufficient_context_result(
                question=normalized_question,
                retrieved_result_count=hybrid_response.result_count,
                assembled_context=assembled_context,
                prompt_package=prompt_package,
                citations=citations,
            )

        generation_output = self.generation_provider.generate(
            system_prompt=prompt_package.system_prompt,
            user_prompt=prompt_package.user_prompt,
        )
        self._validate_generation_output(
            output=generation_output,
            prompt_package=prompt_package,
        )
        citation_markers = validate_citation_markers(
            text=generation_output.text,
            available_source_numbers=[citation.source_number for citation in citations],
            require_citations=self.require_answer_citations,
        )

        return GroundedAnswerResult(
            question=normalized_question,
            answer=generation_output.text,
            model_name=generation_output.model_name,
            citations=citations,
            citation_markers=citation_markers,
            retrieved_result_count=hybrid_response.result_count,
            context_source_count=assembled_context.source_count,
            context_truncated=assembled_context.truncated,
            insufficient_context=prompt_package.insufficient_context,
            input_characters=generation_output.input_characters,
            output_characters=generation_output.output_characters,
            finish_reason=generation_output.finish_reason,
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
