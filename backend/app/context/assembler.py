from collections.abc import Sequence

from app.context.models import AssembledContext, ContextSource
from app.schemas.search import DenseSearchResult


class ContextAssembler:
    """Assemble ranked retrieval results into bounded generation context."""

    def __init__(
        self,
        *,
        max_characters: int = 12000,
        max_sources: int = 8,
        include_metadata_headers: bool = True,
    ) -> None:
        if max_characters <= 0:
            raise ValueError("max_characters must be greater than zero.")

        if max_sources <= 0:
            raise ValueError("max_sources must be greater than zero.")

        self.max_characters = max_characters
        self.max_sources = max_sources
        self.include_metadata_headers = include_metadata_headers

    def assemble(
        self,
        results: Sequence[DenseSearchResult],
    ) -> AssembledContext:
        """Build deterministic context from ranked search results."""
        result_list = list(results)
        chunk_ids = [result.chunk_id for result in result_list]

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("results must not contain duplicate chunk IDs.")

        sources: list[ContextSource] = []
        rendered_blocks: list[str] = []

        for result in result_list:
            if len(sources) >= self.max_sources:
                break

            source = self._create_source(
                result=result,
                source_number=len(sources) + 1,
            )
            block = self._render_source(source)
            candidate_blocks = [*rendered_blocks, block]
            candidate_text = "\n\n".join(candidate_blocks)

            if len(candidate_text) > self.max_characters:
                break

            sources.append(source)
            rendered_blocks.append(block)

        context_text = "\n\n".join(rendered_blocks)
        omitted_result_count = len(result_list) - len(sources)

        return AssembledContext(
            context_text=context_text,
            sources=sources,
            source_count=len(sources),
            total_characters=len(context_text),
            truncated=omitted_result_count > 0,
            omitted_result_count=omitted_result_count,
        )

    def _create_source(
        self,
        *,
        result: DenseSearchResult,
        source_number: int,
    ) -> ContextSource:
        return ContextSource(
            source_number=source_number,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            file_name=result.file_name,
            text=result.text,
            score=result.score,
            chunk_index=result.chunk_index,
            section_index=result.section_index,
            heading=result.heading,
            page_number=result.page_number,
        )

    def _render_source(self, source: ContextSource) -> str:
        if not self.include_metadata_headers:
            return source.text

        heading = source.heading if source.heading is not None else "N/A"
        page_number = (
            str(source.page_number) if source.page_number is not None else "N/A"
        )

        return "\n".join(
            [
                f"[Source {source.source_number}]",
                f"File: {source.file_name}",
                f"Chunk ID: {source.chunk_id}",
                f"Heading: {heading}",
                f"Page: {page_number}",
                "Content:",
                source.text,
            ]
        )
