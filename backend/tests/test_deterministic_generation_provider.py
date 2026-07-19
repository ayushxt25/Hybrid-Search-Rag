from app.context.models import AssembledContext, ContextSource
from app.generation.deterministic import (
    DETERMINISTIC_MODEL_NAME,
    INSUFFICIENT_CONTEXT_TEXT,
    DeterministicAcceptanceGenerationProvider,
)
from app.prompting.builder import GroundedPromptBuilder

DOCUMENT_ID = "d" * 64


def create_context(*, sources: list[ContextSource]) -> AssembledContext:
    rendered_blocks = []
    for source in sources:
        rendered_blocks.append(
            "\n".join(
                [
                    f"[Source {source.source_number}]",
                    f"File: {source.file_name}",
                    f"Chunk ID: {source.chunk_id}",
                    f"Heading: {source.heading or 'N/A'}",
                    f"Page: {source.page_number or 'N/A'}",
                    "Content:",
                    source.text,
                ]
            )
        )
    context_text = "\n\n".join(rendered_blocks)
    return AssembledContext(
        context_text=context_text,
        sources=sources,
        source_count=len(sources),
        total_characters=len(context_text),
        truncated=False,
        omitted_result_count=0,
    )


def create_source(
    *,
    source_number: int = 1,
    text: str,
    chunk_id: str | None = None,
    file_name: str = "policy.txt",
) -> ContextSource:
    return ContextSource(
        source_number=source_number,
        chunk_id=chunk_id or chr(96 + source_number) * 64,
        document_id=DOCUMENT_ID,
        file_name=file_name,
        text=text,
        score=0.9,
        chunk_index=source_number - 1,
        section_index=0,
        heading="Policy",
        page_number=1,
    )


def generate_answer(*, question: str, sources: list[ContextSource]) -> str:
    prompt = GroundedPromptBuilder().build(
        question=question,
        context=create_context(sources=sources),
    )

    output = DeterministicAcceptanceGenerationProvider().generate(
        system_prompt=prompt.system_prompt,
        user_prompt=prompt.user_prompt,
    )

    assert output.model_name == DETERMINISTIC_MODEL_NAME
    assert output.input_characters == prompt.total_prompt_characters
    assert output.output_characters == len(output.text)
    return output.text


def test_frequency_fact_extraction_includes_supported_value() -> None:
    answer = generate_answer(
        question="How often is the aurora ledger reviewed?",
        sources=[
            create_source(
                text="The aurora ledger is reviewed every 23 days.",
                file_name="aurora.txt",
            )
        ],
    )

    assert answer == "The aurora ledger is reviewed every 23 days. [Source 1]"


def test_numeric_fact_extraction_includes_supported_value() -> None:
    answer = generate_answer(
        question="How many review slots are assigned to the beacon queue?",
        sources=[
            create_source(
                text=(
                    "The beacon queue has routine oversight. "
                    "The beacon queue is assigned 17 review slots."
                ),
            )
        ],
    )

    assert answer == "The beacon queue is assigned 17 review slots. [Source 1]"


def test_direct_answer_points_to_supporting_source() -> None:
    answer = generate_answer(
        question="Which team owns the nebula archive?",
        sources=[
            create_source(
                source_number=1,
                text="The comet team owns the release checklist.",
            ),
            create_source(
                source_number=2,
                text="The nebula archive is owned by the atlas team.",
            ),
        ],
    )

    assert answer == "The nebula archive is owned by the atlas team. [Source 2]"


def test_unrelated_question_returns_insufficient_context_without_citation() -> None:
    answer = generate_answer(
        question="What color is the finance dashboard?",
        sources=[
            create_source(
                text="The aurora ledger is reviewed every 23 days.",
            )
        ],
    )

    assert answer == INSUFFICIENT_CONTEXT_TEXT
    assert "[Source" not in answer


def test_context_order_breaks_ties_deterministically() -> None:
    answer = generate_answer(
        question="What is the policy review cadence?",
        sources=[
            create_source(
                source_number=1,
                text="The policy review cadence is monthly.",
            ),
            create_source(
                source_number=2,
                text="The policy review cadence is quarterly.",
            ),
        ],
    )

    assert answer == "The policy review cadence is monthly. [Source 1]"


def test_no_fixture_specific_logic_for_other_frequency_fact() -> None:
    answer = generate_answer(
        question="How often is the beacon rota refreshed?",
        sources=[
            create_source(
                text="The beacon rota is refreshed every 41 weeks.",
            )
        ],
    )

    assert answer == "The beacon rota is refreshed every 41 weeks. [Source 1]"
