from app.generation.base import GenerationProvider
from app.generation.models import GenerationOutput

DETERMINISTIC_MODEL_NAME = "deterministic-acceptance-provider"


class DeterministicAcceptanceGenerationProvider(GenerationProvider):
    """Acceptance-only provider that cites supplied evidence without network access."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> GenerationOutput:
        answer = "The indexed evidence supports this answer. [Source 1]"
        return GenerationOutput(
            text=answer,
            model_name=DETERMINISTIC_MODEL_NAME,
            input_characters=len(system_prompt) + len(user_prompt),
            output_characters=len(answer),
            finish_reason="deterministic",
        )
