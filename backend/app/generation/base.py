from abc import ABC, abstractmethod

from app.generation.models import GenerationOutput


class GenerationProvider(ABC):
    """Provider-neutral text generation interface."""

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> GenerationOutput:
        """Generate text from rendered prompts."""
