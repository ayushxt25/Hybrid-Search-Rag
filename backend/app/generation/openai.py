from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from app.generation.base import GenerationProvider
from app.generation.models import GenerationOutput


class GenerationProviderError(Exception):
    """Raised when generation provider response handling fails."""


class GenerationAuthenticationError(GenerationProviderError):
    """Raised when the generation provider rejects credentials."""


class GenerationRateLimitError(GenerationProviderError):
    """Raised when the generation provider rate limits the request."""


class GenerationConnectionError(GenerationProviderError):
    """Raised when the generation provider cannot be reached."""


class OpenAIGenerationProvider(GenerationProvider):
    """Generation provider backed by the OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        base_url: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be blank.")

        self.api_key = api_key.strip()
        self.base_url = base_url
        self.model_name = model_name
        self.client: OpenAI | None = None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> GenerationOutput:
        if not self.api_key:
            raise RuntimeError("OpenAI API key is not configured.")

        try:
            response = self._client().responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except AuthenticationError as error:
            raise GenerationAuthenticationError(
                "OpenAI authentication failed."
            ) from error
        except RateLimitError as error:
            raise GenerationRateLimitError("OpenAI rate limit exceeded.") from error
        except APIConnectionError as error:
            raise GenerationConnectionError("OpenAI connection failed.") from error
        except (APIError, OpenAIError) as error:
            raise GenerationProviderError("OpenAI generation failed.") from error

        text = response.output_text

        return GenerationOutput(
            text=text,
            model_name=self.model_name,
            input_characters=len(system_prompt) + len(user_prompt),
            output_characters=len(text),
            finish_reason=getattr(response, "status", None),
        )

    def _client(self) -> OpenAI:
        if self.client is None:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        return self.client
