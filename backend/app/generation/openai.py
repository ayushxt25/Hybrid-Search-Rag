from math import isfinite

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
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
        timeout_seconds: float = 30.0,
        max_output_tokens: int | None = None,
        max_retries: int = 2,
        client: OpenAI | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be blank.")
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and greater than zero.")
        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to zero.")

        self.api_key = api_key.strip()
        self.base_url = base_url
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.client = client

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> GenerationOutput:
        if self.client is None and not self.api_key:
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
        except (APIConnectionError, APITimeoutError) as error:
            raise GenerationConnectionError("OpenAI connection failed.") from error
        except (APIStatusError, APIError, OpenAIError) as error:
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
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            )

        return self.client
