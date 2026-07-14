import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from app.generation.openai import (
    GenerationAuthenticationError,
    GenerationConnectionError,
    GenerationProviderError,
    GenerationRateLimitError,
    OpenAIGenerationProvider,
)


@patch("app.generation.openai.OpenAI")
def test_openai_generation_provider_uses_responses_api(openai_class: Mock) -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="Answer [1]",
        status="completed",
    )
    openai_class.return_value = client
    provider = OpenAIGenerationProvider(
        api_key="test-key",
        model_name="gpt-test",
        base_url="https://example.test/v1",
        timeout_seconds=12.5,
        max_retries=4,
    )

    output = provider.generate(system_prompt="system", user_prompt="user")

    openai_class.assert_called_once_with(
        api_key="test-key",
        base_url="https://example.test/v1",
        timeout=12.5,
        max_retries=4,
    )
    client.responses.create.assert_called_once_with(
        model="gpt-test",
        input=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
    )
    assert output.text == "Answer [1]"
    assert output.model_name == "gpt-test"
    assert output.input_characters == len("system") + len("user")
    assert output.output_characters == len("Answer [1]")
    assert output.finish_reason == "completed"


@patch("app.generation.openai.OpenAI")
def test_default_max_retries_is_two(openai_class: Mock) -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="Answer [1]",
        status="completed",
    )
    openai_class.return_value = client
    provider = OpenAIGenerationProvider(api_key="test-key", model_name="gpt-test")

    provider.generate(system_prompt="system", user_prompt="user")

    assert openai_class.call_args.kwargs["max_retries"] == 2


@patch("app.generation.openai.OpenAI")
def test_max_retries_zero_is_allowed(openai_class: Mock) -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="Answer [1]",
        status="completed",
    )
    openai_class.return_value = client
    provider = OpenAIGenerationProvider(
        api_key="test-key",
        model_name="gpt-test",
        max_retries=0,
    )

    provider.generate(system_prompt="system", user_prompt="user")

    assert openai_class.call_args.kwargs["max_retries"] == 0


def test_openai_generation_provider_rejects_blank_configuration() -> None:
    with pytest.raises(ValueError, match="model_name cannot be blank"):
        OpenAIGenerationProvider(api_key="test-key", model_name=" ")

    with pytest.raises(ValueError, match="timeout_seconds"):
        OpenAIGenerationProvider(
            api_key="test-key",
            model_name="gpt-test",
            timeout_seconds=0,
        )

    with pytest.raises(ValueError, match="max_retries"):
        OpenAIGenerationProvider(
            api_key="test-key",
            model_name="gpt-test",
            max_retries=-1,
        )


def test_openai_generation_provider_reports_missing_api_key_at_generation() -> None:
    provider = OpenAIGenerationProvider(api_key="", model_name="gpt-test")

    with pytest.raises(RuntimeError, match="OpenAI API key is not configured"):
        provider.generate(system_prompt="system", user_prompt="user")


@patch("app.generation.openai.OpenAI")
def test_injected_client_is_reused(openai_class: Mock) -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="Answer [1]",
        status="completed",
    )
    provider = OpenAIGenerationProvider(
        api_key="",
        model_name="gpt-test",
        client=client,
    )

    output = provider.generate(system_prompt="system", user_prompt="user")

    assert output.text == "Answer [1]"
    openai_class.assert_not_called()


@pytest.mark.parametrize(
    ("openai_error_name", "expected_error"),
    [
        ("AuthenticationError", GenerationAuthenticationError),
        ("RateLimitError", GenerationRateLimitError),
        ("APIConnectionError", GenerationConnectionError),
        ("APITimeoutError", GenerationConnectionError),
        ("APIStatusError", GenerationProviderError),
        ("APIError", GenerationProviderError),
    ],
)
def test_openai_errors_are_mapped(
    openai_error_name: str,
    expected_error: type[Exception],
) -> None:
    error_class = type(openai_error_name, (Exception,), {})
    client = Mock()
    client.responses.create.side_effect = error_class("provider failed")
    provider = OpenAIGenerationProvider(
        api_key="test-key",
        model_name="gpt-test",
        client=client,
    )

    with patch(f"app.generation.openai.{openai_error_name}", error_class):
        with pytest.raises(expected_error):
            provider.generate(system_prompt="system", user_prompt="user")


def test_local_validation_errors_are_not_converted_to_provider_errors() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="",
        status="completed",
    )
    provider = OpenAIGenerationProvider(
        api_key="test-key",
        model_name="gpt-test",
        client=client,
    )

    with pytest.raises(ValidationError):
        provider.generate(system_prompt="system", user_prompt="user")


def test_no_custom_sleep_or_retry_loop_is_used() -> None:
    source = inspect.getsource(OpenAIGenerationProvider.generate)

    assert "sleep" not in source
    assert "for " not in source
    assert "while " not in source
