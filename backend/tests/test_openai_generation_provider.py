from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.generation.openai import OpenAIGenerationProvider


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
    )

    output = provider.generate(system_prompt="system", user_prompt="user")

    openai_class.assert_called_once_with(
        api_key="test-key",
        base_url="https://example.test/v1",
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


def test_openai_generation_provider_rejects_blank_configuration() -> None:
    with pytest.raises(ValueError, match="model_name cannot be blank"):
        OpenAIGenerationProvider(api_key="test-key", model_name=" ")


def test_openai_generation_provider_reports_missing_api_key_at_generation() -> None:
    provider = OpenAIGenerationProvider(api_key="", model_name="gpt-test")

    with pytest.raises(RuntimeError, match="OpenAI API key is not configured"):
        provider.generate(system_prompt="system", user_prompt="user")
