from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_example_env_contains_no_concrete_secret_values() -> None:
    env_example = read(".env.example")

    assert "sk-" not in env_example
    assert "password=" not in env_example.lower()
    assert "API_AUTH_KEY=" not in env_example
    assert "PRODUCTION_API_AUTH_KEY_SHA256=<sha256-hex-digest-placeholder>" in (
        env_example
    )
    assert "PRODUCTION_OPENAI_API_KEY=<secret-managed-openai-key-placeholder>" in (
        env_example
    )


def test_frontend_defaults_to_relative_api_path_for_production_builds() -> None:
    client = read("frontend/src/lib/api/client.ts")

    assert 'const DEFAULT_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";' in (
        client
    )
    assert '?? "http://127.0.0.1:8000"' not in client
    assert '?? "http://localhost:8000"' not in client


def test_frontend_env_example_documents_local_only_api_url() -> None:
    frontend_env = read("frontend/.env.example")

    assert "VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1" in frontend_env
    assert "Do not put API keys" in frontend_env
    assert "Leave unset for production" in frontend_env
