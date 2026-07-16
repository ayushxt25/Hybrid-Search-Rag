from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_dockerfile_production_shape() -> None:
    dockerfile = read("Dockerfile")

    assert "FROM python:3.11-slim AS builder" in dockerfile
    assert "FROM python:3.11-slim AS runtime" in dockerfile
    assert "python -m build --wheel" in dockerfile
    assert "pip install --no-cache-dir /tmp/*.whl" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert (
        'CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]'
        in dockerfile
    )
    assert "--reload" not in dockerfile
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile
    assert "PYTHONUNBUFFERED=1" in dockerfile
    assert "HF_HOME=/home/appuser/.cache/huggingface" in dockerfile


def test_dockerignore_excludes_local_and_test_artifacts() -> None:
    dockerignore = read(".dockerignore")

    for entry in [".venv", ".git", "backend/tests", "reports", ".env"]:
        assert entry in dockerignore


def test_compose_contains_api_and_qdrant_services() -> None:
    compose = read("docker-compose.yml")

    assert "api:" in compose
    assert "qdrant:" in compose
    assert "QDRANT_URL: http://qdrant:6333" in compose
    assert (
        "QDRANT_HYBRID_COLLECTION_NAME: "
        "${QDRANT_HYBRID_COLLECTION_NAME:-internal_document_chunks_hybrid}" in compose
    )
    assert "8000:8000" in compose
    assert "env_file:" in compose
    assert "condition: service_healthy" in compose


def test_compose_healthchecks_and_security_controls() -> None:
    compose = read("docker-compose.yml")

    assert "/api/v1/health/live" in compose
    assert "/api/v1/health/ready" not in compose
    assert "/dev/tcp/127.0.0.1/6333" in compose
    assert "GET /healthz HTTP/1.1" in compose
    assert "200 OK" in compose
    assert "wget" not in compose
    assert "curl" not in compose
    assert "API_AUTH_KEY_SHA256" not in compose
    assert "privileged: true" not in compose
    assert "/var/run/docker.sock" not in compose
    assert "cap_drop:" in compose
    assert "no-new-privileges:true" in compose


def test_compose_does_not_bind_mount_source_or_configure_multiple_workers() -> None:
    compose = read("docker-compose.yml")

    assert ".:/app" not in compose
    assert "--workers" not in compose
    assert "replicas:" not in compose


def test_model_cache_volume_is_configured() -> None:
    compose = read("docker-compose.yml")

    assert "huggingface_cache:" in compose
    assert "/home/appuser/.cache/huggingface" in compose
