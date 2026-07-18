from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_frontend_dockerfile_is_production_multistage() -> None:
    dockerfile = read("frontend/Dockerfile")
    assert "FROM node:" in dockerfile
    assert "AS build" in dockerfile
    assert "FROM nginxinc/nginx-unprivileged" in dockerfile
    assert "npm ci" in dockerfile
    assert "package-lock.json" in dockerfile
    assert "npm run dev" not in dockerfile
    assert 'CMD ["nginx", "-g", "daemon off;"]' in dockerfile


def test_nginx_spa_proxy_and_security_headers() -> None:
    nginx = read("frontend/nginx.conf")
    assert "try_files $uri $uri/ /index.html" in nginx
    assert "proxy_pass http://api:8000/api/" in nginx
    assert "proxy_pass http://localhost" not in nginx
    assert "X-Content-Type-Options" in nginx
    assert "Referrer-Policy" in nginx
    assert "Content-Security-Policy" in nginx
    assert "X-API-Key" in nginx


def test_docker_compose_frontend_service_preserves_api_qdrant() -> None:
    compose = read("docker-compose.yml")
    assert "frontend:" in compose
    assert '"3000:8080"' in compose
    assert "dockerfile: Dockerfile" in compose
    assert "condition: service_healthy" in compose
    assert "QDRANT_URL: http://qdrant:6333" in compose
    assert "TRUSTED_HOSTS: '${TRUSTED_HOSTS:-" in compose
    assert "qdrant_storage:" in compose
    assert "OPENAI_API_KEY" not in compose
    assert "API_AUTH_KEY" not in compose


def test_frontend_env_and_gitignore_do_not_commit_secrets() -> None:
    env_example = read("frontend/.env.example")
    gitignore = read(".gitignore")
    assert "VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1" in env_example
    assert "Do not put API keys" in env_example
    assert "frontend/.env" in gitignore
    assert "frontend/dist" in gitignore


def test_smoke_script_uses_public_frontend_url() -> None:
    smoke = read("scripts/fullstack_smoke.py")
    assert 'BASE_URL = "http://localhost:3000"' in smoke
    assert "/api/v1/health/live" in smoke
    assert "/api/v1/documents" in smoke
    assert "Full-stack smoke passed" in smoke


def test_huggingface_space_metadata_is_docker_port_7860() -> None:
    readme = read("deployment/huggingface/README.md")

    assert readme.startswith("---\n")
    assert "title: Hybrid Search Studio API" in readme
    assert "sdk: docker" in readme
    assert "app_port: 7860" in readme
    assert "pinned: false" in readme
    assert "QDRANT_API_KEY" not in readme
    assert "API_AUTH_KEY_SHA256" not in readme


def test_huggingface_dockerfile_is_api_only_non_root_port_7860() -> None:
    dockerfile = read("deployment/huggingface/Dockerfile")
    pyproject = read("pyproject.toml")

    assert "FROM python:3.11-slim AS builder" in dockerfile
    assert "FROM python:3.11-slim AS runtime" in dockerfile
    assert "COPY backend/app ./backend/app" in dockerfile
    assert "sentence-transformers==5.6.0" in pyproject
    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert '"torch==2.9.1+cpu"' in dockerfile
    assert "SENTENCE_TRANSFORMERS_HOME" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "EXPOSE 7860" in dockerfile
    assert '"0.0.0.0"' in dockerfile
    assert '"7860"' in dockerfile
    assert '"--workers", "1"' in dockerfile
    assert '"app.main:app"' in dockerfile
    assert "npm" not in dockerfile
    assert "qdrant/qdrant" not in dockerfile
    assert "QDRANT_API_KEY" not in dockerfile
    assert "API_AUTH_KEY_SHA256" not in dockerfile
    assert "OPENAI_API_KEY" not in dockerfile


def test_root_dockerfile_and_compose_ports_remain_unchanged() -> None:
    root_dockerfile = read("Dockerfile")
    compose = read("docker-compose.yml")

    assert "EXPOSE 8000" in root_dockerfile
    assert '"8000"' in root_dockerfile
    assert '"8000:8000"' in compose
    assert '"3000:8080"' in compose
    assert "127.0.0.1:6333:6333" in compose
