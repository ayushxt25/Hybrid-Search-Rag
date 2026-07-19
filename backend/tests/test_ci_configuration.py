import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"


def read_workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_ci_workflow_exists() -> None:
    assert WORKFLOW.exists()


def test_ci_triggers_main_and_manual_dispatch() -> None:
    text = read_workflow()
    assert "push:" in text
    assert "pull_request:" in text
    assert "workflow_dispatch:" in text
    assert "branches:" in text
    assert "- main" in text


def test_ci_permissions_and_concurrency_are_restricted() -> None:
    text = read_workflow()
    assert "permissions:" in text
    assert "contents: read" in text
    assert "concurrency:" in text
    assert "cancel-in-progress: true" in text
    assert "contents: write" not in text


def test_quality_job_runs_python_ruff_and_full_pytest() -> None:
    text = read_workflow()
    assert "quality-and-tests:" in text
    assert "runs-on: ubuntu-latest" in text
    assert 'python-version: "3.11"' in text
    assert "cache: pip" in text
    assert "cache-dependency-path: pyproject.toml" in text
    assert 'python -m pip install -e ".[test]"' in text
    assert "python -m ruff format --check backend" in text
    assert "python -m ruff check backend" in text
    assert "python -m pytest" in text
    assert "python -m pytest --maxfail" not in text
    assert "timeout-minutes: 15" in text


def test_container_job_validates_compose_and_builds_image() -> None:
    text = read_workflow()
    assert "container-build:" in text
    assert "needs: quality-and-tests" in text
    assert "timeout-minutes: 20" in text
    assert "docker compose config" in text
    assert "docker build --tag hybrid-search-rag-ci:local ." in text
    assert "QDRANT_URL=http://qdrant:6333" in text
    assert "READINESS_ENABLED=false" in text
    assert "API_AUTH_ENABLED=false" in text
    assert "API_AUTH_KEY_SHA256" not in text
    assert "OPENAI_API_KEY=" in text


def test_ci_does_not_push_or_use_secrets() -> None:
    text = read_workflow().lower()
    assert "docker login" not in text
    assert "docker push" not in text
    assert "secrets." not in text
    assert "api_auth_key_sha256" not in text
    assert "sk-" not in text


def test_ci_uses_only_official_allowed_actions() -> None:
    action_lines = [
        line.strip() for line in read_workflow().splitlines() if "uses:" in line
    ]
    assert action_lines
    assert set(action_lines) <= {
        "uses: actions/checkout@v4",
        "uses: actions/setup-python@v5",
    }


def test_temporary_env_cleanup_exists() -> None:
    text = read_workflow()
    assert "cat > .env" in text
    assert "if: always()" in text
    assert "rm -f .env" in text


def test_no_runtime_smoke_or_readiness_dependency() -> None:
    text = read_workflow().lower()
    assert "/api/v1/health/ready" not in text
    assert "docker compose up" not in text
    assert "docker compose down" not in text
    assert "huggingface" not in text


def test_ci_test_extra_includes_local_embedding_dependencies() -> None:
    pyproject = load_pyproject()
    optional_dependencies = pyproject["project"]["optional-dependencies"]

    assert "local-embeddings" in optional_dependencies
    assert "test" in optional_dependencies
    assert "sentence-transformers==5.6.0" in optional_dependencies["local-embeddings"]
    assert "sentence-transformers==5.6.0" in optional_dependencies["test"]
    assert "sentence-transformers" not in pyproject["project"]["dependencies"]


def test_readme_documents_ci() -> None:
    text = README.read_text(encoding="utf-8")
    assert "## Continuous Integration" in text
    assert "python -m ruff format --check backend" in text
    assert "python -m pytest" in text
    assert "docker compose config" in text
    assert "docker build --tag hybrid-search-rag-ci:local ." in text
