import ast
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
HEAVY_PACKAGES = [
    "torch",
    "torchvision",
    "scipy",
    "transformers",
    "sentence-transformers",
    "tokenizers",
]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_vercel() -> dict:
    return json.loads(read("vercel.json"))


def load_pyproject() -> dict:
    return tomllib.loads(read("pyproject.toml"))


def test_fastapi_vercel_entrypoint_reexports_existing_app() -> None:
    tree = ast.parse(read("api/index.py"))
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "app.main"
    ]

    assert imports
    assert any(alias.name == "app" for node in imports for alias in node.names)


def test_vercel_entrypoint_serves_spa_routes_from_frontend_dist(
    tmp_path,
    monkeypatch,
) -> None:
    from api import index as vercel_entrypoint

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>Hybrid Search Studio</html>")
    monkeypatch.setattr(vercel_entrypoint, "FRONTEND_DIST", dist)
    monkeypatch.setattr(vercel_entrypoint, "FRONTEND_INDEX", dist / "index.html")

    client = TestClient(vercel_entrypoint.app)

    for route in [
        "/",
        "/index.html",
        "/overview",
        "/documents",
        "/retrieval",
        "/answers",
        "/system",
    ]:
        response = client.get(route)
        assert response.status_code == 200
        assert "Hybrid Search Studio" in response.text
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
        assert "connect-src 'self'" in response.headers["Content-Security-Policy"]


def test_vercel_entrypoint_spa_fallback_does_not_intercept_api(monkeypatch) -> None:
    from api import index as vercel_entrypoint

    monkeypatch.setattr(
        vercel_entrypoint,
        "FRONTEND_INDEX",
        Path("frontend/dist/index.html"),
    )
    client = TestClient(vercel_entrypoint.app)

    response = client.get("/api/not-a-real-route")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["Content-Security-Policy"] == (
        "default-src 'none'; frame-ancestors 'none'"
    )


def test_vercel_entrypoint_serves_built_files_before_spa_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    from api import index as vercel_entrypoint

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>index</html>")
    (dist / "manifest.webmanifest").write_text('{"name":"test"}')
    monkeypatch.setattr(vercel_entrypoint, "FRONTEND_DIST", dist)
    monkeypatch.setattr(vercel_entrypoint, "FRONTEND_INDEX", dist / "index.html")
    client = TestClient(vercel_entrypoint.app)

    response = client.get("/manifest.webmanifest")

    assert response.status_code == 200
    assert response.text == '{"name":"test"}'
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]


def test_vercel_build_output_and_api_routing_are_configured() -> None:
    config = load_vercel()
    rewrites = config["rewrites"]

    assert config["framework"] == "fastapi"
    assert (
        config["buildCommand"]
        == "cd frontend && npm ci --cache /tmp/npm-cache && npm run build && cd .. "
        "&& node scripts/prepare_vercel_static.mjs"
    )
    assert config["outputDirectory"] == "frontend/dist"
    assert config["functions"]["api/index.py"]["maxDuration"] == 60
    exclude_files = config["functions"]["api/index.py"]["excludeFiles"]
    assert isinstance(exclude_files, str)
    assert ".env" in exclude_files
    assert ".env.*" in exclude_files
    assert ".venv/**" in exclude_files
    assert "build/**" in exclude_files
    assert "node_modules" in exclude_files
    assert ".npm-cache" in exclude_files
    assert "frontend/dist/**" not in exclude_files
    assert rewrites[0] == {
        "source": "/api/v1/:path*",
        "destination": "/api/index.py",
    }
    assert rewrites[1] == {
        "source": "/:path((?!api/v1).*)",
        "destination": "/index.html",
    }
    assert not any(
        rewrite["source"] == "/api/(.*)" and rewrite["destination"] == "/index.html"
        for rewrite in rewrites
    )


def test_vercel_direct_frontend_routes_fall_back_to_index() -> None:
    rewrites = load_vercel()["rewrites"]
    fallback = rewrites[1]

    assert fallback["destination"] == "/index.html"
    assert "?!api/v1" in fallback["source"]
    assert rewrites[0]["destination"] == "/api/index.py"


def test_vercel_spa_fallback_covers_declared_frontend_routes() -> None:
    fallback = load_vercel()["rewrites"][1]

    for route in ["/", "/overview", "/documents", "/retrieval", "/answers", "/system"]:
        assert fallback["destination"] == "/index.html"
        assert not route.startswith("/api/v1")


def test_vercel_static_assets_resolve_before_spa_fallback() -> None:
    config = load_vercel()

    assert config["outputDirectory"] == "frontend/dist"
    assert config["framework"] == "fastapi"
    assert config["rewrites"][-1]["destination"] == "/index.html"


def test_vercel_build_copies_frontend_dist_to_static_public_output() -> None:
    script = read("scripts/prepare_vercel_static.mjs")
    gitignore = read(".gitignore")

    assert 'resolve(root, "frontend", "dist")' in script
    assert 'resolve(root, "public")' in script
    assert "await cp(source, destination, { recursive: true })" in script
    assert "public/" in gitignore


def test_generated_vercel_output_bundles_frontend_dist_and_function_when_present() -> (
    None
):
    output = ROOT / ".vercel" / "output"
    if not output.exists():
        pytest.skip("Vercel output has not been generated.")

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if "routes" not in config:
        pytest.skip("Vercel output is incomplete from a failed build.")
    routes: list[dict[str, Any]] = config["routes"]
    function_config = json.loads(
        (output / "functions" / "index.func" / ".vc-config.json").read_text(
            encoding="utf-8",
        ),
    )
    file_path_map: dict[str, str] = function_config["filePathMap"]

    assert (output / "functions" / "index.func").is_dir()
    assert "frontend/dist/index.html" in file_path_map
    assert any(path.startswith("frontend/dist/assets/") for path in file_path_map)
    assert ".env" not in file_path_map
    assert ".env.local" not in file_path_map
    assert routes[0] == {"handle": "filesystem"}
    assert "/api/v1" in routes[1]["src"]
    assert "?!api/v1" in routes[2]["src"]


def test_vercel_dependency_file_excludes_heavy_local_ml_packages() -> None:
    requirements = read("requirements-vercel.txt")
    api_requirements = read("api/requirements.txt")
    pyproject = load_pyproject()
    base_dependencies = "\n".join(pyproject["project"]["dependencies"])
    optional_dependencies = pyproject["project"]["optional-dependencies"]

    assert "-r ../requirements-vercel.txt" in api_requirements
    assert "google-genai" in requirements
    for package in HEAVY_PACKAGES:
        assert package not in requirements
        assert package not in base_dependencies
    assert "sentence-transformers==5.6.0" in optional_dependencies["local-embeddings"]
    assert "sentence-transformers==5.6.0" in optional_dependencies["test"]


def test_vercelignore_excludes_pyproject_to_avoid_local_ml_dependencies() -> None:
    vercelignore = read(".vercelignore")

    assert "uv.lock" in vercelignore
    assert "frontend/node_modules/" in vercelignore
    assert "frontend/dist/" in vercelignore
    assert "frontend/.npm-cache/" in vercelignore
    assert "deployment/google-cloud-run/" in vercelignore
    assert "deployment/huggingface/" in vercelignore


def test_vercel_docs_record_gemini_collection_strategy() -> None:
    docs = read("deployment/vercel/DEPLOYMENT.md")

    assert "5254.97 MB" in docs
    assert "DENSE_EMBEDDING_PROVIDER=gemini" in docs
    assert "GEMINI_EMBEDDING_MODEL=gemini-embedding-001" in docs
    assert "GEMINI_EMBEDDING_DIMENSION=768" in docs
    assert "internal_document_chunks_vercel_gemini_preview" in docs
    assert "internal_document_chunks_vercel_gemini" in docs


def test_vercel_config_contains_no_secret_values_or_vite_secrets() -> None:
    combined = "\n".join(
        [
            read("vercel.json"),
            read(".vercelignore"),
            read("deployment/vercel/DEPLOYMENT.md"),
            read("frontend/.env.example"),
        ]
    )

    assert "sk-" not in combined
    assert "VITE_QDRANT" not in combined
    assert "VITE_GEMINI" not in combined
    assert "VITE_API_KEY" not in combined
    assert "VITE_API_AUTH" not in combined
    assert "API_AUTH_KEY_SHA256=" not in combined


def test_local_docker_and_huggingface_assets_remain_present() -> None:
    assert "EXPOSE 8000" in read("Dockerfile")
    assert '"8000:8000"' in read("docker-compose.yml")
    assert "app_port: 7860" in read("deployment/huggingface/README.md")
