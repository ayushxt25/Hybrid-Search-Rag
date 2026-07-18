# Hugging Face Docker Space Deployment

This prepares the FastAPI backend only. Do not copy the React frontend into the
Space as a frontend deployment target; deploy it later on Vercel.

## Architecture

- Hugging Face Docker Space runs `app.main:app` on free CPU hardware.
- The image preinstalls CPU-only Torch before app dependencies to avoid CUDA
  wheels on free CPU hardware.
- The container listens on `0.0.0.0:7860`.
- Qdrant runs outside the Space through `QDRANT_URL`.
- Document persistence is Qdrant-backed; do not rely on Space local disk.
- Liveness is lazy: `/api/v1/health/live` does not load embedding models or call
  Qdrant/OpenAI.

## Required Space Secrets

Set these in Space Settings as Secrets:

- `QDRANT_URL`: HTTPS Qdrant Cloud endpoint.
- `QDRANT_API_KEY`: Qdrant Cloud API key.
- `API_AUTH_KEY_SHA256`: SHA-256 digest of the client API key.

Do not store the plaintext client API key in the Space. The backend stores only
the SHA-256 digest.

## Required Space Variables

Set these in Space Settings as Variables:

- `ENVIRONMENT=production`
- `API_AUTH_ENABLED=true`
- `GENERATION_PROVIDER=deterministic`
- `QDRANT_HYBRID_COLLECTION_NAME=internal_document_chunks_hybrid`
- `TRUSTED_HOSTS=["<space-subdomain>.hf.space"]`
- `CORS_ENABLED=true`
- `CORS_ALLOWED_ORIGINS=["<future-vercel-origin>"]`
- `CORS_ALLOW_CREDENTIALS=false`
- `READINESS_ENABLED=true`

Use `GENERATION_PROVIDER=deterministic` for free CPU deployment unless provider
secrets are added later. With deterministic generation, `OPENAI_API_KEY` is not
required.

## CORS And Trusted Hosts

For initial API-only testing, use a localhost origin only if calling from local
browser tooling:

```text
CORS_ALLOWED_ORIGINS=["http://localhost:5173"]
```

For final deployment, set the exact Vercel frontend origin:

```text
CORS_ALLOWED_ORIGINS=["https://<vercel-app>.vercel.app"]
```

Set trusted hosts to the exact Hugging Face Space host:

```text
TRUSTED_HOSTS=["<space-subdomain>.hf.space"]
```

Avoid wildcard CORS origins and wildcard trusted hosts in final deployment.

## Create The Space

1. Create a new Hugging Face Space.
2. Choose **Docker** as the SDK.
3. Choose free CPU hardware.
4. Clone the Space repository.

## Copy Project Files

The Docker build context must contain the repository root files used by the
Dockerfile:

- `pyproject.toml`
- `README.md`
- `backend/app/**`
- `deployment/huggingface/Dockerfile`
- `deployment/huggingface/README.md`

Copy into the Space repository:

```bash
cp -r backend pyproject.toml README.md <space-repo>/
cp deployment/huggingface/Dockerfile <space-repo>/Dockerfile
cp deployment/huggingface/README.md <space-repo>/README.md
```

Then commit and push to the Space repository.

## Build And Test

After pushing, wait for the Space build to finish. Free Spaces sleep after
inactivity, so the first request after sleep may be slow.

Test liveness:

```text
https://<space-subdomain>.hf.space/api/v1/health/live
```

Readiness may fail until Qdrant Cloud URL, API key, collection name, and backend
API authentication settings are correct.
