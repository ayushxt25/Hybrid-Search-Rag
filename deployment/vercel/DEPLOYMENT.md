# Vercel Full-Stack Deployment

Hybrid Search Studio runs as one Vercel project when dense embeddings use Gemini instead of local Sentence Transformers.

Previous Vercel backend test:

- Local Python embedding bundle: 5254.97 MB
- Vercel Python function limit: 500 MB
- Resolution: Vercel uses `DENSE_EMBEDDING_PROVIDER=gemini` and `requirements-vercel.txt`, which excludes Torch, Transformers, SciPy, tokenizers, and Sentence Transformers.

## Routing

- Frontend: `/`, `/overview`, `/documents`, `/retrieval`, `/answers`, `/system`
- API: `/api/v1/...`
- FastAPI entrypoint: `api/index.py`
- Frontend API base in Vercel: relative `/api/v1`
- Build command: `cd frontend && npm ci --cache /tmp/npm-cache && npm run build`

Local Vite can still use `VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1`.

## Preview Environment

Secrets:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `API_AUTH_KEY_SHA256`
- `GEMINI_API_KEY`

Non-secret Preview values:

- `ENVIRONMENT=production`
- `API_AUTH_ENABLED=true`
- `GENERATION_PROVIDER=deterministic`
- `DENSE_EMBEDDING_PROVIDER=gemini`
- `GEMINI_EMBEDDING_MODEL=gemini-embedding-001`
- `GEMINI_EMBEDDING_DIMENSION=768`
- `GEMINI_EMBEDDING_TIMEOUT_SECONDS=30`
- `QDRANT_HYBRID_COLLECTION_NAME=internal_document_chunks_vercel_gemini_preview`
- `READINESS_ENABLED=true`
- `MAX_DOCUMENT_UPLOAD_BYTES=4000000`
- `VITE_MAX_DOCUMENT_UPLOAD_BYTES=4000000`
- `LOG_LEVEL=INFO`
- `OBSERVABILITY_ENABLED=true`

Production must use `QDRANT_HYBRID_COLLECTION_NAME=internal_document_chunks_vercel_gemini`.

Do not put Qdrant credentials, Gemini keys, API auth digests, or plaintext API keys in `VITE_*` variables.

## Notes

Qdrant Cloud remains the persistent vector store. Vercel local disk is ephemeral and is used only for temporary upload processing. Cached Gemini clients may persist only inside a warm function instance. Rate limiting and document replacement locking remain process-instance-local.

Vercel request and response payloads are limited to 4.5 MB, so the deployment uses `MAX_DOCUMENT_UPLOAD_BYTES=4000000` and `VITE_MAX_DOCUMENT_UPLOAD_BYTES=4000000`.
