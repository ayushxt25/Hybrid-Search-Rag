## Retrieval And Context Assembly

Hybrid retrieval ranks candidate chunks with dense and sparse search. The
context assembly layer then converts those ranked chunks into a deterministic,
citation-ready context package before any generation step.

Dense, sparse, hybrid, and grounded-answer retrieval support scoped retrieval
with the legacy `document_id` field, `document_ids` arrays, and `content_types`
arrays. `document_id` is merged into `document_ids` when both are supplied, with
duplicates removed. Supported content types are `text/plain`, `text/markdown`,
`application/pdf`, and
`application/vnd.openxmlformats-officedocument.wordprocessingml.document`.
Filters are applied inside Qdrant before ranking and fusion; results are not
post-filtered in application code. Unknown but valid document IDs return normal
empty result sets. Actual document IDs and content-type values are not logged.
Requests may include up to 20 document IDs and 10 content types.

Dense search:

```json
{
  "query": "remote work policy",
  "document_ids": ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
  "content_types": ["text/plain"]
}
```

Sparse search:

```json
{
  "query": "remote policy",
  "document_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "content_types": ["text/markdown"]
}
```

Hybrid search:

```json
{
  "query": "travel reimbursement",
  "document_ids": [
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  ],
  "content_types": ["application/pdf"]
}
```

Set `include_score_diagnostics` to `true` on dense, sparse, or hybrid search
requests to include safe score diagnostics on each result. The default is
`false`, which preserves the existing response shape. Dense diagnostics include
the dense raw score and 1-based dense rank. Sparse diagnostics include the
sparse raw score and 1-based sparse rank; sparse scoring comes from the
deterministic hashed lexical sparse provider, not BM25.

Hybrid diagnostics include dense and sparse branch raw scores, branch ranks,
active branch weights, weighted-RRF contributions, fused score, and fused rank.
The contribution formula is `branch_weight / (rrf_k + branch_rank)`, and the
fused score is the sum of branch contributions. Dense and sparse raw scores are
not directly comparable; weighted RRF ranks branch result order rather than
normalizing raw scores. Fused score is not a probability or confidence value.
Higher fused score means stronger combined ranking evidence, and absence from
one branch does not exclude a result.

```json
{
  "query": "remote work policy",
  "include_score_diagnostics": true
}
```

```json
{
  "score": 0.0407,
  "score_diagnostics": {
    "dense": {
      "raw_score": 0.82,
      "rank": 2,
      "weight": 1.5,
      "rrf_contribution": 0.0242
    },
    "sparse": {
      "raw_score": 0.54,
      "rank": 1,
      "weight": 1.0,
      "rrf_contribution": 0.0164
    },
    "fused_score": 0.0407,
    "fused_rank": 1
  }
}
```

Diagnostics exclude embeddings, dense vectors, sparse vectors, raw Qdrant
payloads, prompts, and other sensitive internals. Grounded-answer responses do
not expose score diagnostics.

Context assembly selects complete chunks under a fixed character budget. Chunk
text is never cut, metadata headers and separators count toward the budget, and
source numbers are assigned before generation so answers can cite stable source
numbers. The LLM connection is intentionally a later stage; this layer only
prepares bounded structured context.

## Grounded Prompt Construction

The generation path is staged as retrieval, context assembly, prompt
construction, and future generation. Prompt construction does not call an LLM;
it renders deterministic system and user prompts from the user question and the
assembled context.

Retrieved document text is treated as untrusted evidence, not executable
instructions. Source markers are assigned before generation, insufficient
context is encoded explicitly when no sources are available, and general
knowledge is disabled by default.

## Provider-Neutral Generation

The full answer path is retrieval, context assembly, prompt construction,
provider-neutral generation, and grounded answer result. The generation provider
interface is isolated from retrieval, Qdrant, FastAPI, and context assembly so a
real provider can be added later without changing the earlier stages.

When no context sources are assembled, the answer service skips model invocation
and returns a deterministic insufficient-context answer. Character counts are
used until provider token accounting exists. Citation sources are returned
structurally with the answer, but citation parsing and correctness checking are
future work. No external generation provider is connected yet.

## Citation Marker Validation

Grounded answer results distinguish structured citations from emitted citation
markers. Structured citations list the available evidence sources supplied to
the model, while `citation_markers` records the `[Source N]` references that the
model actually emitted in the answer text.

Generated answers reject unknown or malformed source markers before being
returned. This validation checks marker syntax and source availability only; it
does not yet prove that a cited sentence is factually entailed by the cited
source. Factual citation verification remains a later stage.

## Grounded Answer API

`POST /api/v1/answers/grounded` accepts `question`, `limit`,
`candidate_limit`, optional `document_id`, `document_ids`, and `content_types`,
then returns a grounded answer with structured `citations`, emitted
`citation_markers`, context metadata, and model metadata. If retrieval finds no
evidence, the service returns the deterministic insufficient-context answer
without invoking OpenAI.

```json
{
  "question": "What is the remote work policy?",
  "document_ids": ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
  "content_types": ["text/plain"]
}
```

Only grounded-answer generation is rate limited. By default, each client
address may make 10 requests per 60 seconds. Responses include
`X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`; denied
requests return `429` with `Retry-After`. The limiter is in-memory, resets on
process restart, and is suitable for local or single-process deployment only.
Redis or distributed limiting is future work.

`OPENAI_API_KEY` is required only when generation is actually invoked. Provider
and network errors are returned with sanitized API details; prompts, retrieved
context, keys, raw provider responses, and raw provider exception messages are
not exposed. OpenAI SDK timeout is application-configurable, SDK retries default
to `2`, and setting retries to `0` disables SDK retries. Authentication failures
and local validation failures are not manually retried, and there is no custom
retry loop beyond the official SDK behavior. Streaming is not implemented.

Run `python backend/scripts/grounded_answer_smoke_test.py` for a local
grounded-answer smoke test. Docker/Qdrant must already be running. The script
uses the real local ingestion, embedding, Qdrant retrieval, context assembly,
prompting, answer route, and citation validation pipeline, but replaces OpenAI
with a deterministic in-process stub, so no API key or generation credits are
used. Success validates citation markers and source metadata.

## Docker-Backed Acceptance Test

`python scripts/acceptance/run_acceptance.py` runs a Docker-backed acceptance
workflow against the public HTTP API and real Qdrant service. It verifies
liveness, TXT/Markdown/PDF/DOCX ingestion, document listing/detail,
dense/sparse/hybrid retrieval, score diagnostics, metadata filters, document
replacement, deletion, grounded answers, and cleanup.

Prerequisites: Docker Compose services must be running and reachable at
`ACCEPTANCE_API_BASE_URL` or `http://127.0.0.1:8000`. OpenAI is not required
when the API is started with `GENERATION_PROVIDER=deterministic`; that provider
is acceptance-only and never enabled by default.

Recommended isolated startup:

```bash
QDRANT_HYBRID_COLLECTION_NAME=internal_document_chunks_acceptance \
GENERATION_PROVIDER=deterministic \
READINESS_ENABLED=false \
docker compose up -d --build
python scripts/acceptance/run_acceptance.py
```

Set `ACCEPTANCE_API_KEY` when API-key authentication is enabled. The runner uses
small deterministic fixtures, creates PDF/DOCX fixtures locally at runtime, and
deletes the documents it creates in best-effort cleanup unless `--keep-data` is
passed. Unit tests do not start Docker; the acceptance workflow is an explicit
end-to-end check. Successful output ends with:

```text
Acceptance workflow passed: 14/14 checks
```

## Frontend Foundation

`frontend/` contains the React foundation for Hybrid Search Studio, a
recruiter-facing developer tool for demonstrating ingestion, retrieval,
diagnostics, grounded answers, and health status. The stack is React,
TypeScript, Vite, Tailwind CSS, React Router, TanStack Query, Lucide React,
Vitest, React Testing Library, ESLint, and Prettier.

Setup:

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Configure the backend URL with:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1
```

Leave `VITE_API_BASE_URL` unset for production-style reverse-proxy builds; the
frontend then uses the relative `/api/v1` path and does not embed localhost or
private network addresses. `VITE_*` values are public build-time values, so do
not place API keys, passwords, provider secrets, or other credentials in them.

Production build:

```bash
npm run build
```

Implemented routes are `/overview`, `/documents`, `/retrieval`, `/answers`,
`/system`, and a not-found route. The Documents route supports listing indexed
documents, uploading TXT, Markdown, PDF, and DOCX files, inspecting safe
metadata, replacing an existing document, and deleting indexed chunks. The
Retrieval Playground runs dense, sparse, and weighted-RRF hybrid searches,
supports document and content-type metadata filters, and can show optional score
diagnostics. Diagnostics expose branch ranks, raw branch scores, weighted-RRF
contributions, fused rank, and fused score; fused score is not a probability or
confidence value, and dense/sparse raw scores are not directly comparable. The
UI labels request timing as client-observed duration and does not display
vectors, sparse indices, Qdrant point IDs, raw payloads, or secrets. The
Grounded Answers route asks questions over indexed documents, supports
document/content-type scoping, renders backend citation markers without
renumbering them, and shows source evidence cards from safe returned citation
metadata. Insufficient-context responses are shown as a deliberate “not enough
evidence” state rather than a crash or fabricated answer. The UI states that
answers are generated by the backend provider configured for the environment;
when the backend returns a provider/model label, it is shown as returned and is
not inferred. Grounded-answer summaries use citation/source counts and
client-observed duration, not confidence percentages. Deployment is the next
major frontend stage.

The document UI depends on the backend API at `VITE_API_BASE_URL` in local Vite
development, or `/api/v1` when built behind the reverse proxy. Uploads use
multipart field `file`; replacements use the same upload endpoint with
`replace_document_id`. The displayed upload limit is the backend default
`MAX_DOCUMENT_UPLOAD_BYTES=10485760` (10 MB). Replacement follows backend
semantics: the new file is parsed first and old indexed chunks are removed before
upsert, but the operation is not a database transaction. Deletion removes indexed
chunks for the selected document. If API-key auth is enabled, the page can keep a
key in memory for the current browser page session only; it is never stored in
localStorage, sessionStorage, or cookies.


## Request Observability

HTTP responses include `X-Request-ID`. A valid incoming request ID is preserved;
missing, blank, unsafe, or overly long IDs are replaced with a generated UUID4.
The API emits JSON request lifecycle events named `api_request_started`,
`api_request_completed`, and `api_error`. Request lifecycle logs contain only
`request_id`, `method`, `path`, `status_code`, `duration_ms`, and `timestamp`.
The grounded-answer pipeline records internal retrieval, context assembly,
prompt construction, generation, and total timings. Logs intentionally exclude
questions, document/context text, answers, prompts, identifiers, secrets, and raw
provider responses. Observability logs can be disabled with
`OBSERVABILITY_ENABLED=false`; timings are internal and are not exposed in API
responses.

## Health Checks

The legacy `GET /api/v1/health` endpoint still returns service metadata.
`GET /api/v1/health/live` is a fast process liveness check and returns
`{"status":"alive"}` without creating external clients. `GET
/api/v1/health/ready` checks Qdrant connectivity and hybrid collection
compatibility, OpenAI generation configuration, and embedding dimension
configuration. Readiness does not call OpenAI, load embedding models, index
documents, or run searches. A `503` response means at least one required
component is not ready. Set `READINESS_ENABLED=false` to return ready with a
`not_configured` readiness component. Container liveness probes should use
`/api/v1/health/live`; readiness probes should use `/api/v1/health/ready`.

## Document Lifecycle

Indexed documents can be managed without a separate database. `GET
/api/v1/documents` lists document metadata, `GET /api/v1/documents/{document_id}`
returns detail metadata, and `DELETE /api/v1/documents/{document_id}`
permanently removes all matching chunks from the configured Qdrant collection.
Management responses do not include chunk text or embedding vectors.

Ingestion remains deterministic: identical content produces the same document ID
and upserts idempotently. To replace a logical document with changed content,
submit the upload with multipart field `replace_document_id`; the backend parses
and embeds the new file first, then deletes stale chunks immediately before
upserting the replacement. Qdrant is not transactional, so a process or Qdrant
failure between delete and upsert can still require re-ingestion. Delete and
replace operations use process-local per-document locks only; multiple API
processes need external coordination. Create a backup before destructive
production operations.

## API Security

Trusted host validation defaults to `localhost`, `127.0.0.1`, and `testserver`;
production deployments should set `TRUSTED_HOSTS` to their public hostnames.
CORS is disabled by default and only configured origins are allowed when enabled.
Responses include conservative security headers for content sniffing, framing,
referrer, permissions, resource policy, and CSP protections. HSTS is
intentionally excluded while local HTTP development is supported.

JSON requests are limited to 256 KiB by default and document uploads to 10 MiB.
Malformed `Content-Length` returns `400`; oversized JSON or document uploads
return `413` with sanitized details.

API-key authentication is disabled by default. When enabled, document ingestion
and grounded-answer generation require the configured header, `X-API-Key` by
default; search routes are also protected unless `API_AUTH_PROTECT_SEARCH=false`.
Health endpoints remain public. The application stores only a SHA-256 digest,
not a plaintext key. Generate a digest locally with
`python -c "import hashlib,getpass; print(hashlib.sha256(getpass.getpass('API key: ').encode()).hexdigest())"`.
Missing or invalid credentials return `401` with `WWW-Authenticate: ApiKey`.
This is intended for service-to-service or local deployment; user accounts and
OAuth/JWT remain future work.

## Security Architecture

Authentication is optional API-key authentication for trusted service or local
deployments. When enabled, the backend accepts the configured header
(`X-API-Key` by default), hashes the provided value with SHA-256, and compares it
with the configured digest using constant-time comparison. Plaintext API keys
are never stored by the backend; frontend API keys are held only in page memory.
Health endpoints stay public, and search protection can be disabled separately
with `API_AUTH_PROTECT_SEARCH=false`.

Request tracing is request-ID based. The backend preserves a safe incoming
`X-Request-ID` or generates a UUID4, attaches it to every response, and uses it
as the only correlation value in production logs. Unexpected errors return a
sanitized `500` response with the request ID, without stack traces, internal
paths, provider details, or raw exception messages.

Logging is structured JSON and uses an allowlist of safe fields. Request logs
exclude Authorization headers, cookies, API keys, IP addresses, filenames,
uploaded content, user questions, answers, prompts, embeddings, vectors, Qdrant
payloads, and raw exceptions. Operational route logs use aggregate counts,
status values, and timing metadata only.

Upload validation is extension and size based before indexing. Supported
extensions are `.txt`, `.md`, `.pdf`, and `.docx`; empty, oversized, encrypted,
corrupt, unsupported, or non-extractable documents receive sanitized errors.
Only the basename of the supplied upload name is used for temporary storage, and
upload bytes are written into an isolated temporary directory.

Container hardening uses multi-stage images, no secret values in Compose, a
non-root backend runtime user, an unprivileged nginx frontend image, explicit
health checks, named Qdrant storage, trusted-host validation, disabled-by-default
CORS, conservative response security headers, and Docker secrets or environment
files for runtime secrets.

## Dependency Lifecycle

Heavy dependencies remain lazy and are not created during application startup.
Owned external clients, such as internally constructed Qdrant and OpenAI
clients, are closed during FastAPI shutdown; injected clients remain
caller-owned. Cached dependencies are cleared after shutdown, and repeated
shutdown calls are safe.

## Docker Compose

Create `.env` from `.env.example`, set external secrets such as `OPENAI_API_KEY`
there, then run:

```bash
docker compose up --build
```

Useful commands:

```bash
docker compose ps
docker compose logs -f api
docker compose logs -f qdrant
docker compose down
docker compose build --no-cache api
```

Use `docker compose down -v` only when intentionally deleting persisted Qdrant
data and the Hugging Face model cache. The API is available at
`http://localhost:8000`; useful endpoints are `/api/v1/health/live`,
`/api/v1/health/ready`, and `/docs`.

Local Python development uses `QDRANT_URL=http://127.0.0.1:6333`; Compose sets
`QDRANT_URL=http://qdrant:6333` for the API container. Qdrant data is persisted
in a named volume. Sentence-transformers models are loaded lazily; the first
embedding request may download the model into the `huggingface_cache` volume and
take longer. The API image runs as a non-root user, does not bind-mount source
code, and uses one Uvicorn worker so in-memory rate limits remain process-local.
Docker health uses liveness; readiness may remain `503` until the Qdrant hybrid
collection exists and is compatible.

## Deployment Preparation

This repository is prepared for deployment configuration, but this section does
not imply any service has been deployed. Use `.env.example` as a template and
provide real values through private environment variables, Docker secrets, or a
deployment secret manager.

For full-stack Vercel deployment with Gemini remote embeddings, see
`deployment/vercel/DEPLOYMENT.md`. Google Cloud Run and Hugging Face notes remain
optional deployment references under `deployment/`.

Required backend environment variables for public deployment:

- `ENVIRONMENT=production`
- `API_AUTH_ENABLED=true`
- `API_AUTH_KEY_SHA256=<sha256 digest of the API key>`
- `TRUSTED_HOSTS=["api.example.com","studio.example.com"]`
- `CORS_ENABLED=true` when the frontend and API are on different origins
- `CORS_ALLOWED_ORIGINS=["https://studio.example.com"]`
- `QDRANT_URL=<qdrant cloud or private service URL>`
- `QDRANT_HYBRID_COLLECTION_NAME=<production collection name>`
- `GENERATION_PROVIDER=openai`
- `OPENAI_API_KEY=<secret-managed provider key>`

Local development can use `API_AUTH_ENABLED=false`, local Qdrant at
`http://localhost:6333`, and Vite with
`VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1`. Production should enable API
authentication, set explicit trusted hosts and origins, use a production Qdrant
instance, and keep provider credentials out of source control.

Frontend build configuration is intentionally small. If the frontend is served
behind the included reverse proxy, leave `FRONTEND_API_BASE_URL` and
`VITE_API_BASE_URL` empty so the built app calls relative `/api/v1`. If the
frontend and API are hosted on separate public origins, set the build-time API
base URL to the public HTTPS API prefix, such as
`https://api.example.com/api/v1`. Do not put secrets in `VITE_*` variables
because they are embedded into the browser bundle.

Backend configuration remains environment-driven. CORS, trusted hosts, API-key
authentication, upload limits, request observability, Qdrant connection details,
and provider settings are parsed from environment variables. Security headers,
API-key digest verification, upload validation, and safe error handling should
remain enabled for production.

Qdrant is required before readiness can pass. Use a production Qdrant service or
a managed private deployment, configure `QDRANT_URL`, and use a dedicated
production hybrid collection. Back up production vector data before destructive
document replacement or deletion workflows.

API authentication is required for public deployment. Store only the SHA-256
digest in `API_AUTH_KEY_SHA256`; distribute the plaintext API key through a
private operational channel or secret manager, never through `.env.example`,
frontend variables, logs, or documentation.

Before deployment:

- [ ] configure environment variables
- [ ] configure frontend API URL
- [ ] configure backend allowed origins
- [ ] configure Qdrant connection
- [ ] verify health endpoints
- [ ] run tests
- [ ] build containers

After deployment:

- [ ] verify frontend routes
- [ ] verify API health
- [ ] verify upload
- [ ] verify retrieval
- [ ] verify grounded answers

## Continuous Integration

GitHub Actions runs on pushes to `main`, pull requests targeting `main`, and
manual dispatch. The CI workflow installs full backend test dependencies with
`python -m pip install -e ".[test]"`, checks formatting with
`python -m ruff format --check backend`, lints with `python -m ruff check backend`,
and runs the full test suite with `python -m pytest` on Python 3.11.

The container job validates `docker compose config` and builds the API image with
`docker build --tag hybrid-search-rag-ci:local .`. It does not push images, use
repository secrets, or require an OpenAI API key. Runtime Compose smoke testing is
kept as a local check to avoid CI depending on external model or service state.

## Production-Style Local Stack

Local Vite development remains unchanged:

```bash
# terminal 1
cp .env.example .env
docker compose up -d qdrant api

# terminal 2
cd frontend
cp .env.example .env
npm install
npm run dev
```

`frontend/.env` uses `VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1` for Vite
at `http://localhost:5173`. Leave it unset for reverse-proxy builds so the app
uses `/api/v1`. Keep secrets out of frontend environment variables; the browser
only needs the public API base URL, and any API key entered in the UI is held in
memory for the current page session only.

The production-style local Compose stack serves the built React app with Nginx
and proxies `/api/` to the Docker API service, so browser calls are same-origin
from `http://localhost:3000` and do not require CORS:

```bash
docker compose up -d --build
```

Open `http://localhost:3000`. Direct navigation and refresh work for `/overview`,
`/documents`, `/retrieval`, `/answers`, and `/system`. The frontend container is
a static Nginx image; it does not run the Vite development server. Qdrant data is
kept in the persistent `qdrant_storage` volume, and the Hugging Face cache remains
in `huggingface_cache`.

Useful local `.env` settings:

```text
GENERATION_PROVIDER=deterministic
QDRANT_HYBRID_COLLECTION_NAME=internal_document_chunks_hybrid
API_AUTH_ENABLED=false
```

Use a dedicated `QDRANT_HYBRID_COLLECTION_NAME` for acceptance or demo data when
isolation is needed. Do not place provider keys or API secrets in frontend env
files. If backend API-key auth is enabled, configure it only through backend
`.env`; the frontend can send a session-only key entered by the user.

The System Health page checks liveness automatically through the existing safe
liveness query. Readiness is manual: click **Check readiness** to inspect safe
component statuses without exposing stack traces, filesystem paths, provider
credentials, vectors, or secrets.

Run the concise full-stack smoke test after the stack is up:

```bash
python scripts/fullstack_smoke.py
```

Expected success ends with:

```text
Full-stack smoke passed: 5/5 checks
```

Shutdown:

```bash
docker compose down
```

Public deployment is the next stage; this stack is for local production-style
verification only.
