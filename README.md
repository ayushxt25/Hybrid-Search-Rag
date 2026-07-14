## Retrieval And Context Assembly

Hybrid retrieval ranks candidate chunks with dense and sparse search. The
context assembly layer then converts those ranked chunks into a deterministic,
citation-ready context package before any generation step.

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
`candidate_limit`, and optional `document_id`, then returns a grounded answer
with structured `citations`, emitted `citation_markers`, context metadata, and
model metadata. If retrieval finds no evidence, the service returns the
deterministic insufficient-context answer without invoking OpenAI.

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


## Request Observability

HTTP responses include `X-Request-ID`. A valid incoming request ID is preserved;
missing, blank, unsafe, or overly long IDs are replaced with a generated UUID4.
The grounded-answer pipeline records internal retrieval, context assembly,
prompt construction, generation, and total timings. Logs intentionally exclude
questions, document/context text, answers, prompts, identifiers, secrets, and raw
provider responses. Observability logs can be disabled with
`OBSERVABILITY_ENABLED=false`; timings are internal and are not exposed in API
responses.
