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
