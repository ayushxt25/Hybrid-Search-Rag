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
