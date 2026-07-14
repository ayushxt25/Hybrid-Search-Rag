## Retrieval And Context Assembly

Hybrid retrieval ranks candidate chunks with dense and sparse search. The
context assembly layer then converts those ranked chunks into a deterministic,
citation-ready context package before any generation step.

Context assembly selects complete chunks under a fixed character budget. Chunk
text is never cut, metadata headers and separators count toward the budget, and
source numbers are assigned before generation so answers can cite stable source
numbers. The LLM connection is intentionally a later stage; this layer only
prepares bounded structured context.
