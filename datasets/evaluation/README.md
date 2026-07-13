# Retrieval Evaluation Dataset

`retrieval_cases.example.json` is a format example, not a real benchmark. The chunk IDs are fake placeholders.

To build a real dataset, ingest documents and copy real `chunk_id` values from ingestion responses or search responses. Replace the fake IDs in the example with those real chunk IDs.

One query may have multiple relevant chunks. Add every chunk that should count as relevant for that query to `relevant_chunk_ids`.

Save real cases as `datasets/evaluation/retrieval_cases.json` or pass another path to the evaluation CLI.
