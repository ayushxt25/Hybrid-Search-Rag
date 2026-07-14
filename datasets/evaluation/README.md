# Retrieval Evaluation Dataset

`retrieval_cases.example.json` is a format example, not a real benchmark. The chunk IDs are fake placeholders.

To build a real dataset, ingest documents and copy real `chunk_id` values from ingestion responses or search responses. Replace the fake IDs in the example with those real chunk IDs.

One query may have multiple relevant chunks. Add every chunk that should count as relevant for that query to `relevant_chunk_ids`.

Save real cases as `datasets/evaluation/retrieval_cases.json` or pass another path to the evaluation CLI.

## Controlled Corpus

The `corpus/` directory contains a synthetic controlled corpus intended for local retrieval evaluation. It contains no confidential information.

Keep the facts in this corpus stable so benchmark results remain comparable over time. Changing corpus text changes deterministic document IDs and chunk IDs, so regenerate the corpus manifest after any corpus edit.

Use `backend/scripts/prepare_evaluation_corpus.py` to index the corpus into the configured hybrid collection and export the real document and chunk IDs needed for golden retrieval cases.

## Golden Dataset Validation And Benchmarking

The golden dataset is `retrieval_cases.json`. Each case includes `document_id` so ownership can be validated against the controlled corpus. Keep those `document_id` values in the dataset even when running global-search evaluation, where the search request intentionally ignores them.

The evaluation CLI supports two modes:

- Document-filtered evaluation is the default. It passes each case `document_id` to the retrieval services and measures ranking inside the known correct document.
- Global evaluation uses `--global-search`. It leaves the golden labels unchanged but searches across the full indexed corpus, which is closer to normal RAG retrieval.

Retain both filtered and global reports. The filtered report helps isolate ranking quality within the known document, while the global report shows whether retrieval can find the right chunks without being told which document owns the answer.

Validate the dataset before every benchmark run:

```powershell
python backend/scripts/validate_evaluation_dataset.py --dataset datasets/evaluation/retrieval_cases.json --manifest reports/evaluation_corpus_manifest.json
```

Run retrieval evaluation after validation:

```powershell
python backend/scripts/evaluate_retrieval.py --dataset datasets/evaluation/retrieval_cases.json --top-k 5 --candidate-limit 20 --output reports/retrieval_evaluation.json
```

Run global retrieval evaluation when you want full-corpus retrieval metrics:

```powershell
python backend/scripts/evaluate_retrieval.py \
  --dataset datasets/evaluation/retrieval_cases.json \
  --top-k 1 \
  --candidate-limit 6 \
  --global-search \
  --output reports/retrieval_global_at_1.json
```

## Fusion Weight Tuning

Current controlled-corpus results show equal-weight RRF performing worse than dense retrieval in some global-search cases. Weighted fusion should therefore be evaluated rather than assumed.

Use the offline fusion-weight comparison tool to reuse one dense and one sparse candidate set per case while comparing multiple RRF weight configurations:

```powershell
python backend/scripts/evaluate_fusion_weights.py \
  --dataset datasets/evaluation/retrieval_cases.json \
  --top-k 1 \
  --candidate-limit 6 \
  --global-search \
  --output reports/fusion_weight_comparison.json
```

Treat tuning results with six candidates as diagnostic, not production proof. Production hybrid weights should only change after the comparison reports are reviewed alongside broader retrieval behavior.

Production hybrid search currently uses weighted RRF with dense weight `1.5` and sparse weight `1.0`. This choice came from the controlled evaluation corpus, which is intentionally small and diagnostic. Re-evaluate these weights after major corpus, chunking, embedding-model, or sparse-encoding changes.

Relevance labels must be manually reviewed because only a human can decide which chunk contains the clearest complete answer. Do not assign labels only because a heading or filename seems related.

If the corpus changes, regenerate `reports/evaluation_corpus_manifest.json`, rebuild `retrieval_cases.json` with the new deterministic document and chunk IDs, then rerun validation before comparing benchmark results.
