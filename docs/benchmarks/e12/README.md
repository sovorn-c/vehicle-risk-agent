# Policy retrieval benchmark

This benchmark compares a simple PostgreSQL keyword baseline with the application's hybrid retrieval path. It uses actual local embedding and reranking models and PostgreSQL with pgvector. It does not make paid LLM calls.

## Recorded result

Run [`20261018T121928Z`](retrieval/runs/20261018T121928Z/results.json) completed all 20 evaluation questions for both methods without technical errors. The [CSV](retrieval/runs/20261018T121928Z/per-query.csv) contains 40 method/question rows.

| Metric | Keyword | Hybrid | Hybrid target |
| --- | ---: | ---: | ---: |
| Recall@5 | 0.0667 | 0.9333 | ≥0.90 |
| MRR@5 | 0.0667 | 0.9667 | ≥0.80 |
| Correct unsupported abstentions | 5/5 | 5/5 | ≥4/5 |
| False abstentions on answerable questions | 14/15 | 0/15 | Descriptive |
| Precision@5 | 0.0133 | 0.2133 | Descriptive |
| Citation identity rate | 1.00 | 1.00 | 1.00 |

Recall and MRR use the 15 answerable questions. Unsupported abstention uses five questions. Precision uses five result slots, giving no credit for empty slots. Its label-dependent aggregate ceiling is 0.24. Citation identity checks passage IDs, not whether a generated legal claim is supported.

Hybrid met the declared retrieval targets. This result does not change the separate end-to-end live release verdict.

## Inputs and methods

- [Corpus manifest](corpus/manifest.json): selected material from four official NZ legislative sources, with URLs, source dates, reuse terms, hashes, and model pins.
- [Dataset](retrieval/dataset-v1.json): six development questions and 20 evaluation questions, including 15 answerable and five unsupported questions. Each answerable question lists relevant passage IDs.
- Keyword retrieval uses English PostgreSQL full-text search and `plainto_tsquery`, ranked with `ts_rank_cd`. All remaining query terms must match. No query expansion, disjunctive fallback, or lexical tuning was added for this run.
- Hybrid retrieval combines dense pgvector and keyword candidates with Reciprocal Rank Fusion, then cross-encoder reranking and configured abstention thresholds.

Both methods use the same passage corpus, evaluation questions, and final top-k=5. The JSON records corpus and dataset fingerprints, model revisions, retrieval settings, targets, and Python/platform metadata. It does not record a source commit SHA; the revision wiring fix was uncommitted when this run executed. The revision fix is commit [c909745](https://github.com/sovorn-c/vehicle-risk-agent/commit/c909745).

The models are `sentence-transformers/all-MiniLM-L6-v2` at revision `8b3219a92973c328a8e22fadcfa821b5dc75636a` and `cross-encoder/ms-marco-MiniLM-L-6-v2` at revision `739bce82df32cacea8ff0edf73ab49ae315e5a5f`.

## Reproduce

Use Python 3.12, uv, Docker, and the repository checkout. Start the PostgreSQL service:

```bash
docker compose up -d db
```

The checks use a dedicated database ending in `_test`, separate from the runtime database. They recreate its policy tables. The default is `vehicle_risk_agent_test` on localhost port 54329. A custom URL uses `VEHICLE_RISK_AGENT_TEST_DATABASE_URL`.

Run the actual model-backed checks:

```bash
RUN_E12_CORPUS_SMOKE=1 RUN_E12_RETRIEVAL_BENCHMARK=1 \
  uv run --extra ml pytest \
  tests/integration/test_portfolio_corpus_smoke.py \
  tests/integration/test_portfolio_retrieval_benchmark.py -q
```

Both tests must pass rather than skip. PostgreSQL must be reachable; first execution may download the pinned models. The benchmark writes a new timestamped directory under `docs/benchmarks/e12/retrieval/runs/` and preserves earlier runs.

For the normal credential-free verification suite:

```bash
bash scripts/check.sh
```

The two model-backed checks are opt-in and skipped by default. A passing default suite alone does not prove that retrieval was measured.

For application ingestion with pinned models, set `EMBEDDING_REVISION` and `RERANKER_REVISION` to the matching manifest revisions before creating source snapshots. Defaults remain `main` for compatibility. Corpus activation rejects a mismatch between snapshot embeddings and the declared corpus configuration.

## What the result means

Hybrid retrieved substantially more labelled relevant passages than this simple conjunctive keyword baseline on these questions. The baseline's 14 false abstentions explain much of the gap. This does not compare hybrid against a tuned lexical search system.

The corpus and labels are curated and author-reviewed. The run has 20 questions, uses no independent legal review, and supports no statistical significance claim. Two sources contain section headings only, which support discovery but not interpretation of missing statutory provisions.

Unsupported questions test absence from this corpus, not the absence of applicable law. Returning a valid passage ID does not prove claim support. This benchmark does not measure report quality, model drafting latency/cost, or the complete vehicle-assessment workflow.

The next assessment-quality evidence is a live comparison using the sourced corpus, followed by eight bound human semantic reviews. Vehicle facts remain synthetic and the application retains human approval.
