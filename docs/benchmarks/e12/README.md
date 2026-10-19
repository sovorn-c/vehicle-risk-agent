# Policy retrieval pipeline check

This benchmark exercises PostgreSQL retrieval with pgvector and pinned local embedding and reranking models on 11 passages from four NZ acts: 2,989 passage-text characters. It compares two lexical query strategies with the application's hybrid path. It does not make paid LLM calls.

## Recorded result

Run [`20261019T052909Z`](retrieval/runs/20261019T052909Z/results.json) completed all 20 evaluation questions for each of three methods without technical errors. The [CSV](retrieval/runs/20261019T052909Z/per-query.csv) contains 60 method/question rows.

| Metric | Keyword (AND) | Keyword (OR) | Hybrid | Declared hybrid target |
| --- | ---: | ---: | ---: | ---: |
| Recall@5 | 0.0667 | 1.0000 | 0.9333 | ≥0.90 |
| MRR@5 | 0.0667 | 0.8500 | 0.9667 | ≥0.80 |
| Correct unsupported abstentions | 5/5 | 0/5 | 5/5 | ≥4/5 |
| False abstentions on answerable questions | 14/15 | 0/15 | 0/15 | Descriptive |
| Precision@5 | 0.0133 | 0.2400 | 0.2133 | Descriptive |
| Citation identity rate | 1.00 | 1.00 | 1.00 | 1.00 |

Recall, MRR and precision use the 15 answerable questions. Unsupported abstention uses five questions. Precision uses five result slots, giving no credit for empty slots. Its label-dependent aggregate ceiling is 0.24. Citation identity checks passage IDs, not whether a generated legal claim is supported.

Hybrid met the declared retrieval targets. This result does not change the separate end-to-end live release verdict.

## Inputs and methods

- [Corpus manifest](corpus/manifest.json): selected material from four official NZ legislative sources, with URLs, source dates, reuse terms, hashes, and model pins. The 11 passages contain 2,989 characters of passage text, excluding separately stored headings.
- [Dataset](retrieval/dataset-v1.json): six development questions and 20 evaluation questions, including 15 answerable and five unsupported questions. Each answerable question lists relevant passage IDs. Labels, corpus and hybrid settings are unchanged from the first run.
- `keyword` (AND) uses English PostgreSQL `plainto_tsquery`, ranked with `ts_rank_cd`. All remaining query terms must match. Its 14 false abstentions make it a weak baseline for these natural-language questions.
- `keyword_or` splits the query into Unicode word tokens, quotes each word, and joins them with `OR` for English `websearch_to_tsquery`. It uses the same `ts_rank_cd` ranking, snapshot scope, candidate limit, passage-ID tie-breaker and final top-k as AND search. Stop words are removed by PostgreSQL. Neither lexical method has a tuned score threshold: it abstains only when there are no matches.
- `hybrid` combines dense pgvector and conjunctive keyword candidates with Reciprocal Rank Fusion, then cross-encoder reranking and configured abstention thresholds. Adding the OR benchmark does not change the application's hybrid search behavior.

All three methods use identical passages, evaluation questions, and final top-k=5. No baseline score threshold was tuned against these evaluation labels.

The JSON uses schema version 2 and records clean source commit [`4aea76a4719c14c7b1193382031ff01b71247e43`](https://github.com/sovorn-c/vehicle-risk-agent/commit/4aea76a4719c14c7b1193382031ff01b71247e43), `dirty: false`, and a provenance timestamp captured before ingestion and measurement. It also records corpus size, corpus/dataset fingerprints, method definitions, model revisions, retrieval settings, targets, and Python/platform metadata. Git status includes tracked and untracked changes; ignored local files are outside this source check. Generated artifacts are published in a subsequent commit.

The models are `sentence-transformers/all-MiniLM-L6-v2` at revision `8b3219a92973c328a8e22fadcfa821b5dc75636a` and `cross-encoder/ms-marco-MiniLM-L-6-v2` at revision `739bce82df32cacea8ff0edf73ab49ae315e5a5f`.

The [first two-method run](retrieval/runs/20261018T121928Z/results.json) is preserved as historical evidence. It predates the OR baseline and source provenance fields and ran with uncommitted revision wiring changes. The new run supersedes it for the README comparison.

## Reproduce

Use Python 3.12, uv, Docker, and a clean repository checkout. The benchmark refuses uncommitted or untracked source changes. To reproduce the recorded implementation exactly, check out the source commit linked above in a separate worktree or clone. Start PostgreSQL:

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

Both tests must pass rather than skip. PostgreSQL must be reachable; first execution may download the pinned models. The benchmark writes a new timestamped directory under `docs/benchmarks/e12/retrieval/runs/` and preserves earlier runs. The recorded three-method run was executed with only `RUN_E12_RETRIEVAL_BENCHMARK=1` and the benchmark test; that test itself ingests and activates the corpus through the API.

For the normal credential-free verification suite:

```bash
bash scripts/check.sh
```

The two model-backed checks are opt-in and skipped by default. A passing default suite alone does not prove that retrieval was measured.

For application ingestion with pinned models, set `EMBEDDING_REVISION` and `RERANKER_REVISION` to the matching manifest revisions before creating source snapshots. Defaults remain `main` for compatibility. Corpus activation rejects a mismatch between snapshot embeddings and the declared corpus configuration.

## What the result means

OR lexical search recovered every labelled relevant passage in its top five. Hybrid had higher MRR but missed one of two relevant passages on each of two questions. The trade-off on the unsupported set is also visible: OR returned matches for all five questions, while hybrid abstained on all five. This comparison does not isolate the contribution of each hybrid component or compare calibrated abstention thresholds across methods.

The search space is small: five results can cover 45% of 11 passages. Uniformly selecting five distinct passages gives expected Recall@5 of 5/11 (approximately 0.455), regardless of how many relevant passages a question has. This is an analytical reference, not an executed random-baseline run.

The corpus and labels are curated and author-reviewed. The Motor Vehicle Sales Act and Personal Property Securities Act sources contain section headings only; the latter contributes a single passage. These support discovery, not interpretation of missing statutory provisions. The run uses no independent legal review and supports no statistical significance or broad retrieval-superiority claim.

Unsupported questions test absence from this corpus, not absence of applicable law. Returning a valid passage ID does not prove claim support. This benchmark exercises ingestion, activation, retrieval, scoring and export; it does not measure report quality, model drafting latency/cost, or the complete vehicle-assessment workflow.

Corpus expansion, the live assessment comparison, and eight bound human semantic reviews remain parked. Vehicle facts remain synthetic and the application retains human approval.
