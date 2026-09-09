# Vehicle Risk Assessment Agent

### An auditable decision workflow—not a vehicle lookup API.

[![CI](https://github.com/sovorn-c/vehicle-risk-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/sovorn-c/vehicle-risk-agent/actions/workflows/ci.yml)

This portfolio project turns vehicle evidence and policy into a reviewable risk
assessment. It uses a typed LangGraph workflow, Model Context Protocol (MCP),
policy RAG, deterministic scoring, and mandatory human approval.

> **Important:** This is a reference implementation for demonstration and
> engineering discussion. It does not access live restricted registers and
> must not be used to make production decisions from synthetic data.

## Why this project exists

An MCP server can answer questions about a vehicle. A risk system has to do
more: decide whether the evidence is sufficient, apply a versioned policy,
explain the result, preserve provenance, and stop for human review.

This repository owns that decision layer. It is deliberately separate from
the services that ingest and expose vehicle data.

## Three repositories, three responsibilities

| Repository | Responsibility | Engineering focus |
| --- | --- | --- |
| [`nz-vehicle-data-pipeline`](https://github.com/sovorn-c/nz-vehicle-data-pipeline) | Ingests and reconciles vehicle data | Data pipelines and source authority |
| [`vehicle-mcp-server`](https://github.com/sovorn-c/vehicle-mcp-server) | Exposes typed vehicle-intelligence tools | MCP contracts and service boundaries |
| **`vehicle-risk-agent`** | Produces a policy-grounded, reviewable assessment | Workflow orchestration and decision safety |

The system boundary is:

```text
Pipeline  →  MCP Server  →  Vehicle Risk Agent  →  Human-approved report
  data        capability       decision workflow       outcome
```

The Agent never calls the Pipeline database or API directly. MCP is the only
vehicle-evidence boundary.

## Architecture

```text
                         ┌─────────────────────────┐
                         │  Client / Reviewer       │
                         └────────────┬────────────┘
                                      │ HTTP + SSE
                                      ▼
                         ┌─────────────────────────┐
                         │ FastAPI API boundary     │
                         │ auth, roles, rate limits │
                         │ idempotency, audit       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Typed LangGraph workflow │
                         └──────┬─────────┬────────┘
                                │         │
                    vehicle facts│         │policy passages
                                ▼         ▼
                   ┌────────────────┐  ┌─────────────────┐
                   │ Vehicle MCP    │  │ PostgreSQL +     │
                   │ Server         │  │ pgvector policy  │
                   └────────────────┘  │ knowledge        │
                                       └─────────────────┘
                                ┌──────────────┴──────────────┐
                                ▼                             ▼
                    Deterministic risk engine       Report drafting adapter
                    score and risk band              citations and explanation
                                └──────────────┬──────────────┘
                                               ▼
                                  AWAITING_REVIEW checkpoint
                                               │
                              approve / reject / reinvestigate
                                               ▼
                                    Released report or next run
```

## The assessment lifecycle

Each request becomes a persisted assessment with an explicit run state:

```text
PENDING
  → COLLECTING_EVIDENCE
  → EVALUATING_SUFFICIENCY
      ├─ INCOMPLETE: withhold score and explain missing evidence
      └─ COMPLETE
           → RETRIEVING_POLICY
           → EVALUATING_RISK
           → DRAFTING_REPORT
           → AWAITING_REVIEW
                ├─ RELEASED
                ├─ REJECTED
                └─ IN_PROGRESS: additive reinvestigation, up to three runs
```

The workflow is stateful and interruptible. It is not a single prompt wrapped
in an HTTP endpoint.

## What this demonstrates

### Typed workflow orchestration

- LangGraph state is explicit and validated.
- Nodes are thin, asynchronous, and independently testable.
- Evidence branches merge through deterministic reducers.
- PostgreSQL-backed checkpoints support restart recovery.
- Server-sent events expose safe progress without leaking internals.

### Evidence-grounded reasoning

- MCP responses are validated at the external boundary with Pydantic models.
- Evidence snapshots preserve revision IDs, source observations, timestamps,
  confidence, conflicts, material hashes, and synthetic notices.
- Missing, unavailable, and conflicting evidence remain distinct outcomes.
- No evidence is fabricated when a service fails.

### Policy RAG with abstention

- Policy passages are retrieved with dense search and keyword search.
- Reciprocal Rank Fusion combines candidates before reranking.
- Reports preserve policy source and section citations.
- Weak or empty retrieval produces an explicit abstention instead of an
  invented citation.

### Deterministic risk decisions

The language model can draft prose. It cannot calculate or override the risk
score. The risk engine applies versioned weights and thresholds in ordinary,
testable Python code and records a calculation hash.

### Human approval as a safety boundary

A completed draft is not a released report. An authorized reviewer must
explicitly approve, reject, or request reinvestigation. Concurrent review
actions are serialized and idempotent.

### Deliberate scope

The first release uses one typed workflow, not a swarm of autonomous personas.
The engineering challenge is traceable state transition, evidence integrity, and
safe review—not adding agents for visual complexity.

## Quickstart

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- `curl` for the hosted-MCP reachability check
- Docker and Docker Compose v2.20+

### Option 1: hosted MCP quickstart

This is the fastest way to try the Agent from a standalone clone. It starts
only the Agent API, its PostgreSQL/pgvector database, and the seed job.

```bash
cp .env.example .env
bash scripts/smoke-quickstart.sh
```

The script performs a bounded `tools/list` probe against the default endpoint,
`https://vehicle-mcp.chhlatbot.com/mcp`, then starts
`compose.quickstart.yaml` and runs the Agent smoke journey.

To use another compatible HTTP(S) MCP endpoint:

```bash
QUICKSTART_MCP_SERVER_URL=https://example.invalid/mcp \
  bash scripts/smoke-quickstart.sh
```

The hosted endpoint is optional, may be rate-limited or unavailable, and has
no production SLA. It may expose **synthetic fixtures**. Do not use hosted
results for production decisions or infer access to **live restricted-register data**.

The quickstart never silently switches to fake or local evidence. If the
hosted check fails, clone the sibling repositories and use the full-local path.

### Configuration

Copy `.env.example` for local defaults. The main variables are:

| Variable | Used by | Purpose |
| --- | --- | --- |
| `MCP_SERVER_URL` | Full-local Compose | Local MCP service, normally `http://localhost:8080` |
| `QUICKSTART_MCP_SERVER_URL` | Hosted quickstart | Explicit hosted MCP `/mcp` override |
| `DATABASE_URL` | Local tools | PostgreSQL connection, normally port `54329` |
| `ANTHROPIC_API_KEY` | Optional drafting | Enables live model drafting; offline drafting remains available |

The development bearer tokens in `.env.example` are for local demonstrations
only. Do not use them as production credentials.

### Option 2: full-local Compose

The full-local path is the authoritative reproducible development setup. Clone
the sibling services beside this repository:

```bash
git clone git@github.com:sovorn-c/vehicle-mcp-server.git ../vehicle-mcp-server
git clone git@github.com:sovorn-c/nz-vehicle-data-pipeline.git ../nz-vehicle-data-pipeline
docker compose up -d --build --wait
```

Services:

| Service | Purpose | Local address |
| --- | --- | --- |
| `db` | PostgreSQL 16 with pgvector | `localhost:54329` |
| `pipeline` | Vehicle data pipeline | `localhost:8000` |
| `mcp` | Vehicle Intelligence MCP server | `localhost:8080` |
| `agent-api` | This service | `localhost:8001` |

Check service health:

```bash
curl -f http://localhost:8001/health
curl -f http://localhost:8001/ready
```

## API shape

The public API is asynchronous: create an assessment, observe its run, then
let a reviewer decide what happens to the draft.

```bash
curl -X POST http://localhost:8001/api/v1/assessments \
  -H 'Authorization: Bearer dev-requester-token' \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: demo-assessment-001' \
  -d '{
    "vin": "1HGCR2F85HA000000",
    "context": {
      "sale_type": "DEALER",
      "intended_use": "Evaluate a vehicle before purchase",
      "questions": ["Are there material risk indicators?"]
    }
  }'
```

The response contains an assessment ID. Use it with:

| Operation | Endpoint |
| --- | --- |
| Read lifecycle state | `GET /api/v1/assessments/{assessment_id}` |
| Stream progress | `GET /api/v1/assessments/{assessment_id}/events` |
| Read audit history | `GET /api/v1/assessments/{assessment_id}/history` |
| Read released report | `GET /api/v1/assessments/{assessment_id}/report` |
| Approve draft | `POST /api/v1/assessments/{assessment_id}/review/approve` |
| Reject draft | `POST /api/v1/assessments/{assessment_id}/review/reject` |
| Request reinvestigation | `POST /api/v1/assessments/{assessment_id}/review/reinvestigate` |

Interactive OpenAPI documentation is available at
`http://localhost:8001/docs` while the Agent is running.

## Demonstration scenarios

Run the complete local scenario suite:

```bash
bash scripts/smoke-local.sh
```

Or run the Python smoke runner directly:

```bash
uv run python -m vehicle_risk_agent.cli.smoke
```

The scenarios demonstrate:

1. A clean synthetic vehicle assessment completing normally.
2. A risky synthetic vehicle producing explicit contributing findings.
3. Incomplete evidence withholding the score and risk band.
4. Reviewer approval releasing a report.
5. Reviewer rejection preventing release.
6. Additive reinvestigation creating a new audited run.
7. Idempotent requests, SSE progress, restart behavior, and teardown.

Seed the development database explicitly when needed:

```bash
uv run python -m vehicle_risk_agent.cli.seed
```

## Provenance and safety properties

- Every evidence snapshot is point-in-time, immutable, and integrity-checked.
- Evidence and policy versions are pinned to an assessment run.
- Report claims retain evidence and policy references where available.
- Synthetic data remains labelled through report generation.
- Invalid, unavailable, or incomplete evidence cannot become a scored result.
- Review actions have bounded rationale, role checks, idempotency, and hashes.
- Errors are mapped to safe public categories without stack traces or prompts.
- Telemetry redacts credentials and sensitive identifiers.
- The database remains internal to the quickstart Compose network.

## Security boundaries

The project treats VINs, MCP responses, policy text, prompts, and assessment
context as untrusted input. It uses strict Pydantic validation, bearer-token
roles, bounded intake rate limits, safe error responses, timeouts, and
structured redacted telemetry.

Development tokens in `.env.example` are for local demonstrations only. Replace
them with real secret management and identity controls before any deployment.

## Quality and CI

Run the local seven-stage preflight gate:

```bash
bash scripts/check.sh
```

It checks formatting, Ruff, strict mypy typing, database migrations, Compose
contracts, the full pytest suite, and package building. The current local main
branch passes all stages with **445 tests**.

GitHub Actions runs the same deterministic preflight on pushes and pull
requests. It does not depend on public MCP uptime. An optional live model
evaluation can be triggered manually and requires an `ANTHROPIC_API_KEY`
secret; it is separate from the deterministic gate.

## Repository map

```text
src/vehicle_risk_agent/
├── api/             FastAPI routes, schemas, auth, and review boundaries
├── workflow/        Typed LangGraph state, nodes, and runner
├── evidence/        MCP contracts, snapshots, provenance, and sufficiency
├── policy/          Policy sources, corpora, citations, and lifecycle
├── retrieval/       Dense/keyword retrieval, fusion, and reranking
├── risk/            Versioned deterministic scoring and policy rules
├── reporting/       Grounded report models and drafting adapters
├── review/          Approval, rejection, reinvestigation, and release
├── persistence/     PostgreSQL repositories and event storage
└── evaluation/      Labelled scenarios, graders, and release evidence

tests/               Unit, integration, acceptance, security, and workflow tests
compose.yaml         Full-local four-service stack
compose.quickstart.yaml
                     Standalone hosted-MCP demonstration stack
```

## Limitations

- This is a portfolio/reference implementation, not a production deployment.
- The hosted MCP endpoint is optional, externally dependent, and has no SLA.
- Hosted and seeded scenarios may use synthetic fixtures.
- The system does not access live restricted-register data.
- No report is auto-approved or presented as professional legal, financial, or
  insurance advice.
- Provider, policy, retention, privacy, and operational controls need further
  governance before production use.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| Hosted MCP check fails | Use the documented sibling clone commands and full-local Compose path. |
| `pgvector` is unavailable | Confirm the stack uses `pgvector/pgvector:pg16`. |
| Port `54329` is busy | Stop the conflicting PostgreSQL service or change the full-local mapping. |
| Drafting uses offline mode | Set `ANTHROPIC_API_KEY` only when live model drafting is required. |
| Agent is unhealthy | Check `docker compose ps` and inspect the database/seed health dependencies. |

## Cleanup

Remove the hosted quickstart containers, network, and volume:

```bash
docker compose -p vehicle-risk-agent-quickstart -f compose.quickstart.yaml down -v
```

Remove the full-local stack:

```bash
docker compose down -v
```

## Portfolio talking points

- **MCP integration:** separates vehicle-data capability from decision ownership.
- **LangGraph:** models a resumable, observable workflow rather than a chat loop.
- **RAG discipline:** retrieves and cites policy; it does not treat retrieved text
  as executable instructions.
- **Decision safety:** deterministic scoring, abstention, immutable snapshots,
  and mandatory human approval prevent a fluent model response from becoming an
  unreviewed consequential decision.
- **Production thinking:** authentication, idempotency, concurrency control,
  safe failures, observability, migrations, evaluation, and deterministic CI
  are part of the feature—not afterthoughts.
