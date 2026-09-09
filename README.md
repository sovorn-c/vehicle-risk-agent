# Vehicle Risk Assessment Agent

[![CI](https://github.com/sovorn-c/vehicle-risk-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/sovorn-c/vehicle-risk-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful%20Orchestration-1C3C3C)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791?logo=postgresql&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-445%20passing-success)

**A policy-grounded risk assessment service that evaluates vehicle evidence through typed LangGraph workflows, deterministic scoring, and human review gates.**

An MCP server can answer questions about a vehicle. A consequential risk system must do more: verify whether evidence is legally and empirically sufficient, retrieve and cite applicable policies, calculate deterministic risk scores without LLM hallucination, generate an auditable 9-section report, and halt at human approval checkpoints before release.

This service owns that decision layer. It sits downstream from data ingestion and MCP tool protocols to deliver an auditable, human-in-the-loop assessment workflow.

> [!IMPORTANT]
> This is a reference implementation for demonstration and engineering evaluation. It operates on synthetic fixtures, does not access live restricted-register data, and must not be used to make actual legal, financial, or purchase decisions.

---

## Three repositories, three responsibilities

| Repository | Responsibility | Engineering focus |
|---|---|---|
| [`nz-vehicle-data-pipeline`](https://github.com/sovorn-c/nz-vehicle-data-pipeline) | Ingests and reconciles vehicle data across disparate sources | Data pipelines, source authority, and canonical revisions |
| [`vehicle-mcp-server`](https://github.com/sovorn-c/vehicle-mcp-server) | Exposes typed vehicle-intelligence tools over stdio and Streamable HTTP | MCP contracts, transport parity, and safe service boundaries |
| **`vehicle-risk-agent`** | Produces policy-grounded, reviewable vehicle risk assessments | Typed workflow orchestration, policy RAG, deterministic scoring, and decision safety |

```text
Pipeline  ──>  MCP Server  ──>  Vehicle Risk Agent  ──>  Human-Approved Report
  data          capability          decision workflow               outcome
```

The Agent never queries the pipeline database or API directly. The MCP server remains the only vehicle-evidence boundary.

---

## Scenario guide

The test manifest seeds four standard scenarios that exercise each branch of the assessment workflow:

| Scenario | VIN | Input context | Expected outcome | Review action |
|---|---|---|---|---|
| **Clean** | `1HGCR2F85HA000000` | Dealer purchase | `LOW_RISK` (Score: 12/100). Full evidence available, no register matches. | Reviewer `APPROVE` releases report. |
| **Risky** | `1FA6P8CF8H5000000` | Private purchase | `HIGH_RISK` (Score: 88/100). Active PPSR interest, stolen listing, and statutory write-off. | Reviewer `REJECT` blocks release. |
| **Incomplete evidence** | `JM0BL10F000000000` | Auction evaluation | `INCOMPLETE` (Score withheld). Missing required register records; policy sufficiency rule triggers. | Reviewer `APPROVE` requires explicit missing-evidence waiver. |
| **Reinvestigation** | `1HGCR2F85HA000000` | Re-check registers | Run 1 flagged -> Reviewer requests targeted reinvestigation -> Run 2 created and completed. | Reviewer `APPROVE` releases Run 2 report. |

### Inspect an assessment report

When an assessment completes, the resulting draft contains nine structured sections with attribution citations, risk band, and synthetic notices:

```json
{
  "id": "asmt_01J8N2Q1X...",
  "vin": "1HGCR2F85HA000000",
  "lifecycle_state": "RELEASED",
  "current_run_number": 1,
  "report": {
    "status": "RELEASED",
    "risk_score": 12,
    "risk_band": "LOW_RISK",
    "assessment_outcome": "SCORED",
    "sections": [
      {
        "section_type": "EXECUTIVE_SUMMARY",
        "title": "Executive Summary",
        "content": "Vehicle 1HGCR2F85HA000000 exhibits low overall risk based on available evidence..."
      },
      {
        "section_type": "MANDATORY_REVIEW_FINDINGS",
        "title": "Mandatory Review Findings",
        "content": "No statutory write-off, active stolen listing, or registered security interest detected."
      },
      {
        "section_type": "POLICY_CITATIONS",
        "title": "Applicable Policy Citations",
        "citations": [
          {
            "citation_id": "pol_ppsr_clearance_v1",
            "source_title": "PPSR Security Clearance Policy",
            "section_reference": "Section 3.2: Clean Encumbrance Verification"
          }
        ]
      },
      {
        "section_type": "SYNTHETIC_DATA_NOTICE",
        "title": "Synthetic Data Notice",
        "content": "This record represents no real vehicle, person, police report, insurance decision, or financial obligation."
      }
    ]
  }
}
```

---

## Engineering decisions

| Decision | Rationale |
|---|---|
| **Deterministic scoring outside the LLM** | LLMs cannot calculate or adjust risk scores. Versioned weights, thresholds, and calculation hashes run in pure, testable Python code to guarantee auditability. |
| **Mandatory human-in-the-loop approval** | Consequential risk assessments are never auto-released. Authorized reviewers must explicitly approve, reject, or request reinvestigation via serialized, idempotent endpoints. |
| **Stateful LangGraph checkpointing** | LangGraph state transitions are strictly validated with Pydantic and persisted to PostgreSQL. This enables interruptible review checkpoints, multi-run reinvestigations, and crash recovery. |
| **Hybrid Policy RAG with abstention** | Combines dense vector search (`pgvector`) and BM25 keyword search using Reciprocal Rank Fusion (RRF). Weak retrieval produces explicit abstention rather than hallucinated citations. |
| **Sufficiency gate before risk scoring** | Missing or conflicting required evidence halts progression at `EVALUATING_SUFFICIENCY`, withholding scoring (`INCOMPLETE`) instead of assuming an unverified vehicle is clean. |
| **Strict MCP protocol isolation** | The agent consumes vehicle intelligence strictly through MCP tool calls (`lookup_vehicle`, `explain_vehicle_field`). It has zero access to pipeline databases or internals. |

---

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
                    vehicle facts│         │ policy passages
                                ▼         ▼
                   ┌────────────────┐  ┌─────────────────┐
                   │ Vehicle MCP    │  │ PostgreSQL +     │
                   │ Server         │  │ pgvector policy  │
                   └────────────────┘  │ knowledge        │
                                       └─────────────────┘
                                ┌──────────────┴──────────────┐
                                ▼                             ▼
                    Deterministic risk engine       Report drafting adapter
                    score and risk band             citations and explanation
                                └──────────────┬──────────────┘
                                               ▼
                                  AWAITING_REVIEW checkpoint
                                               │
                               approve / reject / reinvestigate
                                               ▼
                                    Released report or next run
```

### The assessment lifecycle

Each evaluation follows an explicit, interruptible state machine:

```text
PENDING
  → COLLECTING_EVIDENCE
  → EVALUATING_SUFFICIENCY
      ├─ INCOMPLETE: withhold score, generate missing-evidence disclosure
      └─ COMPLETE
           → RETRIEVING_POLICY
           → EVALUATING_RISK
           → DRAFTING_REPORT
           → AWAITING_REVIEW
                ├─ RELEASED: reviewer approved
                ├─ REJECTED: reviewer rejected
                └─ IN_PROGRESS: additive reinvestigation (up to 3 audited runs)
```

---

## Quickstart

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose v2.20+
- `curl`

### Option 1: Hosted MCP quickstart (fastest)

The hosted quickstart evaluates the Agent against the hosted MCP demonstration endpoint. It boots only the Agent API, PostgreSQL with `pgvector`, and the seed worker:

```bash
cp .env.example .env
bash scripts/smoke-quickstart.sh
```

The script executes a bounded `tools/list` probe against `https://vehicle-mcp.chhlatbot.com/mcp`, launches `compose.quickstart.yaml`, and runs the full live smoke journey.

To point the quickstart at another compatible HTTP MCP endpoint:

```bash
QUICKSTART_MCP_SERVER_URL=https://example.invalid/mcp \
  bash scripts/smoke-quickstart.sh
```

> [!NOTE]
> The hosted endpoint is optional and provides **synthetic fixtures** without production SLAs. It does not access **live restricted-register data**. If the hosted probe fails, clone the sibling repositories and use the **full-local** Compose path below.

### Option 2: Full-local Compose (authoritative)

To run all three layers locally without external network dependencies, clone the sibling services side-by-side:

```bash
git clone git@github.com:sovorn-c/vehicle-mcp-server.git ../vehicle-mcp-server
git clone git@github.com:sovorn-c/nz-vehicle-data-pipeline.git ../nz-vehicle-data-pipeline
docker compose -f compose.yaml up -d --build --wait
```

#### Services and local ports

| Service | Purpose | Local address |
|---|---|---|
| `db` | PostgreSQL 16 with pgvector (psycopg driver) | `localhost:54329` |
| `pipeline` | NZ Vehicle Data Pipeline | `localhost:8000` |
| `mcp` | Vehicle Intelligence MCP server | `localhost:8080` |
| `agent-api` | Vehicle Risk Assessment Agent | `localhost:8001` |

Check service health:

```bash
curl -f http://localhost:8001/health
curl -f http://localhost:8001/ready
```

---

## API reference

The API uses role-based bearer tokens, strict request schemas, and idempotency keys on mutating routes.

### Create an assessment

```bash
curl -X POST http://localhost:8001/api/v1/assessments \
  -H 'Authorization: Bearer dev-requester-token' \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: demo-asmt-001' \
  -d '{
    "vin": "1HGCR2F85HA000000",
    "context": {
      "sale_type": "DEALER",
      "intended_use": "Evaluate a vehicle before purchase",
      "questions": ["Are there material risk indicators?"]
    }
  }'
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/assessments` | Create a new risk assessment (`Idempotency-Key` required) |
| `GET` | `/api/v1/assessments/{id}` | Retrieve lifecycle status, run history, and current phase |
| `GET` | `/api/v1/assessments/{id}/events` | Stream real-time run progress via Server-Sent Events (SSE) |
| `GET` | `/api/v1/assessments/{id}/history` | Audit trail of all runs, state transitions, and reviewer actions |
| `GET` | `/api/v1/assessments/{id}/report` | Retrieve the final released report (404 until approved) |
| `POST` | `/api/v1/assessments/{id}/review/approve` | Authorize and release a completed report draft |
| `POST` | `/api/v1/assessments/{id}/review/reject` | Reject a draft with documented rationale |
| `POST` | `/api/v1/assessments/{id}/review/reinvestigate` | Trigger an additive re-run with targeted evidence questions |

Interactive OpenAPI documentation is served at [http://localhost:8001/docs](http://localhost:8001/docs).

---

## Verification and smoke testing

### Preflight quality gate

Run the seven-stage verification gate before committing:

```bash
bash scripts/check.sh
```

The gate enforces code formatting, Ruff linting, strict mypy type checking, database migrations, Compose contracts, the full pytest test suite (**445 passing tests**), and package building.

### Automated smoke journeys

Run the end-to-end integration smoke runner against a running local stack:

```bash
bash scripts/smoke-local.sh
```

Or invoke the Python test runner directly:

```bash
uv run python -m vehicle_risk_agent.cli.smoke
```

Seed the database with versioned policies and test scenarios:

```bash
uv run python -m vehicle_risk_agent.cli.seed
```

---

## Provenance and safety properties

- **Immutable evidence snapshots:** Every MCP observation is hashed and frozen at collection time. Assessments evaluate point-in-time snapshots, never mutating live state.
- **Strict deterministic scoring:** Scoring weights and thresholds are versioned. Calculation hashes ensure reproducibility and prevent arbitrary score inflation.
- **Traceable policy citations:** Analytical statements link directly to policy IDs, section references, and source observation IDs.
- **Human approval safety boundary:** Unreviewed reports remain in `AWAITING_REVIEW`. Consequential assessment decisions require active human authorization.
- **Safe error boundaries and telemetry:** Exceptions, prompts, and stack traces never leak across API responses. Telemetry enforces automatic credential and VIN redaction.

---

## Security boundaries

- Untrusted input (VINs, context payloads, policy text, MCP tool results) is validated against strict Pydantic schemas.
- Role-based authorization isolates `requester` roles from `reviewer` approval routes.
- Rate-limiting and request-size bounds protect intake endpoints.
- Database access uses least-privilege configurations with parameterized queries.
- Development tokens in `.env.example` are strictly for local testing.

---

## Limitations

- **Reference implementation:** Designed for demonstration, technical evaluation, and portfolio audit. Not an active commercial vehicle registry.
- **Synthetic data notice:** Operates on synthetic fixtures; does not connect to live NZTA, PPSR, or Police databases.
- **No professional advice:** Output reports do not constitute formal legal, financial, or mechanical inspection advice.
- **Hosted MCP availability:** The hosted demo endpoint operates without an SLA; local Compose remains the authoritative fallback.

---

## Troubleshooting

| Symptom | Resolution |
|---|---|
| Hosted MCP probe fails | Run `bash scripts/smoke-local.sh` using the full-local Compose stack. |
| `pgvector` extension missing | Verify the database container runs `pgvector/pgvector:pg16`. |
| Port `54329` collision | Free local port `54329` or update `.env` and `compose.yaml` port bindings. |
| Report uses offline drafting fallback | Set `ANTHROPIC_API_KEY` in `.env` if live LLM drafting is desired; offline drafting remains default. |
| Agent API reports unhealthy | Inspect logs with `docker compose logs agent-api` and verify database migrations ran cleanly. |

---

## Cleanup and teardown

Remove the hosted quickstart stack, volumes, and networks:

```bash
docker compose -p vehicle-risk-agent-quickstart -f compose.quickstart.yaml down -v
```

Remove the full-local stack:

```bash
docker compose -f compose.yaml down -v
```

---

## Architectural guarantees

- **MCP service boundary:** Isolates upstream vehicle intelligence from the assessment decision engine.
- **LangGraph orchestration:** Coordinates multi-stage, stateful, resumable workflows rather than unbounded prompt loops.
- **Grounded policy RAG:** Enforces evidence sufficiency and policy citations; abstains rather than inventing findings.
- **Auditable review governance:** Deterministic calculations, immutable snapshots, and mandatory human checkpoints prevent unreviewed automated decisions.
- **Enterprise rigor:** Strict typing, idempotency, concurrency controls, safe error boundaries, and 445 automated tests built into core implementation.
