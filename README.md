# Vehicle Risk Assessment Agent

Auditable vehicle risk assessment service combining typed LangGraph orchestration, Model Context Protocol (MCP) vehicle evidence, hybrid policy RAG, deterministic risk scoring, and human-in-the-loop review.

---

## 1. Architecture

The Vehicle Risk Assessment Agent is engineered as a secure, auditable, multi-stage assessment system:

```
[ HTTP Client / Reviewer ]
          │
          ▼
┌─────────────────────────┐
│       FastAPI API       │  ◄── Enforces authentication, rate limits, and audit boundaries
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Typed LangGraph Flow   │  ◄── Orchestrates evidence lookup, sufficiency, RAG, and reporting
├─────────────────────────┤
│ 1. Evidence Collection  │ ──► Model Context Protocol (MCP) Server
│ 2. Sufficiency Check    │ ──► Withholds scoring if required evidence is missing
│ 3. Policy Retrieval     │ ──► PostgreSQL / pgvector Hybrid RAG (Dense + BM25 + Reciprocal Rank Fusion)
│ 4. Deterministic Scoring│ ──► Fixed formula scoring outside LLM reach (Never hallucinates risk)
│ 5. Report Drafting      │ ──► Anthropic Claude / Offline fallback with policy citations
│ 6. Human Review Gate    │ ──► Enforces mandatory human approval before report release
└─────────────────────────┘
```

- **API Boundary:** FastAPI application hosting assessment creation, event streaming (SSE), draft inspection, and review decision endpoints.
- **Workflow Engine:** Typed LangGraph state graph with deterministic checkpointing and resumability across execution runs.
- **Evidence Authority:** Model Context Protocol (MCP) client consuming verified vehicle revisions from upstream pipeline services.
- **Policy Knowledge:** Vector-indexed authoritative legal corpus (pgvector 384-dimensional embeddings) with hybrid dense/sparse retrieval and reranking.
- **Deterministic Risk Engine:** Rule-based scoring calculator adhering to strictly defined policy weights and thresholds; language models never compute or override risk scores.
- **Human Approval Gate:** Strict human-in-the-loop checkpoint ensuring consequential reports cannot be auto-approved.

---

## 2. Policy Provenance & Legal Knowledge

All regulatory and risk evaluations are anchored in attributable New Zealand legal authorities:

| Authority | Issuing Body | Classification | Reuse License |
|---|---|---|---|
| **Fair Trading Act 1986** | Parliament of New Zealand | Primary Legislation | Creative Commons Attribution 4.0 (CC BY 4.0) |
| **Consumer Guarantees Act 1993** | Parliament of New Zealand | Primary Legislation | Creative Commons Attribution 4.0 (CC BY 4.0) |
| **Personal Property Securities Act 1999** | Parliament of New Zealand | Primary Legislation | Creative Commons Attribution 4.0 (CC BY 4.0) |
| **Land Transport Act 1998** | Parliament of New Zealand | Primary Legislation | Creative Commons Attribution 4.0 (CC BY 4.0) |

- **Attributable Passages:** Legal texts are chunked into section-bounded passages with cryptographic content hashes (`SHA-256`).
- **Corpus Versioning:** Immutable `PolicyCorpusManifest` pins snapshot IDs and retrieval parameters (`sentence-transformers/all-MiniLM-L6-v2`, candidate caps, fusion weights).
- **Explicit Abstention:** When legal queries do not reach citation confidence thresholds, the system emits an explicit abstention notice rather than fabricating policy references.

---

## 3. Security Boundaries & Safe Telemetry

- **No Secret Leaks:** No API keys, credentials, database passwords, or auth tokens are logged or included in reports.
- **Strict Data Redaction:** Telemetry spans and structured JSON logs automatically redact sensitive customer data and check-digit-sensitive identifiers.
- **Safe Failure Classification:** Exceptions are caught at API and graph boundaries and mapped to safe error codes (`SafeFailureCategory`); raw internal stack traces and prompts are never exposed to clients.
- **Role-Based Principal Authorization:** Endpoints validate caller identity and roles (`principal:requester`, `principal:reviewer`).
- **Container Hardening:** Runs as a dedicated unprivileged user (`appuser`, UID 10001) in a read-only root multi-stage Docker container.

---

## 4. Operational Limitations

- **No Live Restricted Registers Access:** The service does not claim or simulate direct access to live restricted government registers. All data originates from MCP evidence snapshots or seeded evaluation corpora.
- **Deterministic Scoring Separation:** Language models generate draft prose and explanations but CANNOT modify, compute, or override risk scores or risk bands.
- **Synthetic Data Notices:** Synthetic vehicle records and evaluation fixtures carry mandatory synthetic notices preserved through report generation.
- **Three-Run Reinvestigation Cap:** Reviewers may request additive reinvestigation up to a maximum of 3 runs per assessment to prevent infinite loops and unbounded resource usage.
- **Mandatory Human Review:** No assessment report is released to customers without an explicit signed review action from an authorized reviewer.

---

## 5. Quickstart

### Prerequisites

- Python 3.12+
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker and Docker Compose (v2.20+)

### Environment Setup

Copy example environment variables:
```bash
cp .env.example .env
```

Key environment variables:
- `VEHICLE_RISK_AGENT_DATABASE_URL`: PostgreSQL connection string (default: `postgresql+asyncpg://postgres:postgres@localhost:5432/vehicle_risk`)
- `ANTHROPIC_API_KEY`: Optional Anthropic API key for live report drafting (offline drafting adapter used when absent)
- `VEHICLE_RISK_AGENT_LOG_LEVEL`: Logging verbosity (`INFO`, `DEBUG`)

### Running Locally with Docker Compose

Start all pinned services using `compose.yaml`:
```bash
docker compose up -d
```

Services started:
- `db`: PostgreSQL 16 with pgvector extension (`localhost:5432`)
- `pipeline`: Upstream vehicle data pipeline service (`localhost:8001`)
- `mcp`: Vehicle intelligence MCP server (`localhost:8002`)
- `agent-api`: Vehicle risk assessment service (`localhost:8000`)

Check health endpoints:
```bash
curl -f http://localhost:8000/health
curl -f http://localhost:8000/ready
```

---

## 6. Verification & Demonstration Scenarios

### Automated Preflight Check

Run the 7-step fail-fast preflight gate (formatting, linting, strict typing, migrations, container contract, full test suite, and package build):
```bash
bash scripts/check.sh
```

### Idempotent Database Seed

Seed the database with default Risk Policies, Policy Corpora, passages, and evaluation fixtures:
```bash
uv run python -m vehicle_risk_agent.cli.seed
```

### End-to-End Smoke Verification

Execute the complete local smoke suite exercising all domain paths:
```bash
bash scripts/smoke-local.sh
```
Or directly run the Python smoke runner:
```bash
uv run python -m vehicle_risk_agent.cli.smoke
```

### Seven Demonstrated Scenarios

The smoke script exercises:
1. **Clean Vehicle Assessment:** Low-risk vehicle (score 0, Band `LOW`) verified against official register data.
2. **Risky Vehicle Assessment:** High/Critical-risk vehicle with registered security interest and stolen record (score 100, Band `CRITICAL`).
3. **Incomplete Evidence Assessment:** Missing mandatory fields triggers immediate scoring withholding and emits an `INCOMPLETE` report draft with missing evidence notices.
4. **Human Review Approval:** Reviewer records `APPROVE_REPORT` on clean draft, transitioning assessment to `RELEASED`.
5. **Human Review Rejection:** Reviewer records `REJECT_REPORT` on risky draft, transitioning assessment to `REJECTED`.
6. **Additive Reinvestigation:** Reviewer triggers Run 2 with targeted evidence requests (`odometer_reading`), creating an audited sequential run.
7. **Restart Idempotency & Teardown:** Re-running seeds confirms non-destructive idempotency, followed by clean removal of generated test records.

---

## 7. Reviewer Human-in-the-Loop Flow

```
     Assessment Created
             │
             ▼
      Graph Execution
             │
             ▼
     Draft Generated ──► [ Assessment Lifecycle: AWAITING_REVIEW ]
             │
   ┌─────────┴─────────┐
   ▼                   ▼
Approve             Reject           Request Reinvestigation
   │                   │                        │
   ▼                   ▼                        ▼
[ RELEASED ]     [ REJECTED ]        [ IN_PROGRESS (Run n+1) ]
```

- Each review action is immutably recorded with reviewer ID, timestamp, disposition, and rationale.
- The released report is signed with an immutable release hash locking the exact policy, risk result, and draft version.

---

## 8. Troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `pgvector extension missing` | PostgreSQL container started without pgvector | Use image `pgvector/pgvector:pg16` in `compose.yaml` |
| `Alembic version conflict` | Database schema out of sync | Run `uv run alembic upgrade head` |
| `Port 5432 in use` | Existing local postgres service | Stop conflicting daemon or adjust `POSTGRES_PORT` |
| `OpenTelemetry span leak` | Custom span exporter recording raw exceptions | Use `trace_boundary` helper from `vehicle_risk_agent.observability.telemetry` |
| `Drafting fallback active` | `ANTHROPIC_API_KEY` not set | Expected in offline mode; system uses `OfflineReportDraftingAdapter` |

---

## 9. Cleanup & Teardown

Stop and remove all containers, networks, and volumes:
```bash
docker compose down -v
```

To clean local build artifacts and cache:
```bash
rm -rf dist build .pytest_cache .ruff_cache
```
