# Vehicle Risk Assessment Agent — AI Agents

Read `CONVENTIONS.md` before any Git or GitHub operation.
Read `specs/` before changing production code.

## Codebase Navigation

- Use Cymbal first for exact code navigation: symbols, definitions, references, implementations, impact, and diffs.
- Use Graphify for broad architecture and codebase questions when it is installed and available.
- Prefer the existing Graphify graph/query fast path when available; do not re-index unnecessarily.
- Treat source files and tests as authoritative when tool output differs.
- Fall back to normal repository exploration when Cymbal or Graphify is unavailable.

<!-- BEGIN bigpowers:project -->
## Project

Build a production-grade vehicle risk assessment service with typed LangGraph workflows.
Use MCP evidence and policy RAG to produce auditable reports with human approval.

Stack: Python 3.12, LangGraph, FastAPI, Pydantic v2, MCP Python SDK, PostgreSQL, pgvector, uv, pytest, Ruff, mypy, and Docker.

Primary plan: `/Users/sovorn/sovorn-os/sources/jobs_hunting/nz-applied-ai-market-and-readiness-plan.md`.
Upstream MCP server: `/Users/sovorn/dev/portfolio/vehicle-mcp-server`.
Upstream pipeline: `/Users/sovorn/dev/portfolio/nz-vehicle-data-pipeline`.

## Commands

| Action | Command |
|---|---|
| Run | `uv run vehicle-risk-agent` |
| Test | `uv run pytest` |
| Build | `uv build` |
| Lint | `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests` |
| Preflight | `bash scripts/check.sh` |
| CI | `gh pr checks` |

## Architecture

FastAPI owns request, status, and human-approval boundaries.
A typed LangGraph workflow coordinates MCP evidence, policy RAG, deterministic scoring, reports, and approval checkpoints.
Adapters isolate MCP, LLM, vector-store, and persistence dependencies.

## Conventions

- Use a `src/vehicle_risk_agent/` package layout.
- Use `snake_case` for modules, functions, variables, and graph nodes.
- Use `PascalCase` for Pydantic models, graph states, protocols, and exceptions.
- Define explicit typed state for every LangGraph workflow.
- Keep graph nodes thin, asynchronous, and independently testable.
- Validate every external boundary with strict Pydantic models.
- Keep deterministic scoring outside prompts and LLM responses.
- Preserve evidence citations, uncertainty, conflicts, and synthetic notices.
- Mirror source modules under `tests/` where practical.
- Use Conventional Commits: `<type>(<scope>): <description>`.

## Never

- NEVER invent vehicle evidence, policy citations, or assessment findings.
- NEVER bypass the MCP server to access pipeline storage.
- NEVER let an LLM define deterministic risk scores.
- NEVER auto-approve a consequential vehicle assessment.
- NEVER hide missing evidence, uncertainty, conflicts, or synthetic provenance.
- NEVER claim access to live restricted registers.
- NEVER commit secrets, personal information, or production assessment data.
- NEVER block asynchronous paths with synchronous network or database I/O.
- NEVER expose credentials, raw exceptions, prompts, or internal stack traces.
- NEVER modify either upstream repository without separate approval.
- Never add an AI or agent co-author or attribution to a commit. Never include `Co-authored-by:` or `Co-Authored-By:` trailers.

## Agent rules

- MUST use bigpowers skills for feature and defect work.
- MUST write approved plans under `specs/` before production code.
- MUST use `scope-work`, `slice-tasks`, and `plan-work` before implementation.
- MUST use `develop-tdd` or `execute-plan` for implementation.
- MUST use `investigate-bug` before fixing reported defects.
- MUST keep Preflight and CI green.
- MUST run relevant checks after every change.
- MUST show verification evidence before declaring completion.
- MUST write the minimum code that solves approved scope.
- MUST ask one focused question when ambiguity changes behavior.
- MUST check background work every five minutes without a completion notification.
<!-- END bigpowers:project -->

<!-- BEGIN bigpowers:context-routing -->
## Context routing

| Glob | Read first |
|---|---|
| `src/vehicle_risk_agent/api/**` | Inspect API schemas, authorization, rate limits, and approval boundaries. |
| `src/vehicle_risk_agent/graph/**` | Inspect typed state, node contracts, transitions, and checkpoint rules. |
| `src/vehicle_risk_agent/evidence/**` | Inspect MCP tool contracts and evidence-preservation rules. |
| `src/vehicle_risk_agent/policy/**` | Inspect retrieval, citation, chunking, and evaluation contracts. |
| `src/vehicle_risk_agent/risk/**` | Inspect deterministic scoring rules and report schemas. |
| `src/vehicle_risk_agent/adapters/**` | Inspect external dependency protocols, timeouts, and retries. |
| `tests/**` | Follow F.I.R.S.T. rules and test public behavior. |
| `specs/**` | Follow the owning cockpit document before updating state. |
<!-- END bigpowers:context-routing -->

<!-- BEGIN bigpowers:learned-preferences -->
## Learned user preferences

- Build meaningful implementation before batching audit ceremony.
- Present hiring artifacts as real public products.
- Keep agent guidance and the planning cockpit out of GitHub.
- Keep `CLAUDE.md` as a pointer to `AGENTS.md`.
- Plan future implementation waves without speculative detail.

## Workspace facts

- Workflow mode is `solo-git`.
- The planning cockpit lives under `specs/`.
- The agent consumes vehicle evidence through the MCP server.
- The MCP server remains the only vehicle-intelligence integration boundary.
- The pipeline remains the vehicle-data authority.
<!-- END bigpowers:learned-preferences -->
