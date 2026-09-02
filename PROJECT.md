# Project: vehicle-risk-agent Epic e04 (Grounded Risk Reporting)

## Architecture
Epic e04 implements the grounded risk scoring and cited report drafting engine for `vehicle-risk-agent`.
The architecture consists of three integrated layers plus workflow orchestration:

1. **Deterministic Risk Policy & Scoring Layer (`src/vehicle_risk_agent/risk/`)**:
   - Risk Policy domain model with lifecycle management (`DRAFT`, `READY`, `ACTIVE`, `RETIRED`).
   - Operator-only activation endpoint with database-enforced single active policy constraint.
   - Deterministic, monotonic, replayable scoring calculation engine applying Policy v1 weights (`MATCH=30`, `LISTED=45`, `REPAIRABLE=20`, `STATUTORY=40`, max 100) and bands (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
   - Mandatory-Review Findings for all positive factors.
   - Strict `INCOMPLETE` handling (scoring withheld, numeric fields forbidden).
   - Immutable persistence of `RiskResultRecord` with unique run constraint.

2. **Cited Report Drafting Layer (`src/vehicle_risk_agent/reporting/`)**:
   - Structured `ReportDraft` schemas with 9 standard sections.
   - Precise citation tracking (`evidence_refs`, `policy_citation_refs`, `risk_factor_refs`), limitations, missing evidence notices, abstentions, and synthetic data notices.
   - Pure deterministic `OfflineReportDraftingAdapter` for both `SCORED` and `INCOMPLETE` assessments.
   - Persistent `ReportDraftRecord` tied to assessment run, atomically transitioning the `Assessment` aggregate lifecycle state to `AWAITING_REVIEW`.

3. **Live Model Validation & Safe Grounding Layer (`src/vehicle_risk_agent/reporting/` & `src/vehicle_risk_agent/adapters/`)**:
   - `AnthropicReportDraftingAdapter` using Claude (`claude-sonnet-4-6`) with strict prompt minimization, 30s timeout, token limits, and structured outputs.
   - Strict `GroundingValidator` enforcing that all cited evidence IDs and policy citation IDs strictly exist in the pinned run state.
   - Exactly one bounded text-only repair attempt without altering scores, bands, or findings.
   - Fail-closed terminal error handling transitioning to `FAILED` with no draft persisted and no internal error leakage.

4. **LangGraph Workflow Integration (`src/vehicle_risk_agent/graph/`)**:
   - Seamless integration of risk calculation and report drafting nodes into the assessment execution graph.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Risk Policy v1 Model & Weights | Pydantic domain models for Risk Policy with weights (MATCH=30, LISTED=45, REPAIRABLE=20, STATUTORY=40, cap 100), bands (LOW 0-19, MEDIUM 20-39, HIGH 40-69, CRITICAL 70-100), mandatory review rules | M1 (e04s01) | e04s01 spec §1 |
| 2 | Risk Policy Lifecycle & Activation | Lifecycle transitions (DRAFT -> READY -> ACTIVE -> RETIRED), operator-only activation API (`POST /api/v1/risk-policies/{id}/activate`), single-active DB constraint | M1 (e04s01) | e04s01 spec §2 |
| 3 | Deterministic Risk Scoring Engine | Monotonic, order-independent risk score calculator producing immutable `RiskResult` with factor items, bands, and findings; strict INCOMPLETE exclusion | M1 (e04s01) | e04s01 spec §3 |
| 4 | Risk Result DB Persistence | SQLAlchemy models and Alembic migration for `risk_policies` and `risk_results` tables with run idempotency | M1 (e04s01) | e04s01 spec §4 |
| 5 | Structured Report Draft Schemas | Pydantic models for Report Drafts with 9 sections, claim references (evidence, policy, factor refs), limitations, missing evidence notices, synthetic notices | M2 (e04s02) | e04s02 spec §1 |
| 6 | Offline Report Drafting Adapter | Deterministic drafting adapter supporting SCORED and INCOMPLETE reports without external network/LLM dependencies | M2 (e04s02) | e04s02 spec §2 |
| 7 | Report Draft Persistence & State Transition | Persistent `ReportDraftRecord` with unique run constraint; atomic transition of Assessment aggregate to `AWAITING_REVIEW` | M2 (e04s02) | e04s02 spec §3 |
| 8 | Workflow Node Integration for Drafting | LangGraph workflow node executing risk scoring and offline report drafting sequentially | M2 (e04s02) | e04s02 spec §4 |
| 9 | Anthropic Report Drafting Adapter | LLM-based drafting adapter using Anthropic API with structured output, timeouts, token limits, and telemetry | M3 (e04s03) | e04s03 spec §1 |
| 10 | Grounding Validator & Bounded Repair | Strict validator ensuring all claim citations exist in pinned evidence/policy; max 1 bounded text-only repair attempt without altering scores/bands/findings | M3 (e04s03) | e04s03 spec §2 |
| 11 | Safe Failure to FAILED | Terminal failure handling on timeout, ungrounded claims, or provider errors: transitions to FAILED, persists no draft, leaks no internal errors | M3 (e04s03) | e04s03 spec §3 |
| 12 | End-to-End Integration & Preflight | Full regression test suite, integration tests across all 3 stories, clean `bash scripts/check.sh` preflight | M4 | e04 acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: e04s01 Deterministic Risk Policy & Result | Risk Policy domain models, scoring engine, lifecycle API, DB tables & persistence | none | DONE |
| 2 | M2: e04s02 Offline Cited Report Drafting | Report Draft schemas, citation tracking, offline adapter, draft DB persistence, Assessment state transition to AWAITING_REVIEW | M1 | PLANNED |
| 3 | M3: e04s03 Live Model Draft Validation, Grounding & Safe Repair | Anthropic adapter, strict grounding validation, 1 bounded repair attempt, fail-closed handling | M2 | PLANNED |
| 4 | M4: Final Integration, E2E Verification & Preflight | Full preflight (`bash scripts/check.sh`), test suite verification, documentation, specs/state.yaml update | M1, M2, M3 | PLANNED |

## Interface Contracts
### `vehicle_risk_agent.risk` ↔ `vehicle_risk_agent.graph`
- `calculate_risk_result(policy: RiskPolicy, evidence: Sequence[EvidenceItem], policy_citations: Sequence[PolicyCitation]) -> RiskResult`
- Pure function, deterministic, returns immutable `RiskResult` with `score: int | None`, `band: RiskBand | None`, `findings: list[MandatoryFinding]`, `factor_breakdown: list[RiskFactorResult]`.
- If evidence completeness status is `INCOMPLETE`, `score` and `band` are `None`.

### `vehicle_risk_agent.reporting` ↔ `vehicle_risk_agent.graph`
- `ReportDraftingProtocol`: `async def draft_report(context: ReportDraftingContext) -> ReportDraft`
- `ReportDraftingContext`: contains `assessment_id: UUID`, `run_id: UUID`, `vehicle_id: str`, `evidence_items: list[EvidenceItem]`, `policy_citations: list[PolicyCitation]`, `risk_result: RiskResult`.
- `OfflineReportDraftingAdapter`: implements `ReportDraftingProtocol` deterministically.
- `AnthropicReportDraftingAdapter`: implements `ReportDraftingProtocol` with live model and grounding validation.

### `vehicle_risk_agent.reporting.grounding` ↔ `vehicle_risk_agent.reporting`
- `GroundingValidator.validate(draft: ReportDraft, context: ReportDraftingContext) -> GroundingValidationResult`
- Returns `is_grounded: bool`, `unreferenced_evidence_ids: list[str]`, `unreferenced_citation_ids: list[str]`, `invalid_claims: list[str]`.

## Code Layout
- `src/vehicle_risk_agent/risk/`:
  - `models.py`: `RiskPolicy`, `RiskPolicyLifecycleState`, `RiskFactor`, `RiskBand`, `RiskResult`, `MandatoryFinding`
  - `calculator.py`: `calculate_risk_result`
  - `service.py`: `RiskPolicyService`
  - `repository.py`: `RiskPolicyRepository`, `RiskResultRepository`
- `src/vehicle_risk_agent/reporting/`:
  - `models.py`: `ReportDraft`, `ReportSection`, `ClaimReference`, `EvidenceRef`, `PolicyCitationRef`, `RiskFactorRef`, `MissingEvidenceNotice`, `SyntheticNotice`
  - `offline.py`: `OfflineReportDraftingAdapter`
  - `anthropic.py`: `AnthropicReportDraftingAdapter`
  - `grounding.py`: `GroundingValidator`, `repair_draft`
  - `repository.py`: `ReportDraftRepository`
- `src/vehicle_risk_agent/api/`:
  - `risk_policy.py`: Risk policy endpoints (`GET`, `POST /activate`)
- `src/vehicle_risk_agent/db/models/`:
  - `risk_policy.py`: SQLAlchemy `RiskPolicyRecord`
  - `risk_result.py`: SQLAlchemy `RiskResultRecord`
  - `report_draft.py`: SQLAlchemy `ReportDraftRecord`
- `tests/risk/`:
  - `test_risk_policy.py`, `test_risk_result_persistence.py`, `test_risk_scoring.py`
- `tests/api/`:
  - `test_risk_policy_activation.py`
- `tests/reporting/`:
  - `test_report_models.py`, `test_offline_report.py`, `test_anthropic_adapter.py`, `test_grounding.py`
- `tests/workflow/`:
  - `test_report_drafting.py`, `test_model_failures.py`
