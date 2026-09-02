# Original User Request

## Initial Request — 2026-09-01T11:31:08Z

You are the Project Orchestrator for Epic e04 (Grounded Risk Reporting) in vehicle-risk-agent.

Your working directory is: /Users/sovorn/dev/portfolio/vehicle-risk-agent/.agents/orchestrator
The workspace root is: /Users/sovorn/dev/portfolio/vehicle-risk-agent
The authoritative user request is at: /Users/sovorn/dev/portfolio/vehicle-risk-agent/ORIGINAL_REQUEST.md

Mission:
Execute Epic e04 across all three stories following bigpowers TDD conventions:
1. e04s01 (Deterministic Risk Policy and Result Calculation):
   - Review specs: specs/epics/e04-grounded-risk-reporting/e04s01-produce-deterministic-risk-result.md, e04s01-tasks.yaml
   - Immutable Risk Policy v1 with factor weights (MATCH=30, LISTED=45, REPAIRABLE=20, STATUTORY=40), score cap (100), bands (LOW 0-19, MEDIUM 20-39, HIGH 40-69, CRITICAL 70-100), mandatory-review findings.
   - Lifecycle states (DRAFT -> READY -> ACTIVE -> RETIRED) with operator-only activation.
   - Immutable Risk Results with monotonic, order-independent, replayable scoring.
   - Tests: tests/risk/test_risk_policy.py, tests/api/test_risk_policy_activation.py, tests/risk/test_risk_result_persistence.py.

2. e04s02 (Offline Cited Report Drafting):
   - Review specs: specs/epics/e04-grounded-risk-reporting/e04s02-draft-offline-cited-reports.md, e04s02-tasks.yaml
   - Report Draft schemas with structured sections, claim references (evidence_refs, policy_citation_refs, risk_factor_refs), limitations, missing evidence notices, abstentions, synthetic notices.
   - Offline drafting adapter for SCORED and INCOMPLETE reports without LLM dependency.
   - Persist exactly one immutable Report Draft per completed run, transition Assessment state to AWAITING_REVIEW.
   - Tests: tests/reporting/test_report_models.py, tests/reporting/test_offline_report.py, tests/workflow/test_report_drafting.py.

3. e04s03 (Live Model Draft Validation, Grounding & Safe Repair):
   - Review specs: specs/epics/e04-grounded-risk-reporting/e04s03-validate-live-model-drafts.md, e04s03-tasks.yaml
   - Anthropic drafting adapter with bounded timeouts, token limits, structured outputs, telemetry.
   - Strict grounding validation against pinned evidence items and policy citations, permitting exactly one bounded text-only repair attempt without altering scores/bands/findings.
   - Safe failure to FAILED on invalid output, ungrounded claims, timeouts, provider unavailability without persisting drafts or leaking internal errors.
   - Tests: tests/reporting/test_anthropic_adapter.py, tests/reporting/test_grounding.py, tests/workflow/test_model_failures.py.

Process:
- Initialize kickoff-branch on `feat/e04-grounded-risk-reporting`.
- Execute all stories via strict red-green TDD, updating story task YAML files and specs/state.yaml.
- Keep `progress.md` updated at least every 5-10 minutes with current status and changes.
- Ensure full preflight (`bash scripts/check.sh`) passes cleanly with 0 errors (Ruff lint/format, strict Mypy, all pytest tests, package build).
- When complete, notify parent with full completion evidence and summary.
