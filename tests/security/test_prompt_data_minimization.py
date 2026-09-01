"""Security verification tests for prompt and workflow data minimization."""

import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from vehicle_risk_agent.adapters.mcp import FakeVehicleMcpAdapter
from vehicle_risk_agent.api.models import AssessmentContext, SaleType
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    ProvenanceLink,
    SourceObservationResponse,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.snapshot import create_evidence_snapshot
from vehicle_risk_agent.security.minimization import assert_data_minimization
from vehicle_risk_agent.workflow.runner import AssessmentRunner


@pytest.mark.asyncio
async def test_raw_payload_never_enters_graph_state_events_or_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Prove raw_payload never enters graph state, progress events, or log records."""
    caplog.set_level(logging.DEBUG)
    secret_marker = "SECRET_RAW_SENSITIVE_SOURCE_PAYLOAD_XYZ"

    now = datetime.now(UTC)
    obs = SourceObservationResponse(
        observation_id="obs-1",
        source_system="PRIVATE_REGISTRY",
        source_record_id="rec-1",
        ingestion_run_id="run-1",
        raw_payload=f'{{"sensitive": "{secret_marker}"}}',
        payload_hash_sha256="e" * 64,
        retrieved_at=now,
        synthetic=True,
    )
    p = ProvenanceLink(
        observation_id="obs-1",
        source_system="PRIVATE_REGISTRY",
        source_record_id="rec-1",
        retrieved_at=now,
        synthetic=True,
    )
    rev = VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-1",
        revision_number=1,
        material_hash="e" * 64,
        canonical_fields={
            "make": "TOYOTA",
            "model": "COROLLA",
            "year": 2018,
            "ppsr_result": "NO_FINANCE_REGISTERED",
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NOT_WRITTEN_OFF",
        },
        field_provenance={"make": [p]},
        conflicts=[],
        confidence=ConfidenceAssessment(
            score=90,
            band=ConfidenceBand.HIGH,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="verified",
        ),
        as_of=now,
        published_at=now,
    )

    mcp_adapter = FakeVehicleMcpAdapter()
    mcp_adapter.seed_vehicle(rev)
    mcp_adapter.seed_source_observation(obs)

    runner = AssessmentRunner(mcp_adapter=mcp_adapter)
    final_state = await runner.run(
        assessment_id="asmt-sec-01",
        run_number=1,
        vin="7AT0BK00X00000001",
        context=AssessmentContext(sale_type=SaleType.PRIVATE),
    )

    # 1. State check
    assert_data_minimization(final_state, forbidden_substring=secret_marker)

    # 2. Events check
    for evt in final_state.get("events", []):
        assert secret_marker not in evt.safe_message

    # 3. Logs check
    for record in caplog.records:
        assert secret_marker not in record.getMessage()
