"""Tests for crash recovery at foundation phases without duplicate effects."""

import uuid

import pytest
from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import create_async_engine

from vehicle_risk_agent.api.models import AssessmentContext, SaleType
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.persistence.models import Base
from vehicle_risk_agent.workflow.graph import build_assessment_graph
from vehicle_risk_agent.workflow.runner import AssessmentWorkflowRunner
from vehicle_risk_agent.workflow.state import AssessmentGraphState

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest.mark.asyncio
async def test_recovery_from_intermediate_phase_without_backward_transition() -> None:
    """Verify resuming an interrupted run starts from its checkpoint and reaches COMPLETED."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

    settings = Settings(database_url=TEST_DB_URL)
    assessment_id = f"asmt-recover-{uuid.uuid4().hex[:8]}"
    thread_id = f"{assessment_id}:1"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    initial_state: AssessmentGraphState = {
        "assessment_id": assessment_id,
        "run_number": 1,
        "vin": "1HGCR2F85HA000000",
        "context": AssessmentContext(sale_type=SaleType.DEALER),
        "phase": AssessmentRunPhase.PENDING,
        "visited_phases": [AssessmentRunPhase.PENDING],
        "events": [],
    }

    # Step 1: Run workflow with an interrupt before 'evaluating_risk' to simulate crash/interruption
    async with AssessmentWorkflowRunner.create(settings) as runner:
        interrupted_app = build_assessment_graph().compile(
            checkpointer=runner.checkpointer,
            interrupt_before=["evaluating_risk"],
        )
        interrupted_result = await interrupted_app.ainvoke(initial_state, config=config)
        assert interrupted_result["phase"] == AssessmentRunPhase.RETRIEVING_POLICY

        saved = await interrupted_app.aget_state(config)
        assert saved is not None
        assert saved.values["phase"] == AssessmentRunPhase.RETRIEVING_POLICY
        assert saved.next == ("evaluating_risk",)

    # Step 2: "Process restart" - create fresh runner and resume execution from saved checkpoint
    async with AssessmentWorkflowRunner.create(settings) as restarted_runner:
        # Resuming without supplying initial state (ainvoke(None, config=config))
        resumed_result = await restarted_runner.app.ainvoke(None, config=config)

        assert resumed_result["phase"] == AssessmentRunPhase.COMPLETED
        visited = resumed_result["visited_phases"]
        # Visited phases must advance monotonically without backward steps
        assert visited == [
            AssessmentRunPhase.PENDING,
            AssessmentRunPhase.COLLECTING_EVIDENCE,
            AssessmentRunPhase.RETRIEVING_POLICY,
            AssessmentRunPhase.EVALUATING_RISK,
            AssessmentRunPhase.DRAFTING_REPORT,
            AssessmentRunPhase.COMPLETED,
        ]
