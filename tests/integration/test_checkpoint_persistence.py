"""Integration tests for LangGraph PostgreSQL checkpoint persistence with runner."""

import pytest
from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import create_async_engine

from vehicle_risk_agent.api.models import AssessmentContext, SaleType
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.persistence.models import Base
from vehicle_risk_agent.workflow.runner import AssessmentWorkflowRunner
from vehicle_risk_agent.workflow.state import AssessmentGraphState

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest.mark.asyncio
async def test_checkpoint_persistence_under_stable_run_identifier() -> None:
    """Verify workflow state is saved in PostgreSQL under thread_id = assessment_id:run_number."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

    settings = Settings(database_url=TEST_DB_URL)
    async with AssessmentWorkflowRunner.create(settings) as runner:
        thread_id = runner.get_thread_id("asmt-runner-001", 1)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        initial_state: AssessmentGraphState = {
            "assessment_id": "asmt-runner-001",
            "run_number": 1,
            "vin": "1HGCR2F85HA000000",
            "context": AssessmentContext(sale_type=SaleType.DEALER),
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }

        result = await runner.app.ainvoke(initial_state, config=config)
        assert result["phase"] == AssessmentRunPhase.COMPLETED

        # Inspect checkpoint saved in database
        saved_state = await runner.app.aget_state(config)
        assert saved_state is not None
        assert saved_state.values["phase"] == AssessmentRunPhase.COMPLETED
        assert saved_state.values["assessment_id"] == "asmt-runner-001"
