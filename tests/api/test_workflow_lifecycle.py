"""Regression tests for application-owned workflow execution lifecycle."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

import vehicle_risk_agent.api.app as app_module
from vehicle_risk_agent.api.models import AssessmentContext, SaleType
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import (
    Assessment,
    AssessmentLifecycleState,
    AssessmentRun,
    AssessmentRunPhase,
)


class _DummyRunner:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        self.started.set()
        await self.release.wait()
        self.events.append("task-done")
        return {"phase": AssessmentRunPhase.COMPLETED, "state": initial_state}


@pytest.mark.asyncio
async def test_lifespan_owns_and_drains_background_workflow_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Application shutdown must wait for scheduled runs before runner teardown."""
    events: list[str] = []
    runner = _DummyRunner(events)

    class _RunnerFactory:
        @classmethod
        @asynccontextmanager
        async def create(cls, *_args: Any, **_kwargs: Any) -> AsyncIterator[_DummyRunner]:
            yield runner
            events.append("runner-exit")

    monkeypatch.setattr(app_module, "AssessmentWorkflowRunner", _RunnerFactory)
    app = app_module.create_app(
        Settings(database_url="postgresql+psycopg://postgres:postgres@localhost:54329/postgres")
    )
    now = datetime.now(UTC)
    assessment = Assessment(
        id="asmt-lifecycle",
        requester_id="principal-requester-1",
        vin="1HGCR2F85HA000000",
        context=AssessmentContext(sale_type=SaleType.DEALER),
        lifecycle_state=AssessmentLifecycleState.IN_PROGRESS,
        current_run_number=1,
        runs=[
            AssessmentRun(
                id="run-lifecycle",
                assessment_id="asmt-lifecycle",
                run_number=1,
                phase=AssessmentRunPhase.PENDING,
                created_at=now,
                updated_at=now,
            )
        ],
        created_at=now,
        updated_at=now,
    )

    async with app.router.lifespan_context(app):
        assert app.state.workflow_runner is runner
        assert app.state.schedule_workflow_run(assessment) is True
        await runner.started.wait()
        assert len(app.state.workflow_tasks) == 1
        runner.release.set()
        await asyncio.gather(*app.state.workflow_tasks)

    assert events == ["task-done", "runner-exit"]
