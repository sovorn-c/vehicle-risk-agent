"""Assessment workflow runner integrating LangGraph and domain repositories."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph

from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.workflow.graph import build_assessment_graph


class AssessmentWorkflowRunner:
    """Manages compiled assessment workflow execution with PostgreSQL checkpoints."""

    def __init__(self, checkpointer: AsyncPostgresSaver) -> None:
        self.checkpointer = checkpointer
        self._graph = build_assessment_graph()
        self._app: CompiledStateGraph = self._graph.compile(checkpointer=self.checkpointer)  # type: ignore[type-arg]

    @classmethod
    @asynccontextmanager
    async def create(cls, settings: Settings) -> AsyncIterator["AssessmentWorkflowRunner"]:
        """Create and initialize a runner with AsyncPostgresSaver."""
        # Convert sqlalchemy format postgresql+psycopg:// to psycopg format postgresql://
        conn_string = settings.database_url.replace("+psycopg", "")
        async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
            await checkpointer.setup()
            yield cls(checkpointer=checkpointer)

    @property
    def app(self) -> CompiledStateGraph:  # type: ignore[type-arg]
        """Return the compiled workflow application."""
        return self._app

    @staticmethod
    def get_thread_id(assessment_id: str, run_number: int) -> str:
        """Format a stable, bounded thread_id identifier (< 255 chars)."""
        return f"{assessment_id}:{run_number}"
