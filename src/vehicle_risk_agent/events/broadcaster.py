"""In-memory pub/sub broadcaster for live workflow progress events."""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from vehicle_risk_agent.domain.events import WorkflowProgressEvent


class ProgressEventBroadcaster:
    """Publishes live WorkflowProgressEvent instances to subscribed SSE clients."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[WorkflowProgressEvent]]] = defaultdict(set)
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def publish(self, event: WorkflowProgressEvent) -> None:
        """Broadcast event to all active subscriber queues for this assessment."""
        queues = self._subscribers.get(event.assessment_id, set())
        for q in list(queues):
            with suppress(asyncio.QueueFull):
                q.put_nowait(event)

    @asynccontextmanager
    async def subscribe(
        self, assessment_id: str, max_queue_size: int = 100
    ) -> AsyncIterator[asyncio.Queue[WorkflowProgressEvent]]:
        """Subscribe to live progress events for an assessment ID."""
        q: asyncio.Queue[WorkflowProgressEvent] = asyncio.Queue(maxsize=max_queue_size)
        lock = self._get_lock()
        async with lock:
            self._subscribers[assessment_id].add(q)
        try:
            yield q
        finally:
            async with lock:
                if assessment_id in self._subscribers:
                    self._subscribers[assessment_id].discard(q)
                    if not self._subscribers[assessment_id]:
                        del self._subscribers[assessment_id]

    @property
    def subscriber_count(self) -> int:
        """Total number of active subscriber queues across all assessments."""
        return sum(len(queues) for queues in self._subscribers.values())
