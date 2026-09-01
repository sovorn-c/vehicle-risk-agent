"""FastAPI dependencies for authentication, authorization, database sessions, and rate limiting."""

import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from vehicle_risk_agent.auth import Principal, Role, authenticate_bearer_token
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.events.broadcaster import ProgressEventBroadcaster
from vehicle_risk_agent.retrieval.adapters import EmbeddingAdapter, RerankerAdapter


class RateLimiter:
    """In-memory sliding window rate limiter per principal ID."""

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> bool:
        """Return True if under limit, False if rate exceeded."""
        now = time.time()
        window_start = now - self.window_seconds
        valid_timestamps = [ts for ts in self.requests[key] if ts > window_start]
        self.requests[key] = valid_timestamps

        if len(valid_timestamps) >= self.max_requests:
            return False

        self.requests[key].append(now)
        return True

    def reset(self) -> None:
        """Clear all recorded request timestamps."""
        self.requests.clear()


intake_rate_limiter = RateLimiter(max_requests=10, window_seconds=60.0)
global_event_broadcaster = ProgressEventBroadcaster()


def get_settings(request: Request) -> Settings:
    """Extract settings from app state."""
    return request.app.state.settings  # type: ignore[no-any-return]


def get_embedding_adapter(request: Request) -> EmbeddingAdapter:
    """Extract the configured policy embedding adapter."""
    return request.app.state.embedding_adapter  # type: ignore[no-any-return]


def get_reranker_adapter(request: Request) -> RerankerAdapter:
    """Extract the configured policy reranker adapter."""
    return request.app.state.reranker_adapter  # type: ignore[no-any-return]


def get_event_broadcaster(request: Request) -> ProgressEventBroadcaster:
    """Extract event broadcaster from app state or return global fallback."""
    broadcaster = getattr(request.app.state, "event_broadcaster", None)
    if broadcaster is None:
        return global_event_broadcaster
    return broadcaster  # type: ignore[no-any-return]


def get_current_principal(
    request: Request,
    authorization: str | None = Header(default=None),
) -> Principal:
    """Authenticate bearer token from header."""
    settings = get_settings(request)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token"},
        )

    token = authorization.removeprefix("Bearer ").strip()
    principal = authenticate_bearer_token(token, settings)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Invalid authentication credentials"},
        )

    return principal


def require_role(allowed_role: Role) -> Callable[[Principal], Principal]:
    """Enforce that authenticated Principal has the required role."""

    def _role_checker(principal: Principal = Depends(get_current_principal)) -> Principal:
        if principal.role != allowed_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "FORBIDDEN", "message": f"Operation requires role {allowed_role}"},
            )
        return principal

    return _role_checker


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield an async database session from engine session factory."""
    session_factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with session_factory() as session:
        yield session
