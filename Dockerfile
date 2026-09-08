# syntax=docker/dockerfile:1
FROM python:3.12.11-slim-bookworm AS builder

# Install uv from official binary
COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

ARG INSTALL_ML=false

WORKDIR /app

# Install dependencies using lockfile
COPY pyproject.toml uv.lock ./
RUN if [ "$INSTALL_ML" = "true" ]; then \
      uv sync --frozen --no-dev --no-install-project --extra ml; \
    else \
      uv sync --frozen --no-dev --no-install-project; \
    fi

# Copy application source and sync project
COPY README.md ./
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./
RUN if [ "$INSTALL_ML" = "true" ]; then \
      uv sync --frozen --no-dev --extra ml; \
    else \
      uv sync --frozen --no-dev; \
    fi

# Final minimal runtime image
FROM python:3.12.11-slim-bookworm AS runtime

WORKDIR /app

# Run as non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -u 10001 appuser

# Copy virtual environment and app from builder
COPY --from=builder --chown=appuser:appgroup /app /app
COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /bin/uv

USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8001

HEALTHCHECK --interval=5s --timeout=3s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["uvicorn", "vehicle_risk_agent.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8001"]
