# syntax=docker/dockerfile:1
FROM python:3.12.11-slim-bookworm AS builder

# Install uv from official binary
COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface

ARG INSTALL_ML=false
ARG INSTALL_AZURE=false
# Model revisions measured by the e12 retrieval benchmark (docs/benchmarks/e12/README.md).
ARG EMBEDDING_REVISION=8b3219a92973c328a8e22fadcfa821b5dc75636a
ARG RERANKER_REVISION=739bce82df32cacea8ff0edf73ab49ae315e5a5f

WORKDIR /app

# Install dependencies using lockfile
COPY pyproject.toml uv.lock ./
RUN extras=""; \
    if [ "$INSTALL_ML" = "true" ]; then extras="$extras --extra ml"; fi; \
    if [ "$INSTALL_AZURE" = "true" ]; then extras="$extras --extra azure"; fi; \
    uv sync --frozen --no-dev --no-install-project $extras

# Copy application source and sync project
COPY README.md ./
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./
RUN extras=""; \
    if [ "$INSTALL_ML" = "true" ]; then extras="$extras --extra ml"; fi; \
    if [ "$INSTALL_AZURE" = "true" ]; then extras="$extras --extra azure"; fi; \
    uv sync --frozen --no-dev $extras

# Bake the pinned retrieval models so production never downloads at runtime
RUN if [ "$INSTALL_ML" = "true" ]; then \
      .venv/bin/python -c "import sys; from sentence_transformers import CrossEncoder, SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision=sys.argv[1]); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', revision=sys.argv[2])" \
        "$EMBEDDING_REVISION" "$RERANKER_REVISION"; \
    fi

# Final minimal runtime image
FROM python:3.12.11-slim-bookworm AS runtime

ARG EMBEDDING_REVISION=8b3219a92973c328a8e22fadcfa821b5dc75636a
ARG RERANKER_REVISION=739bce82df32cacea8ff0edf73ab49ae315e5a5f

WORKDIR /app

# Run as non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -u 10001 appuser

# Copy virtual environment, baked model cache, and app from builder
COPY --from=builder --chown=appuser:appgroup /app /app
COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /bin/uv

USER appuser

# Revision and offline settings only affect images built with INSTALL_ML=true
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBEDDING_REVISION=$EMBEDDING_REVISION \
    RERANKER_REVISION=$RERANKER_REVISION

EXPOSE 8001

HEALTHCHECK --interval=5s --timeout=3s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["uvicorn", "vehicle_risk_agent.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8001"]
