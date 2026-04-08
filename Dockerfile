# =============================================================================
# CrewAI Dockerfile
# Multi-stage build for containerized crewAI deployment.
# Eliminates common dependency issues (lancedb, litellm, chromadb, etc.)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — Builder
# Install build tools and compile native extensions in an isolated layer.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# System packages required to compile native wheels
# (lancedb needs protobuf/cmake, chromadb needs build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, deterministic dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy workspace definition and lock file first for better layer caching
COPY pyproject.toml uv.lock ./
COPY lib/crewai/pyproject.toml lib/crewai/pyproject.toml
COPY lib/crewai-tools/pyproject.toml lib/crewai-tools/pyproject.toml
COPY lib/crewai-files/pyproject.toml lib/crewai-files/pyproject.toml
COPY lib/devtools/pyproject.toml lib/devtools/pyproject.toml

# Copy full source (needed for editable installs / hatch version discovery)
COPY lib/ lib/

# Install crewAI with default dependencies.
# Pass CREWAI_EXTRAS build arg to include optional extras (e.g. "tools,litellm").
ARG CREWAI_EXTRAS=""
RUN if [ -z "$CREWAI_EXTRAS" ]; then \
      uv sync --locked --no-dev; \
    else \
      uv sync --locked --no-dev --extra "$CREWAI_EXTRAS"; \
    fi

# ---------------------------------------------------------------------------
# Stage 2 — Runtime
# Slim image with only the installed packages and runtime libraries.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Runtime libraries required by native extensions (lancedb, chromadb, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd --gid 1000 crewai \
    && useradd --uid 1000 --gid crewai --create-home crewai

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy source code (needed at runtime for crewAI CLI templates, etc.)
COPY --from=builder /app/lib /app/lib

# Ensure the virtual-env Python is on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER crewai

ENTRYPOINT ["crewai"]
CMD ["--help"]
