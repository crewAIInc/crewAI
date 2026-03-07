FROM python:3.12-slim-bookworm

ENV HOME=/root \
    CODEX_HOME=/root/.codex \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_HTTP_RETRIES=5 \
    UV_HTTP_TIMEOUT=300 \
    CREWAI_DISABLE_TELEMETRY=true \
    OTEL_SDK_DISABLED=true

WORKDIR /workspace

COPY pyproject.toml uv.lock /tmp/workspace/
COPY lib/crewai/pyproject.toml /tmp/workspace/lib/crewai/pyproject.toml
COPY lib/crewai-tools/pyproject.toml /tmp/workspace/lib/crewai-tools/pyproject.toml
COPY lib/crewai-files/pyproject.toml /tmp/workspace/lib/crewai-files/pyproject.toml
COPY lib/devtools/pyproject.toml /tmp/workspace/lib/devtools/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    python -m venv /opt/venv \
 && /opt/venv/bin/pip install uv \
 && install -d \
      /tmp/workspace/lib/crewai \
      /tmp/workspace/lib/crewai-tools \
      /tmp/workspace/lib/crewai-files \
      /tmp/workspace/lib/devtools \
 && cd /tmp/workspace \
 && UV_PROJECT_ENVIRONMENT=/opt/venv \
    /opt/venv/bin/uv sync \
      --locked \
      --no-dev \
      --package crewai \
      --no-editable \
      --no-install-package crewai

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/workspace/lib/crewai/src:/workspace/scripts"

CMD ["sh"]
