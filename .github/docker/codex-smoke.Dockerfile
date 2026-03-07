FROM python:3.12-slim-bookworm

ENV HOME=/root \
    CODEX_HOME=/root/.codex \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace

COPY lib/crewai /workspace/lib/crewai

# Rebuild an isolated runtime from repository sources inside the container.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -e /workspace/lib/crewai

COPY scripts /workspace/scripts

ENV PATH="/opt/venv/bin:${PATH}"

CMD ["sh"]
