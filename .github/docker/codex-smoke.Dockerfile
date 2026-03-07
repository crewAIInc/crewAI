FROM python:3.12-slim-bookworm

ENV HOME=/root \
    CODEX_HOME=/root/.codex \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CREWAI_DISABLE_TELEMETRY=true \
    OTEL_SDK_DISABLED=true

WORKDIR /workspace

COPY lib/crewai/pyproject.toml /tmp/crewai-pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && python - <<'PY' >/tmp/requirements.txt
import pathlib
import tomllib

payload = tomllib.loads(pathlib.Path("/tmp/crewai-pyproject.toml").read_text())
for requirement in payload["project"]["dependencies"]:
    print(requirement)
PY
 && /opt/venv/bin/pip install -r /tmp/requirements.txt

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/workspace/lib/crewai/src:/workspace/scripts"

CMD ["sh"]
