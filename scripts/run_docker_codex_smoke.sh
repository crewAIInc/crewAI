#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-crewai-codex-smoke:local}"
HOST_CODEX_HOME="${CODEX_HOME:-${HOME}/.codex}"
HOST_AUTH_JSON="${HOST_CODEX_HOME%/}/auth.json"
TEMP_CODEX_HOME="$(mktemp -d "${TMPDIR:-/tmp}/crewai-codex-home.XXXXXX")"
TEMP_AUTH_JSON="${TEMP_CODEX_HOME}/auth.json"

cleanup() {
  if [[ -f "${TEMP_AUTH_JSON}" ]]; then
    install -m 600 "${TEMP_AUTH_JSON}" "${HOST_AUTH_JSON}"
  fi
  rm -rf "${TEMP_CODEX_HOME}"
}

trap cleanup EXIT

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is required for the isolated codex smoke test." >&2
  exit 1
fi

if [[ ! -f "${HOST_AUTH_JSON}" ]]; then
  echo "ERROR: Codex auth.json not found at ${HOST_AUTH_JSON}." >&2
  exit 1
fi

install -d -m 700 "${TEMP_CODEX_HOME}"
install -m 600 "${HOST_AUTH_JSON}" "${TEMP_AUTH_JSON}"

docker build \
  --tag "${IMAGE_TAG}" \
  --file .github/docker/codex-smoke.Dockerfile \
  .

docker run --rm \
  -e HOME=/root \
  -e CODEX_HOME=/root/.codex \
  -e CREWAI_TRACING_ENABLED=false \
  -v "${TEMP_CODEX_HOME}:/root/.codex" \
  "${IMAGE_TAG}" \
  sh -c '
    set -eu
    python scripts/check_model_access.py --model gpt-5.3-codex --api both
    python scripts/demo_openai_codex_hi.py --model openai-codex/gpt-5.3-codex --prompt Hi
  '
