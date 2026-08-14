#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the CrewAI uv workspace.
set -euo pipefail

# Install uv (the project's package manager) if it is not already available.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Make uv available on PATH for the rest of this script.
if [ -f "$HOME/.local/bin/env" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.local/bin/env"
fi

# Sync the whole workspace: every package, the dev dependency group, and all
# extras. This matches the documented contributor setup and is a no-op when the
# environment is already up to date.
uv sync --all-groups --all-extras
