"""The detection helpers moved to crewai_core; their old paths must still work.

``crewai.telemetry.utils`` and ``crewai.utilities.constants`` are the
established import paths for these names, and both are public surface for
anyone who reached for them. Moving the implementation must not break them.
"""

from __future__ import annotations

from crewai_core import runtime_env
import pytest


def test_detection_helpers_still_import_from_telemetry_utils() -> None:
    from crewai.telemetry.utils import (
        KNOWN_CODING_AGENTS,
        KNOWN_RUNTIME_CONTEXTS,
        detect_coding_agent,
        detect_runtime_context,
    )

    assert detect_coding_agent is runtime_env.detect_coding_agent
    assert detect_runtime_context is runtime_env.detect_runtime_context
    assert KNOWN_CODING_AGENTS is runtime_env.KNOWN_CODING_AGENTS
    assert KNOWN_RUNTIME_CONTEXTS is runtime_env.KNOWN_RUNTIME_CONTEXTS


@pytest.mark.parametrize(
    "name",
    [
        "ANTIGRAVITY_ENV_VARS",
        "AUGMENT_ENV_VARS",
        "CC_ENV_VAR",
        "CC_ENV_VARS",
        "CI_ENV_VARS",
        "CLINE_ENV_VARS",
        "CODEX_ENV_VARS",
        "CODING_AGENT_ENV_MARKERS",
        "CONTAINER_ENV_VARS",
        "CURSOR_ENV_VARS",
        "GEMINI_CLI_ENV_VARS",
        "GENERIC_AGENT_ENV_VARS",
        "HOSTED_IDE_ENV_VARS",
        "JUNIE_ENV_VARS",
        "NOTEBOOK_ENV_VARS",
        "OPENCODE_ENV_VARS",
        "PAAS_ENV_VARS",
        "RUNTIME_CONTEXT_ENV_MARKERS",
        "SERVERLESS_ENV_VARS",
    ],
)
def test_marker_tables_still_import_from_utilities_constants(name: str) -> None:
    from crewai.utilities import constants

    assert getattr(constants, name) is getattr(runtime_env, name)
    assert name in constants.__all__


def test_env_context_still_reads_the_shared_table() -> None:
    """get_env_context() walks the same table, so both paths stay in agreement."""
    from crewai.utilities.env import CODING_AGENT_ENV_MARKERS

    assert CODING_AGENT_ENV_MARKERS is runtime_env.CODING_AGENT_ENV_MARKERS
