"""Test that dependency constraints are properly configured to avoid conflicts."""

from pathlib import Path

import tomli


def test_openai_version_constraint_allows_mem0ai_compatibility():
    """Test that the openai version constraint allows versions >= 1.90.0.

    This test ensures that the openai dependency constraint is flexible enough
    to allow installation alongside packages like mem0ai that require openai>=1.90.0.

    See: https://github.com/crewAIInc/crewAI/issues/4098
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)

    dependencies = pyproject.get("project", {}).get("dependencies", [])

    openai_dep = None
    for dep in dependencies:
        if dep.startswith("openai"):
            openai_dep = dep
            break

    assert openai_dep is not None, "openai dependency not found in pyproject.toml"

    assert "~=" not in openai_dep, (
        f"openai dependency uses ~= operator which is too restrictive: {openai_dep}. "
        "This causes conflicts with packages like mem0ai that require openai>=1.90.0"
    )

    assert ">=" in openai_dep, (
        f"openai dependency should use >= operator for minimum version: {openai_dep}"
    )

    assert "1.83.0" in openai_dep, (
        f"openai dependency should have minimum version 1.83.0: {openai_dep}"
    )


def test_openai_imports_are_available():
    """Test that all OpenAI imports used by CrewAI are available.

    This test verifies that the OpenAI SDK version installed provides
    all the imports that CrewAI depends on.
    """
    from openai import APIConnectionError, AsyncOpenAI, NotFoundError, OpenAI, Stream
    from openai.lib.streaming.chat import ChatCompletionStream
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import ChoiceDelta

    assert OpenAI is not None
    assert AsyncOpenAI is not None
    assert Stream is not None
    assert APIConnectionError is not None
    assert NotFoundError is not None
    assert ChatCompletion is not None
    assert ChatCompletionChunk is not None
    assert ChatCompletionStream is not None
    assert Choice is not None
    assert ChoiceDelta is not None
