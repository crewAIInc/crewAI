"""Tests for OpenAI SDK version compatibility.

These tests verify that crewAI's openai dependency constraint allows
installation alongside packages that require openai 2.x (like litellm[proxy]).

Related to GitHub issue #4079: CrewAI dependency conflict with litellm[proxy]
"""

import sys
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet


def test_openai_version_constraint_allows_2x():
    """Test that the openai version constraint in pyproject.toml allows openai 2.x.

    This test verifies the fix for issue #4079 where crewAI could not be installed
    alongside litellm[proxy]>=1.74.9 due to conflicting openai version requirements.

    The constraint should allow openai>=1.83.0,<3 to support both:
    - Existing users on openai 1.83.x
    - Users who need openai 2.x for litellm[proxy] compatibility
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    # Find the pyproject.toml file
    tests_dir = Path(__file__).parent
    pyproject_path = tests_dir.parent / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    dependencies = pyproject.get("project", {}).get("dependencies", [])

    # Find the openai dependency
    openai_dep = None
    for dep in dependencies:
        if dep.startswith("openai"):
            openai_dep = dep
            break

    assert openai_dep is not None, "openai dependency not found in pyproject.toml"

    # Extract the version specifier from the dependency string
    # e.g., "openai>=1.83.0,<3" -> ">=1.83.0,<3"
    version_spec = openai_dep.replace("openai", "").strip()
    specifier = SpecifierSet(version_spec)

    # Test that the specifier allows openai 2.8.0 (required by litellm[proxy])
    assert "2.8.0" in specifier, (
        f"openai constraint '{openai_dep}' does not allow version 2.8.0 "
        "which is required by litellm[proxy]>=1.74.9"
    )

    # Test that the specifier still allows openai 1.83.0 (backward compatibility)
    assert "1.83.0" in specifier, (
        f"openai constraint '{openai_dep}' does not allow version 1.83.0 "
        "which breaks backward compatibility"
    )

    # Test that the specifier has an upper bound (to prevent future breaks)
    assert "<3" in version_spec or "<3.0" in version_spec, (
        f"openai constraint '{openai_dep}' should have an upper bound <3 "
        "to prevent future breaking changes"
    )


def test_openai_provider_imports_are_valid():
    """Test that all imports used by the OpenAI provider are valid.

    This test verifies that the OpenAI SDK exports all the classes and types
    that crewAI's OpenAI provider depends on. This ensures compatibility
    across different openai SDK versions.
    """
    # Test core client imports
    from openai import APIConnectionError, AsyncOpenAI, NotFoundError, OpenAI, Stream

    assert OpenAI is not None
    assert AsyncOpenAI is not None
    assert Stream is not None
    assert APIConnectionError is not None
    assert NotFoundError is not None

    # Test streaming imports
    from openai.lib.streaming.chat import ChatCompletionStream

    assert ChatCompletionStream is not None

    # Test type imports
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import ChoiceDelta

    assert ChatCompletion is not None
    assert ChatCompletionChunk is not None
    assert Choice is not None
    assert ChoiceDelta is not None


def test_openai_client_instantiation():
    """Test that OpenAI clients can be instantiated with a test API key.

    This verifies that the OpenAI SDK client initialization is compatible
    with crewAI's usage patterns.
    """
    from openai import AsyncOpenAI, OpenAI

    # Test sync client instantiation
    client = OpenAI(api_key="test-key-for-instantiation-test")
    assert client is not None
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")

    # Test async client instantiation
    async_client = AsyncOpenAI(api_key="test-key-for-instantiation-test")
    assert async_client is not None
    assert hasattr(async_client, "chat")
    assert hasattr(async_client.chat, "completions")


def test_openai_completion_provider_can_be_imported():
    """Test that crewAI's OpenAI completion provider can be imported.

    This verifies that the OpenAI provider module loads correctly with
    the installed openai SDK version.
    """
    from crewai.llms.providers.openai.completion import OpenAICompletion

    assert OpenAICompletion is not None


def test_openai_completion_provider_instantiation():
    """Test that OpenAICompletion can be instantiated.

    This verifies that crewAI's OpenAI provider works correctly with
    the installed openai SDK version.
    """
    from crewai.llms.providers.openai.completion import OpenAICompletion

    # Instantiate with a test API key
    completion = OpenAICompletion(
        model="gpt-4o",
        api_key="test-key-for-instantiation-test",
    )

    assert completion is not None
    assert completion.model == "gpt-4o"
    assert completion.client is not None
    assert completion.async_client is not None
