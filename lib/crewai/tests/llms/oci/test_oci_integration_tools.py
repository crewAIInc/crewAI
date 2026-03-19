"""Live integration tests for OCI Generative AI tool calling.

Run with:
    OCI_AUTH_TYPE=API_KEY OCI_AUTH_PROFILE=API_KEY_AUTH \
    OCI_COMPARTMENT_ID=<compartment> OCI_REGION=us-chicago-1 \
    OCI_TEST_TOOL_MODELS="meta.llama-3.3-70b-instruct" \
    uv run pytest tests/llms/oci/test_oci_integration_tools.py -v
"""

from __future__ import annotations

import os

import pytest

from crewai.llms.providers.oci.completion import OCICompletion


def _env_models(env_var: str, fallback: str, default: str) -> list[str]:
    raw = os.getenv(env_var) or os.getenv(fallback) or default
    return [m.strip() for m in raw.split(",") if m.strip()]


def _skip_unless_live():
    compartment = os.getenv("OCI_COMPARTMENT_ID")
    if not compartment:
        pytest.skip("OCI_COMPARTMENT_ID not set")
    region = os.getenv("OCI_REGION")
    endpoint = os.getenv("OCI_SERVICE_ENDPOINT")
    if not region and not endpoint:
        pytest.skip("Set OCI_REGION or OCI_SERVICE_ENDPOINT")
    config: dict[str, str] = {"compartment_id": compartment}
    if endpoint:
        config["service_endpoint"] = endpoint
    if os.getenv("OCI_AUTH_TYPE"):
        config["auth_type"] = os.getenv("OCI_AUTH_TYPE", "API_KEY")
    if os.getenv("OCI_AUTH_PROFILE"):
        config["auth_profile"] = os.getenv("OCI_AUTH_PROFILE", "DEFAULT")
    return config


TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


@pytest.fixture(
    params=_env_models("OCI_TEST_TOOL_MODELS", "OCI_TEST_TOOL_MODEL", "meta.llama-3.3-70b-instruct"),
    ids=lambda m: m,
)
def oci_tool_model(request):
    return request.param


@pytest.fixture()
def oci_tool_config():
    return _skip_unless_live()


def test_oci_live_tool_call_returns_raw(oci_tool_model: str, oci_tool_config: dict):
    """Without available_functions, tool calls should be returned raw."""
    llm = OCICompletion(model=oci_tool_model, **oci_tool_config)
    result = llm.call(
        messages=[{"role": "user", "content": "What is 3 + 7? Use the add_numbers tool."}],
        tools=TOOL_SPEC,
    )

    assert isinstance(result, list)
    assert len(result) >= 1
    assert result[0]["function"]["name"] == "add_numbers"


def test_oci_live_tool_call_with_execution(oci_tool_model: str, oci_tool_config: dict):
    """With available_functions, tools should execute and model should respond."""
    def add_numbers(a: float, b: float) -> str:
        return str(float(a) + float(b))

    llm = OCICompletion(model=oci_tool_model, **oci_tool_config)
    result = llm.call(
        messages=[{"role": "user", "content": "What is 3 + 7? Use the add_numbers tool."}],
        tools=TOOL_SPEC,
        available_functions={"add_numbers": add_numbers},
    )

    assert isinstance(result, str)
    assert "10" in result
