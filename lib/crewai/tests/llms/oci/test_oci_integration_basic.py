"""Live integration tests for OCI Generative AI basic text completion.

Run with:
    OCI_AUTH_TYPE=API_KEY OCI_AUTH_PROFILE=API_KEY_AUTH \
    OCI_COMPARTMENT_ID=<compartment> OCI_REGION=us-chicago-1 \
    OCI_TEST_MODELS="meta.llama-3.3-70b-instruct,cohere.command-r-plus-08-2024,google.gemini-2.5-flash" \
    uv run pytest tests/llms/oci/test_oci_integration_basic.py -v
"""

from __future__ import annotations

import pytest

from crewai.llms.providers.oci.completion import OCICompletion


def test_oci_live_basic_call(oci_chat_model: str, oci_live_config: dict):
    """Synchronous text completion with a live OCI model."""
    llm = OCICompletion(model=oci_chat_model, **oci_live_config)
    result = llm.call(messages=[{"role": "user", "content": "Say 'hello world' in one sentence."}])

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_oci_live_async_call(oci_chat_model: str, oci_live_config: dict):
    """Async text completion with a live OCI model."""
    llm = OCICompletion(model=oci_chat_model, **oci_live_config)
    result = await llm.acall(messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}])

    assert isinstance(result, str)
    assert len(result) > 0
