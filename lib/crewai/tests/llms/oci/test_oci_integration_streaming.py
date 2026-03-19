"""Live integration tests for OCI Generative AI streaming.

Run with:
    OCI_AUTH_TYPE=API_KEY OCI_AUTH_PROFILE=API_KEY_AUTH \
    OCI_COMPARTMENT_ID=<compartment> OCI_REGION=us-chicago-1 \
    OCI_TEST_MODELS="meta.llama-3.3-70b-instruct,cohere.command-r-plus-08-2024" \
    uv run pytest tests/llms/oci/test_oci_integration_streaming.py -v
"""

from __future__ import annotations

import pytest

from crewai.llms.providers.oci.completion import OCICompletion


def test_oci_live_streaming_call(oci_chat_model: str, oci_live_config: dict):
    """Streaming text completion with a live OCI model."""
    llm = OCICompletion(model=oci_chat_model, stream=True, **oci_live_config)
    result = llm.call(
        messages=[{"role": "user", "content": "Count from 1 to 5, one per line."}]
    )

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_oci_live_astream(oci_chat_model: str, oci_live_config: dict):
    """Async streaming should yield text chunks from a live OCI model."""
    llm = OCICompletion(model=oci_chat_model, **oci_live_config)
    chunks: list[str] = []
    async for chunk in llm.astream(
        messages=[{"role": "user", "content": "Say hello in three words."}]
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert len(full_text) > 0
