"""Tests for OCI true async support (aiohttp-based)."""

from __future__ import annotations

import os

import pytest

from crewai.llms.providers.oci.completion import OCICompletion


def _skip_unless_live() -> dict[str, str]:
    compartment = os.getenv("OCI_COMPARTMENT_ID")
    if not compartment:
        pytest.skip("OCI_COMPARTMENT_ID not set")
    region = os.getenv("OCI_REGION")
    if not region:
        pytest.skip("OCI_REGION not set")
    config: dict[str, str] = {"compartment_id": compartment}
    if os.getenv("OCI_AUTH_TYPE"):
        config["auth_type"] = os.getenv("OCI_AUTH_TYPE", "API_KEY")
    if os.getenv("OCI_AUTH_PROFILE"):
        config["auth_profile"] = os.getenv("OCI_AUTH_PROFILE", "DEFAULT")
    return config


@pytest.fixture()
def oci_async_config():
    return _skip_unless_live()


@pytest.mark.asyncio
async def test_oci_true_async_client_is_used(oci_async_config: dict):
    """Verify the true async client is initialized with a real OCI SDK client."""
    from crewai.utilities.oci_async import OCIAsyncClient

    llm = OCICompletion(
        model="meta.llama-3.3-70b-instruct",
        **oci_async_config,
    )
    assert llm._async_client is not None
    assert isinstance(llm._async_client, OCIAsyncClient)


@pytest.mark.asyncio
async def test_oci_true_async_acall(oci_async_config: dict):
    """True async acall should return a text response without blocking threads."""
    llm = OCICompletion(
        model="meta.llama-3.3-70b-instruct",
        **oci_async_config,
    )
    result = await llm.acall(
        messages=[{"role": "user", "content": "Say hello in one word."}]
    )
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_oci_true_async_astream(oci_async_config: dict):
    """True async astream should yield chunks without thread bridges."""
    llm = OCICompletion(
        model="meta.llama-3.3-70b-instruct",
        **oci_async_config,
    )
    chunks: list[str] = []
    async for chunk in llm.astream(
        messages=[{"role": "user", "content": "Count to 3."}]
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    full = "".join(chunks)
    assert len(full) > 0


@pytest.mark.asyncio
async def test_oci_true_async_concurrent_calls(oci_async_config: dict):
    """Multiple concurrent acall should run without blocking each other."""
    import asyncio

    llm = OCICompletion(
        model="meta.llama-3.3-70b-instruct",
        **oci_async_config,
    )

    results = await asyncio.gather(
        llm.acall(messages=[{"role": "user", "content": "Say 'one'"}]),
        llm.acall(messages=[{"role": "user", "content": "Say 'two'"}]),
    )

    assert len(results) == 2
    assert all(isinstance(r, str) and len(r) > 0 for r in results)
