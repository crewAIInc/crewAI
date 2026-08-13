from concurrent.futures import ThreadPoolExecutor
import threading
import time

import httpx
import pytest

from crewai.llms.providers.openai import completion as openai_completion
from crewai.llms.providers.openai.completion import OpenAICompletion


def test_shared_ssl_context_is_initialized_once_across_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first access must return one shared context."""
    workers = 8
    start = threading.Barrier(workers)
    calls_lock = threading.Lock()
    contexts: list[object] = []

    def create_ssl_context() -> object:
        context = object()
        with calls_lock:
            contexts.append(context)
        time.sleep(0.02)
        return context

    def get_context(_: int) -> object:
        start.wait()
        return openai_completion._shared_ssl_context()

    monkeypatch.setattr(openai_completion, "_SHARED_SSL_CONTEXT", None)
    monkeypatch.setattr(
        openai_completion.httpx, "create_ssl_context", create_ssl_context
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(get_context, range(workers)))

    assert len(contexts) == 1
    assert all(context is results[0] for context in results)


@pytest.mark.asyncio
async def test_close_closes_only_provider_owned_http_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle hooks close both HTTP clients created by the provider."""
    sync_http_client = httpx.Client()
    async_http_client = httpx.AsyncClient()
    monkeypatch.setattr(
        openai_completion,
        "DefaultHttpxClient",
        lambda **_: sync_http_client,
    )
    monkeypatch.setattr(
        openai_completion,
        "DefaultAsyncHttpxClient",
        lambda **_: async_http_client,
    )

    llm = OpenAICompletion(model="gpt-4o", api_key="test-key")

    llm.close()
    await llm.aclose()

    assert sync_http_client.is_closed
    assert async_http_client.is_closed
    assert llm._client is None
    assert llm._async_client is None


@pytest.mark.asyncio
async def test_close_preserves_user_supplied_http_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle hooks must not close clients passed through client_params."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = OpenAICompletion(model="gpt-4o")
    llm.api_key = "test-key"

    sync_http_client = httpx.Client()
    llm.client_params = {"http_client": sync_http_client}
    llm._client = llm._build_sync_client()
    llm.close()

    async_http_client = httpx.AsyncClient()
    llm.client_params = {"http_client": async_http_client}
    llm._async_client = llm._build_async_client()
    await llm.aclose()

    assert not sync_http_client.is_closed
    assert not async_http_client.is_closed

    sync_http_client.close()
    await async_http_client.aclose()
