from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import copy as shallow_copy, deepcopy
import gc
import threading
import time
from typing import Any

import httpx
import pytest

from crewai.llms.hooks.base import BaseInterceptor
from crewai.llms.providers.openai import completion as openai_completion
from crewai.llms.providers.openai.completion import OpenAICompletion


class _PassThroughInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        return message

    async def aon_outbound(self, message: httpx.Request) -> httpx.Request:
        return message

    async def aon_inbound(self, message: httpx.Response) -> httpx.Response:
        return message


class _FailOnceSyncClient:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("sync close failed")


class _FailOnceAsyncClient:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("async close failed")


class _TrackingSyncAPIClient:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _TrackingAsyncAPIClient:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


def test_shared_ssl_context_is_initialized_once_across_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first access must return one shared context."""
    workers = 32
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


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_lazy_client_is_initialized_once_across_threads(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    workers = 32
    start = threading.Barrier(workers)
    calls = 0
    calls_lock = threading.Lock()
    client = object()
    llm = OpenAICompletion(model="gpt-4o", api_key="test-key")

    def build(_: OpenAICompletion) -> object:
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.02)
        return client

    monkeypatch.setattr(OpenAICompletion, f"_build_{mode}_client", build)

    def get_client(_: int) -> object:
        start.wait()
        getter = getattr(llm, f"_get_{mode}_client")
        return getter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(get_client, range(workers)))

    assert calls == 1
    assert all(result is client for result in results)


def test_deep_copy_has_independent_client_locks() -> None:
    llm = OpenAICompletion(model="gpt-4o", api_key="test-key")

    copied = deepcopy(llm)

    assert copied._sync_client_lock is not llm._sync_client_lock
    assert copied._async_client_lock is not llm._async_client_lock


def test_shallow_copy_before_client_creation_has_independent_runtime_state() -> None:
    llm = OpenAICompletion(model="gpt-4o", api_key="test-key")

    copied = shallow_copy(llm)

    assert copied.model_dump() == llm.model_dump()
    assert copied._sync_client_lock is not llm._sync_client_lock
    assert copied._async_client_lock is not llm._async_client_lock
    assert copied._client is None
    assert copied._async_client is None
    assert not copied._owns_sync_http_client
    assert not copied._owns_async_http_client
    assert not copied._async_client_closing


def test_shallow_copy_does_not_share_provider_owned_sync_client() -> None:
    llm = OpenAICompletion(model="gpt-4o", api_key="test-key")
    original_client = llm._get_sync_client()

    copied = shallow_copy(llm)
    copied_client = copied._get_sync_client()

    assert copied_client is not original_client
    assert copied._sync_client_lock is not llm._sync_client_lock
    copied.close()
    assert copied_client.is_closed()
    assert not original_client.is_closed()
    llm.close()
    assert original_client.is_closed()

    second = OpenAICompletion(model="gpt-4o", api_key="test-key")
    second_client = second._get_sync_client()
    second_copy = shallow_copy(second)
    second_copy_client = second_copy._get_sync_client()

    second.close()
    assert second_client.is_closed()
    assert not second_copy_client.is_closed()
    second_copy.close()
    assert second_copy_client.is_closed()


@pytest.mark.asyncio
async def test_shallow_copy_does_not_share_provider_owned_async_client() -> None:
    assert not openai_completion._PENDING_ASYNC_CLOSE_TASKS
    llm = OpenAICompletion(model="gpt-4o", api_key="test-key")
    original_client = llm._get_async_client()

    copied = shallow_copy(llm)
    copied_client = copied._get_async_client()

    assert copied_client is not original_client
    assert copied._async_client_lock is not llm._async_client_lock
    await copied.aclose()
    assert copied_client.is_closed()
    assert not original_client.is_closed()
    assert not openai_completion._PENDING_ASYNC_CLOSE_TASKS
    await llm.aclose()
    assert original_client.is_closed()

    second = OpenAICompletion(model="gpt-4o", api_key="test-key")
    second_client = second._get_async_client()
    second_copy = shallow_copy(second)
    second_copy_client = second_copy._get_async_client()

    await second.aclose()
    assert second_client.is_closed()
    assert not second_copy_client.is_closed()
    await second_copy.aclose()
    assert second_copy_client.is_closed()
    assert not openai_completion._PENDING_ASYNC_CLOSE_TASKS


@pytest.mark.asyncio
async def test_shallow_copy_preserves_caller_owned_clients() -> None:
    sync_http_client = httpx.Client()
    sync_llm = OpenAICompletion(
        model="gpt-4o",
        api_key="test-key",
        client_params={"http_client": sync_http_client},
    )
    sync_api_client = sync_llm._get_sync_client()

    sync_copy = shallow_copy(sync_llm)

    assert sync_copy._client is sync_api_client
    assert sync_copy._sync_client_lock is not sync_llm._sync_client_lock
    assert not sync_copy._owns_sync_http_client
    sync_copy.close()
    sync_llm.close()
    assert not sync_http_client.is_closed

    async_http_client = httpx.AsyncClient()
    async_llm = OpenAICompletion(
        model="gpt-4o",
        api_key="test-key",
        client_params={"http_client": async_http_client},
    )
    async_api_client = async_llm._get_async_client()

    async_copy = shallow_copy(async_llm)

    assert async_copy._async_client is async_api_client
    assert async_copy._async_client_lock is not async_llm._async_client_lock
    assert not async_copy._owns_async_http_client
    await async_copy.aclose()
    await async_llm.aclose()
    assert not async_http_client.is_closed

    sync_http_client.close()
    await async_http_client.aclose()


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_cleanup_waits_for_concurrent_first_client_creation(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    started = threading.Event()
    release = threading.Event()
    llm = OpenAICompletion(model="gpt-4o", api_key="test-key")
    client = (
        _TrackingSyncAPIClient() if mode == "sync" else _TrackingAsyncAPIClient()
    )

    def build(_: OpenAICompletion) -> object:
        if mode == "sync":
            llm._owns_sync_http_client = True
        else:
            llm._owns_async_http_client = True
        started.set()
        assert release.wait(timeout=5)
        return client

    monkeypatch.setattr(OpenAICompletion, f"_build_{mode}_client", build)

    def close_client() -> None:
        if mode == "sync":
            llm.close()
        else:
            asyncio.run(llm.aclose())

    with ThreadPoolExecutor(max_workers=2) as pool:
        get_future = pool.submit(getattr(llm, f"_get_{mode}_client"))
        assert started.wait(timeout=5)
        close_future = pool.submit(close_client)
        time.sleep(0.02)
        assert not close_future.done()
        release.set()
        assert get_future.result(timeout=5) is client
        close_future.result(timeout=5)

    assert client.close_calls == 1
    assert getattr(llm, f"_{mode}_client" if mode == "async" else "_client") is None


def test_concurrent_async_cleanup_closes_once_and_rejects_new_getter() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingAsyncClient:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.001)

    llm = OpenAICompletion(model="gpt-4o", api_key="test-key")
    client = BlockingAsyncClient()
    llm._async_client = client
    llm._owns_async_http_client = True

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_close = pool.submit(lambda: asyncio.run(llm.aclose()))
        assert started.wait(timeout=5)
        second_close = pool.submit(lambda: asyncio.run(llm.aclose()))
        time.sleep(0.02)
        assert not second_close.done()
        with pytest.raises(RuntimeError, match="being closed"):
            llm._get_async_client()
        release.set()
        first_close.result(timeout=5)
        second_close.result(timeout=5)

    assert client.close_calls == 1
    assert llm._async_client is None
    assert not llm._owns_async_http_client


def test_sync_client_is_lazy_and_does_not_create_async_client() -> None:
    llm = OpenAICompletion(model="gpt-4o", api_key="test-key")

    assert llm._client is None
    assert llm._async_client is None

    llm._get_sync_client()

    assert llm._client is not None
    assert llm._async_client is None
    llm.close()


@pytest.mark.asyncio
async def test_async_client_is_lazy_and_does_not_create_sync_client() -> None:
    llm = OpenAICompletion(model="gpt-4o", api_key="test-key")

    llm._get_async_client()

    assert llm._client is None
    assert llm._async_client is not None
    await llm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_first", [False, True])
@pytest.mark.parametrize("interceptor", [None, _PassThroughInterceptor()])
async def test_close_order_is_idempotent(
    async_first: bool,
    interceptor: _PassThroughInterceptor | None,
) -> None:
    llm = OpenAICompletion(
        model="gpt-4o", api_key="test-key", interceptor=interceptor
    )
    sync_http_client = llm._get_sync_client()._client
    async_http_client = llm._get_async_client()._client

    if async_first:
        await llm.aclose()
        llm.close()
    else:
        llm.close()
        await llm.aclose()
    llm.close()
    await llm.aclose()

    assert sync_http_client.is_closed
    assert async_http_client.is_closed
    assert llm._client is None
    assert llm._async_client is None


@pytest.mark.asyncio
async def test_close_preserves_user_supplied_http_clients() -> None:
    sync_http_client = httpx.Client()
    sync_llm = OpenAICompletion(
        model="gpt-4o",
        api_key="test-key",
        client_params={"http_client": sync_http_client},
    )
    sync_api_client = sync_llm._get_sync_client()

    sync_llm.close()

    assert sync_llm._client is sync_api_client
    assert not sync_http_client.is_closed

    async_http_client = httpx.AsyncClient()
    async_llm = OpenAICompletion(
        model="gpt-4o",
        api_key="test-key",
        client_params={"http_client": async_http_client},
    )
    async_api_client = async_llm._get_async_client()

    await async_llm.aclose()

    assert async_llm._async_client is async_api_client
    assert not async_http_client.is_closed

    sync_http_client.close()
    await async_http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sync_name", "async_name", "interceptor"),
    [
        ("_SharedSSLHttpxClient", "_SharedSSLAsyncHttpxClient", None),
        (
            "_InterceptorHttpxClient",
            "_InterceptorAsyncHttpxClient",
            _PassThroughInterceptor(),
        ),
    ],
)
async def test_normal_disposal_finalizes_provider_owned_clients(
    monkeypatch: pytest.MonkeyPatch,
    sync_name: str,
    async_name: str,
    interceptor: _PassThroughInterceptor | None,
) -> None:
    closed = {"sync": 0, "async": 0}
    sync_base = getattr(openai_completion, sync_name)
    async_base = getattr(openai_completion, async_name)

    class TrackingSync(sync_base):
        def close(self) -> None:
            closed["sync"] += 1
            super().close()

    class TrackingAsync(async_base):
        async def aclose(self) -> None:
            closed["async"] += 1
            await super().aclose()

    monkeypatch.setattr(openai_completion, sync_name, TrackingSync)
    monkeypatch.setattr(openai_completion, async_name, TrackingAsync)
    llm = OpenAICompletion(
        model="gpt-4o", api_key="test-key", interceptor=interceptor
    )
    llm._get_sync_client()
    llm._get_async_client()

    del llm
    gc.collect()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert closed == {"sync": 1, "async": 1}


@pytest.mark.asyncio
async def test_context_managers_close_only_created_clients() -> None:
    with OpenAICompletion(model="gpt-4o", api_key="test-key") as sync_llm:
        sync_http_client = sync_llm._get_sync_client()._client
        assert sync_llm._async_client is None
    assert sync_http_client.is_closed

    async with OpenAICompletion(model="gpt-4o", api_key="test-key") as async_llm:
        async_http_client = async_llm._get_async_client()._client
        assert async_llm._client is None
    assert async_http_client.is_closed


@pytest.mark.asyncio
async def test_client_initialization_failure_closes_provider_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_http_client = httpx.Client()
    async_http_client = httpx.AsyncClient()
    monkeypatch.setattr(
        openai_completion, "_SharedSSLHttpxClient", lambda **_: sync_http_client
    )
    monkeypatch.setattr(
        openai_completion,
        "_SharedSSLAsyncHttpxClient",
        lambda **_: async_http_client,
    )
    monkeypatch.setattr(
        openai_completion, "OpenAI", lambda **_: (_ for _ in ()).throw(ValueError())
    )

    llm = OpenAICompletion(model="gpt-4o", api_key="test-key")
    with pytest.raises(ValueError):
        llm._get_sync_client()
    assert sync_http_client.is_closed
    assert not llm._owns_sync_http_client

    monkeypatch.setattr(
        openai_completion,
        "AsyncOpenAI",
        lambda **_: (_ for _ in ()).throw(ValueError()),
    )
    with pytest.raises(ValueError):
        llm._get_async_client()
    await asyncio.sleep(0)
    assert async_http_client.is_closed
    assert not llm._owns_async_http_client


def test_async_initialization_failure_without_loop_closes_only_owned_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned_http_client = httpx.AsyncClient()
    caller_http_client = httpx.AsyncClient()
    monkeypatch.setattr(
        openai_completion,
        "_SharedSSLAsyncHttpxClient",
        lambda **_: owned_http_client,
    )
    monkeypatch.setattr(
        openai_completion,
        "AsyncOpenAI",
        lambda **_: (_ for _ in ()).throw(ValueError("initialization failed")),
    )

    owned_llm = OpenAICompletion(model="gpt-4o", api_key="test-key")
    with pytest.raises(ValueError, match="initialization failed"):
        owned_llm._get_async_client()

    assert owned_http_client.is_closed
    assert not owned_llm._owns_async_http_client

    caller_llm = OpenAICompletion(
        model="gpt-4o",
        api_key="test-key",
        client_params={"http_client": caller_http_client},
    )
    with pytest.raises(ValueError, match="initialization failed"):
        caller_llm._get_async_client()

    assert not caller_http_client.is_closed
    asyncio.run(caller_http_client.aclose())


def test_async_initialization_failure_preserves_original_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingAsyncClient:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1
            raise RuntimeError("cleanup failed")

    http_client = FailingAsyncClient()
    monkeypatch.setattr(
        openai_completion,
        "_SharedSSLAsyncHttpxClient",
        lambda **_: http_client,
    )
    monkeypatch.setattr(
        openai_completion,
        "AsyncOpenAI",
        lambda **_: (_ for _ in ()).throw(ValueError("initialization failed")),
    )

    llm = OpenAICompletion(model="gpt-4o", api_key="test-key")
    with pytest.raises(ValueError, match="initialization failed"):
        llm._get_async_client()

    assert http_client.close_calls == 1
    assert not llm._owns_async_http_client


@pytest.mark.asyncio
async def test_cleanup_failures_remain_retryable() -> None:
    llm = OpenAICompletion(model="gpt-4o", api_key="test-key")
    sync_client = _FailOnceSyncClient()
    async_client = _FailOnceAsyncClient()
    llm._client = sync_client
    llm._async_client = async_client
    llm._owns_sync_http_client = True
    llm._owns_async_http_client = True

    with pytest.raises(RuntimeError, match="sync close failed"):
        llm.close()
    assert llm._client is sync_client
    assert llm._owns_sync_http_client

    with pytest.raises(RuntimeError, match="async close failed"):
        await llm.aclose()
    assert llm._client is None
    assert llm._async_client is async_client
    assert llm._owns_async_http_client

    await llm.aclose()
    assert llm._async_client is None
