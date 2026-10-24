"""Regression tests: the A2A cancel watcher must close its Redis pubsub.

Every executed task opens a pubsub subscription for its cancel channel. When
the wrapped task finished first (or a cancel message arrived, or the Redis
connection failed), the subscription used to be abandoned — leaking one Redis
connection per task on servers configured with a Redis cache backend.
"""

import asyncio
from typing import Any

import pytest
from a2a.server.agent_execution import RequestContext

import crewai.a2a.utils.task as task_mod
from crewai.a2a.utils.task import cancellable


class FakePubSub:
    def __init__(self, fail_subscribe: Exception | None = None) -> None:
        self.channels: list[str] = []
        self.closed = False
        self.fail_subscribe = fail_subscribe
        self._deliver_message = asyncio.Event()

    async def subscribe(self, *channels: str) -> None:
        if self.fail_subscribe is not None:
            raise self.fail_subscribe
        self.channels.extend(channels)

    async def listen(self):
        await self._deliver_message.wait()
        yield {"type": "message"}

    async def aclose(self) -> None:
        self.closed = True


class FakeRedisClient:
    def __init__(self, pubsub_obj: FakePubSub) -> None:
        self._pubsub = pubsub_obj

    def pubsub(self) -> FakePubSub:
        return self._pubsub


class FakeRedisCache:
    """Stands in for aiocache.RedisCache (the non-memory backend)."""

    def __init__(self, pubsub_obj: FakePubSub) -> None:
        self.client = FakeRedisClient(pubsub_obj)

    async def get(self, key: str) -> Any:
        return None

    async def delete(self, key: str) -> None:
        return None


class FakeCaches:
    def __init__(self, cache: FakeRedisCache) -> None:
        self._cache = cache

    def get(self, name: str) -> FakeRedisCache:
        return self._cache


def _install_fake_cache(monkeypatch: pytest.MonkeyPatch, pubsub: FakePubSub) -> None:
    monkeypatch.setattr(task_mod, "caches", FakeCaches(FakeRedisCache(pubsub)))


@pytest.mark.asyncio
async def test_pubsub_closed_when_task_completes_first(monkeypatch):
    pubsub = FakePubSub()
    _install_fake_cache(monkeypatch, pubsub)

    @cancellable
    async def do_work(context: RequestContext) -> str:
        await asyncio.sleep(0.1)
        return "done"

    result = await do_work(RequestContext(task_id="t1", context_id="c1"))
    await asyncio.sleep(0.05)  # let the cancelled watcher run its cleanup

    assert result == "done"
    assert pubsub.channels == ["cancel:t1"]
    assert pubsub.closed, "pubsub subscription leaked after task completion"


@pytest.mark.asyncio
async def test_pubsub_closed_when_cancel_message_received(monkeypatch):
    pubsub = FakePubSub()
    _install_fake_cache(monkeypatch, pubsub)
    started = asyncio.Event()

    @cancellable
    async def do_work(context: RequestContext) -> str:
        started.set()
        await asyncio.sleep(30.0)
        return "done"

    task = asyncio.create_task(
        do_work(RequestContext(task_id="t2", context_id="c2"))
    )
    await started.wait()
    pubsub._deliver_message.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert pubsub.closed, "pubsub subscription leaked after cancellation"


@pytest.mark.asyncio
async def test_pubsub_closed_when_redis_fails_and_polling_takes_over(monkeypatch):
    pubsub = FakePubSub(fail_subscribe=ConnectionError("redis down"))
    _install_fake_cache(monkeypatch, pubsub)

    @cancellable
    async def do_work(context: RequestContext) -> str:
        await asyncio.sleep(0.15)
        return "done"

    result = await do_work(RequestContext(task_id="t3", context_id="c3"))
    await asyncio.sleep(0.05)

    assert result == "done", "polling fallback should keep the run alive"
    assert pubsub.closed, "pubsub subscription leaked on the error path"
