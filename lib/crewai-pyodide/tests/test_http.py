"""The fetch transports, driven against a fake JavaScript ``fetch``."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest

from crewai_pyodide.http import FetchAsyncTransport, FetchTransport


class _Bytes(bytes):
    def to_bytes(self) -> bytes:
        return bytes(self)


@dataclass
class _Reader:
    chunks: list[bytes]
    cancelled: bool = False

    async def read(self) -> SimpleNamespace:
        if not self.chunks:
            return SimpleNamespace(done=True, value=None)
        return SimpleNamespace(done=False, value=_Bytes(self.chunks.pop(0)))

    async def cancel(self) -> None:
        self.cancelled = True


class _Headers:
    def __init__(self, items: list[tuple[str, str]]) -> None:
        self._items = items

    def entries(self) -> list[tuple[str, str]]:
        return self._items


class _JsException(Exception):
    pass


class _Proxy:
    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self.destroyed = False

    def __call__(self) -> None:
        self._fn()

    def destroy(self) -> None:
        self.destroyed = True


@pytest.fixture
def fake_js(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Install fake ``js`` and ``pyodide.ffi`` modules; returns call records."""
    state = SimpleNamespace(
        requests=[],
        chunks=[b'{"ok": ', b"true}"],
        error=None,
        hang=False,
        aborted=0,
        abort_event=asyncio.Event(),
        timers=set(),
    )

    async def wait_for_abort() -> None:
        # A real fetch rejects its pending promises once its signal aborts.
        await state.abort_event.wait()
        raise _JsException("AbortError")

    state.wait_for_abort = wait_for_abort

    async def fetch(url: str, init: dict[str, Any]) -> SimpleNamespace:
        if state.error:
            raise state.error
        if state.hang:
            await wait_for_abort()
        state.requests.append((url, init))
        state.reader = _Reader(list(state.chunks))
        return SimpleNamespace(
            status=200,
            headers=_Headers(
                [
                    ("content-type", "application/json"),
                    ("content-encoding", "gzip"),
                    ("content-length", "3"),
                ]
            ),
            body=SimpleNamespace(getReader=lambda: state.reader),
        )

    js = ModuleType("js")
    js.fetch = fetch  # type: ignore[attr-defined]
    js.Object = SimpleNamespace(fromEntries=dict)  # type: ignore[attr-defined]

    def abort_controller() -> SimpleNamespace:
        def abort() -> None:
            state.aborted += 1
            state.abort_event.set()

        return SimpleNamespace(signal=object(), abort=abort)

    def set_timeout(callback: _Proxy, ms: float) -> asyncio.TimerHandle:
        def fire() -> None:
            state.timers.discard(handle)
            callback()

        handle = asyncio.get_running_loop().call_later(ms / 1000, fire)
        state.timers.add(handle)
        return handle

    def clear_timeout(handle: asyncio.TimerHandle) -> None:
        handle.cancel()
        state.timers.discard(handle)

    js.AbortController = SimpleNamespace(new=abort_controller)  # type: ignore[attr-defined]
    js.setTimeout = set_timeout  # type: ignore[attr-defined]
    js.clearTimeout = clear_timeout  # type: ignore[attr-defined]
    ffi = ModuleType("pyodide.ffi")
    ffi.JsException = _JsException  # type: ignore[attr-defined]
    ffi.to_js = lambda value, dict_converter=None: value  # type: ignore[attr-defined]
    ffi.create_proxy = _Proxy  # type: ignore[attr-defined]
    ffi.can_run_sync = lambda: state.can_run_sync  # type: ignore[attr-defined]
    state.can_run_sync = True

    def run_sync(coro: Any) -> Any:
        import asyncio

        return asyncio.run(coro)

    ffi.run_sync = run_sync  # type: ignore[attr-defined]
    pyodide = ModuleType("pyodide")
    pyodide.ffi = ffi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "js", js)
    monkeypatch.setitem(sys.modules, "pyodide", pyodide)
    monkeypatch.setitem(sys.modules, "pyodide.ffi", ffi)
    return state


@pytest.mark.asyncio
async def test_async_requests_go_through_fetch(fake_js: SimpleNamespace) -> None:
    async with httpx.AsyncClient(transport=FetchAsyncTransport()) as client:
        response = await client.post(
            "https://api.example.com/v1/chat",
            json={"q": 1},
            headers={"Authorization": "Bearer k"},
        )

    assert response.json() == {"ok": True}
    ((url, init),) = fake_js.requests
    assert url == "https://api.example.com/v1/chat"
    assert init["method"] == "POST"
    assert init["body"] == b'{"q":1}'
    sent = {k.lower() for k, _ in init["headers"]}
    assert "authorization" in sent
    # The browser owns these; fetch rejects or drops them.
    assert not sent & {"host", "content-length", "accept-encoding", "connection"}


@pytest.mark.asyncio
async def test_responses_stream_and_are_not_decoded_twice(
    fake_js: SimpleNamespace,
) -> None:
    async with httpx.AsyncClient(transport=FetchAsyncTransport()) as client:
        async with client.stream("GET", "https://api.example.com/sse") as response:
            chunks = [chunk async for chunk in response.aiter_raw()]

    assert chunks == [b'{"ok": ', b"true}"]
    # fetch already decompressed the body.
    assert "content-encoding" not in response.headers


@pytest.mark.asyncio
async def test_network_failures_raise_connect_error(fake_js: SimpleNamespace) -> None:
    fake_js.error = _JsException("TypeError: Failed to fetch")

    async with httpx.AsyncClient(transport=FetchAsyncTransport()) as client:
        with pytest.raises(httpx.ConnectError, match="Failed to fetch"):
            await client.get("https://api.example.com/")


def test_sync_requests_block_on_fetch(fake_js: SimpleNamespace) -> None:
    with httpx.Client(transport=FetchTransport()) as client:
        assert client.get("https://api.example.com/").json() == {"ok": True}


def test_sync_requests_without_jspi_explain_the_requirement(
    fake_js: SimpleNamespace,
) -> None:
    fake_js.can_run_sync = False

    with httpx.Client(transport=FetchTransport()) as client:
        with pytest.raises(RuntimeError, match="JSPI"):
            client.get("https://api.example.com/")


@pytest.mark.asyncio
async def test_body_failures_raise_read_error(fake_js: SimpleNamespace) -> None:
    async def fail() -> SimpleNamespace:
        raise _JsException("TypeError: network error")

    async with httpx.AsyncClient(transport=FetchAsyncTransport()) as client:
        async with client.stream("GET", "https://api.example.com/sse") as response:
            fake_js.reader.read = fail
            with pytest.raises(httpx.ReadError, match="network error"):
                await response.aread()


def test_sync_requests_send_streamed_bodies(fake_js: SimpleNamespace) -> None:
    with httpx.Client(transport=FetchTransport()) as client:
        client.post("https://api.example.com/", content=iter([b"a", b"b"]))

    ((_, init),) = fake_js.requests
    assert init["body"] == b"ab"


@pytest.mark.asyncio
async def test_unanswered_requests_time_out_and_abort_the_fetch(
    fake_js: SimpleNamespace,
) -> None:
    fake_js.hang = True

    async with httpx.AsyncClient(transport=FetchAsyncTransport(), timeout=0.05) as client:
        with pytest.raises(httpx.ReadTimeout):
            await client.get("https://api.example.com/")

    assert fake_js.aborted == 1


@pytest.mark.asyncio
async def test_stalled_bodies_time_out_and_abort_the_fetch(
    fake_js: SimpleNamespace,
) -> None:
    async def stall() -> SimpleNamespace:
        await fake_js.wait_for_abort()
        raise AssertionError

    async with httpx.AsyncClient(transport=FetchAsyncTransport(), timeout=0.05) as client:
        async with client.stream("GET", "https://api.example.com/sse") as response:
            fake_js.reader.read = stall
            with pytest.raises(httpx.ReadTimeout):
                await response.aread()

    assert fake_js.aborted == 1


@pytest.mark.asyncio
async def test_completed_requests_clear_their_timers(fake_js: SimpleNamespace) -> None:
    async with httpx.AsyncClient(transport=FetchAsyncTransport(), timeout=30) as client:
        response = await client.get("https://api.example.com/")

    assert response.json() == {"ok": True}
    assert fake_js.timers == set()
    assert fake_js.aborted == 0
