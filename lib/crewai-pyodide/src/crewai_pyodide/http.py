"""httpx transports backed by the JavaScript ``fetch`` API.

Under Pyodide there are no sockets, so httpx's default transports (and every
SDK built on them, such as ``openai`` and ``anthropic``) cannot make requests.
These transports send requests through ``fetch`` instead. Responses stream, so
server-sent events from LLM providers arrive incrementally.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable
import importlib
from typing import Any

import httpx


# Headers the browser controls itself; fetch rejects or silently drops them.
_FORBIDDEN_REQUEST_HEADERS = frozenset(
    {"host", "connection", "content-length", "accept-encoding", "keep-alive"}
)
# fetch has already decoded the body, so these would make httpx decode it again.
_STALE_RESPONSE_HEADERS = frozenset({"content-encoding", "content-length"})


class _FetchTimeoutError(Exception):
    pass


async def _abort_after(
    awaitable: Awaitable[Any], timeout: float | None, abort_controller: Any
) -> Any:
    """Await ``awaitable``, aborting the fetch if ``timeout`` seconds pass first.

    Raises _FetchTimeoutError on expiry. The timer is a JavaScript one that is
    cleared on every exit: Pyodide's event loop never clears the ``setTimeout``
    behind a cancelled asyncio timer, so ``asyncio.wait_for`` would leave one
    pending per call and keep Node.js alive until it fires.
    """
    if timeout is None:
        return await awaitable

    from pyodide.ffi import JsException, create_proxy  # type: ignore[import-not-found]

    js: Any = importlib.import_module("js")
    expired = False

    def expire() -> None:
        nonlocal expired
        expired = True
        abort_controller.abort()

    callback = create_proxy(expire)
    timer = js.setTimeout(callback, timeout * 1000)
    try:
        return await awaitable
    except JsException as e:
        if expired:
            raise _FetchTimeoutError from e
        raise
    finally:
        js.clearTimeout(timer)
        callback.destroy()


class _FetchByteStream(httpx.AsyncByteStream):
    """Response body read chunk by chunk from a ``ReadableStream``.

    ``read_timeout`` bounds the wait for each chunk, as httpx's read timeout
    does; on expiry ``abort_controller`` cancels the underlying fetch.
    """

    def __init__(
        self, js_response: Any, abort_controller: Any, read_timeout: float | None
    ) -> None:
        body = js_response.body
        self._reader = body.getReader() if body is not None else None
        self._abort_controller = abort_controller
        self._read_timeout = read_timeout

    async def __aiter__(self) -> AsyncIterator[bytes]:
        from pyodide.ffi import JsException

        if self._reader is None:
            return
        while True:
            try:
                chunk = await _abort_after(
                    self._reader.read(), self._read_timeout, self._abort_controller
                )
            except _FetchTimeoutError as e:
                raise httpx.ReadTimeout("Timed out reading the response body") from e
            except JsException as e:
                # The connection dropped mid-body; let httpx and SDKs retry.
                raise httpx.ReadError(str(e)) from e
            if chunk.done:
                return
            yield chunk.value.to_bytes()

    async def aclose(self) -> None:
        if self._reader is not None:
            await self._reader.cancel()
            self._reader = None


class FetchAsyncTransport(httpx.AsyncBaseTransport):
    """Async httpx transport that sends requests with ``fetch``.

    Accepts and ignores the keyword arguments of ``httpx.AsyncHTTPTransport``
    (TLS, proxies, connection limits), which the browser manages itself.
    """

    def __init__(self, **_: Any) -> None:
        pass

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        from pyodide.ffi import JsException, to_js

        js: Any = importlib.import_module("js")  # Pyodide's bridge to JavaScript

        body = await request.aread()
        headers = [
            (k, v)
            for k, v in request.headers.multi_items()
            if k.lower() not in _FORBIDDEN_REQUEST_HEADERS
        ]
        # fetch has no separate connect phase, so httpx's read timeout bounds
        # both the wait for response headers and each body chunk.
        read_timeout = request.extensions.get("timeout", {}).get("read")
        abort_controller = js.AbortController.new()
        init: dict[str, Any] = {
            "method": request.method,
            "headers": headers,
            "signal": abort_controller.signal,
        }
        if body:
            init["body"] = to_js(body)
        try:
            js_response = await _abort_after(
                js.fetch(
                    str(request.url),
                    to_js(init, dict_converter=js.Object.fromEntries),
                ),
                read_timeout,
                abort_controller,
            )
        except _FetchTimeoutError as e:
            raise httpx.ReadTimeout(
                "Timed out waiting for the response", request=request
            ) from e
        except JsException as e:
            # Network failures and CORS rejections both surface as a TypeError.
            raise httpx.ConnectError(str(e), request=request) from e

        response_headers = [
            (k, v)
            for k, v in js_response.headers.entries()
            if k.lower() not in _STALE_RESPONSE_HEADERS
        ]
        return httpx.Response(
            status_code=js_response.status,
            headers=response_headers,
            stream=_FetchByteStream(js_response, abort_controller, read_timeout),
            request=request,
        )


class FetchTransport(httpx.BaseTransport):
    """Sync httpx transport that sends requests with ``fetch``.

    Blocks on the async fetch with ``pyodide.ffi.run_sync``, which needs a
    runtime with JavaScript Promise Integration (JSPI).
    """

    def __init__(self, **_: Any) -> None:
        self._async = FetchAsyncTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        from pyodide.ffi import can_run_sync, run_sync

        if not can_run_sync():
            from crewai_pyodide import JSPI_REQUIRED_MESSAGE

            raise RuntimeError(JSPI_REQUIRED_MESSAGE)

        # Read a streamed sync body here: the async transport can only await it.
        request.read()

        async def send() -> httpx.Response:
            response = await self._async.handle_async_request(request)
            content = await response.aread()
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                content=content,
                request=request,
            )

        result: httpx.Response = run_sync(send())
        return result


def install_fetch_transports() -> None:
    """Make every new httpx client use the fetch-backed transports."""
    httpx.HTTPTransport = FetchTransport  # type: ignore[misc,assignment]
    httpx.AsyncHTTPTransport = FetchAsyncTransport  # type: ignore[misc,assignment]
    # Clients resolve their default transport through this module's globals.
    httpx._client.HTTPTransport = FetchTransport  # type: ignore[attr-defined,assignment]
    httpx._client.AsyncHTTPTransport = FetchAsyncTransport  # type: ignore[attr-defined,assignment]
