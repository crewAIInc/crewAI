from __future__ import annotations

import concurrent.futures

import httpx
import pytest

import crewai_pyodide


@pytest.mark.asyncio
async def test_bootstrap_does_nothing_outside_pyodide() -> None:
    executor_class = concurrent.futures.ThreadPoolExecutor
    transport_class = httpx.AsyncHTTPTransport

    crewai_pyodide.bootstrap()

    assert concurrent.futures.ThreadPoolExecutor is executor_class
    assert httpx.AsyncHTTPTransport is transport_class


@pytest.mark.asyncio
async def test_run_awaits_inside_a_task() -> None:
    import asyncio

    async def current_task() -> asyncio.Task[object] | None:
        return asyncio.current_task()

    assert await crewai_pyodide.run(current_task()) is not None


@pytest.mark.asyncio
async def test_install_rejects_unknown_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import ModuleType

    monkeypatch.setitem(sys.modules, "micropip", ModuleType("micropip"))

    with pytest.raises(ValueError, match="nope"):
        await crewai_pyodide.install(extras=["nope"])
