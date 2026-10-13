"""Run crewai in the browser (or Node.js) on Pyodide/WebAssembly.

Inside Pyodide, after installing this package with micropip::

    import crewai_pyodide

    await crewai_pyodide.install()

    from crewai import Agent, Crew, Task

    crew = Crew(agents=[...], tasks=[...])
    result = await crewai_pyodide.run(crew.akickoff())

See the package README for what works and what does not under WebAssembly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Sequence
import os
import sys
from typing import TypeVar

from crewai_pyodide._entropy import audit_entropy, reseed


__version__ = "1.15.27"

T = TypeVar("T")

# Installed from Pyodide's package index, or PyPI for pure-Python wheels.
# Version ranges mirror crewai's own requirements where Pyodide's builds allow.
REQUIREMENTS: tuple[str, ...] = (
    "pydantic>=2.11.9,<2.13",
    "pydantic-settings>=2.14.2,<3",
    "httpx~=0.28.1",
    "anyio",
    "sniffio",
    "distro",
    "tqdm",
    "jiter",
    "typing-extensions",
    "pyyaml",
    "rich>=13.7.1",
    "click>=8.1.7,<9",
    "packaging>=23.0",
    "cryptography",
    "pyjwt>=2.15.0,<3",
    "appdirs~=1.4.4",
    "jsonref~=1.1.0",
    "json-repair~=0.60.1",
    "json5~=0.10.0",
    "tomli~=2.0.2",
    "python-dotenv>=1.2.2,<2",
    "aiofiles~=24.1.0",
    "aiosqlite~=0.21.0",
    "sqlite3",
    "ssl",
    "jinja2",
    "requests",
    "numpy",
    "opentelemetry-api>=1.42,<2",
    "opentelemetry-sdk>=1.42,<2",
    "opentelemetry-exporter-otlp-proto-http>=1.42,<2",
)

# Installed without dependency resolution: their pins cannot all be met by
# Pyodide's builds (openai wants jiter>=0.10; crewai pulls in chromadb,
# lancedb and other native packages that have no WebAssembly wheels).
NO_DEPS_REQUIREMENTS: tuple[str, ...] = ("openai>=2.30.0,<3",)

# Optional LLM provider SDKs that work under Pyodide.
EXTRAS: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic~=0.73.0", "docstring-parser"),
}

# Native packages crewai imports at module level that have no Pyodide build.
UNAVAILABLE_PACKAGES: tuple[str, ...] = ("chromadb", "portalocker")

JSPI_REQUIRED_MESSAGE = (
    "crewai needs WebAssembly JavaScript Promise Integration (JSPI) to run "
    "under Pyodide: use Chrome/Edge 137+ or Node.js with "
    "--experimental-wasm-jspi, and run Python with runPythonAsync()."
)


def is_pyodide() -> bool:
    """Return True when running on Pyodide/Emscripten."""
    return sys.platform == "emscripten"


async def install(
    crewai: Sequence[str] = (),
    *,
    extras: Sequence[str] = (),
) -> None:
    """Install crewai and its WebAssembly-compatible dependencies, then
    :func:`bootstrap` the runtime.

    Args:
        crewai: Requirements or wheel URLs for ``crewai`` and ``crewai-core``.
            Defaults to the release matching this package's version on PyPI.
            Pass wheel URLs to run an unreleased build.
        extras: Names from :data:`EXTRAS` to install as well, e.g.
            ``["anthropic"]``.
    """
    import micropip  # type: ignore[import-not-found]

    unknown = set(extras) - EXTRAS.keys()
    if unknown:
        raise ValueError(f"Unknown extras: {sorted(unknown)}")

    requirements = list(REQUIREMENTS)
    for extra in extras:
        requirements.extend(EXTRAS[extra])
    await micropip.install(requirements)

    crewai = crewai or (f"crewai=={__version__}", f"crewai-core=={__version__}")
    await micropip.install([*NO_DEPS_REQUIREMENTS, *crewai], deps=False)

    bootstrap()


_bootstrapped = False
_reseeded = False


def bootstrap() -> None:
    """Adapt the runtime so crewai can be imported and run under Pyodide.

    Must run before ``crewai`` is imported. :func:`install` calls it; call it
    directly when the packages are already installed (e.g. from a snapshot).
    Does nothing outside Pyodide, and nothing on repeated calls.
    """
    global _bootstrapped
    if _bootstrapped or not is_pyodide():
        return

    from crewai_pyodide._entropy import install_deterministic_imports
    from crewai_pyodide._locks import install_in_process_locks
    from crewai_pyodide._threads import install_inline_threads
    from crewai_pyodide._unavailable import install_placeholders
    from crewai_pyodide.http import install_fetch_transports

    # Telemetry exports from a background thread, which cannot run here.
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    # Trace exporters also batch from a background thread, and flush() cannot
    # wait for handlers on the host loop.
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    # The PyPI update check uses urllib, which has no socket support here.
    os.environ.setdefault("CREWAI_DISABLE_VERSION_CHECK", "true")
    install_inline_threads()
    install_fetch_transports()
    install_placeholders(UNAVAILABLE_PACKAGES)
    install_deterministic_imports()
    # Imports crewai_core.lock_store, so the placeholders must be in place.
    install_in_process_locks()
    _bootstrapped = True


async def run(awaitable: Awaitable[T]) -> T:
    """Await ``awaitable`` inside an asyncio task.

    Pyodide's top-level ``await`` does not run inside a task, but parts of
    crewai's async stack (e.g. ``asyncio.timeout``) require one::

        result = await crewai_pyodide.run(crew.akickoff())

    Raises:
        RuntimeError: If the runtime cannot block on async work, which crewai
            needs for the synchronous helpers its async path still calls.
    """
    global _reseeded
    if is_pyodide():
        from pyodide.ffi import can_run_sync  # type: ignore[import-not-found]

        if not can_run_sync():
            raise RuntimeError(JSPI_REQUIRED_MESSAGE)
        if not _reseeded:
            reseed()
            _reseeded = True

    async def _await() -> T:
        return await awaitable

    return await asyncio.ensure_future(_await())


__all__ = [
    "EXTRAS",
    "NO_DEPS_REQUIREMENTS",
    "REQUIREMENTS",
    "UNAVAILABLE_PACKAGES",
    "audit_entropy",
    "bootstrap",
    "install",
    "is_pyodide",
    "reseed",
    "run",
]
