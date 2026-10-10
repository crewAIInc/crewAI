"""Keep ``import crewai`` free of entropy, so the interpreter can be snapshotted.

Runtimes that snapshot an isolate after its imports (e.g. workerd) forbid
reading randomness at that point: anything random would be baked into the
snapshot and shared by every isolate restored from it. Two dependencies read
randomness while being imported:

- ``opentelemetry.context`` suffixes its context keys with ``uuid4()``. The keys
  only need to be unique within the process, so it gets counter-based UUIDs.
- ``openai.lib.bedrock`` draws an HMAC key from ``os.urandom``. Nothing uses it
  during import, so it gets a placeholder that :func:`reseed` replaces.

While those modules execute, this hook serves their own calls deterministic
stand-ins, then puts the real functions back.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
import importlib.abc
import importlib.machinery
import itertools
import os
import random
import sys
import traceback
from types import ModuleType
from typing import Any
import uuid


_counter = itertools.count()


def _deterministic_uuid4() -> uuid.UUID:
    return uuid.UUID(int=next(_counter), version=4)


def _zero_bytes(n: int) -> bytes:
    return bytes(n)


def _patch_opentelemetry_context(module: ModuleType) -> None:
    # Other opentelemetry modules create keys while they are imported too.
    # Keys only need to be unique within the process, which a counter gives.
    module.uuid4 = _deterministic_uuid4  # type: ignore[attr-defined]


def _reseed_openai_bedrock(module: ModuleType) -> None:
    module._LEGACY_SIGNATURE_KEY = os.urandom(32)  # type: ignore[attr-defined]


# Module name -> (called right after it executes, called by reseed()).
_DETERMINISTIC_MODULES: dict[
    str, tuple[Callable[[ModuleType], None] | None, Callable[[ModuleType], None] | None]
] = {
    "opentelemetry.context": (_patch_opentelemetry_context, None),
    "openai.lib.bedrock": (None, _reseed_openai_bedrock),
}


@contextmanager
def _without_entropy(module_name: str) -> Iterator[None]:
    """Serve deterministic values to calls made directly from ``module_name``.

    Calls from anywhere else, such as modules it imports, get real entropy and
    stay visible to :func:`audit_entropy`.
    """
    real_uuid4, real_urandom = uuid.uuid4, os.urandom
    # Names bound with ``from os import urandom`` keep these stand-ins after
    # the import, so they must hand out real entropy from then on.
    active = True

    def from_module() -> bool:
        return active and sys._getframe(2).f_globals.get("__name__") == module_name

    def uuid4() -> uuid.UUID:
        return _deterministic_uuid4() if from_module() else real_uuid4()

    def urandom(n: int) -> bytes:
        return _zero_bytes(n) if from_module() else real_urandom(n)

    uuid.uuid4, os.urandom = uuid4, urandom
    try:
        yield
    finally:
        active = False
        uuid.uuid4, os.urandom = real_uuid4, real_urandom


class _DeterministicLoader(importlib.abc.Loader):
    def __init__(
        self, loader: importlib.abc.Loader, after: Callable[[ModuleType], None] | None
    ) -> None:
        self._loader = loader
        self._after = after

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        with _without_entropy(module.__name__):
            self._loader.exec_module(module)
        if self._after is not None:
            self._after(module)


class _DeterministicFinder(importlib.abc.MetaPathFinder):
    def __init__(self, skip: frozenset[str]) -> None:
        self._skip = skip

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in _DETERMINISTIC_MODULES or fullname in self._skip:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                after, _ = _DETERMINISTIC_MODULES[fullname]
                spec.loader = _DeterministicLoader(spec.loader, after)
                return spec
        return None


def _host_patched_modules() -> frozenset[str]:
    """Modules whose import the host already makes snapshot-safe.

    workerd (with its Python SDK) allows the entropy some packages read at
    import, and raises if an allowance goes unused, so those are left to it.
    """
    try:
        from _cloudflare import import_patch_manager  # type: ignore[import-not-found]
    except ImportError:
        return frozenset()
    return frozenset(import_patch_manager.patches)


def install_deterministic_imports() -> None:
    """Import the modules above without reading randomness."""
    sys.meta_path.insert(0, _DeterministicFinder(_host_patched_modules()))


def reseed() -> None:
    """Replace randomness drawn before a snapshot with fresh values.

    Call once in each isolate restored from a snapshot, before doing work.
    :func:`crewai_pyodide.run` does this on first use.
    """
    random.seed()
    for name, (_, on_reseed) in _DETERMINISTIC_MODULES.items():
        module = sys.modules.get(name)
        if module is not None and on_reseed is not None:
            on_reseed(module)


@contextmanager
def audit_entropy() -> Iterator[list[str]]:
    """Record the callers of Python's entropy sources inside the block.

    Yields a list that fills with one ``"file:line"`` entry per call. Covers
    ``os.urandom``, ``uuid.uuid4`` and the ``random`` module's functions;
    randomness read by C extensions or ``random.SystemRandom`` is not visible.
    """
    calls: list[str] = []
    originals: list[tuple[Any, str, Any]] = []

    def record(owner: Any, name: str) -> None:
        real = getattr(owner, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # The nearest caller outside this module (which wraps these too).
            frame = next(
                f for f in reversed(traceback.extract_stack()) if f.filename != __file__
            )
            calls.append(f"{frame.filename}:{frame.lineno}")
            return real(*args, **kwargs)

        originals.append((owner, name, real))
        setattr(owner, name, wrapper)

    record(os, "urandom")
    record(uuid, "uuid4")
    for name in ("random", "randint", "randrange", "choice", "getrandbits", "shuffle"):
        record(random, name)
    try:
        yield calls
    finally:
        for owner, name, real in reversed(originals):
            setattr(owner, name, real)
