"""Deterministic imports, reseeding and the entropy audit."""

from __future__ import annotations

import importlib
import os
import sys
import textwrap
import uuid

import pytest

from crewai_pyodide import _entropy
from crewai_pyodide._entropy import audit_entropy, reseed


@pytest.fixture
def fake_packages(tmp_path, monkeypatch):
    """A package whose modules read entropy at import, standing in for
    ``opentelemetry.context`` and ``openai.lib.bedrock``."""
    pkg = tmp_path / "entropy_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "keys.py").write_text(
        textwrap.dedent(
            """
            from uuid import uuid4
            import entropy_pkg.helper  # draws real entropy at its own import
            KEY = "k-" + str(uuid4())
            def create_key(name):
                return name + "-" + str(uuid4())
            """
        )
    )
    (pkg / "signing.py").write_text("import os\nSECRET = os.urandom(32)\n")
    (pkg / "helper.py").write_text("import os\nSALT = os.urandom(8)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        _entropy,
        "_DETERMINISTIC_MODULES",
        {
            "entropy_pkg.keys": (_entropy._patch_opentelemetry_context, None),
            "entropy_pkg.signing": (
                None,
                lambda module: setattr(module, "SECRET", os.urandom(32)),
            ),
        },
    )
    meta_path = list(sys.meta_path)
    _entropy.install_deterministic_imports()
    yield
    sys.meta_path[:] = meta_path
    for name in [m for m in sys.modules if m.split(".")[0] == "entropy_pkg"]:
        del sys.modules[name]


def test_listed_modules_import_without_reading_entropy(fake_packages) -> None:
    with audit_entropy() as calls:
        importlib.import_module("entropy_pkg.signing")
        keys = importlib.import_module("entropy_pkg.keys")

    # Only helper.py, which is not listed, read real entropy.
    package_calls = [c for c in calls if "entropy_pkg" in c]
    assert len(package_calls) == 1
    assert "helper.py" in package_calls[0]
    assert uuid.UUID(keys.KEY[2:]).version == 4


def test_keys_stay_unique_and_entropy_free_after_import(fake_packages) -> None:
    keys = importlib.import_module("entropy_pkg.keys")

    with audit_entropy() as calls:
        first, second = keys.create_key("a"), keys.create_key("a")

    assert first != second
    assert calls == []


def test_real_functions_are_restored_after_import(fake_packages) -> None:
    importlib.import_module("entropy_pkg.signing")

    assert os.urandom(4) != bytes(4) or os.urandom(4) != bytes(4)
    assert uuid.uuid4() != uuid.uuid4()


def test_reseed_replaces_placeholder_values(fake_packages) -> None:
    signing = importlib.import_module("entropy_pkg.signing")
    assert signing.SECRET == bytes(32)

    reseed()

    assert signing.SECRET != bytes(32)


def test_audit_records_callers_and_restores_functions() -> None:
    real_urandom = os.urandom

    with audit_entropy() as calls:
        os.urandom(1)
        uuid.uuid4()

    assert len(calls) == 3  # uuid4 reads os.urandom itself
    assert all(":" in c for c in calls)
    assert os.urandom is real_urandom


def test_modules_the_host_already_patches_are_left_to_it(
    tmp_path, monkeypatch
) -> None:
    """workerd's SDK allows these modules' entropy itself and raises if an
    allowance goes unused, so they must import untouched."""
    import types

    pkg = tmp_path / "hosted_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "signing.py").write_text("import os\nSECRET = os.urandom(32)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        _entropy, "_DETERMINISTIC_MODULES", {"hosted_pkg.signing": (None, None)}
    )
    cloudflare = types.ModuleType("_cloudflare")
    manager = types.ModuleType("_cloudflare.import_patch_manager")
    manager.patches = {"hosted_pkg.signing": object()}
    cloudflare.import_patch_manager = manager
    monkeypatch.setitem(sys.modules, "_cloudflare", cloudflare)
    monkeypatch.setitem(sys.modules, "_cloudflare.import_patch_manager", manager)
    meta_path = list(sys.meta_path)
    _entropy.install_deterministic_imports()
    try:
        signing = importlib.import_module("hosted_pkg.signing")
    finally:
        sys.meta_path[:] = meta_path
        for name in [m for m in sys.modules if m.split(".")[0] == "hosted_pkg"]:
            del sys.modules[name]

    assert signing.SECRET != bytes(32)
