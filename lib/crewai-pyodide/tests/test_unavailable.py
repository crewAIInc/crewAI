from __future__ import annotations

import importlib
import sys

from pydantic import BaseModel
import pytest

from crewai_pyodide._unavailable import (
    UnavailableDependencyError,
    install_placeholders,
)


@pytest.fixture
def placeholder_package():
    name = "crewai_pyodide_missing_pkg"
    meta_path = list(sys.meta_path)
    install_placeholders([name])
    yield name
    sys.meta_path[:] = meta_path
    for module in [m for m in sys.modules if m.split(".")[0] == name]:
        del sys.modules[module]


def test_imports_of_missing_packages_succeed(placeholder_package: str) -> None:
    module = importlib.import_module(f"{placeholder_package}.utils.functions")
    from crewai_pyodide_missing_pkg.config import Settings  # noqa: F401

    assert isinstance(module.SomeClass, type)


def test_constructing_a_placeholder_names_the_missing_package(
    placeholder_package: str,
) -> None:
    from crewai_pyodide_missing_pkg.config import Settings

    with pytest.raises(UnavailableDependencyError, match=placeholder_package):
        Settings()


def test_placeholders_work_in_annotations_and_pydantic_models(
    placeholder_package: str,
) -> None:
    from crewai_pyodide_missing_pkg.api import Client, Documents

    class Model(BaseModel):
        client: Client | None = None
        docs: Documents[str] | None = None

    assert Model().client is None


def test_installed_packages_are_left_alone() -> None:
    meta_path = list(sys.meta_path)
    install_placeholders(["json"])
    try:
        assert sys.meta_path == meta_path
    finally:
        sys.meta_path[:] = meta_path
