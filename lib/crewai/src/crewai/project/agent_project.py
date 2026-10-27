"""Load a standalone Python agent project."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import keyword
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType
from typing import TYPE_CHECKING

from crewai_core.project import normalize_package_name
import tomli


if TYPE_CHECKING:
    from crewai.agent import Agent


_IMPORT_LOCK = RLock()


def load_agent_project(
    project_path: str | Path = ".", project_name: str | None = None
) -> Agent:
    """Load the exported ``agent`` from ``src/<project_name>/agent.py``.

    The package is imported afresh on every call, including relative tool
    imports, so managed kickoffs do not reuse an earlier agent or its tools.
    When omitted, ``project_name`` is the normalized ``[project].name`` from
    the project's ``pyproject.toml``. Import errors in project code propagate.
    """
    from crewai.agent import Agent

    root = Path(project_path).resolve()
    if project_name is None:
        with (root / "pyproject.toml").open("rb") as config_file:
            project_name = tomli.load(config_file)["project"]["name"]
    if not isinstance(project_name, str) or any(
        separator in project_name for separator in ("/", "\\")
    ):
        raise ValueError("Agent project_name must be a Python package name")
    project_name = normalize_package_name(project_name)
    if not project_name.isidentifier() or keyword.iskeyword(project_name):
        raise ValueError("Agent project_name must be a Python package name")

    package_path = root / "src" / project_name
    agent_path = package_path / "agent.py"
    if not agent_path.is_file():
        raise FileNotFoundError(f"Agent entrypoint not found: {agent_path}")

    namespace = (
        "_crewai_agent_" + hashlib.sha256(str(package_path).encode()).hexdigest()[:16]
    )
    # ponytail: serialize imports; use per-project locks if loading becomes a bottleneck.
    with _IMPORT_LOCK:
        # A private package namespace avoids replacing an application's imports.
        for name in list(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                del sys.modules[name]
        importlib.invalidate_caches()
        package = ModuleType(namespace)
        package.__path__ = [str(package_path)]
        package.__package__ = namespace
        sys.modules[namespace] = package
        try:
            init_path = package_path / "__init__.py"
            if init_path.is_file():
                spec = importlib.util.spec_from_file_location(
                    namespace, init_path, submodule_search_locations=[str(package_path)]
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot import agent package: {package_path}")
                package = importlib.util.module_from_spec(spec)
                sys.modules[namespace] = package
                spec.loader.exec_module(package)
            module = importlib.import_module(f"{namespace}.agent")
            agent = getattr(module, "agent", None)
            if not isinstance(agent, Agent):
                raise ValueError(f"{agent_path} must export `agent = Agent(...)`")
        except BaseException:
            for name in list(sys.modules):
                if name == namespace or name.startswith(namespace + "."):
                    del sys.modules[name]
            raise
    return agent
