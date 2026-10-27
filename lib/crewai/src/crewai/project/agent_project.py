"""Load a standalone Python, JSON, or YAML agent project."""

from __future__ import annotations

import hashlib
import importlib
from importlib.machinery import SourceFileLoader
import importlib.util
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType
from typing import TYPE_CHECKING

from crewai_core.project import resolve_agent_project_entrypoint


if TYPE_CHECKING:
    from crewai.agent import Agent


_IMPORT_LOCK = RLock()


def load_agent_project(
    project_path: str | Path = ".", project_name: str | None = None
) -> Agent:
    """Load the standalone agent selected by the project's entrypoint.

    ``[tool.crewai].definition`` selects a Python, JSON, or YAML file. Without
    it, the shared project resolver discovers a single supported entrypoint.
    Python files export ``agent = Agent(...)``; data files contain Agent
    constructor fields and receive their messages at kickoff time.

    Python packages and relative tool imports are imported afresh on each call
    so managed kickoffs do not reuse an earlier agent or its tools. Project
    code executes during loading and must come from a trusted source.
    """
    from crewai.agent import Agent

    root = Path(project_path).resolve()
    agent_path = resolve_agent_project_entrypoint(root, project_name)
    if agent_path.suffix.lower() != ".py":
        from crewai.project.json_loader import load_agent

        with _IMPORT_LOCK:
            loaded = load_agent(agent_path, project_root=root)
        if not isinstance(loaded, Agent):
            raise ValueError(f"{agent_path} must define an Agent")
        return loaded

    package_path = agent_path.parent
    try:
        source_path = agent_path.relative_to(root / "src")
    except ValueError:
        pass
    else:
        if len(source_path.parts) > 1:
            package_path = root / "src" / source_path.parts[0]

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
            parent_name = namespace
            for component in agent_path.parent.relative_to(package_path).parts:
                parent_name = f"{parent_name}.{component}"
                importlib.import_module(parent_name)
            module_leaf = (
                agent_path.stem
                if agent_path.stem.isidentifier()
                else "_agent_entrypoint"
            )
            module_name = f"{parent_name}.{module_leaf}"
            module = sys.modules.get(module_name)
            if (
                module is None
                or Path(getattr(module, "__file__", "") or "").resolve() != agent_path
            ):
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    agent_path,
                    loader=SourceFileLoader(module_name, str(agent_path)),
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot import agent entrypoint: {agent_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            agent = getattr(module, "agent", None)
            if not isinstance(agent, Agent):
                raise ValueError(f"{agent_path} must export `agent = Agent(...)`")
        except BaseException:
            for name in list(sys.modules):
                if name == namespace or name.startswith(namespace + "."):
                    del sys.modules[name]
            raise
    return agent
