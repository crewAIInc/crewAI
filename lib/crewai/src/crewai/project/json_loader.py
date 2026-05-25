"""Loader utilities for JSON/JSONC agent and tool definitions."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def strip_jsonc_comments(text: str) -> str:
    """Strip ``//`` and ``/* */`` comments from JSONC text, then fix trailing commas.

    Args:
        text: Raw JSONC string potentially containing comments and trailing commas.

    Returns:
        Clean JSON string ready for ``json.loads``.
    """
    result = re.sub(r"(?<!:)//.*?$", "", text, flags=re.MULTILINE)
    result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)
    result = re.sub(r",\s*([}\]])", r"\1", result)
    return result


def load_agent(source: str | Path) -> Any:
    """Load an existing ``Agent`` from a ``.json`` / ``.jsonc`` definition file.

    The definition file should contain at minimum ``role``, ``goal``, and
    ``backstory`` keys.  Optional keys such as ``llm``, ``tools``, ``verbose``,
    ``max_iter``, ``allow_delegation``, ``memory``, and ``max_rpm`` are passed
    through when present.

    Args:
        source: Path to a ``.json`` or ``.jsonc`` agent definition file.

    Returns:
        A configured ``Agent`` instance.
    """
    from crewai import Agent

    path = Path(source)
    raw = path.read_text(encoding="utf-8")
    clean = strip_jsonc_comments(raw)
    defn: dict[str, Any] = json.loads(clean)

    agent_kwargs: dict[str, Any] = {
        "role": defn["role"],
        "goal": defn["goal"],
        "backstory": defn.get("backstory", ""),
    }

    # Settings can be nested under "settings" or flat at the top level.
    _SETTINGS_TO_AGENT = {
        "memory": "memory",
        "verbose": "verbose",
        "allow_delegation": "allow_delegation",
        "max_iter": "max_iter",
        "max_tokens": "max_tokens",
        "max_execution_time": "max_execution_time",
        "max_rpm": "max_rpm",
        "respect_context_window": "respect_context_window",
        "max_retry_limit": "max_retry_limit",
        "planning": "planning",
        "cache": "cache",
        "use_system_prompt": "use_system_prompt",
    }

    settings = defn.get("settings", {})
    for json_key, agent_key in _SETTINGS_TO_AGENT.items():
        if json_key in settings:
            agent_kwargs[agent_key] = settings[json_key]
        elif json_key in defn:
            agent_kwargs[agent_key] = defn[json_key]

    # LLM -- accept a string model identifier or leave as default
    if "llm" in defn and defn["llm"] is not None:
        agent_kwargs["llm"] = defn["llm"]
    if "function_calling_llm" in defn and defn["function_calling_llm"] is not None:
        agent_kwargs["function_calling_llm"] = defn["function_calling_llm"]

    # Tools
    if "tools" in defn:
        agent_kwargs["tools"] = _resolve_tools(defn["tools"])

    # Embedder
    if "embedder" in defn and defn["embedder"] is not None:
        agent_kwargs["embedder"] = defn["embedder"]

    return Agent(**agent_kwargs)


def _resolve_tools(tool_names: list[str]) -> list[Any]:
    """Resolve a list of tool name strings into tool instances.

    Each name is first looked up in ``crewai_tools`` by converting
    ``snake_case`` to ``PascalCaseTool`` (e.g. ``"serper_dev"`` ->
    ``SerperDevTool``).  A direct attribute lookup is tried as fallback.

    Names prefixed with ``custom:`` are resolved from a ``tools/`` directory
    relative to the current working directory.

    Args:
        tool_names: List of tool identifier strings.

    Returns:
        List of instantiated tool objects.  Unresolvable names are logged
        as warnings and skipped.
    """
    tools: list[Any] = []
    for name in tool_names:
        if not name:
            continue
        if name.startswith("custom:"):
            custom_tool = _resolve_custom_tool(name[7:])
            if custom_tool is not None:
                tools.append(custom_tool)
            continue
        try:
            tool_cls = _find_tool_class(name)
            if tool_cls:
                tools.append(tool_cls())
        except Exception as e:
            logger.warning("Failed to resolve tool '%s': %s", name, e)
    return tools


_tool_class_cache: dict[str, type | None] = {}


def _find_tool_class(name: str) -> type | None:
    """Look up a tool class by name from the ``crewai_tools`` package.

    Accepts direct class names (``SerperDevTool``), snake_case names
    (``serper_dev``), or names without the Tool suffix (``SerperDev``).

    Uses lazy per-class imports to avoid loading the entire crewai_tools
    package (220+ tool classes) on startup.

    Args:
        name: Tool class name in any supported format.

    Returns:
        The tool class, or ``None`` if not found.
    """
    if name in _tool_class_cache:
        return _tool_class_cache[name]

    # Build candidate class names to try
    candidates = [name]
    if not name.endswith("Tool"):
        candidates.append(name + "Tool")
    snake_pascal = "".join(word.capitalize() for word in name.split("_")) + "Tool"
    if snake_pascal not in candidates:
        candidates.append(snake_pascal)

    for class_name in candidates:
        cls = _try_import_tool(class_name)
        if cls is not None:
            _tool_class_cache[name] = cls
            return cls

    _tool_class_cache[name] = None
    return None


def _try_import_tool(class_name: str) -> type | None:
    """Attempt to import a single tool class without loading all of crewai_tools."""
    # Map PascalCase class name to its module (snake_case)
    # e.g. SerperDevTool → crewai_tools.tools.serper_dev_tool.serper_dev_tool
    import re as _re

    base = class_name.removesuffix("Tool") if class_name.endswith("Tool") else class_name
    snake = _re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", base).lower()
    tool_snake = snake + "_tool" if not snake.endswith("_tool") else snake

    module_paths = [
        f"crewai_tools.tools.{tool_snake}.{tool_snake}",
        f"crewai_tools.tools.{tool_snake}",
    ]

    for mod_path in module_paths:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            cls = getattr(mod, class_name, None)
            if cls is not None:
                return cls
        except (ImportError, ModuleNotFoundError):
            continue

    # Final fallback: import crewai_tools top-level (slow path)
    try:
        import crewai_tools

        return getattr(crewai_tools, class_name, None)
    except ImportError:
        return None


def _resolve_custom_tool(tool_name: str) -> Any:
    """Resolve a custom tool from the project's ``tools/`` directory.

    Args:
        tool_name: Name of the tool (without ``custom:`` prefix).  Expected to
            map to ``tools/<tool_name>.py`` containing a ``BaseTool`` subclass.

    Returns:
        An instantiated tool, or ``None`` if resolution fails.
    """
    tools_dir = Path.cwd() / "tools"
    tool_file = tools_dir / f"{tool_name}.py"
    if not tool_file.exists():
        logger.warning("Custom tool file not found: %s", tool_file)
        return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(f"custom_tools.{tool_name}", tool_file)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        from crewai.tools.base_tool import BaseTool

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseTool) and attr is not BaseTool:
                return attr()
        logger.warning("No BaseTool subclass found in %s", tool_file)
        return None
    except Exception as e:
        logger.warning("Failed to load custom tool '%s': %s", tool_name, e)
        return None
