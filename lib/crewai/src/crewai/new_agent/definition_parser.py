"""Parser for declarative agent definitions (JSON/JSONC)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any


logger = logging.getLogger(__name__)


def strip_jsonc_comments(text: str) -> str:
    """Strip // and /* */ comments from JSONC text, then fix trailing commas."""
    result = re.sub(r"(?<!:)//.*?$", "", text, flags=re.MULTILINE)
    result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)
    result = re.sub(r",\s*([}\]])", r"\1", result)
    return result


def _validate_against_schema(definition: dict[str, Any]) -> None:
    """Validate agent definition against the JSON schema.

    Logs a warning on validation failure rather than raising, so
    existing definitions continue to work (graceful degradation).
    """
    try:
        import jsonschema
    except ImportError:
        logger.debug("jsonschema not installed, skipping validation")
        return

    schema_path = Path(__file__).parent / "agent_schema.json"
    if not schema_path.exists():
        logger.debug("agent_schema.json not found, skipping validation")
        return

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(definition, schema)
    except jsonschema.ValidationError as e:
        logger.warning("Agent definition validation failed: %s", e.message)
    except Exception as e:
        logger.debug("Schema validation skipped: %s", e)


def parse_agent_definition(source: str | Path | dict) -> dict[str, Any]:
    """Parse an agent definition from a file path, JSON string, or dict.

    Args:
        source: Path to a .json/.jsonc file, a JSON string, or a dict.

    Returns:
        Parsed and validated agent definition dict.
    """
    if isinstance(source, dict):
        defn = source
    elif isinstance(source, Path) or (
        isinstance(source, str) and source.endswith((".json", ".jsonc"))
    ):
        path = Path(source)
        raw = path.read_text(encoding="utf-8")
        clean = strip_jsonc_comments(raw)
        defn = json.loads(clean)
    else:
        raw = source
        clean = strip_jsonc_comments(raw)
        defn = json.loads(clean)

    # GAP-65: validate against schema (warn-only)
    _validate_against_schema(defn)

    return defn


def load_agent_from_definition(
    source: str | Path | dict,
    agents_dir: Path | None = None,
    _loading_chain: set[str] | None = None,
) -> Any:
    """Load a NewAgent from a declarative definition.

    Args:
        source: Agent definition (file path, JSON string, or dict).
        agents_dir: Directory to resolve local coworker refs from.
        _loading_chain: Internal — tracks agent names being loaded to
            detect circular coworker references.

    Returns:
        A configured NewAgent instance.
    """
    from crewai.new_agent.models import AgentSettings
    from crewai.new_agent.new_agent import NewAgent

    if _loading_chain is None:
        _loading_chain = set()

    defn = parse_agent_definition(source)

    agent_name = defn.get("name", "")
    if agent_name and agent_name in _loading_chain:
        logger.debug(
            "Skipping coworker back-reference '%s' (already in loading chain)",
            agent_name,
        )
        return None

    if agent_name:
        _loading_chain.add(agent_name)

    # Build settings
    settings_raw = defn.get("settings", {})
    settings_kwargs = {}
    settings_map = {
        "memory": "memory_enabled",
        "reasoning": "reasoning_enabled",
        "self_improving": "self_improving",
        "planning": "planning_enabled",
        "auto_plan": "auto_plan",
        "can_spawn_copies": "can_spawn_copies",
        "max_spawn_depth": "max_spawn_depth",
        "max_concurrent_spawns": "max_concurrent_spawns",
        "max_history_messages": "max_history_messages",
        "narration_guard": "narration_guard",
        "dreaming_interval_hours": "dreaming_interval_hours",
        "dreaming_trigger_threshold": "dreaming_trigger_threshold",
        "dreaming_llm": "dreaming_llm",
        "provenance_detail": "provenance_detail",
        "spawn_timeout": "spawn_timeout",
        "can_create_knowledge": "can_create_knowledge",
        "can_build_skills": "can_build_skills",
        "can_schedule": "can_schedule",
        "memory_read_only": "memory_read_only",
        "narration_max_retries": "narration_max_retries",
        "respect_context_window": "respect_context_window",
        "cache_tool_results": "cache_tool_results",
        "max_retry_limit": "max_retry_limit",
        "share_data": "share_data",
    }
    for json_key, model_key in settings_map.items():
        if json_key in settings_raw:
            settings_kwargs[model_key] = settings_raw[json_key]

    settings = AgentSettings(**settings_kwargs)

    try:
        # Resolve coworkers (pass loading chain to detect circular refs)
        coworkers = _resolve_coworkers(
            defn.get("coworkers", []), agents_dir, _loading_chain
        )

        # Resolve guardrail
        guardrail = _resolve_guardrail(defn.get("guardrail"))

        # Resolve knowledge sources
        knowledge_sources = _resolve_knowledge_sources(
            defn.get("knowledge_sources", [])
        )

        # Build agent
        agent_kwargs: dict[str, Any] = {
            "role": defn["role"],
            "goal": defn["goal"],
            "backstory": defn.get("backstory", ""),
            "settings": settings,
            "verbose": defn.get("verbose", False),
            "max_iter": defn.get("max_iter", 25),
        }

        if "llm" in defn:
            agent_kwargs["llm"] = defn["llm"]
        if "function_calling_llm" in defn:
            agent_kwargs["function_calling_llm"] = defn["function_calling_llm"]
        if "tools" in defn:
            agent_kwargs["tools"] = _resolve_tools(defn["tools"])
        if "mcps" in defn:
            agent_kwargs["mcps"] = _resolve_mcps(defn["mcps"])
        if "apps" in defn:
            agent_kwargs["apps"] = defn["apps"]
        if coworkers:
            agent_kwargs["coworkers"] = coworkers
        if guardrail is not None:
            agent_kwargs["guardrail"] = guardrail
        if "max_tokens" in defn:
            agent_kwargs["max_tokens"] = defn["max_tokens"]
        if "max_execution_time" in defn:
            agent_kwargs["max_execution_time"] = defn["max_execution_time"]

        if knowledge_sources:
            agent_kwargs["knowledge_sources"] = knowledge_sources

        if "skills" in defn:
            from pathlib import Path as _Path

            agent_kwargs["skills"] = [_Path(p) for p in defn["skills"]]

        if "response_model" in defn:
            resolved_model = _resolve_response_model(defn["response_model"])
            if resolved_model is not None:
                agent_kwargs["response_model"] = resolved_model

        memory_setting = settings_raw.get("memory", True)
        agent_kwargs["memory"] = memory_setting

        return NewAgent(**agent_kwargs)
    finally:
        if agent_name:
            _loading_chain.discard(agent_name)


def _resolve_tools(tool_names: list[str]) -> list[Any]:
    """Resolve tool names into tool instances."""
    tools = []
    for name in tool_names:
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
            logger.warning(f"Failed to resolve tool '{name}': {e}")
    return tools


def _find_tool_class(name: str) -> type | None:
    """Look up a tool class by name from the crewai_tools package."""
    try:
        import crewai_tools

        # Convert snake_case name to PascalCase + Tool suffix
        class_name = "".join(word.capitalize() for word in name.split("_")) + "Tool"
        cls = getattr(crewai_tools, class_name, None)
        if cls is not None:
            return cls
        # Try direct attribute lookup
        cls = getattr(crewai_tools, name, None)
        return cls
    except ImportError:
        return None


def _resolve_coworkers(
    coworker_defs: list[dict[str, Any]],
    agents_dir: Path | None,
    _loading_chain: set[str] | None = None,
) -> list[Any]:
    """Resolve coworker definitions into NewAgent instances or handles."""
    coworkers = []
    for cw in coworker_defs:
        if isinstance(cw, str):
            coworkers.append(cw)
        elif "ref" in cw:
            ref_name = cw["ref"]
            if _loading_chain and ref_name in _loading_chain:
                logger.debug(
                    "Skipping coworker back-reference '%s' (already in loading chain)",
                    ref_name,
                )
                continue
            if agents_dir:
                for ext in (".json", ".jsonc"):
                    ref_path = agents_dir / f"{ref_name}{ext}"
                    if ref_path.exists():
                        result = load_agent_from_definition(
                            ref_path,
                            agents_dir,
                            set(_loading_chain) if _loading_chain else None,
                        )
                        if result is not None:
                            coworkers.append(result)
                        break
                else:
                    logger.warning(
                        f"Coworker ref '{ref_name}' not found in {agents_dir}"
                    )
            else:
                logger.warning(
                    f"Cannot resolve coworker ref '{ref_name}' — no agents_dir specified"
                )
        elif "amp" in cw:
            # AMP handle — pass as string for resolution at construction time
            # Support overrides: {"amp": "handle", "llm": "...", "settings": {...}}
            amp_handle = cw["amp"]
            overrides = {k: v for k, v in cw.items() if k != "amp"}
            if overrides:
                coworkers.append({"handle": amp_handle, "overrides": overrides})
            else:
                coworkers.append(amp_handle)
        elif "a2a" in cw:
            # A2A remote — would need A2AClientConfig
            try:
                from crewai.a2a.config import A2AClientConfig

                coworkers.append(A2AClientConfig(url=cw["a2a"]))
            except ImportError:
                logger.warning(f"A2A support not available for coworker {cw['a2a']}")
        else:
            logger.warning(f"Unknown coworker definition format: {cw}")
    return coworkers


def _resolve_guardrail(guardrail_def: dict[str, Any] | str | None) -> Any:
    """Resolve guardrail definition.

    Supports:
    - String shorthand: converted to an LLM guardrail with the string as instructions.
    - Dict with type "llm": creates an LLMGuardrail.
    - Dict with type "code": resolves a dotted function path.
    """
    if guardrail_def is None:
        return None

    # GAP-91: String shorthand -> LLM guardrail
    if isinstance(guardrail_def, str):
        guardrail_def = {"type": "llm", "instructions": guardrail_def}

    if not isinstance(guardrail_def, dict):
        return None

    guard_type = guardrail_def.get("type", "")
    if guard_type == "llm":
        from crewai.tasks.llm_guardrail import LLMGuardrail
        from crewai.utilities.llm_utils import create_llm

        llm_ref = guardrail_def.get("llm", "openai/gpt-4o-mini")
        llm = create_llm(llm_ref) if isinstance(llm_ref, str) else llm_ref
        return LLMGuardrail(
            description=guardrail_def.get("instructions", ""),
            llm=llm,
        )

    # GAP-106: Code guardrail — resolve dotted function path
    if guard_type == "code":
        import importlib

        code_path = guardrail_def.get("function", guardrail_def.get("path", ""))
        if code_path:
            try:
                module_path, func_name = code_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)
                return func
            except Exception as e:
                logger.warning(f"Failed to resolve code guardrail '{code_path}': {e}")
        return None

    return None


def _resolve_custom_tool(tool_name: str) -> Any:
    """Resolve a custom tool from the project's tools/ directory."""
    tools_dir = Path.cwd() / "tools"
    tool_file = tools_dir / f"{tool_name}.py"
    if not tool_file.exists():
        logger.warning(f"Custom tool file not found: {tool_file}")
        return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            f"custom_tools.{tool_name}", tool_file
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        from crewai.tools.base_tool import BaseTool

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseTool)
                and attr is not BaseTool
            ):
                return attr()
        logger.warning(f"No BaseTool subclass found in {tool_file}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load custom tool '{tool_name}': {e}")
        return None


def _resolve_knowledge_sources(sources: list[dict[str, Any]]) -> list[Any]:
    """Resolve knowledge source definitions into knowledge source instances."""
    resolved = []
    for src in sources:
        path_str = src.get("path", "")
        if not path_str:
            continue
        path = Path(path_str)
        try:
            if path.is_dir():
                from crewai.knowledge.source.directory_knowledge_source import (
                    DirectoryKnowledgeSource,
                )

                resolved.append(DirectoryKnowledgeSource(path=path_str))
            elif path.suffix.lower() == ".csv":
                from crewai.knowledge.source.csv_knowledge_source import (
                    CSVKnowledgeSource,
                )

                resolved.append(CSVKnowledgeSource(file_paths=[path_str]))
            elif path.suffix.lower() == ".pdf":
                from crewai.knowledge.source.pdf_knowledge_source import (
                    PDFKnowledgeSource,
                )

                resolved.append(PDFKnowledgeSource(file_paths=[path_str]))
            elif path.suffix.lower() in (".xls", ".xlsx"):
                from crewai.knowledge.source.excel_knowledge_source import (
                    ExcelKnowledgeSource,
                )

                resolved.append(ExcelKnowledgeSource(file_paths=[path_str]))
            elif path.suffix.lower() == ".json":
                from crewai.knowledge.source.json_knowledge_source import (
                    JSONKnowledgeSource,
                )

                resolved.append(JSONKnowledgeSource(file_paths=[path_str]))
            elif path.suffix.lower() == ".txt":
                from crewai.knowledge.source.text_file_knowledge_source import (
                    TextFileKnowledgeSource,
                )

                resolved.append(TextFileKnowledgeSource(file_paths=[path_str]))
            else:
                from crewai.knowledge.source.text_file_knowledge_source import (
                    TextFileKnowledgeSource,
                )

                resolved.append(TextFileKnowledgeSource(file_paths=[path_str]))
        except Exception as e:
            logger.warning(f"Failed to resolve knowledge source '{path_str}': {e}")
    return resolved


def _resolve_response_model(dotted_path: str) -> type | None:
    """Resolve a dotted path string to a Pydantic BaseModel class."""
    try:
        import importlib

        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        from pydantic import BaseModel

        if isinstance(cls, type) and issubclass(cls, BaseModel):
            return cls
        logger.warning(f"response_model '{dotted_path}' is not a BaseModel subclass")
        return None
    except Exception as e:
        logger.warning(f"Failed to resolve response_model '{dotted_path}': {e}")
        return None


def _resolve_mcps(mcp_defs: list[Any]) -> list[Any]:
    """Resolve MCP definitions into proper config objects."""
    resolved = []
    for mcp in mcp_defs:
        if isinstance(mcp, str):
            resolved.append(mcp)
        elif isinstance(mcp, dict):
            url = mcp.get("url", "")
            if url:
                try:
                    from crewai.mcp import MCPServerConfig

                    resolved.append(MCPServerConfig(url=url, name=mcp.get("name", "")))
                except ImportError:
                    resolved.append(url)
            else:
                resolved.append(mcp)
        else:
            resolved.append(mcp)
    return resolved
