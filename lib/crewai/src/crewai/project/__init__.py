"""Project package for CrewAI."""

from crewai.project.annotations import (
    after_kickoff,
    agent,
    before_kickoff,
    cache_handler,
    callback,
    crew,
    llm,
    output_json,
    output_pydantic,
    task,
    tool,
)
from crewai.project.crew_base import CrewBase
from crewai.project.crew_definition import (
    CrewAgentDefinition,
    CrewDefinition,
    CrewTaskDefinition,
    PythonReferenceDefinition,
)
from crewai.project.crew_loader import load_crew, load_crew_and_kickoff
from crewai.project.json_loader import load_agent, strip_jsonc_comments


__all__ = [
    "CrewAgentDefinition",
    "CrewBase",
    "CrewDefinition",
    "CrewTaskDefinition",
    "PythonReferenceDefinition",
    "after_kickoff",
    "agent",
    "before_kickoff",
    "cache_handler",
    "callback",
    "crew",
    "llm",
    "load_agent",
    "load_crew",
    "load_crew_and_kickoff",
    "output_json",
    "output_pydantic",
    "strip_jsonc_comments",
    "task",
    "tool",
]
