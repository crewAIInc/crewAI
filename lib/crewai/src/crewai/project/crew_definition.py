"""Definition models for inline CrewAI crew configurations."""

from __future__ import annotations

from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from crewai.agent.planning_config import PlanningConfig


__all__ = [
    "AgentDefinition",
    "CrewAgentDefinition",
    "CrewDefinition",
    "CrewTaskDefinition",
    "LLMDefinition",
    "PythonReferenceDefinition",
]


class PythonReferenceDefinition(BaseModel):
    """Dotted Python reference used by crew definitions."""

    python: str = Field(
        description="Dotted Python import path to load.",
        examples=["my_project.schemas.SupportReply"],
    )

    @field_validator("python")
    @classmethod
    def _validate_python_ref(cls, value: str) -> str:
        path = value.strip()
        if not path:
            raise ValueError("Python reference 'python' must be a string")
        if "." not in path:
            raise ValueError(
                f"Python reference '{path}' must be a dotted import path "
                "like 'module.attribute'"
            )
        return path


class LLMDefinition(BaseModel):
    """LLM configuration used by inline agent definitions."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(
        description="Model identifier used to instantiate the LLM.",
        examples=["openai/gpt-4o-mini"],
    )
    max_tokens: int | None = Field(
        default=None,
        description=(
            "Maximum number of tokens the LLM can generate. If null, CrewAI "
            "does not set an explicit output token cap and the provider's "
            "default applies."
        ),
        examples=[4096],
    )


class CrewAgentDefinition(BaseModel):
    """Inline agent definition used by a crew definition."""

    model_config = ConfigDict(extra="allow")

    role: str | None = Field(
        default=None,
        description=(
            "Crew agent role. Crew inputs are interpolated with `{name}` "
            "placeholders such as `{topic}`; this is not CEL."
        ),
        examples=["Research analyst"],
    )
    goal: str | None = Field(
        default=None,
        description=(
            "Crew agent goal. Crew inputs are interpolated with `{name}` "
            "placeholders such as `{topic}`; this is not CEL."
        ),
        examples=["Research {topic}"],
    )
    backstory: str | None = Field(
        default=None,
        description=(
            "Crew agent backstory. Crew inputs are interpolated with `{name}` "
            "placeholders such as `{topic}`; this is not CEL."
        ),
        examples=["Expert at concise technical research."],
    )
    type: str | PythonReferenceDefinition | None = Field(
        default=None,
        description="Optional built-in type or Python reference used to load the agent.",
        examples=["agent", {"python": "my_project.agents.ResearchAgent"}],
    )
    from_repository: str | None = Field(
        default=None,
        description=(
            "Agent repository name to load. Repository values supply missing "
            "agent configuration; explicitly provided local fields override the "
            "repository values."
        ),
        examples=["researcher"],
    )
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional agent settings passed to the loader.",
        examples=[{"llm": "openai/gpt-4o-mini"}],
    )
    llm: str | LLMDefinition | None = Field(
        default=None,
        description=(
            "Language model that runs the agent. Use a string model name or an "
            "object with model settings such as max_tokens."
        ),
        examples=[{"model": "openai/gpt-4o-mini", "max_tokens": 4096}],
    )
    planning_config: PlanningConfig | None = Field(
        default=None,
        description="Configuration for agent planning before task execution.",
        examples=[{"max_attempts": 3}],
    )
    allow_delegation: bool | None = Field(
        default=None,
        description="Enable agent to delegate and ask questions among each other.",
        examples=[False],
    )
    max_iter: int | None = Field(
        default=None,
        description="Maximum iterations for an agent to execute a task",
        examples=[25],
    )
    max_rpm: int | None = Field(
        default=None,
        description=(
            "Maximum number of requests per minute for the agent execution to be "
            "respected."
        ),
        examples=[10],
    )
    max_execution_time: int | None = Field(
        default=None,
        description="Maximum execution time in seconds for an agent to execute a task",
        examples=[300],
    )
    tools: list[str | dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Tool refs or serialized tool definitions available to this agent. "
            "String refs can use CrewAI tool names, `custom:<name>`, or fully "
            "qualified `module:Class` references."
        ),
        examples=[["crewai_tools:SerperDevTool", "custom:file_read"]],
    )
    apps: list[str] | None = Field(
        default=None,
        description=(
            "Platform apps available to this agent. Can contain app names such as "
            "`gmail` or app/action refs such as `gmail/send_email`."
        ),
        examples=[["gmail", "slack/send_message"]],
    )
    mcps: list[str | dict[str, Any]] | None = Field(
        default=None,
        description=(
            "MCP server refs or serialized MCP server configs available to this "
            "agent. String refs can use HTTPS URLs, connected MCP integration "
            "slugs, or refs with a `#tool_name` suffix for specific tools."
        ),
        examples=[
            [
                "https://api.weather.com/mcp#get_current_weather",
                "snowflake",
                "stripe#list_invoices",
                {
                    "url": "https://api.example.com/mcp",
                    "headers": {"Authorization": "Bearer your_token"},
                    "streamable": True,
                    "cache_tools_list": True,
                },
            ]
        ],
    )

    @field_validator("settings", mode="before")
    @classmethod
    def _validate_settings(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, dict):
            raise ValueError("agent.settings must be a mapping")
        return value or {}


class AgentDefinition(CrewAgentDefinition):
    """Inline individual agent definition used outside of a crew."""

    role: str | None = Field(
        default=None,
        description="Individual agent role used by a Flow agent action outside of a crew.",
        examples=["Support specialist"],
    )
    goal: str | None = Field(
        default=None,
        description="Individual agent goal for the Flow agent action outside of a crew.",
        examples=["Draft a concise customer reply"],
    )
    backstory: str | None = Field(
        default=None,
        description=(
            "Individual agent backstory used to shape behavior outside of a crew."
        ),
        examples=["Expert at resolving SaaS support questions."],
    )
    type: str | PythonReferenceDefinition | None = Field(
        default=None,
        description="Optional built-in type or Python reference used to load the agent.",
        examples=["agent", {"python": "my_project.agents.SupportAgent"}],
    )
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional agent settings passed to the loader.",
        examples=[{"llm": "openai/gpt-4o-mini"}],
    )
    input: str = Field(
        description="Input passed to the individual agent kickoff outside of a crew.",
        examples=["${state.ticket.body}"],
    )
    response_format: PythonReferenceDefinition | None = Field(
        default=None,
        description="Optional Python reference to a Pydantic response format.",
        examples=[{"python": "my_project.schemas.SupportReply"}],
    )

    @field_validator("input", mode="before")
    @classmethod
    def _validate_input(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("agent.input must be a string")
        return value


class CrewTaskDefinition(BaseModel):
    """Task definition used by a crew definition."""

    model_config = ConfigDict(extra="allow")

    description: str = Field(
        description=(
            "Task instructions. Crew inputs are interpolated with `{name}` "
            "placeholders such as `{topic}`; this is not CEL."
        ),
        examples=["Research {topic}."],
    )
    expected_output: str = Field(
        description=(
            "Expected task output. Crew inputs are interpolated with `{name}` "
            "placeholders such as `{topic}`; this is not CEL."
        ),
        examples=["Key findings about {topic}."],
    )
    name: str | None = Field(
        default=None,
        description="Optional task name used by context references.",
        examples=["research_task"],
    )
    agent: str | None = Field(
        default=None,
        description="Name of the crew agent assigned to this task.",
        examples=["researcher"],
    )
    context: list[str] | None = Field(
        default=None,
        description="Names of previous tasks whose outputs should be used as context.",
        examples=[["research_task"]],
    )
    type: str | PythonReferenceDefinition | None = Field(
        default=None,
        description="Optional built-in type or Python reference used to load the task.",
        examples=["task", {"python": "my_project.tasks.ResearchTask"}],
    )


_CrewAgentsInput: TypeAlias = dict[str, CrewAgentDefinition] | list[dict[str, Any]]


class CrewDefinition(BaseModel):
    """In-memory JSON/YAML crew definition with inline agents and tasks."""

    model_config = ConfigDict(extra="allow")

    agents: dict[str, CrewAgentDefinition] = Field(
        description="Inline crew agents keyed by agent name.",
        examples=[
            {
                "researcher": {
                    "role": "Research analyst",
                    "goal": "Research {topic}",
                    "backstory": "Expert at concise technical research.",
                }
            }
        ],
    )
    tasks: list[CrewTaskDefinition] = Field(
        description="Ordered crew tasks.",
        examples=[
            [
                {
                    "name": "research_task",
                    "description": "Research {topic}.",
                    "expected_output": "Key findings about {topic}.",
                    "agent": "researcher",
                }
            ]
        ],
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Default crew inputs. Values are available to crew agent and task "
            "interpolation as `{name}` placeholders, for example `{topic}`."
        ),
        examples=[{"topic": "AI agents"}],
    )
    manager_agent: str | PythonReferenceDefinition | None = Field(
        default=None,
        description="Optional manager agent name or Python reference.",
        examples=["manager", {"python": "my_project.agents.ManagerAgent"}],
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def _validate_inputs(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, dict):
            raise ValueError("crew.inputs must be a mapping")
        return value or {}

    @field_validator(
        "agents",
        mode="before",
        json_schema_input_type=_CrewAgentsInput,
    )
    @classmethod
    def _validate_inline_agents(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return value
        if not isinstance(value, list):
            return value

        agents: dict[str, Any] = {}
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"agents[{index}] must be an inline agent mapping")

            if "name" in item:
                name = item["name"]
                if not isinstance(name, str) or not name:
                    raise ValueError(f"agents[{index}].name must be a non-empty string")
                agents[name] = {key: val for key, val in item.items() if key != "name"}
                continue

            if len(item) != 1:
                raise ValueError(
                    f"agents[{index}] must include a name field or be a one-key mapping"
                )
            name, definition = next(iter(item.items()))
            agents[str(name)] = definition

        return agents

    @model_validator(mode="after")
    def _validate_inline_shape(self) -> CrewDefinition:
        if not self.agents:
            raise ValueError("crew action requires inline agent definitions")

        if not self.tasks:
            raise ValueError("crew action requires a non-empty tasks list")
        return self
