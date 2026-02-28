from pydantic import Field, PrivateAttr
from typing import Any, Optional, Dict, Union, Callable, List

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.agent_adapters.base_agent_adapter import BaseAgentAdapter
from crewai import Agent as CrewAIAgent
from crewai import Flow as CrewAIFlow
from crewai.tools.base_tool import BaseTool, Tool as CrewAITool
from crewai.utilities.import_utils import import_and_validate_definition
from crewai.utilities.types import LLMMessage


class AgentSpecAgentAdapter(BaseAgentAdapter):
    """
    Adapter that lets CrewAI import agents defined using Oracle's AgentSpec specification language.
    (https://github.com/oracle/agent-spec.git)

    This adapter wraps around the crewaiagentspecadapter which provides all required
    conversion methods for loading an AgentSpec representation into a CrewAI Agent.
    (https://github.com/oracle/agent-spec/tree/main/adapters/crewaiagentspecadapter)

    When the conversion is done, this adapter delegates required methods to corresponding
    methods of the underlying converted agent.

    Supported features:
    - ReAct-style agents
    - Tools
    - Flows (without inputs)

    Not currently supported:
    - Multi-agent patterns

    Installation:
    1) git clone https://github.com/oracle/agent-spec.git
    2) cd agent-spec
    3) pip install pyagentspec
    4) pip install adapters/crewaiagentspecadapter
    """

    _crewai_component: CrewAIAgent | CrewAIFlow = PrivateAttr()
    function_calling_llm: Any = Field(default=None)
    step_callback: Any = Field(default=None)

    def __init__(
        self,
        agentspec_agent_json: str,
        tool_registry: Optional[Dict[str, Union[Callable, CrewAITool]]] = None,
        **kwargs: Any,
    ):
        agent_spec_loader: type[Any] = import_and_validate_definition(
            "pyagentspec.adapters.crewai.AgentSpecLoader"
        )
        loader = agent_spec_loader(tool_registry=tool_registry)
        crewai_component = loader.load_json(agentspec_agent_json)

        init_kwargs = {
            "role": getattr(crewai_component, "role", "AgentSpec Agent"),
            "goal": getattr(crewai_component, "goal", "Execute tasks defined by AgentSpec"),
            "backstory": getattr(
                crewai_component, "backstory", "Adapter wrapper around AgentSpec-generated CrewAI agent"
            ),
            "llm": getattr(crewai_component, "llm", None),
            "function_calling_llm": getattr(crewai_component, "llm", None),
            "tools": getattr(crewai_component, "tools", None),
            "verbose": getattr(crewai_component, "verbose", False),
            "max_iter": getattr(crewai_component, "max_iter", 25),
        }
        init_kwargs.update(kwargs or {})
        super().__init__(**{k: v for k, v in init_kwargs.items() if v is not None})

        self._crewai_component = crewai_component


    # --- Abstract methods of BaseAgentAdapter ---

    def configure_tools(self, tools: list[BaseTool] | None = None) -> None:
        # Nothing to do, tools were already converted by AgentSpecLoader
        pass

    @property
    def last_messages(self) -> list[LLMMessage]:
        return getattr(self._crewai_component, "last_messages", [])


    # --- Abstract methods of BaseAgent ---
    # We just delegate to the underlying agent's methods, since it's all already
    # created by AgentSpecLoader (the output is crewai.Agent which is derived from BaseAgent)

    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> Any:
        if isinstance(self._crewai_component, CrewAIAgent):
            return self._crewai_component.execute_task(task, context=context, tools=tools)
        elif isinstance(self._crewai_component, CrewAIFlow):
            return self._crewai_component.kickoff({})
        raise TypeError(
            f"Expected underlying component to be an Agent or a Flow but received {type(self._crewai_component)}"
        )

    def create_agent_executor(self, tools: Optional[List[Any]] = None) -> None:
        if isinstance(self._crewai_component, CrewAIAgent):
            self._crewai_component.create_agent_executor(tools=tools)

    def get_delegation_tools(self, agents: List[BaseAgent]) -> List[Any]:
        if isinstance(self._crewai_component, CrewAIAgent):
            return self._crewai_component.get_delegation_tools(agents)
        return []

    def get_platform_tools(self, apps: List[Any]) -> List[Any]:
        if isinstance(self._crewai_component, CrewAIAgent):
            return self._crewai_component.get_platform_tools(apps)
        return []

    def get_mcp_tools(self, mcps: List[Any]) -> List[Any]:
        if isinstance(self._crewai_component, CrewAIAgent):
            return self._crewai_component.get_mcp_tools(mcps)
        return []
