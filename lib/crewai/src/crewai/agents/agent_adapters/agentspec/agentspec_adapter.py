from pydantic import Field, PrivateAttr
from typing import Any, Optional, Dict, Union, Callable, List

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.agent_adapters.base_agent_adapter import BaseAgentAdapter
from crewai import Agent as CrewAIAgent
from crewai.tools.base_tool import BaseTool, Tool as CrewAITool
from crewai.utilities.import_utils import import_and_validate_definition
from crewai.utilities.types import LLMMessage


class AgentSpecAgentAdapter(BaseAgentAdapter):

    _crewai_agent: CrewAIAgent = PrivateAttr()
    function_calling_llm: Any = Field(default=None)
    step_callback: Any = Field(default=None)

    def __init__(
        self,
        agentspec_agent_json: str,
        tool_registry: Optional[Dict[str, Union[Callable, CrewAITool]]] = None,
        **kwargs: Any,
    ):
        agent_spec_loader: type[Any] = import_and_validate_definition(
            "crewai_agentspec_adapter.AgentSpecLoader"
        )
        loader = agent_spec_loader(tool_registry=tool_registry)
        crewai_agent = loader.load_json(agentspec_agent_json)

        init_kwargs = {
            "role": getattr(crewai_agent, "role", "AgentSpec Agent"),
            "goal": getattr(crewai_agent, "goal", "Execute tasks defined by AgentSpec"),
            "backstory": getattr(
                crewai_agent, "backstory", "Adapter wrapper around AgentSpec-generated CrewAI agent"
            ),
            "llm": getattr(crewai_agent, "llm", None),
            "function_calling_llm": getattr(crewai_agent, "llm", None),
            "tools": getattr(crewai_agent, "tools", None),
            "verbose": getattr(crewai_agent, "verbose", False),
            "max_iter": getattr(crewai_agent, "max_iter", 25),
        }
        init_kwargs.update(kwargs or {})
        super().__init__(**{k: v for k, v in init_kwargs.items() if v is not None})

        self.function_calling_llm = getattr(crewai_agent, "llm", None)
        self._crewai_agent = crewai_agent


    # --- Abstract methods of BaseAgentAdapter ---

    def configure_tools(self, tools: list[BaseTool] | None = None) -> None:
        # Nothing to do, tools were already converted by AgentSpecLoader
        pass

    @property
    def last_messages(self) -> list[LLMMessage]:
        return self._crewai_agent.last_messages


    # --- Abstract methods of BaseAgent ---
    # We just delegate to the underlying agent's methods, since it's all already
    # created by AgentSpecLoader (the output is crewai.Agent which is derived from BaseAgent)

    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> Any:
        return self._crewai_agent.execute_task(task, context=context, tools=tools)

    def create_agent_executor(self, tools: Optional[List[Any]] = None) -> None:
        self._crewai_agent.create_agent_executor(tools=tools)

    def get_delegation_tools(self, agents: List[BaseAgent]) -> List[Any]:
        return self._crewai_agent.get_delegation_tools(agents)

    def get_platform_tools(self, apps: List[Any]) -> List[Any]:
        return self._crewai_agent.get_platform_tools(apps)

    def get_mcp_tools(self, mcps: List[Any]) -> List[Any]:
        return self._crewai_agent.get_mcp_tools(mcps)
