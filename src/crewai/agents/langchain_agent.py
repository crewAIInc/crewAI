from typing import Any, Optional, List, Callable
import warnings

from langchain_core.messages import AIMessage
from crewai.agents.agent_interface import AgentWrapperParent


class LangchainCrewAgent(AgentWrapperParent):

    def __init__(
        self,
        agent: Any,
        role: str,
        allow_delegation: bool = False,
        tools: List[Any] | None = None,
        **data: Any,
    ):
        super().__init__(role=role, allow_delegation=allow_delegation, **data)
        self.data.update(data)
        self.data["agent"] = agent
        # store tools by name to eliminate duplicates
        self.data["tools"] = {}
        self.tools = tools or []

    def execute_task(
        self,
        task: str,
        context: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        used_tools = self.tools + (tools or [])

        if context:
            context = [AIMessage(content=ctx) for ctx in context]
        else:
            context = []
        # https://github.com/langchain-ai/langchain/discussions/17403
        return self.data["agent"].invoke(
            {"input": task, "chat_history": context, "tools": used_tools}
        )["output"]

    @property
    def tools(self) -> List[Any]:
        return list(self.data["tools"].values())

    @tools.setter
    def tools(self, tools: List[Any]) -> None:
        for tool in tools:
            if tool.name not in self.data["tools"]:
                self.data["tools"][tool.name] = tool
            else:
                warnings.warn(f"Tool {tool.name} already exists in the agent.")
