from typing import Any, Optional, List, Callable

from langchain_core.messages import AIMessage
from crewai.agents.agent_interface import AgentWrapperParent


class LangchainAgent(AgentWrapperParent):

    def __init__(
        self,
        agent_from_tools: Callable[[Any], Any],
        tools: List[Any],
        role: str,
        allow_delegation: bool = False,
        **data: Any
    ):
        super().__init__(role=role, allow_delegation=allow_delegation, **data)
        self.data.update(data)

        self.data["agent_from_tools"] = agent_from_tools
        # store tools by name to eliminate duplicates
        self.data["tools"] = {}
        self.add_tools(tools)
        self.create_agent_executor()

    def execute_task(
        self,
        task: str,
        context: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        self.tools += tools or []

        if context:
            context = [AIMessage(content=ctx) for ctx in context]
        else:
            context = []

        return self.data["agent_executor"].invoke(
            {"input": task, "chat_history": context}
        )["output"]

    @property
    def tools(self) -> List[Any]:
        return list(self.data["tools"].values())

    @tools.setter
    def tools(self, tools: List[Any]) -> None:
        added = self.add_tools(tools)
        if added:
            # Is there a way to add tools without re-creating the agent?
            self.create_agent_executor()

    def add_tools(self, tools: List[Any] | None) -> int:
        if tools is None:
            return 0
        added = 0
        for tool in tools:
            if tool.name not in self.data["tools"]:
                self.data["tools"][tool.name] = tool
                added += 1
        return added

    def create_agent_executor(self) -> None:
        self.data["agent_executor"] = self.data["agent_from_tools"](self.tools)
