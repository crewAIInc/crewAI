import hashlib
from typing import Any, List, Optional

from pydantic import BaseModel

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools.base_tool import BaseTool


class MockAgent(BaseAgent):
    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> str:
        return ""

    def create_agent_executor(self, tools=None) -> None: ...

    def get_delegation_tools(self, agents: List["BaseAgent"]): ...

    def get_output_converter(
        self, llm: Any, text: str, model: type[BaseModel] | None, instructions: str
    ): ...


def test_key():
    agent = MockAgent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
    )
    hash = hashlib.md5("test role|test goal|test backstory".encode()).hexdigest()
    assert agent.key == hash
