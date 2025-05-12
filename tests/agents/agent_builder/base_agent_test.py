import hashlib
from typing import Any

from pydantic import BaseModel

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools.base_tool import BaseTool


class MockAgent(BaseAgent):
    def execute_task(
        self,
        task: Any,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> str:
        return ""

    def create_agent_executor(self, tools=None) -> None: ...

    def get_delegation_tools(self, agents: list["BaseAgent"]) -> None: ...

    def get_output_converter(
        self, llm: Any, text: str, model: type[BaseModel] | None, instructions: str,
    ) -> None: ...


def test_key() -> None:
    agent = MockAgent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
    )
    hash = hashlib.md5(b"test role|test goal|test backstory").hexdigest()
    assert agent.key == hash
