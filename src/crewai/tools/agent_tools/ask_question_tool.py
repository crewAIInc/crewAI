from typing import List, Optional

from pydantic import BaseModel, Field

from crewai.tools.agent_tools.base_agent_tools import BaseAgentTool
from crewai.tools.base_tool import BaseTool


class AskQuestionToolSchema(BaseModel):
    question: str = Field(..., description="The question to ask")
    context: str = Field(..., description="The context for the question")
    coworker: str = Field(..., description="The role/name of the coworker to ask")


class AskQuestionTool(BaseAgentTool):
    """Tool for asking questions to coworkers"""

    name: str = "Ask question to coworker"
    args_schema: type[BaseModel] = AskQuestionToolSchema
    _agent_tools: Optional[List[BaseTool]] = None

    def _run(
        self,
        question: str,
        context: str,
        coworker: Optional[str] = None,
        **kwargs,
    ) -> str:
        coworker = self._get_coworker(coworker, **kwargs)
        tools = self._get_tools(**kwargs)
        return self._execute(coworker, question, context, tools)
