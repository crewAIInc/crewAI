from pydantic import BaseModel, Field

from crewai.tools.agent_tools.base_agent_tools import BaseAgentTool


class AskQuestionToolSchema(BaseModel):
    question: str = Field(..., description="De vraag om te stellen")
    context: str = Field(..., description="De context voor de vraag")
    coworker: str = Field(..., description="De rol/naam van de collega om te vragen")


class AskQuestionTool(BaseAgentTool):
    """Tool voor het stellen van vragen aan collega's"""

    name: str = "Vraag stellen aan collega"
    args_schema: type[BaseModel] = AskQuestionToolSchema

    def _run(
        self,
        question: str,
        context: str,
        coworker: str | None = None,
        **kwargs,
    ) -> str:
        coworker = self._get_coworker(coworker, **kwargs)
        return self._execute(coworker, question, context)
