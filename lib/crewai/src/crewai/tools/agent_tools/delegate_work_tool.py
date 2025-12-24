from pydantic import BaseModel, Field

from crewai.tools.agent_tools.base_agent_tools import BaseAgentTool


class DelegateWorkToolSchema(BaseModel):
    task: str = Field(..., description="De taak om te delegeren")
    context: str = Field(..., description="De context voor de taak")
    coworker: str = Field(
        ..., description="De rol/naam van de collega om naar te delegeren"
    )


class DelegateWorkTool(BaseAgentTool):
    """Tool voor het delegeren van werk aan collega's"""

    name: str = "Werk delegeren aan collega"
    args_schema: type[BaseModel] = DelegateWorkToolSchema

    def _run(
        self,
        task: str,
        context: str,
        coworker: str | None = None,
        **kwargs,
    ) -> str:
        coworker = self._get_coworker(coworker, **kwargs)
        return self._execute(coworker, task, context)
