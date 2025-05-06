from typing import Optional

from pydantic import BaseModel, Field

from crewai.tools.agent_tools.base_agent_tools import BaseAgentTool


class DelegateWorkToolSchema(BaseModel):
    task: str = Field(..., description="The task to delegate")
    context: str = Field(..., description="The context for the task")
    coworker: str = Field(
        ..., description="The role/name of the coworker to delegate to"
    )


class DelegateWorkTool(BaseAgentTool):
    """Tool for delegating work to other agents in the crew.
    
    Attributes:
        result_as_answer (bool): When True, returns the delegated agent's result
            as the final answer instead of metadata about delegation.
    """

    name: str = "Delegate work to coworker"
    args_schema: type[BaseModel] = DelegateWorkToolSchema
    result_as_answer: bool = True

    def _run(
        self,
        task: str,
        context: str,
        coworker: Optional[str] = None,
        **kwargs,
    ) -> str:
        coworker = self._get_coworker(coworker, **kwargs)
        return self._execute(coworker, task, context)
