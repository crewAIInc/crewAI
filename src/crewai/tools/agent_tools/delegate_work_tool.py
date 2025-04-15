from typing import Optional, Union, Dict, Any

from pydantic import BaseModel, Field

from crewai.tools.agent_tools.base_agent_tools import BaseAgentTool


class DelegateWorkToolSchema(BaseModel):
    task: Union[str, Dict[str, Any]] = Field(..., description="The task to delegate")
    context: str = Field(..., description="The context for the task")
    coworker: str = Field(
        ..., description="The role/name of the coworker to delegate to"
    )


class DelegateWorkTool(BaseAgentTool):
    """Tool for delegating work to coworkers"""

    name: str = "Delegate work to coworker"
    args_schema: type[BaseModel] = DelegateWorkToolSchema

    def _run(
        self,
        task: Union[str, Dict[str, Any]],
        context: Union[str, Dict[str, Any]],
        coworker: Optional[str] = None,
        **kwargs,
    ) -> str:
        # Convert task and context to strings if they're dictionaries
        if isinstance(task, dict) and "description" in task:
            task = task["description"]
        
        if isinstance(context, dict) and "description" in context:
            context = context["description"]
            
        coworker = self._get_coworker(coworker, **kwargs)
        return self._execute(coworker, task, context)
