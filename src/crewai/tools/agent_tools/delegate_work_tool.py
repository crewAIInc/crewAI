import json
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field

from crewai.tools.agent_tools.base_agent_tools import BaseAgentTool


class DelegateWorkToolSchema(BaseModel):
    task: Union[str, Dict[str, Any]] = Field(..., description="The task to delegate")
    context: Union[str, Dict[str, Any]] = Field(..., description="The context for the task")
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
        coworker = self._get_coworker(coworker, **kwargs)
        task_str = json.dumps(task) if isinstance(task, dict) else task
        context_str = json.dumps(context) if isinstance(context, dict) else context
        return self._execute(coworker, task_str, context_str)
