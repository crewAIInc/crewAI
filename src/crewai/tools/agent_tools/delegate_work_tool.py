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
    """Tool for delegating work to coworkers"""

    name: str = "Delegate work to coworker"
    args_schema: type[BaseModel] = DelegateWorkToolSchema

    def _run(
        self,
        task: str,
        context: str,
        coworker: Optional[str] = None,
        **kwargs,
    ) -> str:
        coworker = self._get_coworker(coworker, **kwargs)
        
        if hasattr(self, 'agents') and self.agents:
            delegating_agent = kwargs.get('delegating_agent')
            if delegating_agent and hasattr(delegating_agent, 'responsibility_system'):
                responsibility_system = delegating_agent.responsibility_system
                if responsibility_system and responsibility_system.enabled:
                    task_obj = kwargs.get('task_obj')
                    if task_obj:
                        responsibility_system.delegate_task(
                            delegating_agent=delegating_agent,
                            receiving_agent=coworker,
                            task=task_obj,
                            reason=f"Delegation based on capability match for: {task[:100]}..."
                        )
        
        return self._execute(coworker, task, context)
