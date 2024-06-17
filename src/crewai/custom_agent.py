import inspect
from pydantic import Field
from typing import Any, List, Optional, Dict

from crewai.agent import Agent


class CustomAgent(Agent):
    """Extends the Agent class to create a custom agent (llamaindex, langchain custom agents, etc)"""

    agent_executor: Any = Field(
        default=None,
        description="Bring the agent executor method of a custom agent to execute/run the agent.",
    )

    def __init__(self, agent_executor, **data):
        super().__init__(**data)
        self.agent_executor = agent_executor

    def create_agent_executor(self, tools=None) -> None:
        # overriding the create_agent_executor since custom agents utilize their own
        pass

    def execute_task(self, task, context=None, tools=None, *args, **kwargs):
        if self.tools_handler:
            # type: ignore # Incompatible types in assignment (expression has type "dict[Never, Never]", variable has type "ToolCalling")
            self.tools_handler.last_used_tool = {}
        task_prompt = task.prompt()
        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        sig = inspect.signature(self.agent_executor)
        params = sig.parameters

        # for custom langchain agents and then other third party agents
        for param in params.values():
            print("PARAM", param.annotation)
            if param.annotation == Dict[str, Any]:
                return self.agent_executor({"input": task_prompt})
            elif (
                param.annotation == list
                or param.annotation == Optional[List[Dict[str, Any]]]
            ):
                return self.agent_executor(
                    messages=[{"content": task_prompt, "role": "user"}]
                )
            elif param.annotation == str:
                return self.agent_executor(task_prompt)
            else:
                raise TypeError(f"Unsupported parameter type: {param.annotation}")
        # works with llama index
        return self.agent_executor(task_prompt)
