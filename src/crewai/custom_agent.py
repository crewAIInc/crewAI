import inspect
from pydantic import Field
from typing import Any, List, Optional, Dict

from crewai.agent import Agent


class CustomAgent(Agent):
    """Extends the Agent class to create a custom agent (llamaindex, langchain custom agents, etc)

    Each agent has a role, a goal, a backstory, agent_executor and an optional output_key if method to execute agent returns a dict instead of a string.

    Attributes:
    agent_executor: The chat/execute/generate_reply method of the of the agent you bring in.
    output_key: The key of the output to return if the agent_executor() returns a Dict instead of a string.
    role: The role of the agent.
    goal: The objective of the agent.
    backstory: The backstory of the agent.
    """

    agent_executor: Any = Field(
        default=None,
        description="Bring the agent executor method of a custom agent to execute/run the agent.",
    )
    output_key: Optional[str] = Field(
        default=None,
        description="The key of the output to return if the agent_executor() returns a Dict instead of a string",
    )

    def __init__(self, agent_executor, output_key=None, **data):
        super().__init__(**data)
        self.agent_executor = agent_executor
        self.output_key = output_key

    def create_agent_executor(self, tools=None) -> None:
        # overriding the create_agent_executor since custom agents utilize their own
        pass

    def execute_task(self, task, context=None, tools=None) -> str:
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

        for param in params.values():
            if param.annotation == Dict[str, Any]:
                # Check for default values to determine expected keys
                default_value = (
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else {}
                )
                if isinstance(default_value, dict) and default_value:
                    # Find the key that has a default value of type string
                    for key, value in default_value.items():
                        if isinstance(value, str):
                            result = self.agent_executor({key: task_prompt})
                            break
                    else:
                        result = self.agent_executor({"input": task_prompt})
                else:
                    result = self.agent_executor({"input": task_prompt})
            elif (
                param.annotation == list
                or param.annotation == Optional[List[Dict[str, Any]]]
            ):  # some agents take a list of messages like autogen
                result = self.agent_executor(
                    messages=[{"content": task_prompt, "role": "user"}]
                )
            elif (
                param.annotation == str
            ):  # some agent runners only take a string like llamaindex
                result = self.agent_executor(task_prompt)
            else:
                raise TypeError(f"Unsupported parameter type: {param.annotation}")
            break
        else:
            result = self.agent_executor(task_prompt)

        if self.output_key and isinstance(result, dict):
            return result.get(self.output_key, "")

        return result
