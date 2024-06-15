import inspect
from pydantic import Field
from typing import Any, List

from crewai.agent import Agent


class CustomAgentWrapper(Agent):
    custom_agent: Any = Field(default=None)
    agent_executor: Any = Field(default=None)
    # agent_executor_partial: Any = Field(default=None)
    tools: List[Any] = Field(default=None)

    def __init__(self, custom_agent, agent_executor, **data):
        super().__init__(**data)
        self.custom_agent = custom_agent
        self.agent_executor = agent_executor
        # self.agent_executor_partial = partial(agent_executor, **data)
        self.tools = data.get("tools")

    def create_agent_executor(self, tools=None) -> None:
        pass

    def execute_task(self, task, context=None, tools=None, *args, **kwargs):
        if self.tools_handler:
            # type: ignore # Incompatible types in assignment (expression has type "dict[Never, Never]", variable has type "ToolCalling")
            self.tools_handler.last_used_tool = {}
        task_prompt = task.prompt()
        # print("task_prompt", task_prompt)
        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )
        # print('agent_executor_partial',self.agent_executor)
        # parsed_tools = self._parse_tools(tools)
        # tools_description = render_text_description(parsed_tools)

        sig = inspect.signature(self.agent_executor)
        params = sig.parameters

        if "input" in params:
            return self.agent_executor({"input": task_prompt})

        return self.agent_executor(task_prompt)
