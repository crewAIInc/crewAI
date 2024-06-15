from pydantic import Field
from typing import Any, List

from crewai.agent import Agent


class CustomAgentWrapper(Agent):
    custom_agent: Any = Field(default=None)
    agent_executor: Any = Field(default=None)
    tools: List[Any] = Field(default=None)

    def __init__(self, custom_agent, agent_executor, **data):
        print("data", data)
        print("passed in custom_agent", custom_agent)
        print("passed in agent_executor", agent_executor)
        super().__init__(**data)
        self.custom_agent = custom_agent
        self.agent_executor = agent_executor
        self.tools = data.get("tools")

    def create_agent_executor(self, tools=None) -> None:
        pass

    #     """Create an agent executor for the agent.

    #     Returns:
    #         An instance of the CrewAgentExecutor class.
    #     """
    #     tools = tools or self.tools

    #     executor_args = {
    #         "llm": self.llm,
    #         "i18n": self.i18n,
    #         "crew": self.crew,
    #         "crew_agent": self,
    #         "tools": self._parse_tools(tools),
    #         "verbose": self.verbose,
    #         "original_tools": tools,
    #         "handle_parsing_errors": True,
    #         "max_iterations": self.max_iter,
    #         "max_execution_time": self.max_execution_time,
    #         "step_callback": self.step_callback,
    #         "tools_handler": self.tools_handler,
    #         "function_calling_llm": self.function_calling_llm,
    #         "callbacks": self.callbacks,
    #     }

    #     if self._rpm_controller:
    #         executor_args["request_within_rpm_limit"] = (
    #             self._rpm_controller.check_or_wait
    #         )

    #     prompt = Prompts(
    #         i18n=self.i18n,
    #         tools=tools,
    #         system_template=self.system_template,
    #         prompt_template=self.prompt_template,
    #         response_template=self.response_template,
    #     ).task_execution()

    #     execution_prompt = prompt.partial(
    #         goal=self.goal,
    #         role=self.role,
    #         backstory=self.backstory,
    #     )

    #     stop_words = [self.i18n.slice("observation")]
    #     if self.response_template:
    #         stop_words.append(
    #             self.response_template.split("{{ .Response }}")[1].strip()
    #         )

    #     bind = self.llm.bind(stop=stop_words)
    #     print('self.custom_agent',self.custom_agent)

    def execute_task(self, task, context=None, tools=None):
        print("tools used", tools)
        if self.tools_handler:
            # type: ignore # Incompatible types in assignment (expression has type "dict[Never, Never]", variable has type "ToolCalling")
            self.tools_handler.last_used_tool = {}
        task_prompt = task.prompt()
        print("task_prompt", task_prompt)
        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        return self.agent_executor(task_prompt)
