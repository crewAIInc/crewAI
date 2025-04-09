import os
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, InstanceOf, PrivateAttr, model_validator

from crewai.agents import CacheHandler
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.llm import LLM
from crewai.task import Task
from crewai.tools import BaseTool
from crewai.tools.base_tool import Tool
from crewai.utilities import Converter, Prompts
from crewai.utilities.token_counter_callback import TokenCalcHandler


class LiteAgent(BaseAgent):
    """Represents a lightweight agent in a system.

    Each agent has a role, a goal, a backstory, and an optional language model (llm).
    The agent can execute tasks but with fewer features compared to the full Agent class.

    This is a simplified version of the Agent class with less dependencies and overhead.

    Attributes:
        agent_executor: An instance of the CrewAgentExecutor class.
        role: The role of the agent.
        goal: The objective of the agent.
        backstory: The backstory of the agent.
        llm: The language model that will run the agent.
        max_iter: Maximum number of iterations for an agent to execute a task.
        verbose: Whether the agent execution should be in verbose mode.
        tools: Tools at agent's disposal
    """

    _times_executed: int = PrivateAttr(default=0)
    max_execution_time: Optional[int] = Field(
        default=None,
        description="Maximum execution time for an agent to execute a task",
    )
    cache_handler: InstanceOf[CacheHandler] = Field(
        default=None, description="An instance of the CacheHandler class."
    )
    llm: Union[str, InstanceOf[LLM], Any] = Field(
        description="Language model that will run the agent.", default=None
    )
    max_iter: int = Field(
        default=20,
        description="Maximum number of iterations for an agent to execute a task before giving it's best answer",
    )
    max_retry_limit: int = Field(
        default=2,
        description="Maximum number of retries for an agent to execute a task when an error occurs.",
    )
    tools_results: Optional[List[Any]] = Field(
        default=[], description="Results of the tools used by the agent."
    )

    @model_validator(mode="after")
    def post_init_setup(self):
        if isinstance(self.llm, str):
            self.llm = LLM(model=self.llm)
        elif isinstance(self.llm, LLM):
            pass
        elif self.llm is None:
            model_name = (
                os.environ.get("OPENAI_MODEL_NAME")
                or os.environ.get("MODEL")
                or "gpt-4o-mini"
            )
            llm_params = {"model": model_name}

            api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get(
                "OPENAI_BASE_URL"
            )
            if api_base:
                llm_params["base_url"] = api_base

            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                llm_params["api_key"] = api_key

            self.llm = LLM(**llm_params)
        else:
            llm_params = {
                "model": getattr(self.llm, "model_name", None)
                or getattr(self.llm, "deployment_name", None)
                or str(self.llm),
                "temperature": getattr(self.llm, "temperature", None),
                "max_tokens": getattr(self.llm, "max_tokens", None),
                "api_key": getattr(self.llm, "api_key", None),
                "base_url": getattr(self.llm, "base_url", None),
                "organization": getattr(self.llm, "organization", None),
            }
            llm_params = {k: v for k, v in llm_params.items() if v is not None}
            self.llm = LLM(**llm_params)

        if not self.agent_executor:
            self._setup_agent_executor()

        return self

    def _setup_agent_executor(self):
        if not self.cache_handler:
            self.cache_handler = CacheHandler()
        self.set_cache_handler(self.cache_handler)

    def execute_task(
        self,
        task: Task,
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> str:
        """Execute a task with the agent.

        Args:
            task: Task to execute.
            context: Context to execute the task in.
            tools: Tools to use for the task.

        Returns:
            Output of the agent
        """
        if self.tools_handler:
            self.tools_handler.last_used_tool = {}

        task_prompt = task.prompt()

        if task.output_json or task.output_pydantic:
            if task.output_json:
                schema = Converter.generate_model_description(task.output_json)
            elif task.output_pydantic:
                schema = Converter.generate_model_description(task.output_pydantic)

            task_prompt += "\n" + self.i18n.slice("formatted_task_instructions").format(
                output_format=schema
            )

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        tools = tools or self.tools or []
        self.create_agent_executor(tools=tools, task=task)

        try:
            result = self.agent_executor.invoke(
                {
                    "input": task_prompt,
                    "tool_names": self.agent_executor.tools_names,
                    "tools": self.agent_executor.tools_description,
                    "ask_for_human_input": task.human_input,
                }
            )["output"]
        except Exception as e:
            self._times_executed += 1
            if self._times_executed > self.max_retry_limit:
                raise e
            result = self.execute_task(task, context, tools)

        if self.max_rpm and self._rpm_controller:
            self._rpm_controller.stop_rpm_counter()

        for tool_result in self.tools_results:
            if tool_result.get("result_as_answer", False):
                result = tool_result["result"]

        return result

    def create_agent_executor(
        self, tools: Optional[List[BaseTool]] = None, task=None
    ) -> None:
        """Create an agent executor for the agent.

        Returns:
            An instance of the CrewAgentExecutor class.
        """
        tools = tools or self.tools or []
        parsed_tools = self._parse_tools(tools)

        prompt = Prompts(
            agent=self,
            tools=tools,
            i18n=self.i18n,
        ).task_execution()

        stop_words = [self.i18n.slice("observation")]

        self.agent_executor = CrewAgentExecutor(
            llm=self.llm,
            task=task,
            agent=self,
            crew=self.crew,
            tools=parsed_tools,
            prompt=prompt,
            original_tools=tools,
            stop_words=stop_words,
            max_iter=self.max_iter,
            tools_handler=self.tools_handler,
            tools_names=self.__tools_names(parsed_tools),
            tools_description=self._render_text_description_and_args(parsed_tools),
            respect_context_window=True,
            request_within_rpm_limit=(
                self._rpm_controller.check_or_wait if self._rpm_controller else None
            ),
            callbacks=[TokenCalcHandler(self._token_process)],
        )

    def get_delegation_tools(self, agents: List[BaseAgent]):
        """Stub implementation - LiteAgent doesn't support delegation."""
        return []

    def get_multimodal_tools(self) -> List[Tool]:
        """Stub implementation - LiteAgent doesn't support multimodal tools."""
        return []

    def get_code_execution_tools(self):
        """Stub implementation - LiteAgent doesn't support code execution."""
        return []

    def get_output_converter(self, llm, text, model, instructions):
        """Get the output converter for the agent."""
        return Converter(llm=llm, text=text, model=model, instructions=instructions)

    def _parse_tools(self, tools: List[Any]) -> List[Any]:
        """Parse tools to be used for the task."""
        tools_list = []
        try:
            from crewai.tools import BaseTool as CrewAITool

            for tool in tools:
                if isinstance(tool, CrewAITool):
                    tools_list.append(tool.to_structured_tool())
                else:
                    tools_list.append(tool)
        except ModuleNotFoundError:
            tools_list = []
            for tool in tools:
                tools_list.append(tool)

        return tools_list

    def _render_text_description_and_args(self, tools: List[BaseTool]) -> str:
        """Render the tool name, description, and args in plain text."""
        tool_strings = []
        for tool in tools:
            tool_strings.append(tool.description)

        return "\n".join(tool_strings)

    @staticmethod
    def __tools_names(tools) -> str:
        """Get the names of the tools as a comma-separated string."""
        return ", ".join([t.name for t in tools])

    def __repr__(self):
        return f"LiteAgent(role={self.role}, goal={self.goal}, backstory={self.backstory})"
