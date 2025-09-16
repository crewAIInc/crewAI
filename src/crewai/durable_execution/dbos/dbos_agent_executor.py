"""Agent executor for DBOS durable agents.

Handles agent execution flow including LLM interactions, tool execution,
and memory management. Also automatically wraps LLM calls as DBOS steps.
"""

from collections.abc import Callable
from typing import Any

from dbos import DBOS

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.tools_handler import ToolsHandler
from crewai.tools.structured_tool import CrewStructuredTool


class DBOSAgentExecutor(CrewAgentExecutor):
    """Executor for DBOS agents.

    Manages the execution lifecycle of an agent including prompt formatting,
    LLM interactions, tool execution, and feedback handling. Also automatically wraps LLM calls as DBOS steps.
    """

    def __init__(
        self,
        llm: Any,
        task: Any,
        crew: Any,
        agent: BaseAgent,
        prompt: dict[str, str],
        max_iter: int,
        tools: list[CrewStructuredTool],
        tools_names: str,
        stop_words: list[str],
        tools_description: str,
        tools_handler: ToolsHandler,
        step_callback: Any = None,
        original_tools: list[Any] | None = None,
        function_calling_llm: Any = None,
        respect_context_window: bool = False,
        request_within_rpm_limit: Callable[[], bool] | None = None,
        callbacks: list[Any] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            llm: Language model instance.
            task: Task to execute.
            crew: Crew instance.
            agent: Agent to execute.
            prompt: Prompt templates.
            max_iter: Maximum iterations.
            tools: Available tools.
            tools_names: Tool names string.
            stop_words: Stop word list.
            tools_description: Tool descriptions.
            tools_handler: Tool handler instance.
            step_callback: Optional step callback.
            original_tools: Original tool list.
            function_calling_llm: Optional function calling LLM.
            respect_context_window: Respect context limits.
            request_within_rpm_limit: RPM limit check function.
            callbacks: Optional callbacks list.
            llm_step_config: DBOS step config for LLM calls.
            function_calling_llm_step_config: DBOS step config for function calling LLM calls.
        """
        super().__init__(
            llm=llm,
            task=task,
            crew=crew,
            agent=agent,
            prompt=prompt,
            max_iter=max_iter,
            tools=tools,
            tools_names=tools_names,
            stop_words=stop_words,
            tools_description=tools_description,
            tools_handler=tools_handler,
            step_callback=step_callback,
            original_tools=original_tools,
            function_calling_llm=function_calling_llm,
            respect_context_window=respect_context_window,
            request_within_rpm_limit=request_within_rpm_limit,
            callbacks=callbacks,
        )

        # Overload invoke with DBOS workflow
        @DBOS.workflow(name="dbos_agent_executor_invoke")
        def dbos_invoke(inputs: dict[str, str]) -> dict[str, Any]:
            print("DBOSAgentExecutor invoke called")
            return super(DBOSAgentExecutor, self).invoke(inputs)

        self.dbos_invoke = dbos_invoke

    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        """Invoke the agent executor within a DBOS workflow.

        Args:
            inputs: Input dictionary containing the task input.

        Returns:
            Dictionary with agent output.
        """
        return self.dbos_invoke(inputs)
