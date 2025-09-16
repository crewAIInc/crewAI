from typing import (
    cast,
)

from pydantic import Field, PrivateAttr, model_validator

from crewai.agent import Agent
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.durable_execution.dbos.dbos_agent_executor import DBOSAgentExecutor
from crewai.durable_execution.dbos.dbos_llm import DBOSLLM
from crewai.durable_execution.dbos.dbos_utils import StepConfig
from crewai.task import Task
from crewai.tools import BaseTool
from crewai.utilities.agent_utils import (
    get_tool_names,
    parse_tools,
    render_text_description_and_args,
)
from crewai.utilities.prompts import Prompts
from crewai.utilities.token_counter_callback import TokenCalcHandler


class DBOSAgent(BaseAgent):
    """Wrap an agent for use in DBOS durable workflows.

    Automatically wraps LLM calls (both llm and function_calling_llm) as DBOS steps.

    Attributes:
            wrapped_agent: The underlying agent instance.
            llm_step_config: The DBOS step configuration to use for LLM steps.
            function_calling_llm_step_config:  The DBOS step configuration to use for function calling LLM steps.
    """

    agent_name: str = Field(description="The unique name of the DBOS agent.")
    orig_agent: Agent = Field(exclude=True, description="The original agent instance.")
    llm_step_config: StepConfig | None = Field(
        default=None,
        description="The DBOS step configuration to use for LLM steps.",
    )
    function_calling_llm_step_config: StepConfig | None = Field(
        default=None,
        description="The DBOS step configuration to use for function calling LLM steps.",
    )
    _wrapped_agent: Agent = PrivateAttr()

    @model_validator(mode="before")
    @classmethod
    def pre_init_setup(cls, values):
        if "orig_agent" not in values or not isinstance(values["orig_agent"], Agent):
            raise ValueError("orig_agent must be provided and be an instance of Agent")
        # populate required fields: role, goal, backstory
        agent = cast(Agent, values["orig_agent"])
        values["role"] = agent.role
        values["goal"] = agent.goal
        values["backstory"] = agent.backstory
        return values

    @model_validator(mode="after")
    def post_init_setup(self):
        # Create a deep copy of the agent to avoid mutating the original
        self._wrapped_agent = self.orig_agent.copy()
        # Replace the original create_agent_executor with a DBOS wrap
        object.__setattr__(
            self._wrapped_agent, "create_agent_executor", self.create_agent_executor
        )
        # TODO: wrap LLM/function calling LLM as steps
        self._wrapped_agent.llm = DBOSLLM(
            orig_llm=self._wrapped_agent.llm,
            step_config=self.llm_step_config,
            agent_name=self.agent_name,
        )
        if self._wrapped_agent.function_calling_llm:
            self._wrapped_agent.function_calling_llm = DBOSLLM(
                orig_llm=self._wrapped_agent.function_calling_llm,
                step_config=self.function_calling_llm_step_config,
                agent_name=self.agent_name,
            )
        return self

    def execute_task(
        self,
        task: Task,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> str:
        return self._wrapped_agent.execute_task(
            task=task,
            context=context,
            tools=tools,
        )

    def get_delegation_tools(self, agents: list[BaseAgent]):
        return self._wrapped_agent.get_delegation_tools(agents=agents)

    def create_agent_executor(
        self, tools: list[BaseTool] | None = None, task=None
    ) -> None:
        """Create a DBOS agent executor for the agent.

        Returns:
            An instance of the DBOSAgentExecutor class.
        """
        print("DBOS create_agent_executor called")

        raw_tools: list[BaseTool] = tools or self._wrapped_agent.tools or []
        parsed_tools = parse_tools(raw_tools)

        prompt = Prompts(
            agent=self._wrapped_agent,
            has_tools=len(raw_tools) > 0,
            i18n=self._wrapped_agent.i18n,
            use_system_prompt=self._wrapped_agent.use_system_prompt,
            system_template=self._wrapped_agent.system_template,
            prompt_template=self._wrapped_agent.prompt_template,
            response_template=self._wrapped_agent.response_template,
        ).task_execution()

        stop_words = [self.i18n.slice("observation")]

        if self._wrapped_agent.response_template:
            stop_words.append(
                self._wrapped_agent.response_template.split("{{ .Response }}")[
                    1
                ].strip()
            )

        self._wrapped_agent.agent_executor = DBOSAgentExecutor(
            llm=self._wrapped_agent.llm,
            task=task,
            agent=self._wrapped_agent,
            crew=self._wrapped_agent.crew,
            tools=parsed_tools,
            prompt=prompt,
            original_tools=raw_tools,
            stop_words=stop_words,
            max_iter=self._wrapped_agent.max_iter,
            tools_handler=self._wrapped_agent.tools_handler,
            tools_names=get_tool_names(parsed_tools),
            tools_description=render_text_description_and_args(parsed_tools),
            step_callback=self._wrapped_agent.step_callback,
            function_calling_llm=self._wrapped_agent.function_calling_llm,
            respect_context_window=self._wrapped_agent.respect_context_window,
            request_within_rpm_limit=(
                self._wrapped_agent._rpm_controller.check_or_wait
                if self._wrapped_agent._rpm_controller
                else None
            ),
            callbacks=[TokenCalcHandler(self._wrapped_agent._token_process)],
            agent_name=self.agent_name,
        )
