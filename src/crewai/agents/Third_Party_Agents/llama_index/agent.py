from typing import List, Any
from pydantic import Field

from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

from crewai.agents.third_party_agents.base_agent import BaseAgent
from crewai.memory.contextual.contextual_memory import ContextualMemory
from crewai.agents.third_party_agents.llama_index.tools.task_tools import (
    LlamaAgentTools,
)


class LlamaIndexAgent(BaseAgent):
    llm: OpenAI = Field(
        default_factory=lambda: OpenAI(model="gpt-3.5-turbo-0613"),
        description="Language model that will run the agent.",
    )

    def __init__(
        self,
        tools: List[FunctionTool] = [],
        llm: OpenAI = OpenAI(temperature=0, model="gpt-4o"),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tools = tools
        self.llm = llm

    def execute_task(self, task, context=None, tools=None) -> str:
        task_prompt = task.prompt()
        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        if self.crew and self.crew.memory:
            contextual_memory = ContextualMemory(
                self.crew._short_term_memory,
                self.crew._long_term_memory,
                self.crew._entity_memory,
            )
            memory = contextual_memory.build_context_for_task(task, context)
            if memory.strip() != "":
                task_prompt += self.i18n.slice("memory").format(memory=memory)

        tools = tools or self.tools
        parsed_tools = self._parse_tools(tools)
        self.create_agent_executor(tools=tools)
        self.agent_executor.tools = parsed_tools
        return self.agent_executor.chat(task_prompt)

    def _parse_tools(self, tools: List[Any]) -> List[FunctionTool]:
        """Ensures tools being passed are correct for llama index"""
        tools_list = []
        try:
            from llama_index.core.tools import FunctionTool

            for tool in tools:
                if isinstance(tool, FunctionTool):
                    tools_list.append(tool)
                else:
                    tools_list.append(tool)
        except ModuleNotFoundError:
            for tool in tools:
                tools_list.append(tool)
        return tools_list

    def create_agent_executor(self, tools=None):
        self.agent_executor = OpenAIAgent.from_tools(
            tools=tools,
            llm=self.llm,
            verbose=self.verbose,
        )

    def create_delegate_work_tool(self, agents):
        print("agents to call from delegation", agents)
        coworkers = f"[{', '.join([f'{agent.role}' for agent in agents])}]"
        return FunctionTool.from_defaults(
            fn=self.delegate_work,
            name="Delegate-work-to-coworker",
            description=self.i18n.tools("delegate_work").format(coworkers=coworkers),
        )

    def create_ask_question_tool(self, agents):
        coworkers = f"[{', '.join([f'{agent.role}' for agent in agents])}]"
        return FunctionTool.from_defaults(
            fn=self.ask_question,
            name="Ask-question-to-coworker",
            description=self.i18n.tools("ask_question").format(coworkers=coworkers),
        )

    def set_agent_tools(self, agents: List[BaseAgent]):
        """Set the agent tools and update tools."""

        agent_tools = LlamaAgentTools(agents=agents)
        tools = agent_tools.tools()
        return tools
