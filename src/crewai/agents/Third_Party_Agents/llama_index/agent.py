import tiktoken
from typing import List, Any
from pydantic import Field, model_validator

from crewai.agents.third_party_agents.base_agent import BaseAgent
from crewai.agents.third_party_agents.llama_index.utilities.token_handler import (
    ExtendedTokenCountingHandler,
    TokenProcess,
)
from crewai.memory.contextual.contextual_memory import ContextualMemory
from crewai.agents import CacheHandler
from crewai.agents.third_party_agents.llama_index.tools.task_tools import (
    LlamaAgentTools,
)

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager


class LlamaIndexReActAgent(BaseAgent):
    llm: OpenAI = Field(
        default_factory=lambda: OpenAI(model="gpt-3.5-turbo"),
        description="Language model that will run the agent.",
    )

    token_process: TokenProcess = TokenProcess()

    _token_counter: ExtendedTokenCountingHandler

    def __init__(__pydantic_self__, **data):
        config = data.pop("config", {})
        super().__init__(**config, **data)

    @model_validator(mode="after")
    def set_agent_executor(self) -> "LlamaIndexReActAgent":
        """set agent executor is set."""
        if hasattr(self.llm, "model"):
            self._token_counter = ExtendedTokenCountingHandler(
                tokenizer=tiktoken.encoding_for_model(self.llm.model).encode,
                token_process=self.token_process,
            )

        if not self.agent_executor:
            if not self.cache_handler:
                self.cache_handler = CacheHandler()
            self.set_cache_handler(self.cache_handler)
        return self

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
        result = self.agent_executor.chat(task_prompt)
        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()

        return result

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
        self.agent_executor = ReActAgent.from_llm(
            tools=tools,
            llm=self.llm,
            verbose=self.verbose,
            callback_manager=CallbackManager([self._token_counter]),
            max_iterations=self.max_iter,
        )

    def get_delegation_tools(self, agents: List[BaseAgent]):
        agent_tools = LlamaAgentTools(agents=agents)
        tools = agent_tools.tools()
        return tools
