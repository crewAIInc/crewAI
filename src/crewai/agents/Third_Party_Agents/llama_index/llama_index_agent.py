from llama_index.core.tools import FunctionTool
from typing import List

from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI


from crewai.agents.BaseAgent import BaseAgent
from crewai import Crew, Task
from typing import Any
from pydantic import (
    Field,
)
from crewai.memory.contextual.contextual_memory import ContextualMemory


# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

# picks out the best agent to do the job
# agent = AgentRunner.from_llm([multiply_tool], llm=llm, verbose=True)
# TokenCountingHandler(
#         tokenizer=tiktoken.encoding_for_model(model_name).encode
#     )

# test usage


class LlamaIndexAgent(BaseAgent):
    llm: Any = Field(
        default_factory=lambda: OpenAI(model="gpt-3.5-turbo-0613"),
        description="Language model that will run the agent.",
    )
    # llm = OpenAI(model="gpt-3.5-turbo-0613")
    # agent = OpenAIAgent.from_tools(
    #     [multiply_tool], llm=llm, verbose=True
    # )
    # agent: InstanceOf[AgentRunner] = Field(
    #     default=None, description='llama An instance of the AgentRunner class'
    # )

    def __init__(
        self,
        tools: List[FunctionTool] = [],
        llm: OpenAI = OpenAI(temperature=0, model="gpt-4o"),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.agent = agent
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
        # print('tools to use', tools)
        parsed_tools = self._parse_tools(tools)
        print("parsed_tools", parsed_tools)
        # print('tools returned', parsed_tools)
        self.create_agent_executor(tools=tools)

        return str(self.agent_executor.chat(task_prompt))
        # result = self.agent_executor.from_tools([multiply_tool],llm=self.llm, verbose=self.verbose)
        # return result

    def _parse_tools(self, tools: List[Any]) -> List[FunctionTool]:
        """Parse tools to be used for the task."""
        print("passed in to parse tools:", tools)
        # tentatively try to import from crewai_tools import BaseTool as CrewAITool
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
        print("parsed_tools", tools_list)
        return tools_list

    def create_agent_executor(self, tools=None):
        # Implement the method as required

        # self.agent_executor = AgentRunner(callback_manager=CallbackManager([TokenCountingHandler(
        #     tokenizer=tiktoken.encoding_for_model(self.llm.model).encode
        # )]))
        print("tools to be used for llma agent", tools)
        self.agent_executor = OpenAIAgent.from_tools(
            tools, llm=self.llm, verbose=self.verbose
        )

    def set_cache_handler(self):
        # Implement the method as required
        pass

    def set_rpm_controller(self):
        # Implement the method as required
        pass

    def set_agent_executor(self):
        pass


agent = LlamaIndexAgent(
    role="backstory agent",
    goal="who is joe biden?",
    backstory="agent backstory",
    verbose=True,
    tools=[multiply_tool],
)

task1 = Task(
    expected_output="What is 2123 * 21512?",
    description="What is 2123 * 21512?",
    agent=agent,
)
agent2 = LlamaIndexAgent(
    role="backstory agent",
    goal="who is joe biden?",
    backstory="agent backstory",
    verbose=True,
)

task2 = Task(
    expected_output="a short biography of joe biden",
    description="a short biography of joe biden",
    agent=agent,
)

my_crew = Crew(agents=[agent, agent2], tasks=[task1, task2], verbose=True)

crew = my_crew.kickoff()


# result = agent.execute_task(task1)
# result2 = agent2.execute_task(task2)
# print('result',crew)

# print('result', result)
# print('result2',result2)
