from typing import Any, List, Optional, Type, cast

from pydantic import Field

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.task import Task
from crewai.tools import BaseTool
from crewai.utilities.converter import Converter, generate_model_description


class LangChainAgentAdapter(BaseAgent):
    """
    Adapter class to wrap a LangChain agent and make it compatible with CrewAI's BaseAgent interface.

    Note:
        - This adapter does not require LangChain as a dependency.
        - It wraps an external LangChain agent (passed as any type) and delegates calls
          such as execute_task() to the LangChain agent's invoke() method.
        - Extended logic is added to build prompts, incorporate memory, knowledge, training hints,
          and now a human feedback loop similar to what is done in CrewAgentExecutor.
    """

    langchain_agent: Any = Field(
        ...,
        description="The wrapped LangChain runnable agent instance. It is expected to have an 'invoke' method.",
    )
    function_calling_llm: Optional[Any] = Field(
        default=None, description="Optional function calling LLM."
    )
    step_callback: Optional[Any] = Field(
        default=None,
        description="Callback executed after each step of agent execution.",
    )
    allow_code_execution: Optional[bool] = Field(
        default=False, description="Enable code execution for the agent."
    )
    multimodal: bool = Field(
        default=False, description="Whether the agent is multimodal."
    )
    i18n: Any = None
    crew: Any = None
    knowledge: Any = None
    tools: Optional[List[BaseTool]] = None

    class Config:
        arbitrary_types_allowed = True

    def execute_task(
        self,
        task: Task,
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> str:
        """
        Execute a task by building the full task prompt (with memory, knowledge, tool instructions,
        and training hints) then delegating execution to the wrapped LangChain agent.
        If the task requires human input, a feedback loop is run that mimics the CrewAgentExecutor.
        """
        task_prompt = task.prompt()

        if task.output_json or task.output_pydantic:
            # Choose the output format, preferring output_json if available
            output_format = (
                task.output_json if task.output_json else task.output_pydantic
            )
            schema = generate_model_description(cast(type, output_format))
            instruction = self.i18n.slice("formatted_task_instructions").format(
                output_format=schema
            )
            task_prompt += f"\n{instruction}"

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        if self.crew and self.crew.memory:
            from crewai.memory.contextual.contextual_memory import ContextualMemory

            contextual_memory = ContextualMemory(
                self.crew.memory_config,
                self.crew._short_term_memory,
                self.crew._long_term_memory,
                self.crew._entity_memory,
                self.crew._user_memory,
            )
            memory = contextual_memory.build_context_for_task(task, context)
            if memory.strip():
                task_prompt += self.i18n.slice("memory").format(memory=memory)

        if self.knowledge:
            agent_knowledge_snippets = self.knowledge.query([task.prompt()])
            if agent_knowledge_snippets:
                from crewai.knowledge.utils.knowledge_utils import (
                    extract_knowledge_context,
                )

                agent_knowledge_context = extract_knowledge_context(
                    agent_knowledge_snippets
                )
                if agent_knowledge_context:
                    task_prompt += agent_knowledge_context

        if self.crew:
            knowledge_snippets = self.crew.query_knowledge([task.prompt()])
            if knowledge_snippets:
                from crewai.knowledge.utils.knowledge_utils import (
                    extract_knowledge_context,
                )

                crew_knowledge_context = extract_knowledge_context(knowledge_snippets)
                if crew_knowledge_context:
                    task_prompt += crew_knowledge_context

        tools = tools or self.tools or []
        self.create_agent_executor(tools=tools)

        if self.crew and getattr(self.crew, "_train", False):
            task_prompt = self._training_handler(task_prompt=task_prompt)
        else:
            task_prompt = self._use_trained_data(task_prompt=task_prompt)

        try:
            # Initial invocation of the LangChain agent
            result = self.agent_executor.invoke(
                {
                    "input": task_prompt,
                    "tool_names": getattr(self.agent_executor, "tools_names", ""),
                    "tools": getattr(self.agent_executor, "tools_description", ""),
                    "ask_for_human_input": task.human_input,
                }
            )["output"]

            # If human feedback is required, enter a feedback loop
            if task.human_input:
                result = self._handle_human_feedback(result)
        except Exception as e:
            # Example: you could add retry logic here if desired.
            raise e

        return result

    def _handle_human_feedback(self, current_output: str) -> str:
        """
        Implements a feedback loop that prompts the user for feedback and then instructs
        the underlying LangChain agent to regenerate its answer with the requested changes.
        """
        while True:
            print("\nAgent output:")
            print(current_output)
            # Prompt the user for feedback
            feedback = input("\nEnter your feedback (or press Enter to accept): ")
            if not feedback.strip():
                break  # No feedback provided, exit the loop

            # Construct a new prompt with explicit instructions
            new_prompt = (
                f"Below is your previous answer:\n{current_output}\n\n"
                f"Based on the following feedback: '{feedback}', please regenerate your answer with the requested details. "
                f"Specifically, display 10 bullet points in each section. Provide the complete updated answer below.\n\nUpdated answer:"
            )
            try:
                invocation = self.agent_executor.invoke(
                    {
                        "input": new_prompt,
                        "tool_names": getattr(self.agent_executor, "tools_names", ""),
                        "tools": getattr(self.agent_executor, "tools_description", ""),
                        "ask_for_human_input": True,
                    }
                )
                current_output = invocation["output"]
            except Exception as e:
                print("Error during re-invocation with feedback:", e)
                break

        return current_output

    def _generate_model_description(self, model: Any) -> str:
        """
        Generates a string description (schema) for the expected output.
        This is a placeholder that should call the actual implementation.
        """
        from crewai.utilities.converter import generate_model_description

        return generate_model_description(model)

    def _training_handler(self, task_prompt: str) -> str:
        """
        Append training instructions from Crew data to the task prompt.
        """
        from crewai.utilities.constants import TRAINING_DATA_FILE
        from crewai.utilities.training_handler import CrewTrainingHandler

        data = CrewTrainingHandler(TRAINING_DATA_FILE).load()
        if data:
            agent_id = str(self.id)
            if data.get(agent_id):
                human_feedbacks = [
                    i["human_feedback"] for i in data.get(agent_id, {}).values()
                ]
                task_prompt += (
                    "\n\nYou MUST follow these instructions: \n "
                    + "\n - ".join(human_feedbacks)
                )
        return task_prompt

    def _use_trained_data(self, task_prompt: str) -> str:
        """
        Append pre-trained instructions from Crew data to the task prompt.
        """
        from crewai.utilities.constants import TRAINED_AGENTS_DATA_FILE
        from crewai.utilities.training_handler import CrewTrainingHandler

        data = CrewTrainingHandler(TRAINED_AGENTS_DATA_FILE).load()
        if data and (trained_data_output := data.get(getattr(self, "role", "default"))):
            task_prompt += (
                "\n\nYou MUST follow these instructions: \n - "
                + "\n - ".join(trained_data_output["suggestions"])
            )
        return task_prompt

    def create_agent_executor(self, tools: Optional[List[BaseTool]] = None) -> None:
        """
        Creates an agent executor using LangChain's AgentExecutor.
        """
        from importlib import import_module

        langchain_agents = import_module("langchain.agents")
        AgentExecutor = getattr(langchain_agents, "AgentExecutor")
        used_tools = tools or self.tools or []

        print(f"Creating agent executor for langchain agent: {self.langchain_agent}")
        print("Passing tools: ", used_tools)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.langchain_agent,
            tools=used_tools,
            verbose=getattr(self, "verbose", True),
        )
        print("Created agent executor for langchain agent")

    def _parse_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        return tools

    def get_delegation_tools(self, agents: List["BaseAgent"]) -> List[BaseTool]:
        return []

    def get_output_converter(
        self,
        llm: Any,
        text: str,
        model: Optional[Type] = None,
        instructions: str = "",
    ) -> Converter:
        return Converter(llm=llm, text=text, model=model, instructions=instructions)
