from typing import Any, List, Optional, Type, Union, cast

from crewai.tools.base_tool import Tool

try:
    from langchain_core.tools import Tool as LangChainTool  # type: ignore
except ImportError:
    LangChainTool = None

from pydantic import Field, field_validator

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
    tools: Optional[List[Union[BaseTool, Any]]] = Field(
        default_factory=list,
        description="Tools at the agent's disposal. Accepts both CrewAI BaseTool instances and other tools.",
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

    class Config:
        arbitrary_types_allowed = True

    @field_validator("tools", mode="before")
    def convert_tools(cls, value):
        """Ensure tools are valid CrewAI BaseTool instances."""
        if not value:
            return value
        new_tools = []
        for tool in value:
            # If tool is already a CrewAI BaseTool instance, keep it as is.
            if isinstance(tool, BaseTool):
                new_tools.append(tool)
            else:
                new_tools.append(Tool.from_langchain(tool))
        return new_tools

    def _extract_text(self, message: Any) -> str:
        """
        Helper to extract plain text from a message object.
        This checks if the message is a dict with a "content" key, or has a "content" attribute.
        """
        if isinstance(message, dict) and "content" in message:
            return message["content"]
        elif hasattr(message, "content") and isinstance(
            getattr(message, "content"), str
        ):
            return getattr(message, "content")
        elif isinstance(message, str):
            return message
        return str(message)

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

        self._show_start_logs(task)

        if self.crew and getattr(self.crew, "_train", False):
            task_prompt = self._training_handler(task_prompt=task_prompt)
        else:
            task_prompt = self._use_trained_data(task_prompt=task_prompt)

        init_state = {"messages": [("user", task_prompt)]}
        state = self.agent_executor.invoke(init_state)

        if "structured_response" in state:
            current_output = state["structured_response"]
        elif "messages" in state and state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, tuple):
                current_output = last_message[1]
            else:
                current_output = self._extract_text(last_message)
        else:
            current_output = ""

        if task.human_input:
            current_output = self._handle_human_feedback(current_output)

        return current_output

    def _handle_human_feedback(self, current_output: str) -> str:
        """
        Implements a feedback loop that prompts the user for feedback and then instructs
        the underlying LangChain agent to regenerate its answer with the requested changes.
        Only the inner content of the output is displayed to the user.
        """
        while True:
            print("\nAgent output:")
            # Print only the inner text extracted from current_output.
            print(self._extract_text(current_output))

            feedback = input("\nEnter your feedback (or press Enter to accept): ")
            if not feedback.strip():
                break  # No feedback provided, exit the loop

            extracted_output = self._extract_text(current_output)
            new_prompt = (
                f"Below is your previous answer:\n"
                f"{extracted_output}\n\n"
                f"Based on the following feedback: '{feedback}', please regenerate your answer with the requested details. "
                f"Specifically, display 10 bullet points in each section. Provide the complete updated answer below.\n\n"
                f"Updated answer:"
            )
            try:
                new_state = self.agent_executor.invoke(
                    {"messages": [("user", new_prompt)]}
                )
                if "structured_response" in new_state:
                    new_output = new_state["structured_response"]
                elif "messages" in new_state and new_state["messages"]:
                    last_message = new_state["messages"][-1]
                    if isinstance(last_message, tuple):
                        new_output = last_message[1]
                    else:
                        new_output = self._extract_text(last_message)
                else:
                    new_output = ""
                current_output = new_output
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
        Creates an agent executor using LangGraph's create_react_agent if given an LLM,
        or uses the provided language model directly.
        """
        try:
            from langgraph.prebuilt import create_react_agent
        except ImportError as e:
            raise ImportError(
                "LangGraph library not found. Please run `uv add langgraph` to add LangGraph support."
            ) from e

        # Ensure raw_tools is always a list, even if tools and self.tools are None.
        raw_tools = tools or self.tools or []
        # Fallback: if raw_tools is still empty, try to extract them from the wrapped langchain agent.
        if not raw_tools:
            if hasattr(self.langchain_agent, "agent") and hasattr(
                self.langchain_agent.agent, "tools"
            ):
                raw_tools = self.langchain_agent.agent.tools or []
            else:
                raw_tools = getattr(self.langchain_agent, "tools", []) or []

        used_tools = []
        try:
            # Import the CrewAI Tool class.
            from crewai.tools.base_tool import Tool as CrewTool
        except ImportError:
            CrewTool: Optional[Type[BaseTool]] = None  # Explicitly annotate as Optional

        for tool in raw_tools:
            # If the tool is a CrewAI Tool, convert it to a LangChain compatible tool.
            if CrewTool is not None and isinstance(tool, CrewTool):
                used_tools.append(tool.to_langchain())
            else:
                used_tools.append(tool)

        # Sanitize the agent's role for the "name" field. The allowed pattern is ^[a-zA-Z0-9_-]+$
        import re

        agent_role = getattr(self, "role", "agent")
        sanitized_role = re.sub(r"\s+", "_", agent_role)

        self.agent_executor = create_react_agent(
            model=self.langchain_agent,
            tools=used_tools,
            debug=getattr(self, "verbose", False),
            name=sanitized_role,
        )

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

    def _show_start_logs(self, task: Task) -> None:
        if self.langchain_agent is None:
            raise ValueError("Agent cannot be None")
        # Check if the adapter or its crew is in verbose mode.
        verbose = self.verbose or (self.crew and getattr(self.crew, "verbose", False))
        if verbose:
            from crewai.utilities import Printer

            printer = Printer()
            # Use the adapter's role (inherited from BaseAgent) for logging.
            printer.print(
                content=f"\033[1m\033[95m# Agent:\033[00m \033[1m\033[92m{self.role}\033[00m"
            )
            description = getattr(task, "description", "Not Found")
            printer.print(
                content=f"\033[95m## Task:\033[00m \033[92m{description}\033[00m"
            )
