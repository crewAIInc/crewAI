from typing import Any, List, Optional, Type, Union, cast

from pydantic import Field, field_validator

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.task import Task
from crewai.tools import BaseTool
from crewai.tools.base_tool import Tool
from crewai.utilities.converter import Converter, generate_model_description
from crewai.utilities.token_counter_callback import (
    LangChainTokenCounter,
    LiteLLMTokenCounter,
)


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
    token_process: TokenProcess = Field(default_factory=TokenProcess, exclude=True)
    token_callback: Optional[Any] = None

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
        This checks if the message is a dict with a "content" key, or has a "content" attribute,
        or if it's a tuple from LangGraph's message format.
        """
        # Handle LangGraph message tuple format (role, content)
        if isinstance(message, tuple) and len(message) == 2:
            return str(message[1])

        # Handle dictionary with content key
        elif isinstance(message, dict):
            if "content" in message:
                return message["content"]
            # Handle LangGraph message format with additional metadata
            elif "messages" in message and message["messages"]:
                last_message = message["messages"][-1]
                if isinstance(last_message, tuple) and len(last_message) == 2:
                    return str(last_message[1])
                return self._extract_text(last_message)

        # Handle object with content attribute
        elif hasattr(message, "content") and isinstance(
            getattr(message, "content"), str
        ):
            return getattr(message, "content")

        # Handle string directly
        elif isinstance(message, str):
            return message

        # Default fallback
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

        # Initialize token tracking callback if needed
        if hasattr(self, "token_process") and self.token_callback is None:
            # Determine if we're using LangChain or LiteLLM based on the agent type
            if hasattr(self.langchain_agent, "client") and hasattr(
                self.langchain_agent.client, "callbacks"
            ):
                # This is likely a LiteLLM-based agent
                self.token_callback = LiteLLMTokenCounter(self.token_process)

                # Add our callback to the LLM directly
                if isinstance(self.langchain_agent.client.callbacks, list):
                    self.langchain_agent.client.callbacks.append(self.token_callback)
                else:
                    self.langchain_agent.client.callbacks = [self.token_callback]
            else:
                # This is likely a LangChain-based agent
                self.token_callback = LangChainTokenCounter(self.token_process)

                # Add callback to the LangChain model
                if hasattr(self.langchain_agent, "callbacks"):
                    if self.langchain_agent.callbacks is None:
                        self.langchain_agent.callbacks = [self.token_callback]
                    elif isinstance(self.langchain_agent.callbacks, list):
                        self.langchain_agent.callbacks.append(self.token_callback)
                # For direct LLM models
                elif hasattr(self.langchain_agent, "llm") and hasattr(
                    self.langchain_agent.llm, "callbacks"
                ):
                    if self.langchain_agent.llm.callbacks is None:
                        self.langchain_agent.llm.callbacks = [self.token_callback]
                    elif isinstance(self.langchain_agent.llm.callbacks, list):
                        self.langchain_agent.llm.callbacks.append(self.token_callback)
                # Direct LLM case
                elif not hasattr(self.langchain_agent, "agent"):
                    # This might be a direct LLM, not an agent
                    if (
                        not hasattr(self.langchain_agent, "callbacks")
                        or self.langchain_agent.callbacks is None
                    ):
                        self.langchain_agent.callbacks = [self.token_callback]
                    elif isinstance(self.langchain_agent.callbacks, list):
                        self.langchain_agent.callbacks.append(self.token_callback)

        init_state = {"messages": [("user", task_prompt)]}

        # Estimate input tokens for tracking
        if hasattr(self, "token_process"):
            # Rough estimate based on characters (better than word count)
            estimated_prompt_tokens = len(task_prompt) // 4  # ~4 chars per token
            self.token_process.sum_prompt_tokens(estimated_prompt_tokens)

        state = self.agent_executor.invoke(init_state)

        # Extract output from state based on its structure
        if "structured_response" in state:
            current_output = state["structured_response"]
        elif "messages" in state and state["messages"]:
            last_message = state["messages"][-1]
            current_output = self._extract_text(last_message)
        elif "output" in state:
            current_output = str(state["output"])
        else:
            # Fallback to extracting text from the entire state
            current_output = self._extract_text(state)

        # Estimate completion tokens for tracking if we don't have actual counts
        if hasattr(self, "token_process"):
            # Rough estimate based on characters
            estimated_completion_tokens = len(current_output) // 4  # ~4 chars per token
            self.token_process.sum_completion_tokens(estimated_completion_tokens)
            self.token_process.sum_successful_requests(1)

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

            # Estimate input tokens for tracking
            if hasattr(self, "token_process"):
                # Rough estimate based on characters
                estimated_prompt_tokens = len(new_prompt) // 4  # ~4 chars per token
                self.token_process.sum_prompt_tokens(estimated_prompt_tokens)

            try:
                new_state = self.agent_executor.invoke(
                    {"messages": [("user", new_prompt)]}
                )
                # Extract output from state based on its structure
                if "structured_response" in new_state:
                    new_output = new_state["structured_response"]
                elif "messages" in new_state and new_state["messages"]:
                    last_message = new_state["messages"][-1]
                    new_output = self._extract_text(last_message)
                elif "output" in new_state:
                    new_output = str(new_state["output"])
                else:
                    # Fallback to extracting text from the entire state
                    new_output = self._extract_text(new_state)

                # Estimate completion tokens for tracking
                if hasattr(self, "token_process"):
                    # Rough estimate based on characters
                    estimated_completion_tokens = (
                        len(new_output) // 4
                    )  # ~4 chars per token
                    self.token_process.sum_completion_tokens(
                        estimated_completion_tokens
                    )
                    self.token_process.sum_successful_requests(1)

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

        # Ensure raw_tools is always a list.
        raw_tools: List[Any] = (
            tools
            if tools is not None
            else (self.tools if self.tools is not None else [])
        )
        # Fallback: if raw_tools is still empty, try to extract them from the wrapped LangChain agent.
        if not raw_tools:
            if hasattr(self.langchain_agent, "agent") and hasattr(
                self.langchain_agent.agent, "tools"
            ):
                raw_tools = self.langchain_agent.agent.tools or []
            else:
                raw_tools = getattr(self.langchain_agent, "tools", []) or []

        used_tools = []
        # Use the global CrewAI Tool class (imported at the module level)
        for tool in raw_tools:
            # If the tool is a CrewAI Tool, convert it to a LangChain compatible tool.
            if isinstance(tool, Tool):
                used_tools.append(tool.to_langchain())
            else:
                used_tools.append(tool)

        # Sanitize the agent's role for the "name" field. The allowed pattern is ^[a-zA-Z0-9_-]+$
        import re

        agent_role = getattr(self, "role", "agent")
        sanitized_role = re.sub(r"\s+", "_", agent_role)

        # Initialize token tracking callback if needed
        if hasattr(self, "token_process") and self.token_callback is None:
            # Determine if we're using LangChain or LiteLLM based on the agent type
            if hasattr(self.langchain_agent, "client") and hasattr(
                self.langchain_agent.client, "callbacks"
            ):
                # This is likely a LiteLLM-based agent
                self.token_callback = LiteLLMTokenCounter(self.token_process)

                # Add our callback to the LLM directly
                if isinstance(self.langchain_agent.client.callbacks, list):
                    if self.token_callback not in self.langchain_agent.client.callbacks:
                        self.langchain_agent.client.callbacks.append(
                            self.token_callback
                        )
                else:
                    self.langchain_agent.client.callbacks = [self.token_callback]
            else:
                # This is likely a LangChain-based agent
                self.token_callback = LangChainTokenCounter(self.token_process)

                # Add callback to the LangChain model
                if hasattr(self.langchain_agent, "callbacks"):
                    if self.langchain_agent.callbacks is None:
                        self.langchain_agent.callbacks = [self.token_callback]
                    elif isinstance(self.langchain_agent.callbacks, list):
                        self.langchain_agent.callbacks.append(self.token_callback)
                # For direct LLM models
                elif hasattr(self.langchain_agent, "llm") and hasattr(
                    self.langchain_agent.llm, "callbacks"
                ):
                    if self.langchain_agent.llm.callbacks is None:
                        self.langchain_agent.llm.callbacks = [self.token_callback]
                    elif isinstance(self.langchain_agent.llm.callbacks, list):
                        self.langchain_agent.llm.callbacks.append(self.token_callback)
                # Direct LLM case
                elif not hasattr(self.langchain_agent, "agent"):
                    # This might be a direct LLM, not an agent
                    if (
                        not hasattr(self.langchain_agent, "callbacks")
                        or self.langchain_agent.callbacks is None
                    ):
                        self.langchain_agent.callbacks = [self.token_callback]
                    elif isinstance(self.langchain_agent.callbacks, list):
                        self.langchain_agent.callbacks.append(self.token_callback)

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
