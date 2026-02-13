from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict

from pydantic import BaseModel
from rich.console import Console

from crewai.agents.constants import FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    OutputParserError,
    parse,
)
from crewai.cli.config import Settings
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool as CrewAITool
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_types import ToolResult
from crewai.utilities.errors import AgentRepositoryError
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.i18n import I18N
from crewai.utilities.printer import ColoredText, Printer
from crewai.utilities.pydantic_schema_utils import generate_model_description
from crewai.utilities.string_utils import sanitize_tool_name
from crewai.utilities.token_counter_callback import TokenCalcHandler
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.agents.crew_agent_executor import CrewAgentExecutor
    from crewai.agents.tools_handler import ToolsHandler
    from crewai.experimental.agent_executor import AgentExecutor
    from crewai.lite_agent import LiteAgent
    from crewai.llm import LLM
    from crewai.task import Task

_create_plus_client_hook: Callable[[], Any] | None = None


class SummaryContent(TypedDict):
    """Structure for summary content entries.

    Attributes:
        content: The summarized content.
    """

    content: str


console = Console()

_MULTIPLE_NEWLINES: Final[re.Pattern[str]] = re.compile(r"\n+")


def is_inside_event_loop() -> bool:
    """Check if code is currently running inside an asyncio event loop.

    This is used to detect when code is being called from within an async context
    (e.g., inside a Flow). In such cases, callers should return a coroutine
    instead of executing synchronously to avoid nested event loop errors.

    Returns:
        True if inside a running event loop, False otherwise.
    """
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def parse_tools(tools: list[BaseTool]) -> list[CrewStructuredTool]:
    """Parse tools to be used for the task.

    Args:
        tools: List of tools to parse.

    Returns:
        List of structured tools.

    Raises:
        ValueError: If a tool is not a CrewStructuredTool or BaseTool.
    """
    tools_list: list[CrewStructuredTool] = []

    for tool in tools:
        if isinstance(tool, CrewAITool):
            structured_tool = tool.to_structured_tool()
            structured_tool.current_usage_count = 0
            if structured_tool._original_tool:
                structured_tool._original_tool.current_usage_count = 0
            tools_list.append(structured_tool)
        else:
            raise ValueError("Tool is not a CrewStructuredTool or BaseTool")

    return tools_list


def get_tool_names(tools: Sequence[CrewStructuredTool | BaseTool]) -> str:
    """Get the sanitized names of the tools.

    Args:
        tools: List of tools to get names from.

    Returns:
        Comma-separated string of sanitized tool names.
    """
    return ", ".join([sanitize_tool_name(t.name) for t in tools])


def render_text_description_and_args(
    tools: Sequence[CrewStructuredTool | BaseTool],
) -> str:
    """Render the tool name, description, and args in plain text.

    search: This tool is used for search, args: {"query": {"type": "string"}}
    calculator: This tool is used for math, \
    args: {"expression": {"type": "string"}}

    Args:
        tools: List of tools to render.

    Returns:
        Plain text description of tools.
    """
    tool_strings = [tool.description for tool in tools]
    return "\n".join(tool_strings)


def convert_tools_to_openai_schema(
    tools: Sequence[BaseTool | CrewStructuredTool],
) -> tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]:
    """Convert CrewAI tools to OpenAI function calling format.

    This function converts CrewAI BaseTool and CrewStructuredTool objects
    into the OpenAI-compatible tool schema format that can be passed to
    LLM providers for native function calling.

    Args:
        tools: List of CrewAI tool objects to convert.

    Returns:
        Tuple containing:
        - List of OpenAI-format tool schema dictionaries
        - Dict mapping tool names to their callable run() methods

    Example:
        >>> tools = [CalculatorTool(), SearchTool()]
        >>> schemas, functions = convert_tools_to_openai_schema(tools)
        >>> # schemas can be passed to llm.call(tools=schemas)
        >>> # functions can be passed to llm.call(available_functions=functions)
    """
    openai_tools: list[dict[str, Any]] = []
    available_functions: dict[str, Callable[..., Any]] = {}

    for tool in tools:
        # Get the JSON schema for tool parameters
        parameters: dict[str, Any] = {}
        if hasattr(tool, "args_schema") and tool.args_schema is not None:
            try:
                schema_output = generate_model_description(tool.args_schema)
                parameters = schema_output.get("json_schema", {}).get("schema", {})
                # Remove title and description from schema root as they're redundant
                parameters.pop("title", None)
                parameters.pop("description", None)
            except Exception:
                parameters = {}

        # Extract original description from formatted description
        # BaseTool formats description as "Tool Name: ...\nTool Arguments: ...\nTool Description: {original}"
        description = tool.description
        if "Tool Description:" in description:
            description = description.split("Tool Description:")[-1].strip()

        sanitized_name = sanitize_tool_name(tool.name)

        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": sanitized_name,
                "description": description,
                "parameters": parameters,
                "strict": True,
            },
        }
        openai_tools.append(schema)
        available_functions[sanitized_name] = tool.run  # type: ignore[union-attr]

    return openai_tools, available_functions


def has_reached_max_iterations(iterations: int, max_iterations: int) -> bool:
    """Check if the maximum number of iterations has been reached.

    Args:
        iterations: Current number of iterations.
        max_iterations: Maximum allowed iterations.

    Returns:
        True if maximum iterations reached, False otherwise.
    """
    return iterations >= max_iterations


def handle_max_iterations_exceeded(
    formatted_answer: AgentAction | AgentFinish | None,
    printer: Printer,
    i18n: I18N,
    messages: list[LLMMessage],
    llm: LLM | BaseLLM,
    callbacks: list[TokenCalcHandler],
    verbose: bool = True,
) -> AgentFinish:
    """Handles the case when the maximum number of iterations is exceeded. Performs one more LLM call to get the final answer.

    Args:
        formatted_answer: The last formatted answer from the agent.
        printer: Printer instance for output.
        i18n: I18N instance for internationalization.
        messages: List of messages to send to the LLM.
        llm: The LLM instance to call.
        callbacks: List of callbacks for the LLM call.
        verbose: Whether to print output.

    Returns:
        AgentFinish with the final answer after exceeding max iterations.
    """
    if verbose:
        printer.print(
            content="Maximum iterations reached. Requesting final answer.",
            color="yellow",
        )

    if formatted_answer and hasattr(formatted_answer, "text"):
        assistant_message = (
            formatted_answer.text + f"\n{i18n.errors('force_final_answer')}"
        )
    else:
        assistant_message = i18n.errors("force_final_answer")

    messages.append(format_message_for_llm(assistant_message, role="assistant"))

    # Perform one more LLM call to get the final answer
    answer = llm.call(
        messages,
        callbacks=callbacks,
    )

    if answer is None or answer == "":
        if verbose:
            printer.print(
                content="Received None or empty response from LLM call.",
                color="red",
            )
        raise ValueError("Invalid response from LLM call - None or empty.")

    formatted = format_answer(answer=answer)

    # If format_answer returned an AgentAction, convert it to AgentFinish
    if isinstance(formatted, AgentFinish):
        return formatted
    return AgentFinish(
        thought=formatted.thought,
        output=formatted.text,
        text=formatted.text,
    )


def format_message_for_llm(
    prompt: str, role: Literal["user", "assistant", "system"] = "user"
) -> LLMMessage:
    """Format a message for the LLM.

    Args:
        prompt:  The message content.
        role:  The role of the message sender, either 'user' or 'assistant'.

    Returns:
        A dictionary with 'role' and 'content' keys.

    """
    prompt = prompt.rstrip()
    return {"role": role, "content": prompt}


def format_answer(answer: str) -> AgentAction | AgentFinish:
    """Format a response from the LLM into an AgentAction or AgentFinish.

    Args:
        answer: The raw response from the LLM

    Returns:
        Either an AgentAction or AgentFinish
    """
    try:
        return parse(answer)
    except Exception:
        return AgentFinish(
            thought="Failed to parse LLM response",
            output=answer,
            text=answer,
        )


def enforce_rpm_limit(
    request_within_rpm_limit: Callable[[], bool] | None = None,
) -> None:
    """Enforce the requests per minute (RPM) limit if applicable.

    Args:
        request_within_rpm_limit: Function to enforce RPM limit.
    """
    if request_within_rpm_limit:
        request_within_rpm_limit()


def _prepare_llm_call(
    executor_context: CrewAgentExecutor | AgentExecutor | LiteAgent | None,
    messages: list[LLMMessage],
    printer: Printer,
    verbose: bool = True,
) -> list[LLMMessage]:
    """Shared pre-call logic: run before hooks and resolve messages.

    Args:
        executor_context: Optional executor context for hook invocation.
        messages: The messages to send to the LLM.
        printer: Printer instance for output.
        verbose: Whether to print output.

    Returns:
        The resolved messages list (may come from executor_context).

    Raises:
        ValueError: If a before hook blocks the call.
    """
    if executor_context is not None:
        if not _setup_before_llm_call_hooks(executor_context, printer, verbose=verbose):
            raise ValueError("LLM call blocked by before_llm_call hook")
        messages = executor_context.messages
    return messages


def _validate_and_finalize_llm_response(
    answer: Any,
    executor_context: CrewAgentExecutor | AgentExecutor | LiteAgent | None,
    printer: Printer,
    verbose: bool = True,
) -> str | BaseModel | Any:
    """Shared post-call logic: validate response and run after hooks.

    Args:
        answer: The raw LLM response.
        executor_context: Optional executor context for hook invocation.
        printer: Printer instance for output.
        verbose: Whether to print output.

    Returns:
        The potentially modified response.

    Raises:
        ValueError: If the response is None or empty.
    """
    if not answer:
        if verbose:
            printer.print(
                content="Received None or empty response from LLM call.",
                color="red",
            )
        raise ValueError("Invalid response from LLM call - None or empty.")

    return _setup_after_llm_call_hooks(
        executor_context, answer, printer, verbose=verbose
    )


def get_llm_response(
    llm: LLM | BaseLLM,
    messages: list[LLMMessage],
    callbacks: list[TokenCalcHandler],
    printer: Printer,
    tools: list[dict[str, Any]] | None = None,
    available_functions: dict[str, Callable[..., Any]] | None = None,
    from_task: Task | None = None,
    from_agent: Agent | LiteAgent | None = None,
    response_model: type[BaseModel] | None = None,
    executor_context: CrewAgentExecutor | AgentExecutor | LiteAgent | None = None,
    verbose: bool = True,
) -> str | BaseModel | Any:
    """Call the LLM and return the response, handling any invalid responses.

    Args:
        llm: The LLM instance to call.
        messages: The messages to send to the LLM.
        callbacks: List of callbacks for the LLM call.
        printer: Printer instance for output.
        tools: Optional list of tool schemas for native function calling.
        available_functions: Optional dict mapping function names to callables.
        from_task: Optional task context for the LLM call.
        from_agent: Optional agent context for the LLM call.
        response_model: Optional Pydantic model for structured outputs.
        executor_context: Optional executor context for hook invocation.
        verbose: Whether to print output.

    Returns:
        The response from the LLM as a string, Pydantic model (when response_model is provided),
        or tool call results if native function calling is used.

    Raises:
        Exception: If an error occurs.
        ValueError: If the response is None or empty.
    """
    messages = _prepare_llm_call(executor_context, messages, printer, verbose=verbose)

    try:
        answer = llm.call(
            messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,  # type: ignore[arg-type]
            response_model=response_model,
        )
    except Exception as e:
        raise e

    return _validate_and_finalize_llm_response(
        answer, executor_context, printer, verbose=verbose
    )


async def aget_llm_response(
    llm: LLM | BaseLLM,
    messages: list[LLMMessage],
    callbacks: list[TokenCalcHandler],
    printer: Printer,
    tools: list[dict[str, Any]] | None = None,
    available_functions: dict[str, Callable[..., Any]] | None = None,
    from_task: Task | None = None,
    from_agent: Agent | LiteAgent | None = None,
    response_model: type[BaseModel] | None = None,
    executor_context: CrewAgentExecutor | AgentExecutor | None = None,
    verbose: bool = True,
) -> str | BaseModel | Any:
    """Call the LLM asynchronously and return the response.

    Args:
        llm: The LLM instance to call.
        messages: The messages to send to the LLM.
        callbacks: List of callbacks for the LLM call.
        printer: Printer instance for output.
        tools: Optional list of tool schemas for native function calling.
        available_functions: Optional dict mapping function names to callables.
        from_task: Optional task context for the LLM call.
        from_agent: Optional agent context for the LLM call.
        response_model: Optional Pydantic model for structured outputs.
        executor_context: Optional executor context for hook invocation.
        verbose: Whether to print output.

    Returns:
        The response from the LLM as a string, Pydantic model (when response_model is provided),
        or tool call results if native function calling is used.

    Raises:
        Exception: If an error occurs.
        ValueError: If the response is None or empty.
    """
    messages = _prepare_llm_call(executor_context, messages, printer, verbose=verbose)

    try:
        answer = await llm.acall(
            messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,  # type: ignore[arg-type]
            response_model=response_model,
        )
    except Exception as e:
        raise e

    return _validate_and_finalize_llm_response(
        answer, executor_context, printer, verbose=verbose
    )


def process_llm_response(
    answer: str, use_stop_words: bool
) -> AgentAction | AgentFinish:
    """Process the LLM response and format it into an AgentAction or AgentFinish.

    Args:
        answer: The raw response from the LLM
        use_stop_words: Whether to use stop words in the LLM call

    Returns:
        Either an AgentAction or AgentFinish
    """
    if not use_stop_words:
        try:
            # Preliminary parsing to check for errors.
            format_answer(answer)
        except OutputParserError as e:
            if FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE in e.error:
                answer = answer.split("Observation:")[0].strip()

    return format_answer(answer)


def handle_agent_action_core(
    formatted_answer: AgentAction,
    tool_result: ToolResult,
    messages: list[LLMMessage] | None = None,
    step_callback: Callable | None = None,  # type: ignore[type-arg]
    show_logs: Callable | None = None,  # type: ignore[type-arg]
) -> AgentAction | AgentFinish:
    """Core logic for handling agent actions and tool results.

    Args:
        formatted_answer: The agent's action
        tool_result: The result of executing the tool
        messages: Optional list of messages to append results to
        step_callback: Optional callback to execute after processing
        show_logs: Optional function to show logs

    Returns:
        Either an AgentAction or AgentFinish

    Notes:
        - TODO: Remove messages parameter and its usage.
    """
    if step_callback:
        step_callback(tool_result)

    formatted_answer.text += f"\nObservation: {tool_result.result}"
    formatted_answer.result = tool_result.result

    if tool_result.result_as_answer:
        return AgentFinish(
            thought="",
            output=tool_result.result,
            text=formatted_answer.text,
        )

    if show_logs:
        show_logs(formatted_answer)

    return formatted_answer


def handle_unknown_error(
    printer: Printer, exception: Exception, verbose: bool = True
) -> None:
    """Handle unknown errors by informing the user.

    Args:
        printer: Printer instance for output
        exception: The exception that occurred
        verbose: Whether to print output.
    """
    if not verbose:
        return

    error_message = str(exception)

    if "litellm" in error_message:
        return

    printer.print(
        content="An unknown error occurred. Please check the details below.",
        color="red",
    )
    printer.print(
        content=f"Error details: {error_message}",
        color="red",
    )


def handle_output_parser_exception(
    e: OutputParserError,
    messages: list[LLMMessage],
    iterations: int,
    log_error_after: int = 3,
    printer: Printer | None = None,
    verbose: bool = True,
) -> AgentAction:
    """Handle OutputParserError by updating messages and formatted_answer.

    Args:
        e: The OutputParserError that occurred
        messages: List of messages to append to
        iterations: Current iteration count
        log_error_after: Number of iterations after which to log errors
        printer: Optional printer instance for logging

    Returns:
        AgentAction: A formatted answer with the error
    """
    messages.append({"role": "user", "content": e.error})

    formatted_answer = AgentAction(
        text=e.error,
        tool="",
        tool_input="",
        thought="",
    )

    if verbose and iterations > log_error_after and printer:
        printer.print(
            content=f"Error parsing LLM output, agent will retry: {e.error}",
            color="red",
        )

    return formatted_answer


def is_context_length_exceeded(exception: Exception) -> bool:
    """Check if the exception is due to context length exceeding.

    Args:
        exception: The exception to check

    Returns:
        bool: True if the exception is due to context length exceeding
    """
    return LLMContextLengthExceededError(str(exception))._is_context_limit_error(
        str(exception)
    )


def handle_context_length(
    respect_context_window: bool,
    printer: Printer,
    messages: list[LLMMessage],
    llm: LLM | BaseLLM,
    callbacks: list[TokenCalcHandler],
    i18n: I18N,
    verbose: bool = True,
) -> None:
    """Handle context length exceeded by either summarizing or raising an error.

    Args:
        respect_context_window: Whether to respect context window
        printer: Printer instance for output
        messages: List of messages to summarize
        llm: LLM instance for summarization
        callbacks: List of callbacks for LLM
        i18n: I18N instance for messages

    Raises:
        SystemExit: If context length is exceeded and user opts not to summarize
    """
    if respect_context_window:
        if verbose:
            printer.print(
                content="Context length exceeded. Summarizing content to fit the model context window. Might take a while...",
                color="yellow",
            )
        summarize_messages(
            messages=messages, llm=llm, callbacks=callbacks, i18n=i18n, verbose=verbose
        )
    else:
        if verbose:
            printer.print(
                content="Context length exceeded. Consider using smaller text or RAG tools from crewai_tools.",
                color="red",
            )
        raise SystemExit(
            "Context length exceeded and user opted not to summarize. Consider using smaller text or RAG tools from crewai_tools."
        )


def summarize_messages(
    messages: list[LLMMessage],
    llm: LLM | BaseLLM,
    callbacks: list[TokenCalcHandler],
    i18n: I18N,
    verbose: bool = True,
) -> None:
    """Summarize messages to fit within context window.

    Preserves any files attached to user messages and re-attaches them to
    the summarized message. Files from all user messages are merged.

    Args:
        messages: List of messages to summarize (modified in-place)
        llm: LLM instance for summarization
        callbacks: List of callbacks for LLM
        i18n: I18N instance for messages
    """
    preserved_files: dict[str, Any] = {}
    for msg in messages:
        if msg.get("role") == "user" and msg.get("files"):
            preserved_files.update(msg["files"])

    messages_string = " ".join(
        [str(message.get("content", "")) for message in messages]
    )
    cut_size = llm.get_context_window_size()

    messages_groups = [
        {"content": messages_string[i : i + cut_size]}
        for i in range(0, len(messages_string), cut_size)
    ]

    summarized_contents: list[SummaryContent] = []

    total_groups = len(messages_groups)
    for idx, group in enumerate(messages_groups, 1):
        if verbose:
            Printer().print(
                content=f"Summarizing {idx}/{total_groups}...",
                color="yellow",
            )

        summarization_messages = [
            format_message_for_llm(
                i18n.slice("summarizer_system_message"), role="system"
            ),
            format_message_for_llm(
                i18n.slice("summarize_instruction").format(group=group["content"]),
            ),
        ]
        summary = llm.call(
            summarization_messages,
            callbacks=callbacks,
        )
        summarized_contents.append({"content": str(summary)})

    merged_summary = " ".join(content["content"] for content in summarized_contents)

    messages.clear()
    summary_message = format_message_for_llm(
        i18n.slice("summary").format(merged_summary=merged_summary)
    )
    if preserved_files:
        summary_message["files"] = preserved_files
    messages.append(summary_message)


def show_agent_logs(
    printer: Printer,
    agent_role: str,
    formatted_answer: AgentAction | AgentFinish | None = None,
    task_description: str | None = None,
    verbose: bool = False,
) -> None:
    """Show agent logs for both start and execution states.

    Args:
        printer: Printer instance for output
        agent_role: Role of the agent
        formatted_answer: Optional AgentAction or AgentFinish for execution logs
        task_description: Optional task description for start logs
        verbose: Whether to show verbose output
    """
    if not verbose:
        return

    agent_role = agent_role.partition("\n")[0]

    if formatted_answer is None:
        # Start logs
        printer.print(
            content=[
                ColoredText("# Agent: ", "bold_purple"),
                ColoredText(agent_role, "bold_green"),
            ]
        )
        if task_description:
            printer.print(
                content=[
                    ColoredText("## Task: ", "purple"),
                    ColoredText(task_description, "green"),
                ]
            )
    else:
        # Execution logs
        printer.print(
            content=[
                ColoredText("\n\n# Agent: ", "bold_purple"),
                ColoredText(agent_role, "bold_green"),
            ]
        )

        if isinstance(formatted_answer, AgentAction):
            thought = _MULTIPLE_NEWLINES.sub("\n", formatted_answer.thought)
            formatted_json = json.dumps(
                formatted_answer.tool_input,
                indent=2,
                ensure_ascii=False,
            )
            if thought and thought != "":
                printer.print(
                    content=[
                        ColoredText("## Thought: ", "purple"),
                        ColoredText(thought, "green"),
                    ]
                )
            printer.print(
                content=[
                    ColoredText("## Using tool: ", "purple"),
                    ColoredText(formatted_answer.tool, "green"),
                ]
            )
            printer.print(
                content=[
                    ColoredText("## Tool Input: ", "purple"),
                    ColoredText(f"\n{formatted_json}", "green"),
                ]
            )
            printer.print(
                content=[
                    ColoredText("## Tool Output: ", "purple"),
                    ColoredText(f"\n{formatted_answer.result}", "green"),
                ]
            )
        elif isinstance(formatted_answer, AgentFinish):
            printer.print(
                content=[
                    ColoredText("## Final Answer: ", "purple"),
                    ColoredText(f"\n{formatted_answer.output}\n\n", "green"),
                ]
            )


def _print_current_organization() -> None:
    settings = Settings()
    if settings.org_uuid:
        console.print(
            f"Fetching agent from organization: {settings.org_name} ({settings.org_uuid})",
            style="bold blue",
        )
    else:
        console.print(
            "No organization currently set. We recommend setting one before using: `crewai org switch <org_id>` command.",
            style="yellow",
        )


def load_agent_from_repository(from_repository: str) -> dict[str, Any]:
    """Load an agent from the repository.

    Args:
        from_repository: The name of the agent to load.

    Returns:
        A dictionary of attributes to use for the agent.

    Raises:
        AgentRepositoryError: If the agent cannot be loaded.
    """
    attributes: dict[str, Any] = {}
    if from_repository:
        import importlib

        if callable(_create_plus_client_hook):
            client = _create_plus_client_hook()
        else:
            from crewai.cli.authentication.token import get_auth_token
            from crewai.cli.plus_api import PlusAPI

            client = PlusAPI(api_key=get_auth_token())
        _print_current_organization()
        response = asyncio.run(client.get_agent(from_repository))
        if response.status_code == 404:
            raise AgentRepositoryError(
                f"Agent {from_repository} does not exist, make sure the name is correct or the agent is available on your organization."
                f"\nIf you are using the wrong organization, switch to the correct one using `crewai org switch <org_id>` command.",
            )

        if response.status_code != 200:
            raise AgentRepositoryError(
                f"Agent {from_repository} could not be loaded: {response.text}"
                f"\nIf you are using the wrong organization, switch to the correct one using `crewai org switch <org_id>` command.",
            )

        agent = response.json()
        for key, value in agent.items():
            if key == "tools":
                attributes[key] = []
                for tool in value:
                    try:
                        module = importlib.import_module(tool["module"])
                        tool_class = getattr(module, tool["name"])

                        tool_value = tool_class(**tool["init_params"])

                        if isinstance(tool_value, list):
                            attributes[key].extend(tool_value)
                        else:
                            attributes[key].append(tool_value)

                    except Exception as e:  # noqa: PERF203
                        raise AgentRepositoryError(
                            f"Tool {tool['name']} could not be loaded: {e}"
                        ) from e
            else:
                attributes[key] = value
    return attributes


DELEGATION_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    [
        sanitize_tool_name("Delegate work to coworker"),
        sanitize_tool_name("Ask question to coworker"),
    ]
)


# native tool calling tracking for delegation
def track_delegation_if_needed(
    tool_name: str,
    tool_args: dict[str, Any],
    task: Task | None,
) -> None:
    """Track delegation if the tool is a delegation tool.

    Args:
        tool_name: Name of the tool being executed.
        tool_args: Arguments passed to the tool.
        task: The task being executed (used to track delegations).
    """
    if sanitize_tool_name(tool_name) in DELEGATION_TOOL_NAMES and task is not None:
        coworker = tool_args.get("coworker")
        task.increment_delegations(coworker)


def extract_tool_call_info(
    tool_call: Any,
) -> tuple[str, str, dict[str, Any] | str] | None:
    """Extract tool call ID, name, and arguments from various provider formats.

    Args:
        tool_call: The tool call object to extract info from.

    Returns:
        Tuple of (call_id, func_name, func_args) or None if format is unrecognized.
    """
    if hasattr(tool_call, "function"):
        # OpenAI-style: has .function.name and .function.arguments
        call_id = getattr(tool_call, "id", f"call_{id(tool_call)}")
        return (
            call_id,
            sanitize_tool_name(tool_call.function.name),
            tool_call.function.arguments,
        )
    if hasattr(tool_call, "function_call") and tool_call.function_call:
        # Gemini-style: has .function_call.name and .function_call.args
        call_id = f"call_{id(tool_call)}"
        return (
            call_id,
            sanitize_tool_name(tool_call.function_call.name),
            dict(tool_call.function_call.args) if tool_call.function_call.args else {},
        )
    if hasattr(tool_call, "name") and hasattr(tool_call, "input"):
        # Anthropic format: has .name and .input (ToolUseBlock)
        call_id = getattr(tool_call, "id", f"call_{id(tool_call)}")
        return call_id, sanitize_tool_name(tool_call.name), tool_call.input
    if isinstance(tool_call, dict):
        # Support OpenAI "id", Bedrock "toolUseId", or generate one
        call_id = (
            tool_call.get("id") or tool_call.get("toolUseId") or f"call_{id(tool_call)}"
        )
        func_info = tool_call.get("function", {})
        func_name = func_info.get("name", "") or tool_call.get("name", "")
        func_args = func_info.get("arguments") or tool_call.get("input") or {}
        return call_id, sanitize_tool_name(func_name), func_args
    return None


def is_tool_call_list(response: list[Any]) -> bool:
    """Check if a response from the LLM is a list of tool calls.

    Supports OpenAI, Anthropic, Bedrock, and Gemini formats.

    Args:
        response: The response to check.

    Returns:
        True if the response appears to be a list of tool calls.
    """
    if not response:
        return False
    first_item = response[0]
    # OpenAI-style
    if hasattr(first_item, "function") or (
        isinstance(first_item, dict) and "function" in first_item
    ):
        return True
    # Anthropic-style (ToolUseBlock)
    if (
        hasattr(first_item, "type")
        and getattr(first_item, "type", None) == "tool_use"
    ):
        return True
    if hasattr(first_item, "name") and hasattr(first_item, "input"):
        return True
    # Bedrock-style
    if (
        isinstance(first_item, dict)
        and "name" in first_item
        and "input" in first_item
    ):
        return True
    # Gemini-style
    if hasattr(first_item, "function_call") and first_item.function_call:
        return True
    return False


def check_native_tool_support(llm: Any, original_tools: list[BaseTool] | None) -> bool:
    """Check if the LLM supports native function calling and tools are available.

    Args:
        llm: The LLM instance.
        original_tools: Original BaseTool instances.

    Returns:
        True if native function calling is supported and tools exist.
    """
    return (
        hasattr(llm, "supports_function_calling")
        and callable(getattr(llm, "supports_function_calling", None))
        and llm.supports_function_calling()
        and bool(original_tools)
    )


def setup_native_tools(
    original_tools: list[BaseTool],
) -> tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]:
    """Convert tools to OpenAI schema format for native function calling.

    Args:
        original_tools: Original BaseTool instances.

    Returns:
        Tuple of (openai_tools_schema, available_functions_dict).
    """
    return convert_tools_to_openai_schema(original_tools)


def build_tool_calls_assistant_message(
    tool_calls: list[Any],
) -> tuple[LLMMessage | None, list[dict[str, Any]]]:
    """Build an assistant message containing tool call reports.

    Extracts info from each tool call, builds the standard assistant message
    format, and preserves raw Gemini parts when applicable.

    Args:
        tool_calls: Raw tool call objects from the LLM response.

    Returns:
        Tuple of (assistant_message, tool_calls_to_report).
        assistant_message is None if no valid tool calls found.
    """
    tool_calls_to_report: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        info = extract_tool_call_info(tool_call)
        if not info:
            continue
        call_id, func_name, func_args = info
        tool_calls_to_report.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": func_args
                    if isinstance(func_args, str)
                    else json.dumps(func_args),
                },
            }
        )

    if not tool_calls_to_report:
        return None, []

    assistant_message: LLMMessage = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls_to_report,
    }
    # Preserve raw parts for Gemini compatibility
    if all(type(tc).__qualname__ == "Part" for tc in tool_calls):
        assistant_message["raw_tool_call_parts"] = list(tool_calls)

    return assistant_message, tool_calls_to_report


@dataclass
class NativeToolCallResult:
    """Result from executing a single native tool call."""

    call_id: str
    func_name: str
    result: str
    from_cache: bool = False
    result_as_answer: bool = False
    tool_message: LLMMessage = field(default_factory=dict)  # type: ignore[assignment]


def execute_single_native_tool_call(
    tool_call: Any,
    *,
    available_functions: dict[str, Callable[..., Any]],
    original_tools: list[BaseTool],
    structured_tools: list[CrewStructuredTool] | None,
    tools_handler: ToolsHandler | None,
    agent: Agent | None,
    task: Task | None,
    crew: Any | None,
    event_source: Any,
    printer: Printer | None = None,
    verbose: bool = False,
) -> NativeToolCallResult:
    """Execute a single native tool call with full lifecycle management.

    Handles: arg parsing, tool lookup, max-usage check, cache read/write,
    before/after hooks, event emission, and result_as_answer detection.

    Args:
        tool_call: Raw tool call object from the LLM.
        available_functions: Map of sanitized tool name -> callable.
        original_tools: Original BaseTool list (for cache_function, result_as_answer).
        structured_tools: Structured tools list (for hook context).
        tools_handler: Optional handler with cache.
        agent: The agent instance.
        task: The current task.
        crew: The crew instance.
        event_source: The object to use as event emitter source.
        printer: Optional printer for verbose logging.
        verbose: Whether to print verbose output.

    Returns:
        NativeToolCallResult with all execution details.
    """
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.tool_usage_events import (
        ToolUsageErrorEvent,
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
    )
    from crewai.hooks.tool_hooks import (
        ToolCallHookContext,
        get_after_tool_call_hooks,
        get_before_tool_call_hooks,
    )

    info = extract_tool_call_info(tool_call)
    if not info:
        return NativeToolCallResult(
            call_id="", func_name="", result="Unrecognized tool call format"
        )

    call_id, func_name, func_args = info

    # Parse arguments
    if isinstance(func_args, str):
        try:
            args_dict = json.loads(func_args)
        except json.JSONDecodeError:
            args_dict = {}
    else:
        args_dict = func_args

    agent_key = getattr(agent, "key", "unknown") if agent else "unknown"

    # Find original tool for cache_function and result_as_answer
    original_tool: BaseTool | None = None
    for tool in original_tools:
        if sanitize_tool_name(tool.name) == func_name:
            original_tool = tool
            break

    # Check max usage count
    max_usage_reached = False
    if (
        original_tool
        and original_tool.max_usage_count is not None
        and original_tool.current_usage_count >= original_tool.max_usage_count
    ):
        max_usage_reached = True

    # Check cache
    from_cache = False
    input_str = json.dumps(args_dict) if args_dict else ""
    result = "Tool not found"

    if tools_handler and tools_handler.cache:
        cached_result = tools_handler.cache.read(tool=func_name, input=input_str)
        if cached_result is not None:
            result = (
                str(cached_result) if not isinstance(cached_result, str) else cached_result
            )
            from_cache = True

    # Emit tool started event
    started_at = datetime.now()
    crewai_event_bus.emit(
        event_source,
        event=ToolUsageStartedEvent(
            tool_name=func_name,
            tool_args=args_dict,
            from_agent=agent,
            from_task=task,
            agent_key=agent_key,
        ),
    )

    track_delegation_if_needed(func_name, args_dict, task)

    # Find structured tool for hooks
    structured_tool: CrewStructuredTool | None = None
    for structured in structured_tools or []:
        if sanitize_tool_name(structured.name) == func_name:
            structured_tool = structured
            break

    # Before hooks
    hook_blocked = False
    before_hook_context = ToolCallHookContext(
        tool_name=func_name,
        tool_input=args_dict,
        tool=structured_tool,  # type: ignore[arg-type]
        agent=agent,
        task=task,
        crew=crew,
    )
    try:
        for hook in get_before_tool_call_hooks():
            if hook(before_hook_context) is False:
                hook_blocked = True
                break
    except Exception:  # noqa: S110
        pass

    error_event_emitted = False
    if hook_blocked:
        result = f"Tool execution blocked by hook. Tool: {func_name}"
    elif not from_cache and not max_usage_reached:
        if func_name in available_functions:
            try:
                tool_func = available_functions[func_name]
                raw_result = tool_func(**args_dict)

                # Cache result
                if tools_handler and tools_handler.cache:
                    should_cache = True
                    if original_tool:
                        should_cache = original_tool.cache_function(args_dict, raw_result)
                    if should_cache:
                        tools_handler.cache.add(
                            tool=func_name, input=input_str, output=raw_result
                        )

                result = (
                    str(raw_result) if not isinstance(raw_result, str) else raw_result
                )
            except Exception as e:
                result = f"Error executing tool: {e}"
                if task:
                    task.increment_tools_errors()
                crewai_event_bus.emit(
                    event_source,
                    event=ToolUsageErrorEvent(
                        tool_name=func_name,
                        tool_args=args_dict,
                        from_agent=agent,
                        from_task=task,
                        agent_key=agent_key,
                        error=e,
                    ),
                )
                error_event_emitted = True
    elif max_usage_reached and original_tool:
        result = (
            f"Tool '{func_name}' has reached its usage limit of "
            f"{original_tool.max_usage_count} times and cannot be used anymore."
        )

    # After hooks
    after_hook_context = ToolCallHookContext(
        tool_name=func_name,
        tool_input=args_dict,
        tool=structured_tool,  # type: ignore[arg-type]
        agent=agent,
        task=task,
        crew=crew,
        tool_result=result,
    )
    try:
        for after_hook in get_after_tool_call_hooks():
            hook_result = after_hook(after_hook_context)
            if hook_result is not None:
                result = hook_result
                after_hook_context.tool_result = result
    except Exception:  # noqa: S110
        pass

    # Emit tool finished event (only if error event wasn't already emitted)
    if not error_event_emitted:
        crewai_event_bus.emit(
            event_source,
            event=ToolUsageFinishedEvent(
                output=result,
                tool_name=func_name,
                tool_args=args_dict,
                from_agent=agent,
                from_task=task,
                agent_key=agent_key,
                started_at=started_at,
                finished_at=datetime.now(),
            ),
        )

    # Build tool result message
    tool_message: LLMMessage = {
        "role": "tool",
        "tool_call_id": call_id,
        "name": func_name,
        "content": result,
    }

    if verbose and printer:
        cache_info = " (from cache)" if from_cache else ""
        printer.print(
            content=f"Tool {func_name} executed with result{cache_info}: {result[:200]}...",
            color="green",
        )

    # Check result_as_answer
    is_result_as_answer = bool(
        original_tool
        and hasattr(original_tool, "result_as_answer")
        and original_tool.result_as_answer
    )

    return NativeToolCallResult(
        call_id=call_id,
        func_name=func_name,
        result=result,
        from_cache=from_cache,
        result_as_answer=is_result_as_answer,
        tool_message=tool_message,
    )


def _setup_before_llm_call_hooks(
    executor_context: CrewAgentExecutor | AgentExecutor | LiteAgent | None,
    printer: Printer,
    verbose: bool = True,
) -> bool:
    """Setup and invoke before_llm_call hooks for the executor context.

    Args:
        executor_context: The executor context to setup the hooks for.
        printer: Printer instance for error logging.
        verbose: Whether to print output.

    Returns:
        True if LLM execution should proceed, False if blocked by a hook.
    """
    if executor_context and executor_context.before_llm_call_hooks:
        from crewai.hooks.llm_hooks import LLMCallHookContext

        original_messages = executor_context.messages

        hook_context = LLMCallHookContext(executor_context)
        try:
            for hook in executor_context.before_llm_call_hooks:
                result = hook(hook_context)
                if result is False:
                    if verbose:
                        printer.print(
                            content="LLM call blocked by before_llm_call hook",
                            color="yellow",
                        )
                    return False
        except Exception as e:
            if verbose:
                printer.print(
                    content=f"Error in before_llm_call hook: {e}",
                    color="yellow",
                )

        if not isinstance(executor_context.messages, list):
            if verbose:
                printer.print(
                    content=(
                        "Warning: before_llm_call hook replaced messages with non-list. "
                        "Restoring original messages list. Hooks should modify messages in-place, "
                        "not replace the list (e.g., use context.messages.append() not context.messages = [])."
                    ),
                    color="yellow",
                )
            if isinstance(original_messages, list):
                executor_context.messages = original_messages
            else:
                executor_context.messages = []

    return True


def _setup_after_llm_call_hooks(
    executor_context: CrewAgentExecutor | AgentExecutor | LiteAgent | None,
    answer: str | BaseModel,
    printer: Printer,
    verbose: bool = True,
) -> str | BaseModel:
    """Setup and invoke after_llm_call hooks for the executor context.

    Args:
        executor_context: The executor context to setup the hooks for.
        answer: The LLM response (string or Pydantic model).
        printer: Printer instance for error logging.
        verbose: Whether to print output.

    Returns:
        The potentially modified response (string or Pydantic model).
    """
    if executor_context and executor_context.after_llm_call_hooks:
        from crewai.hooks.llm_hooks import LLMCallHookContext

        original_messages = executor_context.messages

        # For Pydantic models, serialize to JSON for hooks
        if isinstance(answer, BaseModel):
            pydantic_answer = answer
            hook_response: str = pydantic_answer.model_dump_json()
            original_json: str = hook_response
        else:
            pydantic_answer = None
            hook_response = str(answer)

        hook_context = LLMCallHookContext(executor_context, response=hook_response)
        try:
            for hook in executor_context.after_llm_call_hooks:
                modified_response = hook(hook_context)
                if modified_response is not None and isinstance(modified_response, str):
                    hook_response = modified_response

        except Exception as e:
            if verbose:
                printer.print(
                    content=f"Error in after_llm_call hook: {e}",
                    color="yellow",
                )

        if not isinstance(executor_context.messages, list):
            if verbose:
                printer.print(
                    content=(
                        "Warning: after_llm_call hook replaced messages with non-list. "
                        "Restoring original messages list. Hooks should modify messages in-place, "
                        "not replace the list (e.g., use context.messages.append() not context.messages = [])."
                    ),
                    color="yellow",
                )
            if isinstance(original_messages, list):
                executor_context.messages = original_messages
            else:
                executor_context.messages = []

        # If hooks modified the response, update answer accordingly
        if pydantic_answer is not None:
            # For Pydantic models, reparse the JSON if it was modified
            if hook_response != original_json:
                try:
                    model_class: type[BaseModel] = type(pydantic_answer)
                    answer = model_class.model_validate_json(hook_response)
                except Exception as e:
                    if verbose:
                        printer.print(
                            content=f"Warning: Hook modified response but failed to reparse as {type(pydantic_answer).__name__}: {e}. Using original model.",
                            color="yellow",
                        )
        else:
            # For string responses, use the hook-modified response
            answer = hook_response

    return answer
