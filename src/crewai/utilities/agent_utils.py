import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from crewai.agents.parser import (
    FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE,
    AgentAction,
    AgentFinish,
    CrewAgentParser,
    OutputParserException,
)
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool as CrewAITool
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_types import ToolResult
from crewai.utilities import I18N, Printer
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededException,
)


def parse_tools(tools: List[BaseTool]) -> List[CrewStructuredTool]:
    """Parse tools to be used for the task."""
    tools_list = []

    for tool in tools:
        if isinstance(tool, CrewAITool):
            tools_list.append(tool.to_structured_tool())
        else:
            raise ValueError("Tool is not a CrewStructuredTool or BaseTool")

    return tools_list


def get_tool_names(tools: Sequence[Union[CrewStructuredTool, BaseTool]]) -> str:
    """Get the names of the tools."""
    return ", ".join([t.name for t in tools])


def render_text_description_and_args(
    tools: Sequence[Union[CrewStructuredTool, BaseTool]],
) -> str:
    """Render the tool name, description, and args in plain text.
    
        search: This tool is used for search, args: {"query": {"type": "string"}}
        calculator: This tool is used for math, \
        args: {"expression": {"type": "string"}}
    """
    tool_strings = []
    for tool in tools:
        tool_strings.append(tool.description)

    return "\n".join(tool_strings)


def has_reached_max_iterations(iterations: int, max_iterations: int) -> bool:
    """Check if the maximum number of iterations has been reached."""
    return iterations >= max_iterations


def handle_max_iterations_exceeded(
    formatted_answer: Union[AgentAction, AgentFinish, None],
    printer: Printer,
    i18n: I18N,
    messages: List[Dict[str, str]],
    llm: Union[LLM, BaseLLM],
    callbacks: List[Any],
) -> Union[AgentAction, AgentFinish]:
    """
    Handles the case when the maximum number of iterations is exceeded.
    Performs one more LLM call to get the final answer.

    Parameters:
        formatted_answer: The last formatted answer from the agent.

    Returns:
        The final formatted answer after exceeding max iterations.
    """
    printer.print(
        content="Maximum iterations reached. Requesting final answer.",
        color="yellow",
    )

    if formatted_answer and hasattr(formatted_answer, "text"):
        assistant_message = (
            formatted_answer.text + f'\n{i18n.errors("force_final_answer")}'
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
        printer.print(
            content="Received None or empty response from LLM call.",
            color="red",
        )
        raise ValueError("Invalid response from LLM call - None or empty.")

    formatted_answer = format_answer(answer)
    # Return the formatted answer, regardless of its type
    return formatted_answer


def format_message_for_llm(prompt: str, role: str = "user") -> Dict[str, str]:
    prompt = prompt.rstrip()
    return {"role": role, "content": prompt}


def format_answer(answer: str) -> Union[AgentAction, AgentFinish]:
    """Format a response from the LLM into an AgentAction or AgentFinish."""
    try:
        return CrewAgentParser.parse_text(answer)
    except Exception:
        # If parsing fails, return a default AgentFinish
        return AgentFinish(
            thought="Failed to parse LLM response",
            output=answer,
            text=answer,
        )


def enforce_rpm_limit(
    request_within_rpm_limit: Optional[Callable[[], bool]] = None,
) -> None:
    """Enforce the requests per minute (RPM) limit if applicable."""
    if request_within_rpm_limit:
        request_within_rpm_limit()


def get_llm_response(
    llm: Union[LLM, BaseLLM],
    messages: List[Dict[str, str]],
    callbacks: List[Any],
    printer: Printer,
) -> str:
    """Call the LLM and return the response, handling any invalid responses."""
    try:
        answer = llm.call(
            messages,
            callbacks=callbacks,
        )
    except Exception as e:
        printer.print(
            content=f"Error during LLM call: {e}",
            color="red",
        )
        raise e
    if not answer:
        printer.print(
            content="Received None or empty response from LLM call.",
            color="red",
        )
        raise ValueError("Invalid response from LLM call - None or empty.")

    return answer


def process_llm_response(
    answer: str, use_stop_words: bool
) -> Union[AgentAction, AgentFinish]:
    """Process the LLM response and format it into an AgentAction or AgentFinish."""
    if not use_stop_words:
        try:
            # Preliminary parsing to check for errors.
            format_answer(answer)
        except OutputParserException as e:
            if FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE in e.error:
                answer = answer.split("Observation:")[0].strip()

    return format_answer(answer)


def handle_agent_action_core(
    formatted_answer: AgentAction,
    tool_result: ToolResult,
    messages: Optional[List[Dict[str, str]]] = None,
    step_callback: Optional[Callable] = None,
    show_logs: Optional[Callable] = None,
) -> Union[AgentAction, AgentFinish]:
    """Core logic for handling agent actions and tool results.

    Args:
        formatted_answer: The agent's action
        tool_result: The result of executing the tool
        messages: Optional list of messages to append results to
        step_callback: Optional callback to execute after processing
        show_logs: Optional function to show logs

    Returns:
        Either an AgentAction or AgentFinish
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

    if messages is not None:
        messages.append({"role": "assistant", "content": tool_result.result})

    return formatted_answer


def handle_unknown_error(printer: Any, exception: Exception) -> None:
    """Handle unknown errors by informing the user.

    Args:
        printer: Printer instance for output
        exception: The exception that occurred
    """
    printer.print(
        content="An unknown error occurred. Please check the details below.",
        color="red",
    )
    printer.print(
        content=f"Error details: {exception}",
        color="red",
    )


def handle_output_parser_exception(
    e: OutputParserException,
    messages: List[Dict[str, str]],
    iterations: int,
    log_error_after: int = 3,
    printer: Optional[Any] = None,
) -> AgentAction:
    """Handle OutputParserException by updating messages and formatted_answer.

    Args:
        e: The OutputParserException that occurred
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

    if iterations > log_error_after and printer:
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
    return LLMContextLengthExceededException(str(exception))._is_context_limit_error(
        str(exception)
    )


def handle_context_length(
    respect_context_window: bool,
    printer: Any,
    messages: List[Dict[str, str]],
    llm: Any,
    callbacks: List[Any],
    i18n: Any,
) -> None:
    """Handle context length exceeded by either summarizing or raising an error.

    Args:
        respect_context_window: Whether to respect context window
        printer: Printer instance for output
        messages: List of messages to summarize
        llm: LLM instance for summarization
        callbacks: List of callbacks for LLM
        i18n: I18N instance for messages
    """
    if respect_context_window:
        printer.print(
            content="Context length exceeded. Summarizing content to fit the model context window.",
            color="yellow",
        )
        summarize_messages(messages, llm, callbacks, i18n)
    else:
        printer.print(
            content="Context length exceeded. Consider using smaller text or RAG tools from crewai_tools.",
            color="red",
        )
        raise SystemExit(
            "Context length exceeded and user opted not to summarize. Consider using smaller text or RAG tools from crewai_tools."
        )


def summarize_messages(
    messages: List[Dict[str, str]],
    llm: Any,
    callbacks: List[Any],
    i18n: Any,
) -> None:
    """Summarize messages to fit within context window.

    Args:
        messages: List of messages to summarize
        llm: LLM instance for summarization
        callbacks: List of callbacks for LLM
        i18n: I18N instance for messages
    """
    messages_groups = []
    for message in messages:
        content = message["content"]
        cut_size = llm.get_context_window_size()
        for i in range(0, len(content), cut_size):
            messages_groups.append({"content": content[i : i + cut_size]})

    summarized_contents = []
    for group in messages_groups:
        summary = llm.call(
            [
                format_message_for_llm(
                    i18n.slice("summarizer_system_message"), role="system"
                ),
                format_message_for_llm(
                    i18n.slice("summarize_instruction").format(group=group["content"]),
                ),
            ],
            callbacks=callbacks,
        )
        summarized_contents.append({"content": str(summary)})

    merged_summary = " ".join(content["content"] for content in summarized_contents)

    messages.clear()
    messages.append(
        format_message_for_llm(
            i18n.slice("summary").format(merged_summary=merged_summary)
        )
    )


def show_agent_logs(
    printer: Printer,
    agent_role: str,
    formatted_answer: Optional[Union[AgentAction, AgentFinish]] = None,
    task_description: Optional[str] = None,
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

    agent_role = agent_role.split("\n")[0]

    if formatted_answer is None:
        # Start logs
        printer.print(
            content=f"\033[1m\033[95m# Agent:\033[00m \033[1m\033[92m{agent_role}\033[00m"
        )
        if task_description:
            printer.print(
                content=f"\033[95m## Task:\033[00m \033[92m{task_description}\033[00m"
            )
    else:
        # Execution logs
        printer.print(
            content=f"\n\n\033[1m\033[95m# Agent:\033[00m \033[1m\033[92m{agent_role}\033[00m"
        )

        if isinstance(formatted_answer, AgentAction):
            thought = re.sub(r"\n+", "\n", formatted_answer.thought)
            formatted_json = json.dumps(
                formatted_answer.tool_input,
                indent=2,
                ensure_ascii=False,
            )
            if thought and thought != "":
                printer.print(
                    content=f"\033[95m## Thought:\033[00m \033[92m{thought}\033[00m"
                )
            printer.print(
                content=f"\033[95m## Using tool:\033[00m \033[92m{formatted_answer.tool}\033[00m"
            )
            printer.print(
                content=f"\033[95m## Tool Input:\033[00m \033[92m\n{formatted_json}\033[00m"
            )
            printer.print(
                content=f"\033[95m## Tool Output:\033[00m \033[92m\n{formatted_answer.result}\033[00m"
            )
        elif isinstance(formatted_answer, AgentFinish):
            printer.print(
                content=f"\033[95m## Final Answer:\033[00m \033[92m\n{formatted_answer.output}\033[00m\n\n"
            )
