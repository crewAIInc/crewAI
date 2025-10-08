from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict

from rich.console import Console

from crewai.agents.constants import FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    OutputParserError,
    parse,
)
from crewai.cli.config import Settings
from crewai.llm import LLM
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
from crewai.utilities.types import LLMMessage

if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.lite_agent import LiteAgent
    from crewai.task import Task


class SummaryContent(TypedDict):
    """Structure for summary content entries.

    Attributes:
        content: The summarized content.
    """

    content: str


console = Console()

_MULTIPLE_NEWLINES: Final[re.Pattern[str]] = re.compile(r"\n+")


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
            tools_list.append(tool.to_structured_tool())
        else:
            raise ValueError("Tool is not a CrewStructuredTool or BaseTool")

    return tools_list


def get_tool_names(tools: Sequence[CrewStructuredTool | BaseTool]) -> str:
    """Get the names of the tools.

    Args:
        tools: List of tools to get names from.

    Returns:
        Comma-separated string of tool names.
    """
    return ", ".join([t.name for t in tools])


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
    callbacks: list[Callable[..., Any]],
) -> AgentAction | AgentFinish:
    """Handles the case when the maximum number of iterations is exceeded. Performs one more LLM call to get the final answer.

    Args:
        formatted_answer: The last formatted answer from the agent.
        printer: Printer instance for output.
        i18n: I18N instance for internationalization.
        messages: List of messages to send to the LLM.
        llm: The LLM instance to call.
        callbacks: List of callbacks for the LLM call.

    Returns:
        The final formatted answer after exceeding max iterations.
    """
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
        messages,  # type: ignore[arg-type]
        callbacks=callbacks,
    )

    if answer is None or answer == "":
        printer.print(
            content="Received None or empty response from LLM call.",
            color="red",
        )
        raise ValueError("Invalid response from LLM call - None or empty.")

    # Return the formatted answer, regardless of its type
    return format_answer(answer=answer)


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


def get_llm_response(
    llm: LLM | BaseLLM,
    messages: list[LLMMessage],
    callbacks: list[Callable[..., Any]],
    printer: Printer,
    from_task: Task | None = None,
    from_agent: Agent | LiteAgent | None = None,
) -> str:
    """Call the LLM and return the response, handling any invalid responses.

    Args:
        llm: The LLM instance to call
        messages: The messages to send to the LLM
        callbacks: List of callbacks for the LLM call
        printer: Printer instance for output
        from_task: Optional task context for the LLM call
        from_agent: Optional agent context for the LLM call

    Returns:
        The response from the LLM as a string

    Raises:
        Exception: If an error occurs.
        ValueError: If the response is None or empty.
    """
    try:
        answer = llm.call(
            messages,  # type: ignore[arg-type]
            callbacks=callbacks,
            from_task=from_task,
            from_agent=from_agent,
        )
    except Exception as e:
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
    messages: list[dict[str, str]] | None = None,
    step_callback: Callable | None = None,
    show_logs: Callable | None = None,
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


def handle_unknown_error(printer: Printer, exception: Exception) -> None:
    """Handle unknown errors by informing the user.

    Args:
        printer: Printer instance for output
        exception: The exception that occurred
    """
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
    return LLMContextLengthExceededError(str(exception))._is_context_limit_error(
        str(exception)
    )


def handle_context_length(
    respect_context_window: bool,
    printer: Printer,
    messages: list[LLMMessage],
    llm: LLM | BaseLLM,
    callbacks: list[Callable[..., Any]],
    i18n: I18N,
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
        printer.print(
            content="Context length exceeded. Summarizing content to fit the model context window. Might take a while...",
            color="yellow",
        )
        summarize_messages(messages=messages, llm=llm, callbacks=callbacks, i18n=i18n)
    else:
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
    callbacks: list[Callable[..., Any]],
    i18n: I18N,
) -> None:
    """Summarize messages to fit within context window.

    Args:
        messages: List of messages to summarize
        llm: LLM instance for summarization
        callbacks: List of callbacks for LLM
        i18n: I18N instance for messages
    """
    messages_string = " ".join([message["content"] for message in messages])
    cut_size = llm.get_context_window_size()

    messages_groups = [
        {"content": messages_string[i : i + cut_size]}
        for i in range(0, len(messages_string), cut_size)
    ]

    summarized_contents: list[SummaryContent] = []

    total_groups = len(messages_groups)
    for idx, group in enumerate(messages_groups, 1):
        Printer().print(
            content=f"Summarizing {idx}/{total_groups}...",
            color="yellow",
        )

        messages = [
            format_message_for_llm(
                i18n.slice("summarizer_system_message"), role="system"
            ),
            format_message_for_llm(
                i18n.slice("summarize_instruction").format(group=group["content"]),
            ),
        ]
        summary = llm.call(
            messages,  # type: ignore[arg-type]
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

        from crewai.cli.authentication.token import get_auth_token
        from crewai.cli.plus_api import PlusAPI

        client = PlusAPI(api_key=get_auth_token())
        _print_current_organization()
        response = client.get_agent(from_repository)
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
