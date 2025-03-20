from typing import Any, Callable, Dict, List, Optional, Union

from crewai.agents.parser import (
    FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE,
    AgentAction,
    AgentFinish,
    CrewAgentParser,
    OutputParserException,
)
from crewai.llm import LLM
from crewai.tools import BaseTool as CrewAITool
from crewai.tools.base_tool import BaseTool
from crewai.utilities.i18n import I18N
from crewai.utilities.printer import Printer


def parse_tools(tools: List[Any]) -> List[Any]:
    """Parse tools to be used for the task."""
    tools_list = []
    try:
        for tool in tools:
            if isinstance(tool, CrewAITool):
                tools_list.append(tool.to_structured_tool())
            else:
                tools_list.append(tool)
    except ModuleNotFoundError:
        tools_list = []
        for tool in tools:
            tools_list.append(tool)

    return tools_list


def get_tool_names(tools: List[Any]) -> str:
    """Get the names of the tools."""
    return ", ".join([t.name for t in tools])


def render_text_description_and_args(tools: List[BaseTool]) -> str:
    """Render the tool name, description, and args in plain text.

        Output will be in the format of:

        .. code-block:: markdown

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
    llm: LLM,
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
    request_within_rpm_limit: Optional[Callable[[], bool]] = None
) -> None:
    """Enforce the requests per minute (RPM) limit if applicable."""
    if request_within_rpm_limit:
        request_within_rpm_limit()


def get_llm_response(
    llm: LLM, messages: List[Dict[str, str]], callbacks: List[Any], printer: Printer
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
