from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import concurrent.futures
import inspect
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
) -> tuple[
    list[dict[str, Any]],
    dict[str, Callable[..., Any]],
    dict[str, BaseTool | CrewStructuredTool],
]:
    """Convert CrewAI tools to OpenAI function calling format.

    This function converts CrewAI BaseTool and CrewStructuredTool objects
    into the OpenAI-compatible tool schema format that can be passed to
    LLM providers for native function calling.

    Args:
        tools: List of CrewAI tool objects to convert.

    Returns:
        Tuple containing:
        - List of OpenAI-format tool schema dictionaries
        - Dict mapping sanitized tool names to their callable run() methods
        - Dict mapping sanitized tool names to their original tool objects
    """
    openai_tools: list[dict[str, Any]] = []
    available_functions: dict[str, Callable[..., Any]] = {}
    tool_name_mapping: dict[str, BaseTool | CrewStructuredTool] = {}

    for tool in tools:
        # Get the JSON schema for tool parameters
        parameters: dict[str, Any] = {}
        if hasattr(tool, "args_schema") and tool.args_schema is not None:
            try:
                schema_output = generate_model_description(
                    tool.args_schema, strip_null_types=False
                )
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

        if sanitized_name in available_functions:
            counter = 2
            candidate = sanitize_tool_name(f"{sanitized_name}_{counter}")
            while candidate in available_functions:
                counter += 1
                candidate = sanitize_tool_name(f"{sanitized_name}_{counter}")
            sanitized_name = candidate

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
        tool_name_mapping[sanitized_name] = tool

    return openai_tools, available_functions, tool_name_mapping


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

    if executor_context is not None:
        if not _setup_before_llm_call_hooks(executor_context, printer, verbose=verbose):
            raise ValueError("LLM call blocked by before_llm_call hook")
        messages = executor_context.messages

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

    Returns:
        The response from the LLM as a string, Pydantic model (when response_model is provided),
        or tool call results if native function calling is used.

    Raises:
        Exception: If an error occurs.
        ValueError: If the response is None or empty.
    """
    if executor_context is not None:
        if not _setup_before_llm_call_hooks(executor_context, printer, verbose=verbose):
            raise ValueError("LLM call blocked by before_llm_call hook")
        messages = executor_context.messages

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
        cb_result = step_callback(tool_result)
        if inspect.iscoroutine(cb_result):
            asyncio.run(cb_result)

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


def _estimate_token_count(text: str) -> int:
    """Estimate token count using a conservative cross-provider heuristic.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count (roughly 1 token per 4 characters).
    """
    return len(text) // 4


def _format_messages_for_summary(messages: list[LLMMessage]) -> str:
    """Format messages with role labels for summarization.

    Skips system messages. Handles None content, tool_calls, and
    multimodal content blocks.

    Args:
        messages: List of messages to format.

    Returns:
        Role-labeled conversation text.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            continue

        content = msg.get("content")
        if content is None:
            # Check for tool_calls on assistant messages with no content
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tool_names = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = (
                        func.get("name", "unknown")
                        if isinstance(func, dict)
                        else "unknown"
                    )
                    tool_names.append(name)
                content = f"[Called tools: {', '.join(tool_names)}]"
            else:
                content = ""
        elif isinstance(content, list):
            # Multimodal content blocks — extract text parts
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = " ".join(text_parts) if text_parts else "[multimodal content]"

        if role == "assistant":
            label = "[ASSISTANT]:"
        elif role == "tool":
            tool_name = msg.get("name", "unknown")
            label = f"[TOOL_RESULT ({tool_name})]:"
        else:
            label = "[USER]:"

        lines.append(f"{label} {content}")

    return "\n\n".join(lines)


def _split_messages_into_chunks(
    messages: list[LLMMessage], max_tokens: int
) -> list[list[LLMMessage]]:
    """Split messages into chunks at message boundaries.

    Excludes system messages from chunks. Each chunk stays under
    max_tokens based on estimated token count.

    Args:
        messages: List of messages to split.
        max_tokens: Maximum estimated tokens per chunk.

    Returns:
        List of message chunks.
    """
    non_system = [m for m in messages if m.get("role") != "system"]
    if not non_system:
        return []

    chunks: list[list[LLMMessage]] = []
    current_chunk: list[LLMMessage] = []
    current_tokens = 0

    for msg in non_system:
        content = msg.get("content")
        if content is None:
            msg_text = ""
        elif isinstance(content, list):
            msg_text = str(content)
        else:
            msg_text = str(content)

        msg_tokens = _estimate_token_count(msg_text)

        # If adding this message would exceed the limit and we already have
        # messages in the current chunk, start a new chunk
        if current_chunk and (current_tokens + msg_tokens) > max_tokens:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(msg)
        current_tokens += msg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _extract_summary_tags(text: str) -> str:
    """Extract content between <summary></summary> tags.

    Falls back to the full text if no tags are found.

    Args:
        text: Text potentially containing summary tags.

    Returns:
        Extracted summary content, or full text if no tags found.
    """
    match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


async def _asummarize_chunks(
    chunks: list[list[LLMMessage]],
    llm: LLM | BaseLLM,
    callbacks: list[TokenCalcHandler],
    i18n: I18N,
) -> list[SummaryContent]:
    """Summarize multiple message chunks concurrently using asyncio.

    Args:
        chunks: List of message chunks to summarize.
        llm: LLM instance (must support ``acall``).
        callbacks: List of callbacks for the LLM.
        i18n: I18N instance for prompt templates.

    Returns:
        Ordered list of summary contents, one per chunk.
    """

    async def _summarize_one(chunk: list[LLMMessage]) -> SummaryContent:
        conversation_text = _format_messages_for_summary(chunk)
        summarization_messages = [
            format_message_for_llm(
                i18n.slice("summarizer_system_message"), role="system"
            ),
            format_message_for_llm(
                i18n.slice("summarize_instruction").format(
                    conversation=conversation_text
                ),
            ),
        ]
        summary = await llm.acall(summarization_messages, callbacks=callbacks)
        extracted = _extract_summary_tags(str(summary))
        return {"content": extracted}

    results = await asyncio.gather(*[_summarize_one(chunk) for chunk in chunks])
    return list(results)


def summarize_messages(
    messages: list[LLMMessage],
    llm: LLM | BaseLLM,
    callbacks: list[TokenCalcHandler],
    i18n: I18N,
    verbose: bool = True,
) -> None:
    """Summarize messages to fit within context window.

    Uses structured context compaction: preserves system messages,
    splits at message boundaries, formats with role labels, and
    produces structured summaries for seamless task continuation.

    Preserves any files attached to user messages and re-attaches them to
    the summarized message. Files from all user messages are merged.

    Args:
        messages: List of messages to summarize (modified in-place)
        llm: LLM instance for summarization
        callbacks: List of callbacks for LLM
        i18n: I18N instance for messages
        verbose: Whether to print progress.
    """
    # 1. Extract & preserve file attachments from user messages
    preserved_files: dict[str, Any] = {}
    for msg in messages:
        if msg.get("role") == "user" and msg.get("files"):
            preserved_files.update(msg["files"])

    # 2. Extract system messages — never summarize them
    system_messages = [m for m in messages if m.get("role") == "system"]
    non_system_messages = [m for m in messages if m.get("role") != "system"]

    # If there are only system messages (or no non-system messages), nothing to summarize
    if not non_system_messages:
        return

    # 3. Split non-system messages into chunks at message boundaries
    max_tokens = llm.get_context_window_size()
    chunks = _split_messages_into_chunks(non_system_messages, max_tokens)

    # 4. Summarize each chunk with role-labeled formatting
    total_chunks = len(chunks)

    if total_chunks <= 1:
        # Single chunk — no benefit from async overhead
        summarized_contents: list[SummaryContent] = []
        for idx, chunk in enumerate(chunks, 1):
            if verbose:
                Printer().print(
                    content=f"Summarizing {idx}/{total_chunks}...",
                    color="yellow",
                )
            conversation_text = _format_messages_for_summary(chunk)
            summarization_messages = [
                format_message_for_llm(
                    i18n.slice("summarizer_system_message"), role="system"
                ),
                format_message_for_llm(
                    i18n.slice("summarize_instruction").format(
                        conversation=conversation_text
                    ),
                ),
            ]
            summary = llm.call(summarization_messages, callbacks=callbacks)
            extracted = _extract_summary_tags(str(summary))
            summarized_contents.append({"content": extracted})
    else:
        # Multiple chunks — summarize in parallel via asyncio
        if verbose:
            Printer().print(
                content=f"Summarizing {total_chunks} chunks in parallel...",
                color="yellow",
            )
        coro = _asummarize_chunks(
            chunks=chunks, llm=llm, callbacks=callbacks, i18n=i18n
        )
        if is_inside_event_loop():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                summarized_contents = pool.submit(asyncio.run, coro).result()
        else:
            summarized_contents = asyncio.run(coro)

    merged_summary = "\n\n".join(content["content"] for content in summarized_contents)

    # 6. Reconstruct messages: [system messages...] + [summary user message]
    messages.clear()
    messages.extend(system_messages)

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


def parse_tool_call_args(
    func_args: dict[str, Any] | str,
    func_name: str,
    call_id: str,
    original_tool: Any = None,
) -> tuple[dict[str, Any], None] | tuple[None, dict[str, Any]]:
    """Parse tool call arguments from a JSON string or dict.

    Returns:
        ``(args_dict, None)`` on success, or ``(None, error_result)`` on
        JSON parse failure where ``error_result`` is a ready-to-return dict
        with the same shape as ``_execute_single_native_tool_call`` return values.
    """
    if isinstance(func_args, str):
        try:
            return json.loads(func_args), None
        except json.JSONDecodeError as e:
            return None, {
                "call_id": call_id,
                "func_name": func_name,
                "result": (
                    f"Error: Failed to parse tool arguments as JSON: {e}. "
                    f"Please provide valid JSON arguments for the '{func_name}' tool."
                ),
                "from_cache": False,
                "original_tool": original_tool,
            }
    return func_args, None


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
