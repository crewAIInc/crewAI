"""Agent output parsing module for ReAct-style LLM responses.

This module provides parsing functionality for agent outputs that follow
the ReAct (Reasoning and Acting) format, converting them into structured
AgentAction or AgentFinish objects.
"""

from dataclasses import dataclass
import re

from json_repair import repair_json  # type: ignore[import-untyped]
from pydantic import BaseModel

from crewai.agents.constants import (
    ACTION_INPUT_ONLY_REGEX,
    ACTION_INPUT_REGEX,
    ACTION_REGEX,
    FINAL_ANSWER_ACTION,
    MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
    MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
    UNABLE_TO_REPAIR_JSON_RESULTS,
)
from crewai.utilities.i18n import get_i18n


_I18N = get_i18n()


@dataclass
class AgentAction:
    """Represents an action to be taken by an agent."""

    thought: str
    tool: str
    tool_input: str
    text: str
    result: str | None = None


@dataclass
class AgentFinish:
    """Represents the final answer from an agent."""

    thought: str
    output: str | BaseModel
    text: str


class OutputParserError(Exception):
    """Exception raised when output parsing fails.

    Attributes:
        error: The error message.
    """

    def __init__(self, error: str) -> None:
        """Initialize OutputParserError.

        Args:
            error: The error message.
        """
        self.error = error
        super().__init__(error)


def parse(text: str) -> AgentAction | AgentFinish:
    """Parse agent output text into AgentAction or AgentFinish.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    Thought: agent thought here
    Action: search
    Action Input: what is the temperature in SF?

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    Thought: agent thought here
    Final Answer: The temperature is 100 degrees

    Args:
        text: The agent output text to parse.

    Returns:
        AgentAction or AgentFinish based on the content.

    Raises:
        OutputParserError: If the text format is invalid.
    """
    thought = _extract_thought(text)
    includes_answer = FINAL_ANSWER_ACTION in text
    action_match = ACTION_INPUT_REGEX.search(text)

    if includes_answer:
        final_answer = text.split(FINAL_ANSWER_ACTION)[-1].strip()
        final_answer = _strip_trailing_react_blocks(final_answer)
        # Check whether the final answer ends with triple backticks.
        if final_answer.endswith("```"):
            # Count occurrences of triple backticks in the final answer.
            count = final_answer.count("```")
            # If count is odd then it's an unmatched trailing set; remove it.
            if count % 2 != 0:
                final_answer = final_answer[:-3].rstrip()
        return AgentFinish(thought=thought, output=final_answer, text=text)

    if action_match:
        action = action_match.group(1)
        clean_action = _clean_action(action)

        action_input = action_match.group(2).strip()

        tool_input = action_input.strip(" ").strip('"')
        safe_tool_input = _safe_repair_json(tool_input)

        return AgentAction(
            thought=thought, tool=clean_action, tool_input=safe_tool_input, text=text
        )

    if not ACTION_REGEX.search(text):
        raise OutputParserError(
            f"{MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE}\n{_I18N.slice('final_answer_format')}",
        )
    if not ACTION_INPUT_ONLY_REGEX.search(text):
        raise OutputParserError(
            MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
        )
    err_format = _I18N.slice("format_without_tools")
    error = f"{err_format}"
    raise OutputParserError(
        error,
    )


def _extract_thought(text: str) -> str:
    """Extract the thought portion from the text.

    Args:
        text: The full agent output text.

    Returns:
        The extracted thought string.
    """
    thought_index = text.find("\nAction")
    if thought_index == -1:
        thought_index = text.find("\nFinal Answer")
    if thought_index == -1:
        return ""
    thought = text[:thought_index].strip()
    # Remove any triple backticks from the thought string
    return thought.replace("```", "").strip()


def _clean_action(text: str) -> str:
    """Clean action string by removing non-essential formatting characters.

    Args:
        text: The action text to clean.

    Returns:
        The cleaned action string.
    """
    return text.strip().strip("*").strip()


def _safe_repair_json(tool_input: str) -> str:
    """Safely repair JSON input.

    Args:
        tool_input: The tool input string to repair.

    Returns:
        The repaired JSON string or original if repair fails.
    """
    # Skip repair if the input starts and ends with square brackets
    # Explanation: The JSON parser has issues handling inputs that are enclosed in square brackets ('[]').
    # These are typically valid JSON arrays or strings that do not require repair. Attempting to repair such inputs
    # might lead to unintended alterations, such as wrapping the entire input in additional layers or modifying
    # the structure in a way that changes its meaning. By skipping the repair for inputs that start and end with
    # square brackets, we preserve the integrity of these valid JSON structures and avoid unnecessary modifications.
    if tool_input.startswith("[") and tool_input.endswith("]"):
        return tool_input

    # Before repair, handle common LLM issues:
    # 1. Replace """ with " to avoid JSON parser errors

    tool_input = tool_input.replace('"""', '"')

    result = repair_json(tool_input)
    if result in UNABLE_TO_REPAIR_JSON_RESULTS:
        return tool_input

    return str(result)


_CODE_SPAN_RE: re.Pattern[str] = re.compile(
    r"(?is)<pre><code>.*?</code></pre>|<code>.*?</code>"
)
_TRAILING_REACT_MARKER_RE: re.Pattern[str] = re.compile(
    r"(?is)(?:^|[\r\n]|>)[ \t]*(Thought:|Action:|Action Input:|Observation:)"
)


def _strip_trailing_react_blocks(final_answer: str) -> str:
    """Strip accidental ReAct blocks appended after a valid Final Answer.

    Some models will output a valid `Final Answer:` segment and then append
    `Thought:` / `Action:` / `Action Input:` / `Observation:`. Those blocks are
    control text and should never be included in the user-facing answer.
    """
    if not final_answer:
        return final_answer

    code_spans = [(m.start(), m.end()) for m in _CODE_SPAN_RE.finditer(final_answer)]

    def _is_in_code(index: int) -> bool:
        return any(start <= index < end for start, end in code_spans)

    for match in _TRAILING_REACT_MARKER_RE.finditer(final_answer):
        marker_index = match.start(1)
        if _is_in_code(marker_index):
            continue

        # Reduce false positives: only strip if the remaining tail looks like
        # a real ReAct control block (e.g., includes Action Input).
        tail = final_answer[marker_index:]
        marker = match.group(1)
        if marker.startswith("Thought:") and ("Action:" not in tail and "Action Input:" not in tail):
            continue
        if marker.startswith("Action:") and "Action Input:" not in tail:
            continue

        return final_answer[:marker_index].strip()

    return final_answer
