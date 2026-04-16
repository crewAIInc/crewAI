"""Agent output parsing module for ReAct-style LLM responses.

This module provides parsing functionality for agent outputs that follow
the ReAct (Reasoning and Acting) format, converting them into structured
AgentAction or AgentFinish objects.
"""

from dataclasses import dataclass

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
from crewai.utilities.i18n import I18N_DEFAULT as _I18N


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
        if final_answer.endswith("```"):
            count = final_answer.count("```")
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
    if tool_input.startswith("[") and tool_input.endswith("]"):
        return tool_input

    tool_input = tool_input.replace('"""', '"')

    result = repair_json(tool_input)
    if result in UNABLE_TO_REPAIR_JSON_RESULTS:
        return tool_input

    return str(result)
