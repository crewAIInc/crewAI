"""Agent output parsing module for multiple LLM response formats.

This module provides parsing functionality for agent outputs that follow
different formats (ReAct, OpenAI Harmony, etc.), converting them into structured
AgentAction or AgentFinish objects with automatic format detection.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from json_repair import repair_json

from crewai.agents.constants import (
    ACTION_INPUT_ONLY_REGEX,
    ACTION_INPUT_REGEX,
    ACTION_REGEX,
    FINAL_ANSWER_ACTION,
    HARMONY_ANALYSIS_CHANNEL,
    HARMONY_COMMENTARY_CHANNEL,
    HARMONY_FINAL_ANSWER_ERROR_MESSAGE,
    HARMONY_MISSING_CONTENT_ERROR_MESSAGE,
    HARMONY_START_PATTERN,
    MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
    MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
    UNABLE_TO_REPAIR_JSON_RESULTS,
)
from crewai.utilities import I18N

_I18N = I18N()


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
    output: str
    text: str


class OutputParserException(Exception):
    """Exception raised when output parsing fails.

    Attributes:
        error: The error message.
    """

    def __init__(self, error: str) -> None:
        """Initialize OutputParserException.

        Args:
            error: The error message.
        """
        self.error = error
        super().__init__(error)


class BaseOutputParser(ABC):
    """Abstract base class for output parsers."""

    @abstractmethod
    def can_parse(self, text: str) -> bool:
        """Check if this parser can handle the given text format."""

    @abstractmethod
    def parse_text(self, text: str) -> AgentAction | AgentFinish:
        """Parse the text into AgentAction or AgentFinish."""


class OutputFormatRegistry:
    """Registry for managing different output format parsers."""

    def __init__(self):
        self._parsers: dict[str, BaseOutputParser] = {}

    def register(self, name: str, parser: BaseOutputParser) -> None:
        """Register a parser for a specific format."""
        self._parsers[name] = parser

    def detect_and_parse(self, text: str) -> AgentAction | AgentFinish:
        """Automatically detect format and parse with appropriate parser."""
        for parser in self._parsers.values():
            if parser.can_parse(text):
                return parser.parse_text(text)

        return self._parsers.get('react', ReActParser()).parse_text(text)


class ReActParser(BaseOutputParser):
    """Parser for ReAct format outputs."""

    def can_parse(self, text: str) -> bool:
        """Check if text follows ReAct format."""
        return (
            FINAL_ANSWER_ACTION in text or
            ACTION_INPUT_REGEX.search(text) is not None or
            ACTION_REGEX.search(text) is not None
        )

    def parse_text(self, text: str) -> AgentAction | AgentFinish:
        """Parse ReAct format text."""
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
            raise OutputParserException(
                f"{MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE}\n{_I18N.slice('final_answer_format')}",
            )
        if not ACTION_INPUT_ONLY_REGEX.search(text):
            raise OutputParserException(
                MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
            )
        err_format = _I18N.slice("format_without_tools")
        error = f"{err_format}"
        raise OutputParserException(error)


class HarmonyParser(BaseOutputParser):
    """Parser for OpenAI Harmony format outputs."""

    def can_parse(self, text: str) -> bool:
        """Check if text follows OpenAI Harmony format."""
        return HARMONY_START_PATTERN.search(text) is not None

    def parse_text(self, text: str) -> AgentAction | AgentFinish:
        """Parse OpenAI Harmony format text."""
        matches = HARMONY_START_PATTERN.findall(text)

        if not matches:
            raise OutputParserException(HARMONY_MISSING_CONTENT_ERROR_MESSAGE)

        channel, tool_name, content = matches[-1]
        content = content.strip()

        if channel == HARMONY_ANALYSIS_CHANNEL:
            return AgentFinish(
                thought=f"Analysis: {content}",
                output=content,
                text=text
            )

        if channel == HARMONY_COMMENTARY_CHANNEL and tool_name:
            thought_content = content
            tool_input = content

            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    tool_input = json_match.group(0)
                    thought_content = content[:json_match.start()].strip()
                    if not thought_content:
                        thought_content = f"Using tool {tool_name}"
            except Exception:
                tool_input = content

            safe_tool_input = _safe_repair_json(tool_input)

            return AgentAction(
                thought=thought_content,
                tool=tool_name,
                tool_input=safe_tool_input,
                text=text
            )

        raise OutputParserException(HARMONY_FINAL_ANSWER_ERROR_MESSAGE)


_format_registry = OutputFormatRegistry()
_format_registry.register('react', ReActParser())
_format_registry.register('harmony', HarmonyParser())


def parse(text: str) -> AgentAction | AgentFinish:
    """Parse agent output text into AgentAction or AgentFinish.

    Automatically detects the format (ReAct, OpenAI Harmony, etc.) and uses
    the appropriate parser. Maintains backward compatibility with existing ReAct format.

    Supports multiple formats:

    ReAct format:
    Thought: agent thought here
    Action: search
    Action Input: what is the temperature in SF?

    Or for final answers:
    Thought: agent thought here
    Final Answer: The temperature is 100 degrees

    OpenAI Harmony format:
    <|start|>assistant<|channel|>analysis<|message|>The temperature is 100 degrees<|end|>

    Or for tool actions:
    <|start|>assistant<|channel|>commentary to=search<|message|>{"query": "temperature in SF"}<|call|>

    Args:
        text: The agent output text to parse.

    Returns:
        AgentAction or AgentFinish based on the content.

    Raises:
        OutputParserException: If the text format is invalid or unsupported.
    """
    return _format_registry.detect_and_parse(text)


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
