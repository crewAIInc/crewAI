"""Constants for agent-related modules."""

import re
from typing import Final

# crewai.agents.parser constants

FINAL_ANSWER_ACTION: Final[str] = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE: Final[str] = (
    "I did it wrong. Invalid Format: I missed the 'Action:' after 'Thought:'. I will do right next, and don't use a tool I have already used.\n"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE: Final[str] = (
    "I did it wrong. Invalid Format: I missed the 'Action Input:' after 'Action:'. I will do right next, and don't use a tool I have already used.\n"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE: Final[str] = (
    "I did it wrong. Tried to both perform Action and give a Final Answer at the same time, I must do one or the other"
)
UNABLE_TO_REPAIR_JSON_RESULTS: Final[list[str]] = ['""', "{}"]
ACTION_INPUT_REGEX: Final[re.Pattern[str]] = re.compile(
    r"Action\s*\d*\s*:\s*(.*?)\s*Action\s*\d*\s*Input\s*\d*\s*:\s*(.*)", re.DOTALL
)
ACTION_REGEX: Final[re.Pattern[str]] = re.compile(
    r"Action\s*\d*\s*:\s*(.*?)", re.DOTALL
)
ACTION_INPUT_ONLY_REGEX: Final[re.Pattern[str]] = re.compile(
    r"\s*Action\s*\d*\s*Input\s*\d*\s*:\s*(.*)", re.DOTALL
)

HARMONY_START_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"<\|start\|>assistant<\|channel\|>(\w+)(?:\s+to=(\w+))?<\|message\|>(.*?)<\|(?:end|call)\|>", 
    re.DOTALL
)
HARMONY_ANALYSIS_CHANNEL: Final[str] = "analysis"
HARMONY_COMMENTARY_CHANNEL: Final[str] = "commentary"
HARMONY_FINAL_ANSWER_ERROR_MESSAGE: Final[str] = (
    "I did it wrong. Invalid Harmony Format: I need to use proper channel structure."
)
HARMONY_MISSING_CONTENT_ERROR_MESSAGE: Final[str] = (
    "I did it wrong. Invalid Harmony Format: Missing content in message section."
)
