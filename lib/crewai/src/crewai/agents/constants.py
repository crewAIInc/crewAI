"""Constanten voor agent-gerelateerde modules."""

import re
from typing import Final


# crewai.agents.parser constanten

FINAL_ANSWER_ACTION: Final[str] = "Eindantwoord:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE: Final[str] = (
    "Ik deed het fout. Ongeldig Formaat: Ik miste de 'Actie:' na 'Gedachte:'. Ik doe het de volgende keer goed.\n"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE: Final[str] = (
    "Ik deed het fout. Ongeldig Formaat: Ik miste de 'Actie Input:' na 'Actie:'. Ik doe het de volgende keer goed.\n"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE: Final[str] = (
    "Ik deed het fout. Probeerde zowel een Actie als een Eindantwoord te geven tegelijk, ik moet een van beide doen."
)
UNABLE_TO_REPAIR_JSON_RESULTS: Final[list[str]] = ['""', "{}"]
ACTION_INPUT_REGEX: Final[re.Pattern[str]] = re.compile(
    r"Actie\s*\d*\s*:\s*(.*?)\s*Actie\s*\d*\s*Input\s*\d*\s*:\s*(.*)", re.DOTALL
)
ACTION_REGEX: Final[re.Pattern[str]] = re.compile(
    r"Actie\s*\d*\s*:\s*(.*?)", re.DOTALL
)
ACTION_INPUT_ONLY_REGEX: Final[re.Pattern[str]] = re.compile(
    r"\s*Actie\s*\d*\s*Input\s*\d*\s*:\s*(.*)", re.DOTALL
)
