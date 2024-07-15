import re
from typing import Any, Union

from json_repair import repair_json
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from crewai.utilities import I18N

FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = "I did it wrong. Invalid Format: I missed the 'Action:' after 'Thought:'. I will do right next, and don't use a tool I have already used.\n"
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = "I did it wrong. Invalid Format: I missed the 'Action Input:' after 'Action:'. I will do right next, and don't use a tool I have already used.\n"
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = "I did it wrong. Tried to both perform Action and give a Final Answer at the same time, I must do one or the other"


class CrewAgentParser(ReActSingleInputOutputParser):
    """Parses ReAct-style LLM calls that have a single tool input.

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
    """

    _i18n: I18N = I18N()
    agent: Any = None

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )
            action = action_match.group(1)
            clean_action = self._clean_action(action)

            action_input = action_match.group(2).strip()

            tool_input = action_input.strip(" ").strip('"')
            safe_tool_input = self._safe_repair_json(tool_input)

            return AgentAction(clean_action, safe_tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            self.agent.increment_formatting_errors()
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=f"{MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE}\n{self._i18n.slice('final_answer_format')}",
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            self.agent.increment_formatting_errors()
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            format = self._i18n.slice("format_without_tools")
            error = f"{format}"
            self.agent.increment_formatting_errors()
            raise OutputParserException(
                error,
                observation=error,
                llm_output=text,
                send_to_llm=True,
            )

    def _clean_action(self, text: str) -> str:
        """Clean action string by removing non-essential formatting characters."""
        return re.sub(r"^\s*\*+\s*|\s*\*+\s*$", "", text).strip()

    def _safe_repair_json(self, tool_input: str) -> str:
        UNABLE_TO_REPAIR_JSON_RESULTS = ['""', "{}"]

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
