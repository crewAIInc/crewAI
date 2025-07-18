import re
from typing import Any, Optional, Union

from json_repair import repair_json

from crewai.utilities import I18N

FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = "I did it wrong. Invalid Format: I missed the 'Action:' after 'Thought:'. I will do right next, and don't use a tool I have already used.\n"
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = "I did it wrong. Invalid Format: I missed the 'Action Input:' after 'Action:'. I will do right next, and don't use a tool I have already used.\n"
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = "I did it wrong. Tried to both perform Action and give a Final Answer at the same time, I must do one or the other"


class AgentAction:
    thought: str
    tool: str
    tool_input: str
    text: str
    result: str

    def __init__(self, thought: str, tool: str, tool_input: str, text: str):
        self.thought = thought
        self.tool = tool
        self.tool_input = tool_input
        self.text = text


class AgentFinish:
    thought: str
    output: str
    text: str

    def __init__(self, thought: str, output: str, text: str):
        self.thought = thought
        self.output = output
        self.text = text


class OutputParserException(Exception):
    error: str

    def __init__(self, error: str):
        self.error = error


class CrewAgentParser:
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

    def __init__(self, agent: Optional[Any] = None):
        self.agent = agent

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        cleaned_text = self._clean_agent_observations(text)

        thought = self._extract_thought(text)
        includes_answer = FINAL_ANSWER_ACTION in text
        action_match = self._find_last_action_input_pair(cleaned_text)

        # Prevent tool bypassing when both Action and Final Answer are present
        # If the model returns both, we PRIORITIZE the action to force tool execution
        if includes_answer and action_match:
            return self._create_agent_action(thought, action_match, cleaned_text)

        elif includes_answer:
            final_answer = cleaned_text.split(FINAL_ANSWER_ACTION)[-1].strip()
            # Check whether the final answer ends with triple backticks.
            if final_answer.endswith("```"):
                # Count occurrences of triple backticks in the final answer.
                count = final_answer.count("```")
                # If count is odd then it's an unmatched trailing set; remove it.
                if count % 2 != 0:
                    final_answer = final_answer[:-3].rstrip()
            return AgentFinish(thought, final_answer, cleaned_text)

        elif action_match:
            return self._create_agent_action(thought, action_match, cleaned_text)

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", cleaned_text, re.DOTALL):
            raise OutputParserException(
                f"{MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE}\n{self._i18n.slice('final_answer_format')}",
            )
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", cleaned_text, re.DOTALL
        ):
            raise OutputParserException(
                MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
            )
        else:
            format = self._i18n.slice("format_without_tools")
            error = f"{format}"
            raise OutputParserException(
                error,
            )

    def _extract_thought(self, text: str) -> str:
        thought_index = text.find("\nAction")
        if thought_index == -1:
            thought_index = text.find("\nFinal Answer")
        if thought_index == -1:
            return ""
        thought = text[:thought_index].strip()
        # Remove any triple backticks from the thought string
        thought = thought.replace("```", "").strip()
        return thought

    def _clean_action(self, text: str) -> str:
        """Clean action string by removing non-essential formatting characters."""
        return text.strip().strip("*").strip()

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

    def _create_agent_action(self, thought: str, action_match: dict, cleaned_text: str) -> AgentAction:
        action = action_match["action"]
        clean_action = self._clean_action(action)
        action_input = action_match["input"]

        tool_input = action_input.strip(" ").strip('"')
        safe_tool_input = self._safe_repair_json(tool_input)

        return AgentAction(thought, clean_action, safe_tool_input, cleaned_text)

    def _find_last_action_input_pair(self, text: str) -> Optional[dict]:
        """
        Finds the last complete Action / Action Input pair in the given text.
        Useful for handling multiple action/observation cycles.
        """
        def _match_all_pairs(text: str) -> list[tuple[str, str]]:
            pattern = (
                r"Action\s*\d*\s*:\s*([^\n]+)"                            # Action content
                r"\s*[\n]+"                                               # Optional whitespace/newline
                r"Action\s*\d*\s*Input\s*\d*\s*:\s*"                      # Action Input label
                r"([^\n]*(?:\n(?!Observation:|Thought:|Action\s*\d*\s*:|Final Answer:)[^\n]*)*)"
            )
            return re.findall(pattern, text, re.MULTILINE | re.DOTALL)

        def _match_fallback_pair(text: str) -> Optional[dict]:
            fallback_pattern = (
                r"Action\s*\d*\s*:\s*(.*?)"
                r"\s*Action\s*\d*\s*Input\s*\d*\s*:\s*"
                r"(.*?)(?=\n(?:Observation:|Thought:|Action\s*\d*\s*:|Final Answer:)|$)"
            )
            match = re.search(fallback_pattern, text, re.DOTALL)
            if match:
                return {
                    "action": match.group(1).strip(),
                    "input": match.group(2).strip()
                }
            return None

        matches = _match_all_pairs(text)
        if matches:
            last_action, last_input = matches[-1]
            return {
                "action": last_action.strip(),
                "input": last_input.strip()
            }

        return _match_fallback_pair(text)


    def _clean_agent_observations(self, text: str) -> str:
        # Pattern: capture Action/Input lines, then Observation block until next Thought or end-of-string
        obs_pattern = re.compile(
            r'^(\s*Action:.*\n\s*Action Input:.*\n)'   # group 1: Action + Action Input
            r'\s*Observation:.*?(?=(?:\n\s*Thought:|\Z))',  # non-greedy until Thought: or end-of-string
            flags=re.DOTALL | re.MULTILINE
        )

        if obs_pattern.search(text):
            text = obs_pattern.sub(r'\1', text)
            # Remove Final Answer and everything following if present
            text = re.sub(r'\n\s*Final\s+Answer:.*', '', text, flags=re.DOTALL | re.MULTILINE)
            # Normalize blank lines
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text).strip()
        return text
