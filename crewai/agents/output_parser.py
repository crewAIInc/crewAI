import re
from typing import Union

from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from .cache_handler import CacheHandler
from .tools_handler import ToolsHandler

FINAL_ANSWER_ACTION = "Final Answer:"
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)


class CrewAgentOutputParser(ReActSingleInputOutputParser):
    """Parses ReAct-style LLM calls that have a single tool input.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    Thought: agent thought here
    Action: search
    Action Input: what is the temperature in SF?
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    Thought: agent thought here
    Final Answer: The temperature is 100 degrees
    ```

    It also prevents tools from being reused in a roll.
    """

    class Config:
        arbitrary_types_allowed = True

    tools_handler: ToolsHandler
    cache: CacheHandler

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
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            tool_input = tool_input.strip('"')

            last_tool_usage = self.tools_handler.last_used_tool
            if last_tool_usage:
                usage = {
                    "tool": action,
                    "input": tool_input,
                }
                if usage == last_tool_usage:
                    raise OutputParserException(
                        f"""\nI just used the {action} tool with input {tool_input}. So I already know the result of that."""
                    )

            result = self.cache.read(action, tool_input)
            if result:
                return AgentFinish({"output": result}, text)

        return super().parse(text)
