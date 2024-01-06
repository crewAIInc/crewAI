import re
from typing import Union

from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish

from .cache import CacheHandler, CacheHit
from .exceptions import TaskRepeatedUsageException
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

    def parse(self, text: str) -> Union[AgentAction, AgentFinish, CacheHit]:
        FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
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
                    raise TaskRepeatedUsageException(
                        tool=action, tool_input=tool_input, text=text
                    )

            result = self.cache.read(action, tool_input)
            if result:
                action = AgentAction(action, tool_input, text)
                return CacheHit(action=action, cache=self.cache)

        return super().parse(text)
