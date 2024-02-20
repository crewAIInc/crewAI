from typing import Union

from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from crewai.utilities import I18N

TOOL_USAGE_SECTION = "Use Tool:"
FINAL_ANSWER_ACTION = "Final Answer:"
FINAL_ANSWER_AND_TOOL_ERROR_MESSAGE = "I tried to use a tool and give a final answer at the same time, I must choose only one."


class CrewAgentParser(ReActSingleInputOutputParser):
    """Parses Crew-style LLM calls that have a single tool input.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    Use Tool: All context for using the tool here
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    Final Answer: The temperature is 100 degrees
    ```
    """

    _i18n: I18N = I18N()

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        includes_tool = TOOL_USAGE_SECTION in text

        if includes_tool:
            if includes_answer:
                raise OutputParserException(f"{FINAL_ANSWER_AND_TOOL_ERROR_MESSAGE}")

            return AgentAction("", "", text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        format = self._i18n.slice("format_without_tools")
        error = f"{format}"
        raise OutputParserException(
            error,
            observation=error,
            llm_output=text,
            send_to_llm=True,
        )
