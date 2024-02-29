from typing import Any, Union

from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from crewai.utilities import I18N

TOOL_USAGE_SECTION = "Use Tool:"
FINAL_ANSWER_ACTION = "Final Answer:"
FINAL_ANSWER_AND_TOOL_ERROR_MESSAGE = "I tried to use a tool and give a final answer at the same time, I must choose only one."


class CrewAgentParser(ReActSingleInputOutputParser):
    """
    This class is responsible for parsing the output of the Crew-style Language Learning Model (LLM) calls that have a single tool input.
    The output is expected to be in one of two formats.

    If the output signals that an action should be taken, it should be in the below format. 
    This will result in an AgentAction being returned.
    ```
    Use Tool: All context for using the tool here
    ```

    If the output signals that a final answer should be given, it should be in the below format. 
    This will result in an AgentFinish being returned.
    ```
    Final Answer: The temperature is 100 degrees
    ```
    """

    _i18n: I18N = I18N()  # Internationalization object for multilingual support
    agent: Any = None  # The agent that will use this parser

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        This method parses the given text and returns an AgentAction or AgentFinish object based on the content of the text.

        Parameters:
        text: The text to be parsed.

        Returns:
        An AgentAction object if the text includes a tool usage section, or an AgentFinish object if the text includes a final answer action.
        """

        # Check if the text includes a final answer action or a tool usage section
        includes_answer = FINAL_ANSWER_ACTION in text
        includes_tool = TOOL_USAGE_SECTION in text

        # If the text includes a tool usage section, return an AgentAction object
        if includes_tool:
            # If the text also includes a final answer action, increment the number of formatting errors and raise an OutputParserException
            if includes_answer:
                self.agent.increment_formatting_errors()
                raise OutputParserException(f"{FINAL_ANSWER_AND_TOOL_ERROR_MESSAGE}")

            return AgentAction("", "", text)

        # If the text includes a final answer action, return an AgentFinish object
        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        # If the text does not include a tool usage section or a final answer action, increment the number of formatting errors and raise an OutputParserException
        format = self._i18n.slice("format_without_tools")
        error = f"{format}"
        self.agent.increment_formatting_errors()
        raise OutputParserException(
            error,
            observation=error,
            llm_output=text,
            send_to_llm=True,
        )
