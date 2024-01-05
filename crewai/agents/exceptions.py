from langchain_core.exceptions import OutputParserException


class TaskRepeatedUsageException(OutputParserException):
    """Exception raised when a task is used twice in a roll."""

    error: str = "TaskRepeatedUsageException"
    message: str = "\nI just used the {tool} tool with input {tool_input}. So I already know the result of that.\n"

    def __init__(self, tool: str, tool_input: str):
        self.tool = tool
        self.tool_input = tool_input
        self.message = self.message.format(tool=tool, tool_input=tool_input)

        super().__init__(
            error=self.error, observation=self.message, send_to_llm=True, llm_output=""
        )

    def __str__(self):
        return self.message
