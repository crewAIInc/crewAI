from langchain_core.exceptions import OutputParserException


class TaskRepeatedUsageException(OutputParserException):
    """Exception raised when a task is used twice in a roll."""

    error: str = "TaskRepeatedUsageException"
    message: str = "I just used the {tool} tool with input {tool_input}. So I already know the result of that and don't need to use it now.\n"

    def __init__(self, tool: str, tool_input: str, text: str):
        self.text = text
        self.tool = tool
        self.tool_input = tool_input
        self.message = self.message.format(tool=tool, tool_input=tool_input)

        super().__init__(
            error=self.error,
            observation=self.message,
            send_to_llm=True,
            llm_output=self.text,
        )

    def __str__(self):
        return self.message
