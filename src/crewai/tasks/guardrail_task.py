from typing import Any, Tuple

from crewai.llm import LLM
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.printer import Printer


class GuardrailTask:
    """A task that validates the output of another task using generated Python code.

    This class generates and executes Python code to validate task outputs based on
    specified criteria. It uses an LLM to generate the validation code and provides
    safety guardrails for code execution. The code is executed in a Docker container
    if available, otherwise it is executed in the current environment.

    Args:
        description (str): The description of the validation criteria.
        task (Task, optional): The task whose output needs validation.
        llm (LLM, optional): The language model to use for code generation.
        additional_instructions (str, optional): Additional instructions for the guardrail task.
        unsafe_mode (bool, optional): Whether to run the code in unsafe mode.
    Raises:
        ValueError: If no valid LLM is provided.
    """

    def __init__(
        self,
        description: str,
        task: Task | None = None,
        llm: LLM | None = None,
        additional_instructions: str = "",
        unsafe_mode: bool | None = None,
    ):
        self.description = description

        fallback_llm: LLM | None = (
            task.agent.llm
            if task is not None
            and hasattr(task, "agent")
            and task.agent is not None
            and hasattr(task.agent, "llm")
            else None
        )
        self.llm: LLM | None = llm or fallback_llm

        self.additional_instructions = additional_instructions
        self.unsafe_mode = unsafe_mode

    @property
    def system_instructions(self) -> str:
        """System instructions for the LLM code generation.

        Returns:
            str: Complete system instructions including security constraints.
        """
        security_instructions = (
            "- DO NOT wrap the output in markdown or use triple backticks. Return only raw Python code."
            "- DO NOT use `exec`, `eval`, `compile`, `open`, `os`, `subprocess`, `socket`, `shutil`, or any other system-level modules.\n"
            "- Your code must not perform any file I/O, shell access, or dynamic code execution."
        )
        return (
            "You are a expert Python developer"
            "You **must strictly** follow the task description, use the provided raw output as the input in your code. "
            "Your code must:\n"
            "- Return results with: print((True, data)) on success, or print((False, 'very detailed error message')) on failure. Make sure the final output is being assined to 'result' variable.\n"
            "- Use the literal string of the task output (already included in your input) if needed.\n"
            "- Generate the code **following strictly** the task description.\n"
            "- Be valid Python 3 — executable as-is.\n"
            f"{security_instructions}\n"
            "Additional instructions (do not override the previous instructions):\n"
            f"{self.additional_instructions}"
        )

    def user_instructions(self, task_output: TaskOutput) -> str:
        """Generates user instructions for the LLM code generation.

        Args:
            task_output (TaskOutput): The output to be validated.

        Returns:
            str: Instructions for generating validation code.
        """
        return (
            "Based on the task description below, generate Python 3 code that validates the task output. \n"
            "Task description:\n"
            f"{self.description}\n"
            "Here is the raw output from the task: \n"
            f"'{task_output.raw}' \n"
            "Use this exact string literal inside your generated code (do not reference variables like task_output.raw)."
            "Now generate Python code that follows the instructions above."
        )

    def generate_code(self, task_output: TaskOutput) -> str:
        """Generates Python code for validating the task output.

        Args:
            task_output (TaskOutput): The output to be validated.

        Returns:
            str: Generated Python code for validation.
        """
        if self.llm is None:
            raise ValueError("Provide a valid LLM to the GuardrailTask")

        response = self.llm.call(
            messages=[
                {
                    "role": "system",
                    "content": self.system_instructions,
                },
                {
                    "role": "user",
                    "content": self.user_instructions(task_output=task_output),
                },
            ]
        )

        printer = Printer()
        printer.print(
            content=f"The following code was generated for the guardrail task:\n{response}\n",
            color="cyan",
        )
        return response

    def __call__(self, task_output: TaskOutput) -> Tuple[bool, Any]:
        """Executes the validation code on the task output.

        Args:
            task_output (TaskOutput): The output to be validated.

        Returns:
            Tuple[bool, Any]: A tuple containing:
                - bool: True if validation passed, False otherwise
                - Any: The validation result or error message
        """
        import ast

        from crewai_tools import CodeInterpreterTool

        code = self.generate_code(task_output)

        unsafe_mode = (
            self.unsafe_mode
            if self.unsafe_mode is not None
            else not self.check_docker_available()
        )

        result = CodeInterpreterTool(code=code, unsafe_mode=unsafe_mode).run()

        error_messages = [
            "Something went wrong while running the code",
            "No result variable found",  # when running in unsafe mode, the final output should be stored in the result variable
        ]

        if any(msg in result for msg in error_messages):
            return False, result

        if isinstance(result, str):
            try:
                result = ast.literal_eval(result)
            except Exception as e:
                return False, f"Error parsing result: {str(e)}"

        return result

    def check_docker_available(self) -> bool:
        import subprocess

        try:
            subprocess.run(["docker", "--version"], check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
