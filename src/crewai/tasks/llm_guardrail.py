from typing import Any, Tuple

from crewai.task import Task
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.printer import Printer


class LLMGuardrailTask:
    def __init__(
        self, description: str, task: Task | None = None, run_unsafe: bool = False
    ):
        self.description = description
        self.task = task
        self.run_unsafe = run_unsafe

    def generate_code(self, task_output: TaskOutput) -> str:
        response = self.task.agent.llm.call(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a expert Python developer"
                        "You **must strictly** follow the task description, use the provided raw output as the input in your code. "
                        "Your code must:\n"
                        "1. Return results with: print((True, data)) on success, or print((False, 'very detailed error message')) on failure.\n"
                        "2. Use the literal string of the task output (already included in your input) if needed.\n"
                        "3. Generate the code **following strictly** the task description.\n"
                        "4. Be valid Python 3 — executable as-is.\n"
                        "5. DO NOT wrap the output in markdown or use triple backticks. Return only raw Python code."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""Based on the task description below, generate Python 3 code that validates the task output.

Task description:
{self.description}

Here is the raw output from the task:
'{task_output.raw}'

Use this exact string literal inside your generated code (do not reference variables like task_output.raw).

Now generate Python code that follows the instructions above.""",
                },
            ]
        )
        return response

    def __call__(self, task_output: TaskOutput) -> Tuple[bool, Any]:
        import ast

        from crewai_tools import CodeInterpreterTool

        code = self.generate_code(task_output)

        printer = Printer()
        printer.print(
            content=f"Executing the following code to validate the task output:\n{code}\n",
            color="cyan",
        )

        result = CodeInterpreterTool(code=code).run()

        if "Something went wrong while running the code" in result:
            return False, result

        return ast.literal_eval(result)
