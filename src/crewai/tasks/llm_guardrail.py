from typing import Any, Tuple

from pydantic import BaseModel, Field

from crewai.agent import Agent, LiteAgentOutput
from crewai.llm import BaseLLM
from crewai.tasks.task_output import TaskOutput


class LLMGuardrailResult(BaseModel):
    valid: bool = Field(
        description="Whether the task output complies with the guardrail"
    )
    feedback: str | None = Field(
        description="A feedback about the task output if it is not valid",
        default=None,
    )


class LLMGuardrail:
    """It validates the output of another task using an LLM.

    This class is used to validate the output from a Task based on specified criteria.
    It uses an LLM to validate the output and provides a feedback if the output is not valid.

    Args:
        description (str): The description of the validation criteria.
        llm (LLM, optional): The language model to use for code generation.
    """

    def __init__(
        self,
        description: str,
        llm: BaseLLM,
    ):
        self.description = description

        self.llm: BaseLLM = llm

    def _validate_output(self, task_output: TaskOutput) -> LiteAgentOutput:
        agent = Agent(
            role="Guardrail Agent",
            goal="Validate the output of the task",
            backstory="You are a expert at validating the output of a task. By providing effective feedback if the output is not valid.",
            llm=self.llm,
        )

        query = f"""
        Ensure the following task result complies with the given guardrail.

        Task result:
        {task_output.raw}

        Guardrail:
        {self.description}
        
        Your task:
        - Confirm if the Task result complies with the guardrail.
        - If not, provide clear feedback explaining what is wrong (e.g., by how much it violates the rule, or what specific part fails).
        - Focus only on identifying issues — do not propose corrections.
        - If the Task result complies with the guardrail, saying that is valid
        """

        result = agent.kickoff(query, response_format=LLMGuardrailResult)

        return result

    def __call__(self, task_output: TaskOutput) -> Tuple[bool, Any]:
        """Validates the output of a task based on specified criteria.

        Args:
            task_output (TaskOutput): The output to be validated.

        Returns:
            Tuple[bool, Any]: A tuple containing:
                - bool: True if validation passed, False otherwise
                - Any: The validation result or error message
        """

        try:
            result = self._validate_output(task_output)
            assert isinstance(
                result.pydantic, LLMGuardrailResult
            ), "The guardrail result is not a valid pydantic model"

            if result.pydantic.valid:
                return True, task_output.raw
            else:
                return False, result.pydantic.feedback
        except Exception as e:
            return False, f"Error while validating the task output: {str(e)}"
