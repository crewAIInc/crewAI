"""Hallucination Guardrail Placeholder for CrewAI.

This is a no-op version of the HallucinationGuardrail for the open-source repository.

Classes:
    HallucinationGuardrail: Placeholder guardrail that validates task outputs.
"""

from typing import Any

from crewai.llm import LLM
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.logger import Logger


class HallucinationGuardrail:
    """Placeholder for the HallucinationGuardrail feature.

    Attributes:
        context: The reference context that outputs would be checked against.
        llm: The language model that would be used for evaluation.
        threshold: Optional minimum faithfulness score that would be required to pass.
        tool_response: Optional tool response information that would be used in evaluation.

    Examples:
        >>> # Basic usage with default verdict logic
        >>> guardrail = HallucinationGuardrail(
        ...     context="AI helps with various tasks including analysis and generation.",
        ...     llm=agent.llm,
        ... )

        >>> # With custom threshold for stricter validation
        >>> strict_guardrail = HallucinationGuardrail(
        ...     context="Quantum computing uses qubits in superposition.",
        ...     llm=agent.llm,
        ...     threshold=8.0,  # Would require score >= 8 to pass in enterprise version
        ... )

        >>> # With tool response for additional context
        >>> guardrail_with_tools = HallucinationGuardrail(
        ...     context="The current weather data",
        ...     llm=agent.llm,
        ...     tool_response="Weather API returned: Temperature 22°C, Humidity 65%",
        ... )
    """

    def __init__(
        self,
        context: str,
        llm: LLM,
        threshold: float | None = None,
        tool_response: str = "",
    ):
        """Initialize the HallucinationGuardrail placeholder.

        Args:
            context: The reference context that outputs would be checked against.
            llm: The language model that would be used for evaluation.
            threshold: Optional minimum faithfulness score that would be required to pass.
            tool_response: Optional tool response information that would be used in evaluation.
        """
        self.context = context
        self.llm: LLM = llm
        self.threshold = threshold
        self.tool_response = tool_response
        self._logger = Logger(verbose=True)
        self._logger.log(
            "warning",
            """Hallucination detection is a no-op in open source, use it for free at https://app.crewai.com\n""",
            color="red",
        )

    @property
    def description(self) -> str:
        """Generate a description of this guardrail for event logging."""
        return "HallucinationGuardrail (no-op)"

    def __call__(self, task_output: TaskOutput) -> tuple[bool, Any]:
        """Validate a task output against hallucination criteria.

        In the open source, this method always returns that the output is valid.

        Args:
            task_output: The output to be validated.

        Returns:
            A tuple containing:
                - True
                - The raw task output
        """
        self._logger.log(
            "warning",
            "Premium hallucination detection skipped (use for free at https://app.crewai.com)\n",
            color="red",
        )
        return True, task_output.raw
