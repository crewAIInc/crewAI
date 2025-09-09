"""AI Suite LLM integration for CrewAI.

This module provides integration with AI Suite for LLM capabilities.
"""

from typing import Any

import aisuite as ai  # type: ignore

from crewai.llms.base_llm import BaseLLM


class AISuiteLLM(BaseLLM):
    """AI Suite LLM implementation.

    This class provides integration with AI Suite models through the BaseLLM interface.
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the AI Suite LLM.

        Args:
            model: The model identifier for AI Suite.
            temperature: Optional temperature setting for response generation.
            stop: Optional list of stop sequences for generation.
            **kwargs: Additional keyword arguments passed to the AI Suite client.
        """
        super().__init__(model, temperature, stop)
        self.client = ai.Client()
        self.kwargs = kwargs

    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | Any:
        """Call the AI Suite LLM with the given messages.

        Args:
            messages: Input messages for the LLM.
            tools: Optional list of tool schemas for function calling.
            callbacks: Optional list of callback functions.
            available_functions: Optional dict mapping function names to callables.
            from_task: Optional task caller.
            from_agent: Optional agent caller.

        Returns:
            The text response from the LLM.
        """
        completion_params = self._prepare_completion_params(messages, tools)
        response = self.client.chat.completions.create(**completion_params)

        return response.choices[0].message.content

    def _prepare_completion_params(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Prepare parameters for the AI Suite completion call.

        Args:
            messages: Input messages for the LLM.
            tools: Optional list of tool schemas.

        Returns:
            Dictionary of parameters for the completion API.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "tools": tools,
            **self.kwargs,
        }

        if self.stop:
            params["stop"] = self.stop

        return params

    def supports_function_calling(self) -> bool:
        """Check if the LLM supports function calling.

        Returns:
            False, as AI Suite does not currently support function calling.
        """
        return False
