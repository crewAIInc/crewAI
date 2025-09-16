"""DBOS LLM class for CrewAI.

This module provides the DBOS wrapper class for all LLM implementations
in CrewAI, making call() a DBOS step.
"""

from typing import Any

from dbos import DBOS

from crewai.durable_execution.dbos.dbos_utils import StepConfig
from crewai.llms.base_llm import BaseLLM


class DBOSLLM(BaseLLM):
    def __init__(
        self,
        name_prefix: str,
        orig_llm: BaseLLM,
        step_config: StepConfig | None = None,
    ):
        """Initialize the DBOSLLM with an underlying LLM and step configuration.

        Args:
            orig_llm: The original LLM instance to wrap.
            step_config: Optional DBOS step configuration for LLM calls.
        """
        super().__init__(
            model=orig_llm.model,
            temperature=orig_llm.temperature,
            stop=orig_llm.stop,
        )
        self._orig_llm = orig_llm
        self._step_config = step_config or {}

        # Wrap the call method as a DBOS step
        @DBOS.step(name=f"{name_prefix}.{orig_llm.model}", **self._step_config)
        def dbos_call(
            messages: str | list[dict[str, str]],
            tools: list[dict] | None = None,
            callbacks: list[Any] | None = None,
            available_functions: dict[str, Any] | None = None,
            from_task: Any | None = None,
            from_agent: Any | None = None,
        ) -> str | Any:
            return self._orig_llm.call(
                tools=tools,
                messages=messages,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

        self._dbos_call = dbos_call

    # Return whatever attribute the underlying LLM has, except for the call method
    def __getattr__(self, name: str) -> Any:
        if name == "call":
            return self._dbos_call
        return getattr(self._orig_llm, name)

    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | Any:
        return self._dbos_call(
            messages=messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
        )
