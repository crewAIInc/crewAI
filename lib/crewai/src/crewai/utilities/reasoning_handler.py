"""Handles planning/reasoning for agents before task execution."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pydantic import BaseModel, Field

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.reasoning_events import (
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
    AgentReasoningStartedEvent,
)
from crewai.llm import LLM
from crewai.utilities.llm_utils import create_llm
from crewai.utilities.string_utils import sanitize_tool_name


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.agent.planning_config import PlanningConfig
    from crewai.task import Task


class ReasoningPlan(BaseModel):
    """Model representing a reasoning plan for a task."""

    plan: str = Field(description="The detailed reasoning plan for the task.")
    ready: bool = Field(description="Whether the agent is ready to execute the task.")


class AgentReasoningOutput(BaseModel):
    """Model representing the output of the agent reasoning process."""

    plan: ReasoningPlan = Field(description="The reasoning plan for the task.")


# Aliases for backward compatibility
PlanningPlan = ReasoningPlan
AgentPlanningOutput = AgentReasoningOutput


FUNCTION_SCHEMA: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": "create_reasoning_plan",
        "description": "Create or refine a reasoning plan for a task",
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "The detailed reasoning plan for the task.",
                },
                "ready": {
                    "type": "boolean",
                    "description": "Whether the agent is ready to execute the task.",
                },
            },
            "required": ["plan", "ready"],
            "additionalProperties": False,
        },
    },
}


class AgentReasoning:
    """
    Handles the agent planning/reasoning process, enabling an agent to reflect
    and create a plan before executing a task.

    Attributes:
        task: The task for which the agent is planning (optional).
        agent: The agent performing the planning.
        config: The planning configuration.
        llm: The language model used for planning.
        logger: Logger for logging events and errors.
        description: Task description or input text for planning.
        expected_output: Expected output description.
    """

    def __init__(
        self,
        agent: Agent,
        task: Task | None = None,
        *,
        description: str | None = None,
        expected_output: str | None = None,
    ) -> None:
        """Initialize the AgentReasoning with an agent and optional task.

        Args:
            agent: The agent performing the planning.
            task: The task for which the agent is planning (optional).
            description: Task description or input text (used if task is None).
            expected_output: Expected output (used if task is None).
        """
        self.agent = agent
        self.task = task
        # Use task attributes if available, otherwise use provided values
        self._description = description or (
            task.description if task else "Complete the requested task"
        )
        self._expected_output = expected_output or (
            task.expected_output if task else "Complete the task successfully"
        )
        self.config = self._get_planning_config()
        self.llm = self._resolve_llm()
        self.logger = logging.getLogger(__name__)

    @property
    def description(self) -> str:
        """Get the task/input description."""
        return self._description

    @property
    def expected_output(self) -> str:
        """Get the expected output."""
        return self._expected_output

    def _get_planning_config(self) -> PlanningConfig:
        """Get the planning configuration from the agent.

        Returns:
            The planning configuration, using defaults if not set.
        """
        from crewai.agent.planning_config import PlanningConfig

        if self.agent.planning_config is not None:
            return self.agent.planning_config
        # Fallback for backward compatibility
        return PlanningConfig(
            max_attempts=getattr(self.agent, "max_reasoning_attempts", None),
        )

    def _resolve_llm(self) -> LLM:
        """Resolve which LLM to use for planning.

        Returns:
            The LLM to use - either from config or the agent's LLM.
        """
        if self.config.llm is not None:
            if isinstance(self.config.llm, LLM):
                return self.config.llm
            return create_llm(self.config.llm)
        return cast(LLM, self.agent.llm)

    def handle_agent_reasoning(self) -> AgentReasoningOutput:
        """Public method for the planning process that creates and refines a plan
        for the task until the agent is ready to execute it.

        Returns:
            AgentReasoningOutput: The output of the agent planning process.
        """
        task_id = str(self.task.id) if self.task else "kickoff"

        # Emit a planning started event (attempt 1)
        try:
            crewai_event_bus.emit(
                self.agent,
                AgentReasoningStartedEvent(
                    agent_role=self.agent.role,
                    task_id=task_id,
                    attempt=1,
                    from_task=self.task,
                ),
            )
        except Exception:  # noqa: S110
            # Ignore event bus errors to avoid breaking execution
            pass

        try:
            output = self._execute_planning()

            crewai_event_bus.emit(
                self.agent,
                AgentReasoningCompletedEvent(
                    agent_role=self.agent.role,
                    task_id=task_id,
                    plan=output.plan.plan,
                    ready=output.plan.ready,
                    attempt=1,
                    from_task=self.task,
                    from_agent=self.agent,
                ),
            )

            return output
        except Exception as e:
            # Emit planning failed event
            try:
                crewai_event_bus.emit(
                    self.agent,
                    AgentReasoningFailedEvent(
                        agent_role=self.agent.role,
                        task_id=task_id,
                        error=str(e),
                        attempt=1,
                        from_task=self.task,
                        from_agent=self.agent,
                    ),
                )
            except Exception as event_error:
                logging.error(f"Error emitting planning failed event: {event_error}")

            raise

    def _execute_planning(self) -> AgentReasoningOutput:
        """Execute the planning process.

        Returns:
            The output of the agent planning process.
        """
        plan, ready = self._create_initial_plan()
        plan, ready = self._refine_plan_if_needed(plan, ready)

        reasoning_plan = ReasoningPlan(plan=plan, ready=ready)
        return AgentReasoningOutput(plan=reasoning_plan)

    def _create_initial_plan(self) -> tuple[str, bool]:
        """Creates the initial plan for the task.

        Returns:
            The initial plan and whether the agent is ready to execute the task.
        """
        planning_prompt = self._create_planning_prompt()

        if self.llm.supports_function_calling():
            plan, ready = self._call_with_function(planning_prompt, "create_plan")
            return plan, ready

        response = self._call_llm_with_prompt(
            prompt=planning_prompt,
            plan_type="create_plan",
        )

        return self._parse_planning_response(str(response))

    def _refine_plan_if_needed(self, plan: str, ready: bool) -> tuple[str, bool]:
        """Refines the plan if the agent is not ready to execute the task.

        Args:
            plan: The current plan.
            ready: Whether the agent is ready to execute the task.

        Returns:
            The refined plan and whether the agent is ready to execute the task.
        """
        attempt = 1
        max_attempts = self.config.max_attempts
        task_id = str(self.task.id) if self.task else "kickoff"

        while not ready and (max_attempts is None or attempt < max_attempts):
            # Emit event for each refinement attempt
            try:
                crewai_event_bus.emit(
                    self.agent,
                    AgentReasoningStartedEvent(
                        agent_role=self.agent.role,
                        task_id=task_id,
                        attempt=attempt + 1,
                        from_task=self.task,
                    ),
                )
            except Exception:  # noqa: S110
                pass

            refine_prompt = self._create_refine_prompt(plan)

            if self.llm.supports_function_calling():
                plan, ready = self._call_with_function(refine_prompt, "refine_plan")
            else:
                response = self._call_llm_with_prompt(
                    prompt=refine_prompt,
                    plan_type="refine_plan",
                )
                plan, ready = self._parse_planning_response(str(response))

            attempt += 1

            if max_attempts is not None and attempt >= max_attempts:
                self.logger.warning(
                    f"Agent planning reached maximum attempts ({max_attempts}) "
                    "without being ready. Proceeding with current plan."
                )
                break

        return plan, ready

    def _call_with_function(
        self, prompt: str, plan_type: Literal["create_plan", "refine_plan"]
    ) -> tuple[str, bool]:
        """Calls the LLM with function calling to get a plan.

        Args:
            prompt: The prompt to send to the LLM.
            plan_type: The type of plan being created.

        Returns:
            A tuple containing the plan and whether the agent is ready.
        """
        self.logger.debug(f"Using function calling for {plan_type} planning")

        try:
            system_prompt = self._get_system_prompt()

            # Prepare a simple callable that just returns the tool arguments as JSON
            def _create_reasoning_plan(plan: str, ready: bool = True) -> str:
                """Return the planning result in JSON string form."""
                return json.dumps({"plan": plan, "ready": ready})

            response = self.llm.call(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tools=[FUNCTION_SCHEMA],
                available_functions={"create_reasoning_plan": _create_reasoning_plan},
                from_task=self.task,
                from_agent=self.agent,
            )

            try:
                result = json.loads(response)
                if "plan" in result and "ready" in result:
                    return result["plan"], result["ready"]
            except (json.JSONDecodeError, KeyError):
                pass

            response_str = str(response)
            return (
                response_str,
                "READY: I am ready to execute the task." in response_str,
            )

        except Exception as e:
            self.logger.warning(
                f"Error during function calling: {e!s}. Falling back to text parsing."
            )

            try:
                system_prompt = self._get_system_prompt()

                fallback_response = self.llm.call(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    from_task=self.task,
                    from_agent=self.agent,
                )

                fallback_str = str(fallback_response)
                return (
                    fallback_str,
                    "READY: I am ready to execute the task." in fallback_str,
                )
            except Exception as inner_e:
                self.logger.error(f"Error during fallback text parsing: {inner_e!s}")
                return (
                    "Failed to generate a plan due to an error.",
                    True,
                )  # Default to ready to avoid getting stuck

    def _call_llm_with_prompt(
        self,
        prompt: str,
        plan_type: Literal["create_plan", "refine_plan"],
    ) -> str:
        """Calls the LLM with the planning prompt.

        Args:
            prompt: The prompt to send to the LLM.
            plan_type: The type of plan being created.

        Returns:
            The LLM response.
        """
        system_prompt = self._get_system_prompt()

        response = self.llm.call(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            from_task=self.task,
            from_agent=self.agent,
        )
        return str(response)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for planning.

        Returns:
            The system prompt, either custom or from i18n.
        """
        if self.config.system_prompt is not None:
            return self.config.system_prompt

        # Try new "planning" section first, fall back to "reasoning" for compatibility
        try:
            return self.agent.i18n.retrieve("planning", "system_prompt")
        except (KeyError, AttributeError):
            # Fallback to reasoning section for backward compatibility
            return self.agent.i18n.retrieve("reasoning", "initial_plan").format(
                role=self.agent.role,
                goal=self.agent.goal,
                backstory=self._get_agent_backstory(),
            )

    def _get_agent_backstory(self) -> str:
        """Safely gets the agent's backstory, providing a default if not available.

        Returns:
            The agent's backstory or a default value.
        """
        return getattr(self.agent, "backstory", "No backstory provided")

    def _create_planning_prompt(self) -> str:
        """Creates a prompt for the agent to plan the task.

        Returns:
            The planning prompt.
        """
        available_tools = self._format_available_tools()

        # Use custom prompt if provided
        if self.config.plan_prompt is not None:
            return self.config.plan_prompt.format(
                role=self.agent.role,
                goal=self.agent.goal,
                backstory=self._get_agent_backstory(),
                description=self.description,
                expected_output=self.expected_output,
                tools=available_tools,
                max_steps=self.config.max_steps,
            )

        # Try new "planning" section first
        try:
            return self.agent.i18n.retrieve("planning", "create_plan_prompt").format(
                description=self.description,
                expected_output=self.expected_output,
                tools=available_tools,
                max_steps=self.config.max_steps,
            )
        except (KeyError, AttributeError):
            # Fallback to reasoning section for backward compatibility
            return self.agent.i18n.retrieve("reasoning", "create_plan_prompt").format(
                role=self.agent.role,
                goal=self.agent.goal,
                backstory=self._get_agent_backstory(),
                description=self.description,
                expected_output=self.expected_output,
                tools=available_tools,
            )

    def _format_available_tools(self) -> str:
        """Formats the available tools for inclusion in the prompt.

        Returns:
            Comma-separated list of tool names.
        """
        try:
            # Try task tools first, then agent tools
            tools = []
            if self.task:
                tools = self.task.tools or []
            if not tools:
                tools = getattr(self.agent, "tools", []) or []
            if not tools:
                return "No tools available"
            return ", ".join([sanitize_tool_name(tool.name) for tool in tools])
        except (AttributeError, TypeError):
            return "No tools available"

    def _create_refine_prompt(self, current_plan: str) -> str:
        """Creates a prompt for the agent to refine its plan.

        Args:
            current_plan: The current plan.

        Returns:
            The refine prompt.
        """
        # Use custom prompt if provided
        if self.config.refine_prompt is not None:
            return self.config.refine_prompt.format(
                role=self.agent.role,
                goal=self.agent.goal,
                backstory=self._get_agent_backstory(),
                current_plan=current_plan,
                max_steps=self.config.max_steps,
            )

        # Try new "planning" section first
        try:
            return self.agent.i18n.retrieve("planning", "refine_plan_prompt").format(
                current_plan=current_plan,
            )
        except (KeyError, AttributeError):
            # Fallback to reasoning section for backward compatibility
            return self.agent.i18n.retrieve("reasoning", "refine_plan_prompt").format(
                role=self.agent.role,
                goal=self.agent.goal,
                backstory=self._get_agent_backstory(),
                current_plan=current_plan,
            )

    @staticmethod
    def _parse_planning_response(response: str) -> tuple[str, bool]:
        """Parses the planning response to extract the plan and readiness.

        Args:
            response: The LLM response.

        Returns:
            The plan and whether the agent is ready to execute the task.
        """
        if not response:
            return "No plan was generated.", False

        plan = response
        ready = "READY: I am ready to execute the task." in response

        return plan, ready

    # Deprecated methods for backward compatibility
    def __handle_agent_reasoning(self) -> AgentReasoningOutput:
        """Deprecated: Use _execute_planning instead."""
        return self._execute_planning()

    def _handle_agent_reasoning(self) -> AgentReasoningOutput:
        """Deprecated method for backward compatibility.
        Use handle_agent_reasoning() instead.

        Returns:
            AgentReasoningOutput: The output of the agent planning process.
        """
        self.logger.warning(
            "The _handle_agent_reasoning method is deprecated. "
            "Use handle_agent_reasoning instead."
        )
        return self.handle_agent_reasoning()


# Alias for backward compatibility
AgentPlanning = AgentReasoning


def _call_llm_with_reasoning_prompt(
    llm: LLM,
    prompt: str,
    task: Task,
    reasoning_agent: Agent,
    backstory: str,
    plan_type: Literal["initial_plan", "refine_plan"],
) -> str:
    """Deprecated: Calls the LLM with the reasoning prompt.

    This function is kept for backward compatibility.

    Args:
        llm: The language model to use.
        prompt: The prompt to send to the LLM.
        task: The task for which the agent is reasoning.
        reasoning_agent: The agent performing the reasoning.
        backstory: The agent's backstory.
        plan_type: The type of plan being created.

    Returns:
        The LLM response.
    """
    system_prompt = reasoning_agent.i18n.retrieve("reasoning", plan_type).format(
        role=reasoning_agent.role,
        goal=reasoning_agent.goal,
        backstory=backstory,
    )

    response = llm.call(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        from_task=task,
        from_agent=reasoning_agent,
    )
    return str(response)
