import json
import logging
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.reasoning_events import (
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
    AgentReasoningStartedEvent,
)
from crewai.llm import LLM
from crewai.task import Task
from crewai.utilities.string_utils import sanitize_tool_name


class ReasoningPlan(BaseModel):
    """Model representing a reasoning plan for a task."""

    plan: str = Field(description="The detailed reasoning plan for the task.")
    ready: bool = Field(description="Whether the agent is ready to execute the task.")


class AgentReasoningOutput(BaseModel):
    """Model representing the output of the agent reasoning process."""

    plan: ReasoningPlan = Field(description="The reasoning plan for the task.")


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
        },
    },
}


class AgentReasoning:
    """
    Handles the agent reasoning process, enabling an agent to reflect and create a plan
    before executing a task.

    Attributes:
        task: The task for which the agent is reasoning.
        agent: The agent performing the reasoning.
        llm: The language model used for reasoning.
        logger: Logger for logging events and errors.
    """

    def __init__(self, task: Task, agent: Agent) -> None:
        """Initialize the AgentReasoning with a task and an agent.

        Args:
            task: The task for which the agent is reasoning.
            agent: The agent performing the reasoning.
        """
        self.task = task
        self.agent = agent
        self.llm = cast(LLM, agent.llm)
        self.logger = logging.getLogger(__name__)

    def handle_agent_reasoning(self) -> AgentReasoningOutput:
        """Public method for the reasoning process that creates and refines a plan for the task until the agent is ready to execute it.

        Returns:
            AgentReasoningOutput: The output of the agent reasoning process.
        """
        # Emit a reasoning started event (attempt 1)
        try:
            crewai_event_bus.emit(
                self.agent,
                AgentReasoningStartedEvent(
                    agent_role=self.agent.role,
                    task_id=str(self.task.id),
                    attempt=1,
                    from_task=self.task,
                ),
            )
        except Exception:  # noqa: S110
            # Ignore event bus errors to avoid breaking execution
            pass

        try:
            output = self.__handle_agent_reasoning()

            crewai_event_bus.emit(
                self.agent,
                AgentReasoningCompletedEvent(
                    agent_role=self.agent.role,
                    task_id=str(self.task.id),
                    plan=output.plan.plan,
                    ready=output.plan.ready,
                    attempt=1,
                    from_task=self.task,
                    from_agent=self.agent,
                ),
            )

            return output
        except Exception as e:
            # Emit reasoning failed event
            try:
                crewai_event_bus.emit(
                    self.agent,
                    AgentReasoningFailedEvent(
                        agent_role=self.agent.role,
                        task_id=str(self.task.id),
                        error=str(e),
                        attempt=1,
                        from_task=self.task,
                        from_agent=self.agent,
                    ),
                )
            except Exception as e:
                logging.error(f"Error emitting reasoning failed event: {e}")

            raise

    def __handle_agent_reasoning(self) -> AgentReasoningOutput:
        """Private method that handles the agent reasoning process.

        Returns:
            The output of the agent reasoning process.
        """
        plan, ready = self.__create_initial_plan()

        plan, ready = self.__refine_plan_if_needed(plan, ready)

        reasoning_plan = ReasoningPlan(plan=plan, ready=ready)
        return AgentReasoningOutput(plan=reasoning_plan)

    def __create_initial_plan(self) -> tuple[str, bool]:
        """Creates the initial reasoning plan for the task.

        Returns:
            The initial plan and whether the agent is ready to execute the task.
        """
        reasoning_prompt = self.__create_reasoning_prompt()

        if self.llm.supports_function_calling():
            plan, ready = self.__call_with_function(reasoning_prompt, "initial_plan")
            return plan, ready
        response = _call_llm_with_reasoning_prompt(
            llm=self.llm,
            prompt=reasoning_prompt,
            task=self.task,
            reasoning_agent=self.agent,
            backstory=self.__get_agent_backstory(),
            plan_type="initial_plan",
        )

        return self.__parse_reasoning_response(str(response))

    def __refine_plan_if_needed(self, plan: str, ready: bool) -> tuple[str, bool]:
        """Refines the reasoning plan if the agent is not ready to execute the task.

        Args:
            plan: The current reasoning plan.
            ready: Whether the agent is ready to execute the task.

        Returns:
            The refined plan and whether the agent is ready to execute the task.
        """
        attempt = 1
        max_attempts = self.agent.max_reasoning_attempts

        while not ready and (max_attempts is None or attempt < max_attempts):
            # Emit event for each refinement attempt
            try:
                crewai_event_bus.emit(
                    self.agent,
                    AgentReasoningStartedEvent(
                        agent_role=self.agent.role,
                        task_id=str(self.task.id),
                        attempt=attempt + 1,
                        from_task=self.task,
                    ),
                )
            except Exception:  # noqa: S110
                pass

            refine_prompt = self.__create_refine_prompt(plan)

            if self.llm.supports_function_calling():
                plan, ready = self.__call_with_function(refine_prompt, "refine_plan")
            else:
                response = _call_llm_with_reasoning_prompt(
                    llm=self.llm,
                    prompt=refine_prompt,
                    task=self.task,
                    reasoning_agent=self.agent,
                    backstory=self.__get_agent_backstory(),
                    plan_type="refine_plan",
                )
                plan, ready = self.__parse_reasoning_response(str(response))

            attempt += 1

            if max_attempts is not None and attempt >= max_attempts:
                self.logger.warning(
                    f"Agent reasoning reached maximum attempts ({max_attempts}) without being ready. Proceeding with current plan."
                )
                break

        return plan, ready

    def __call_with_function(self, prompt: str, prompt_type: str) -> tuple[str, bool]:
        """Calls the LLM with function calling to get a reasoning plan.

        Args:
            prompt: The prompt to send to the LLM.
            prompt_type: The type of prompt (initial_plan or refine_plan).

        Returns:
            A tuple containing the plan and whether the agent is ready.
        """
        self.logger.debug(f"Using function calling for {prompt_type} reasoning")

        try:
            system_prompt = self.agent.i18n.retrieve("reasoning", prompt_type).format(
                role=self.agent.role,
                goal=self.agent.goal,
                backstory=self.__get_agent_backstory(),
            )

            # Prepare a simple callable that just returns the tool arguments as JSON
            def _create_reasoning_plan(plan: str, ready: bool = True) -> str:
                """Return the reasoning plan result in JSON string form."""
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

            self.logger.debug(f"Function calling response: {response[:100]}...")

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
                system_prompt = self.agent.i18n.retrieve(
                    "reasoning", prompt_type
                ).format(
                    role=self.agent.role,
                    goal=self.agent.goal,
                    backstory=self.__get_agent_backstory(),
                )

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

    def __get_agent_backstory(self) -> str:
        """
        Safely gets the agent's backstory, providing a default if not available.

        Returns:
            str: The agent's backstory or a default value.
        """
        return getattr(self.agent, "backstory", "No backstory provided")

    def __create_reasoning_prompt(self) -> str:
        """
        Creates a prompt for the agent to reason about the task.

        Returns:
            str: The reasoning prompt.
        """
        available_tools = self.__format_available_tools()

        return self.agent.i18n.retrieve("reasoning", "create_plan_prompt").format(
            role=self.agent.role,
            goal=self.agent.goal,
            backstory=self.__get_agent_backstory(),
            description=self.task.description,
            expected_output=self.task.expected_output,
            tools=available_tools,
        )

    def __format_available_tools(self) -> str:
        """
        Formats the available tools for inclusion in the prompt.

        Returns:
            str: Comma-separated list of tool names.
        """
        try:
            return ", ".join(
                [sanitize_tool_name(tool.name) for tool in (self.task.tools or [])]
            )
        except (AttributeError, TypeError):
            return "No tools available"

    def __create_refine_prompt(self, current_plan: str) -> str:
        """
        Creates a prompt for the agent to refine its reasoning plan.

        Args:
            current_plan: The current reasoning plan.

        Returns:
            str: The refine prompt.
        """
        return self.agent.i18n.retrieve("reasoning", "refine_plan_prompt").format(
            role=self.agent.role,
            goal=self.agent.goal,
            backstory=self.__get_agent_backstory(),
            current_plan=current_plan,
        )

    @staticmethod
    def __parse_reasoning_response(response: str) -> tuple[str, bool]:
        """
        Parses the reasoning response to extract the plan and whether
        the agent is ready to execute the task.

        Args:
            response: The LLM response.

        Returns:
            The plan and whether the agent is ready to execute the task.
        """
        if not response:
            return "No plan was generated.", False

        plan = response
        ready = False

        if "READY: I am ready to execute the task." in response:
            ready = True

        return plan, ready

    def _handle_agent_reasoning(self) -> AgentReasoningOutput:
        """
        Deprecated method for backward compatibility.
        Use handle_agent_reasoning() instead.

        Returns:
            AgentReasoningOutput: The output of the agent reasoning process.
        """
        self.logger.warning(
            "The _handle_agent_reasoning method is deprecated. Use handle_agent_reasoning instead."
        )
        return self.handle_agent_reasoning()


def _call_llm_with_reasoning_prompt(
    llm: LLM,
    prompt: str,
    task: Task,
    reasoning_agent: Agent,
    backstory: str,
    plan_type: Literal["initial_plan", "refine_plan"],
) -> str:
    """Calls the LLM with the reasoning prompt.

    Args:
        llm: The language model to use.
        prompt: The prompt to send to the LLM.
        task: The task for which the agent is reasoning.
        reasoning_agent: The agent performing the reasoning.
        backstory: The agent's backstory.
        plan_type: The type of plan being created ("initial_plan" or "refine_plan").

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
