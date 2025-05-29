import logging
import json
from typing import Tuple, cast

from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.task import Task
from crewai.utilities import I18N
from crewai.llm import LLM
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.reasoning_events import (
    AgentReasoningStartedEvent,
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
)


class ReasoningPlan(BaseModel):
    """Model representing a reasoning plan for a task."""
    plan: str = Field(description="The detailed reasoning plan for the task.")
    ready: bool = Field(description="Whether the agent is ready to execute the task.")


class AgentReasoningOutput(BaseModel):
    """Model representing the output of the agent reasoning process."""
    plan: ReasoningPlan = Field(description="The reasoning plan for the task.")


class ReasoningFunction(BaseModel):
    """Model for function calling with reasoning."""
    plan: str = Field(description="The detailed reasoning plan for the task.")
    ready: bool = Field(description="Whether the agent is ready to execute the task.")


class AgentReasoning:
    """
    Handles the agent reasoning process, enabling an agent to reflect and create a plan
    before executing a task.
    """
    def __init__(self, task: Task, agent: Agent):
        if not task or not agent:
            raise ValueError("Both task and agent must be provided.")
        self.task = task
        self.agent = agent
        self.llm = cast(LLM, agent.llm)
        self.logger = logging.getLogger(__name__)
        self.i18n = I18N()

    def handle_agent_reasoning(self) -> AgentReasoningOutput:
        """
        Public method for the reasoning process that creates and refines a plan
        for the task until the agent is ready to execute it.

        Returns:
            AgentReasoningOutput: The output of the agent reasoning process.
        """
        import time
        start_time = time.time()

        # Emit a reasoning started event (attempt 1)
        try:
            crewai_event_bus.emit(
                self.agent,
                AgentReasoningStartedEvent(
                    agent_role=self.agent.role,
                    task_id=str(self.task.id),
                    attempt=1,
                ),
            )
        except Exception:
            # Ignore event bus errors to avoid breaking execution
            pass

        try:
            output = self.__handle_agent_reasoning()

            duration_seconds = time.time() - start_time

            # Emit reasoning completed event
            try:
                crewai_event_bus.emit(
                    self.agent,
                    AgentReasoningCompletedEvent(
                        agent_role=self.agent.role,
                        task_id=str(self.task.id),
                        plan=output.plan.plan,
                        ready=output.plan.ready,
                        attempt=1,
                        duration_seconds=duration_seconds,
                    ),
                )
            except Exception:
                pass

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
                    ),
                )
            except Exception:
                pass

            raise

    def __handle_agent_reasoning(self) -> AgentReasoningOutput:
        """
        Private method that handles the agent reasoning process.

        Returns:
            AgentReasoningOutput: The output of the agent reasoning process.
        """
        plan, ready = self.__create_initial_plan()

        plan, ready = self.__refine_plan_if_needed(plan, ready)

        reasoning_plan = ReasoningPlan(plan=plan, ready=ready)
        return AgentReasoningOutput(plan=reasoning_plan)

    def __create_initial_plan(self) -> Tuple[str, bool]:
        """
        Creates the initial reasoning plan for the task.

        Returns:
            Tuple[str, bool]: The initial plan and whether the agent is ready to execute the task.
        """
        reasoning_prompt = self.__create_reasoning_prompt()

        if self.llm.supports_function_calling():
            plan, ready = self.__call_with_function(reasoning_prompt, "initial_plan")
            return plan, ready
        else:
            system_prompt = self.i18n.retrieve("reasoning", "initial_plan").format(
                role=self.agent.role,
                goal=self.agent.goal,
                backstory=self.__get_agent_backstory()
            )

            response = self.llm.call(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": reasoning_prompt}
                ]
            )

            return self.__parse_reasoning_response(str(response))

    def __refine_plan_if_needed(self, plan: str, ready: bool) -> Tuple[str, bool]:
        """
        Refines the reasoning plan if the agent is not ready to execute the task.

        Args:
            plan: The current reasoning plan.
            ready: Whether the agent is ready to execute the task.

        Returns:
            Tuple[str, bool]: The refined plan and whether the agent is ready to execute the task.
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
                    ),
                )
            except Exception:
                pass

            refine_prompt = self.__create_refine_prompt(plan)

            if self.llm.supports_function_calling():
                plan, ready = self.__call_with_function(refine_prompt, "refine_plan")
            else:
                system_prompt = self.i18n.retrieve("reasoning", "refine_plan").format(
                    role=self.agent.role,
                    goal=self.agent.goal,
                    backstory=self.__get_agent_backstory()
                )

                response = self.llm.call(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": refine_prompt}
                    ]
                )
                plan, ready = self.__parse_reasoning_response(str(response))

            attempt += 1

            if max_attempts is not None and attempt >= max_attempts:
                self.logger.warning(
                    f"Agent reasoning reached maximum attempts ({max_attempts}) without being ready. Proceeding with current plan."
                )
                break

        return plan, ready

    def __call_with_function(self, prompt: str, prompt_type: str) -> Tuple[str, bool]:
        """
        Calls the LLM with function calling to get a reasoning plan.

        Args:
            prompt: The prompt to send to the LLM.
            prompt_type: The type of prompt (initial_plan or refine_plan).

        Returns:
            Tuple[str, bool]: A tuple containing the plan and whether the agent is ready.
        """
        self.logger.debug(f"Using function calling for {prompt_type} reasoning")

        function_schema = {
            "type": "function",
            "function": {
                "name": "create_reasoning_plan",
                "description": "Create or refine a reasoning plan for a task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "string",
                            "description": "The detailed reasoning plan for the task."
                        },
                        "ready": {
                            "type": "boolean",
                            "description": "Whether the agent is ready to execute the task."
                        }
                    },
                    "required": ["plan", "ready"]
                }
            }
        }

        try:
            system_prompt = self.i18n.retrieve("reasoning", prompt_type).format(
                role=self.agent.role,
                goal=self.agent.goal,
                backstory=self.__get_agent_backstory()
            )

            # Prepare a simple callable that just returns the tool arguments as JSON
            def _create_reasoning_plan(plan: str, ready: bool):  # noqa: N802
                """Return the reasoning plan result in JSON string form."""
                return json.dumps({"plan": plan, "ready": ready})

            response = self.llm.call(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                tools=[function_schema],
                available_functions={"create_reasoning_plan": _create_reasoning_plan},
            )

            self.logger.debug(f"Function calling response: {response[:100]}...")

            try:
                result = json.loads(response)
                if "plan" in result and "ready" in result:
                    return result["plan"], result["ready"]
            except (json.JSONDecodeError, KeyError):
                pass

            response_str = str(response)
            return response_str, "READY: I am ready to execute the task." in response_str

        except Exception as e:
            self.logger.warning(f"Error during function calling: {str(e)}. Falling back to text parsing.")

            try:
                system_prompt = self.i18n.retrieve("reasoning", prompt_type).format(
                    role=self.agent.role,
                    goal=self.agent.goal,
                    backstory=self.__get_agent_backstory()
                )

                fallback_response = self.llm.call(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )

                fallback_str = str(fallback_response)
                return fallback_str, "READY: I am ready to execute the task." in fallback_str
            except Exception as inner_e:
                self.logger.error(f"Error during fallback text parsing: {str(inner_e)}")
                return "Failed to generate a plan due to an error.", True  # Default to ready to avoid getting stuck

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

        return self.i18n.retrieve("reasoning", "create_plan_prompt").format(
            role=self.agent.role,
            goal=self.agent.goal,
            backstory=self.__get_agent_backstory(),
            description=self.task.description,
            expected_output=self.task.expected_output,
            tools=available_tools
        )

    def __format_available_tools(self) -> str:
        """
        Formats the available tools for inclusion in the prompt.

        Returns:
            str: Comma-separated list of tool names.
        """
        try:
            return ', '.join([tool.name for tool in (self.task.tools or [])])
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
        return self.i18n.retrieve("reasoning", "refine_plan_prompt").format(
            role=self.agent.role,
            goal=self.agent.goal,
            backstory=self.__get_agent_backstory(),
            current_plan=current_plan
        )

    def __parse_reasoning_response(self, response: str) -> Tuple[str, bool]:
        """
        Parses the reasoning response to extract the plan and whether
        the agent is ready to execute the task.

        Args:
            response: The LLM response.

        Returns:
            Tuple[str, bool]: The plan and whether the agent is ready to execute the task.
        """
        if not response:
            return "No plan was generated.", False

        plan = response
        ready = False

        if "READY: I am ready to execute the task." in response or "READY: I am ready to continue executing the task." in response:
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

    def _emit_reasoning_event(self, event_class, **kwargs):
        """Centralized method for emitting reasoning events."""
        try:
            reasoning_trigger = "interval"
            if hasattr(self.agent, 'adaptive_reasoning') and self.agent.adaptive_reasoning:
                reasoning_trigger = "adaptive"

            crewai_event_bus.emit(
                self.agent,
                event_class(
                    agent_role=self.agent.role,
                    task_id=str(self.task.id),
                    reasoning_trigger=reasoning_trigger,
                    **kwargs
                ),
            )
        except Exception:
            # Ignore event bus errors to avoid breaking execution
            pass

    def handle_mid_execution_reasoning(
        self,
        current_steps: int,
        tools_used: list,
        current_progress: str,
        iteration_messages: list
    ) -> AgentReasoningOutput:
        """
        Handle reasoning during task execution with context about current progress.

        Args:
            current_steps: Number of steps executed so far
            tools_used: List of tools that have been used
            current_progress: Summary of progress made so far
            iteration_messages: Recent conversation messages

        Returns:
            AgentReasoningOutput: Updated reasoning plan based on current context
        """
        import time
        start_time = time.time()

        from crewai.utilities.events.reasoning_events import AgentMidExecutionReasoningStartedEvent

        self._emit_reasoning_event(
            AgentMidExecutionReasoningStartedEvent,
            current_step=current_steps
        )

        try:
            output = self.__handle_mid_execution_reasoning(
                current_steps, tools_used, current_progress, iteration_messages
            )

            duration_seconds = time.time() - start_time

            # Emit completed event
            from crewai.utilities.events.reasoning_events import AgentMidExecutionReasoningCompletedEvent

            self._emit_reasoning_event(
                AgentMidExecutionReasoningCompletedEvent,
                current_step=current_steps,
                updated_plan=output.plan.plan,
                duration_seconds=duration_seconds
            )

            return output
        except Exception as e:
            # Emit failed event
            from crewai.utilities.events.reasoning_events import AgentReasoningFailedEvent

            self._emit_reasoning_event(
                AgentReasoningFailedEvent,
                error=str(e),
                attempt=1
            )

            raise

    def __handle_mid_execution_reasoning(
        self,
        current_steps: int,
        tools_used: list,
        current_progress: str,
        iteration_messages: list
    ) -> AgentReasoningOutput:
        """
        Private method that handles the mid-execution reasoning process.

        Args:
            current_steps: Number of steps executed so far
            tools_used: List of tools that have been used
            current_progress: Summary of progress made so far
            iteration_messages: Recent conversation messages

        Returns:
            AgentReasoningOutput: The output of the mid-execution reasoning process.
        """
        mid_execution_prompt = self.__create_mid_execution_prompt(
            current_steps, tools_used, current_progress, iteration_messages
        )

        if self.llm.supports_function_calling():
            plan, ready = self.__call_with_function(mid_execution_prompt, "mid_execution_plan")
        else:
            # Use the same prompt for system context
            system_prompt = self.i18n.retrieve("reasoning", "mid_execution_plan").format(
                role=self.agent.role,
                goal=self.agent.goal,
                backstory=self.__get_agent_backstory()
            )

            response = self.llm.call(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": mid_execution_prompt}
                ]
            )

            plan, ready = self.__parse_reasoning_response(str(response))

        reasoning_plan = ReasoningPlan(plan=plan, ready=ready)
        return AgentReasoningOutput(plan=reasoning_plan)

    def __create_mid_execution_prompt(
        self,
        current_steps: int,
        tools_used: list,
        current_progress: str,
        iteration_messages: list
    ) -> str:
        """
        Creates a prompt for the agent to reason during task execution.

        Args:
            current_steps: Number of steps executed so far
            tools_used: List of tools that have been used
            current_progress: Summary of progress made so far
            iteration_messages: Recent conversation messages

        Returns:
            str: The mid-execution reasoning prompt.
        """
        tools_used_str = ", ".join(tools_used) if tools_used else "No tools used yet"

        recent_messages = ""
        if iteration_messages:
            recent_msgs = iteration_messages[-6:] if len(iteration_messages) > 6 else iteration_messages
            for msg in recent_msgs:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content:
                    recent_messages += f"{role.upper()}: {content[:200]}...\n\n"

        return self.i18n.retrieve("reasoning", "mid_execution_reasoning").format(
            description=self.task.description,
            expected_output=self.task.expected_output,
            current_steps=current_steps,
            tools_used=tools_used_str,
            current_progress=current_progress,
            recent_messages=recent_messages
        )

    def should_adaptive_reason_llm(
        self,
        current_steps: int,
        tools_used: list,
        current_progress: str
    ) -> bool:
        """
        Use LLM function calling to determine if adaptive reasoning should be triggered.

        Args:
            current_steps: Number of steps executed so far
            tools_used: List of tools that have been used
            current_progress: Summary of progress made so far

        Returns:
            bool: True if reasoning should be triggered, False otherwise.
        """
        try:
            decision_prompt = self.__create_adaptive_reasoning_decision_prompt(
                current_steps, tools_used, current_progress
            )

            if self.llm.supports_function_calling():
                should_reason, reasoning_expl = self.__call_adaptive_reasoning_function(decision_prompt)
            else:
                should_reason, reasoning_expl = self.__call_adaptive_reasoning_text(decision_prompt)

            # Emit an event so the UI/console can display the decision
            try:
                from crewai.utilities.events.reasoning_events import AgentAdaptiveReasoningDecisionEvent

                self._emit_reasoning_event(
                    AgentAdaptiveReasoningDecisionEvent,
                    should_reason=should_reason,
                    reasoning=reasoning_expl,
                )
            except Exception:
                # Ignore event bus errors to avoid breaking execution
                pass

            return should_reason
        except Exception as e:
            self.logger.warning(f"Error during adaptive reasoning decision: {str(e)}. Defaulting to no reasoning.")
            return False

    def __call_adaptive_reasoning_function(self, prompt: str) -> tuple[bool, str]:
        """Call LLM with function calling for adaptive reasoning decision."""
        function_schema = {
            "type": "function",
            "function": {
                "name": "decide_reasoning_need",
                "description": "Decide whether reasoning is needed based on current task execution context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "should_reason": {
                            "type": "boolean",
                            "description": "Whether reasoning/re-planning is needed at this point in task execution."
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why reasoning is or isn't needed."
                        }
                    },
                    "required": ["should_reason", "reasoning"]
                }
            }
        }

        def _decide_reasoning_need(should_reason: bool, reasoning: str):
            """Return the reasoning decision result in JSON string form."""
            return json.dumps({"should_reason": should_reason, "reasoning": reasoning})

        system_prompt = self.i18n.retrieve("reasoning", "adaptive_reasoning_decision").format(
            role=self.agent.role,
            goal=self.agent.goal,
            backstory=self.__get_agent_backstory()
        )

        response = self.llm.call(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            tools=[function_schema],
            available_functions={"decide_reasoning_need": _decide_reasoning_need},
        )

        try:
            result = json.loads(response)
            return result.get("should_reason", False), result.get("reasoning", "No explanation provided")
        except (json.JSONDecodeError, KeyError):
            return False, "No explanation provided"

    def __call_adaptive_reasoning_text(self, prompt: str) -> tuple[bool, str]:
        """Fallback text-based adaptive reasoning decision."""
        system_prompt = self.i18n.retrieve("reasoning", "adaptive_reasoning_decision").format(
            role=self.agent.role,
            goal=self.agent.goal,
            backstory=self.__get_agent_backstory()
        )

        response = self.llm.call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt + "\n\nRespond with 'YES' if reasoning is needed, 'NO' if not."}
        ])

        return "YES" in str(response).upper(), "No explanation provided"

    def __create_adaptive_reasoning_decision_prompt(
        self,
        current_steps: int,
        tools_used: list,
        current_progress: str
    ) -> str:
        """Create prompt for adaptive reasoning decision."""
        tools_used_str = ", ".join(tools_used) if tools_used else "No tools used yet"

        # Use the prompt from i18n and format it with the current context
        base_prompt = self.i18n.retrieve("reasoning", "adaptive_reasoning_decision").format(
            role=self.agent.role,
            goal=self.agent.goal,
            backstory=self.__get_agent_backstory()
        )

        context_prompt = self.i18n.retrieve("reasoning", "adaptive_reasoning_context").format(
            description=self.task.description,
            expected_output=self.task.expected_output,
            current_steps=current_steps,
            tools_used=tools_used_str,
            current_progress=current_progress
        )

        prompt = base_prompt + context_prompt

        return prompt
