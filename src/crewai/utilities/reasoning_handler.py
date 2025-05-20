import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.task import Task
from crewai.utilities import I18N


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
        self.logger = logging.getLogger(__name__)
        self.i18n = I18N()

    def handle_agent_reasoning(self) -> AgentReasoningOutput:
        """
        Public method for the reasoning process that creates and refines a plan
        for the task until the agent is ready to execute it.
        
        Returns:
            AgentReasoningOutput: The output of the agent reasoning process.
        """
        return self.__handle_agent_reasoning()
    
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
        
        if self.agent.llm.supports_function_calling():
            response = self.__call_with_function(reasoning_prompt, "initial_plan")
            return response.plan, response.ready
        else:
            response = self.agent.llm.call(
                [
                    {"role": "system", "content": self.i18n.retrieve("reasoning", "initial_plan")},
                    {"role": "user", "content": reasoning_prompt}
                ]
            )
            
            return self.__parse_reasoning_response(response)
    
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
            refine_prompt = self.__create_refine_prompt(plan)
            
            if self.agent.llm.supports_function_calling():
                response = self.__call_with_function(refine_prompt, "refine_plan")
                plan, ready = response.plan, response.ready
            else:
                response = self.agent.llm.call(
                    [
                        {"role": "system", "content": self.i18n.retrieve("reasoning", "refine_plan")},
                        {"role": "user", "content": refine_prompt}
                    ]
                )
                plan, ready = self.__parse_reasoning_response(response)
                
            attempt += 1
            
            if max_attempts is not None and attempt >= max_attempts:
                self.logger.warning(
                    f"Agent reasoning reached maximum attempts ({max_attempts}) without being ready. Proceeding with current plan."
                )
                break
        
        return plan, ready

    def __call_with_function(self, prompt: str, prompt_type: str) -> ReasoningFunction:
        """
        Calls the LLM with function calling to get a reasoning plan.
        
        Args:
            prompt: The prompt to send to the LLM.
            prompt_type: The type of prompt (initial_plan or refine_plan).
            
        Returns:
            ReasoningFunction: The reasoning function response.
        """
        self.logger.debug(f"Using function calling for {prompt_type} reasoning")
        
        function_schema = {
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
        
        try:
            response = self.agent.llm.call(
                [
                    {"role": "system", "content": self.i18n.retrieve("reasoning", prompt_type)},
                    {"role": "user", "content": prompt}
                ],
                tools=[function_schema]
            )
            
            self.logger.debug(f"Function calling response: {response[:100]}...")
            
            try:
                result = json.loads(response)
                if "plan" in result and "ready" in result:
                    return ReasoningFunction(plan=result["plan"], ready=result["ready"])
            except (json.JSONDecodeError, KeyError):
                pass
                
            return ReasoningFunction(
                plan=response,
                ready="READY: I am ready to execute the task." in response
            )
            
        except Exception as e:
            self.logger.warning(f"Error during function calling: {str(e)}. Falling back to text parsing.")
            
            try:
                fallback_response = self.agent.llm.call(
                    [
                        {"role": "system", "content": self.i18n.retrieve("reasoning", prompt_type)},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                return ReasoningFunction(
                    plan=fallback_response,
                    ready="READY: I am ready to execute the task." in fallback_response
                )
            except Exception as inner_e:
                self.logger.error(f"Error during fallback text parsing: {str(inner_e)}")
                return ReasoningFunction(
                    plan="Failed to generate a plan due to an error.",
                    ready=True  # Default to ready to avoid getting stuck
                )

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
