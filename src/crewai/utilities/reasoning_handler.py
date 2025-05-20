import logging
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.task import Task


class ReasoningPlan(BaseModel):
    """Model representing a reasoning plan for a task."""
    plan: str = Field(description="The detailed reasoning plan for the task.")
    ready: bool = Field(description="Whether the agent is ready to execute the task.")


class AgentReasoningOutput(BaseModel):
    """Model representing the output of the agent reasoning process."""
    plan: ReasoningPlan = Field(description="The reasoning plan for the task.")


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
        self.__system_prompts = {
            "initial_plan": "You are a helpful assistant that helps an agent create a plan for a task.",
            "refine_plan": "You are a helpful assistant that helps an agent refine a plan for a task."
        }

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
        
        response = self.agent.llm.call(
            [
                {"role": "system", "content": self.__system_prompts["initial_plan"]},
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
            response = self.agent.llm.call(
                [
                    {"role": "system", "content": self.__system_prompts["refine_plan"]},
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

    def __create_reasoning_prompt(self) -> str:
        """
        Creates a prompt for the agent to reason about the task.
        
        Returns:
            str: The reasoning prompt.
        """
        available_tools = self.__format_available_tools()
        
        return f"""
        You are functioning as {self.agent.role}. Your goal is: {self.agent.goal}.
        
        You have been assigned the following task:
        {self.task.description}
        
        Expected output:
        {self.task.expected_output}
        
        Available tools: {available_tools}
        
        Before executing this task, create a detailed plan that outlines:
        1. Your understanding of the task
        2. The key steps you'll take to complete it
        3. How you'll approach any challenges that might arise
        4. How you'll use the available tools
        5. The expected outcome
        
        After creating your plan, assess whether you feel ready to execute the task.
        Conclude with one of these statements:
        - "READY: I am ready to execute the task."
        - "NOT READY: I need to refine my plan because [specific reason]."
        """

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
        return f"""
        You are functioning as {self.agent.role}. Your goal is: {self.agent.goal}.
        
        You created the following plan for this task:
        {current_plan}
        
        However, you indicated that you're not ready to execute the task yet.
        
        Please refine your plan further, addressing any gaps or uncertainties.
        
        After refining your plan, assess whether you feel ready to execute the task.
        Conclude with one of these statements:
        - "READY: I am ready to execute the task."
        - "NOT READY: I need to refine my plan further because [specific reason]."
        """

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
