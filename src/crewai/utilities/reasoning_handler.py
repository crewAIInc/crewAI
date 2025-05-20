import logging
from typing import Any, Optional

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
        self.task = task
        self.agent = agent
        self.logger = logging.getLogger(__name__)

    def _handle_agent_reasoning(self) -> AgentReasoningOutput:
        """
        Handles the agent reasoning process by creating a detailed plan for the task
        and determining if the agent is ready to execute it.
        """
        reasoning_prompt = self._create_reasoning_prompt()
        
        response = self.agent.llm.call(
            [
                {"role": "system", "content": "You are a helpful assistant that helps an agent create a plan for a task."},
                {"role": "user", "content": reasoning_prompt}
            ]
        )
        
        plan, ready = self._parse_reasoning_response(response)
        
        attempt = 1
        max_attempts = self.agent.max_reasoning_attempts
        
        while not ready and (max_attempts is None or attempt < max_attempts):
            refine_prompt = self._create_refine_prompt(plan)
            response = self.agent.llm.call(
                [
                    {"role": "system", "content": "You are a helpful assistant that helps an agent refine a plan for a task."},
                    {"role": "user", "content": refine_prompt}
                ]
            )
            plan, ready = self._parse_reasoning_response(response)
            attempt += 1
            
            if max_attempts is not None and attempt >= max_attempts:
                self.logger.warning(
                    f"Agent reasoning reached maximum attempts ({max_attempts}) without being ready. Proceeding with current plan."
                )
                break
        
        reasoning_plan = ReasoningPlan(plan=plan, ready=ready)
        return AgentReasoningOutput(plan=reasoning_plan)

    def _create_reasoning_prompt(self) -> str:
        """Creates a prompt for the agent to reason about the task."""
        return f"""
        You are functioning as {self.agent.role}. Your goal is: {self.agent.goal}.
        
        You have been assigned the following task:
        {self.task.description}
        
        Expected output:
        {self.task.expected_output}
        
        Available tools: {', '.join([tool.name for tool in (self.task.tools or [])])}
        
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

    def _create_refine_prompt(self, current_plan: str) -> str:
        """Creates a prompt for the agent to refine its reasoning plan."""
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

    def _parse_reasoning_response(self, response: str) -> tuple[str, bool]:
        """
        Parses the reasoning response to extract the plan and whether
        the agent is ready to execute the task.
        """
        plan = response
        ready = False
        
        if "READY: I am ready to execute the task." in response:
            ready = True
        
        return plan, ready
