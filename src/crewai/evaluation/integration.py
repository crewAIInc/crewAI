"""Integration utilities for agent evaluation.

This module provides utilities for integrating the evaluation framework
with the agent execution process, including callbacks for collecting
execution traces and utilities for running evaluations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable

from crewai.agent import Agent
from crewai.task import Task
from crewai.crew import Crew
from crewai.process import Process
from crewai.tools import BaseTool
from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events.crewai_event_bus import CrewAIEventsBus
from crewai.utilities.events.agent_events import (
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent
)
from crewai.utilities.events.tool_usage_events import (
    ToolUsageStartedEvent,
    ToolUsageFinishedEvent
)
from crewai.utilities.events.knowledge_events import (
    KnowledgeRetrievalStartedEvent,
    KnowledgeRetrievalCompletedEvent
)
from crewai.utilities.events.llm_events import (
    LLMCallStartedEvent,
    LLMCallCompletedEvent
)

# Import only what's needed for type hints, avoid importing AgentEvaluator to prevent circular import
from crewai.evaluation.base_evaluator import (
    AgentEvaluationResult,
    BaseEvaluator,
    EvaluationScore,
    MetricCategory
)


class EvaluationTraceCallback(BaseEventListener):
    """Event listener for collecting execution traces for evaluation.

    This listener attaches to the event bus to collect detailed information
    about the execution process, including agent steps, tool uses, knowledge
    retrievals, and final output - all for use in agent evaluation.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the evaluation trace event listener."""
        if not hasattr(self, "_initialized") or not self._initialized:
            super().__init__()
            self.traces: Dict[str, Dict[str, Any]] = {}
            self.current_agent_id = None
            self.current_task_id = None
            self._initialized = True

    def setup_listeners(self, event_bus: CrewAIEventsBus):
        """Set up listeners for relevant events.

        Args:
            event_bus: The CrewAI events bus to listen to
        """
        @event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(source, event: AgentExecutionStartedEvent):
            self.on_agent_start(event.agent, event.task)

        @event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(source, event: AgentExecutionCompletedEvent):
            self.on_agent_finish(event.agent, event.task, event.output)

        @event_bus.on(ToolUsageStartedEvent)
        def on_tool_started(source, event: ToolUsageStartedEvent):
            self.on_step_start("tool", {"tool_name": event.tool_name, "tool_input": event.tool_args})

        @event_bus.on(ToolUsageFinishedEvent)
        def on_tool_completed(source, event: ToolUsageFinishedEvent):
            self.on_tool_use(event.tool_name, event.tool_args, event.output)
            self.on_step_end(event.output)

        @event_bus.on(KnowledgeRetrievalStartedEvent)
        def on_knowledge_started(source, event: KnowledgeRetrievalStartedEvent):
            self.on_step_start("knowledge_retrieval", {"query": event.query})

        @event_bus.on(KnowledgeRetrievalCompletedEvent)
        def on_knowledge_completed(source, event: KnowledgeRetrievalCompletedEvent):
            self.on_knowledge_retrieval(event.query, event.documents)
            self.on_step_end(event.documents)

        @event_bus.on(LLMCallStartedEvent)
        def on_llm_call_started(source, event: LLMCallStartedEvent):
            self.on_llm_call_start(event.messages, event.tools)

        @event_bus.on(LLMCallCompletedEvent)
        def on_llm_call_completed(source, event: LLMCallCompletedEvent):
            self.on_llm_call_end(event.messages, event.response)

    def on_agent_start(self, agent: Agent, task: Task):
        """Called when an agent starts executing a task.

        Args:
            agent: The agent that is starting execution
            task: The task being executed
        """
        self.current_agent_id = agent.id
        self.current_task_id = task.id

        # Initialize trace for this agent-task pair
        trace_key = f"{agent.id}_{task.id}"
        self.traces[trace_key] = {
            "agent_id": agent.id,
            "task_id": task.id,
            "steps": [],
            "tool_uses": [],
            "knowledge_retrievals": [],
            "llm_calls": [],
            "start_time": datetime.now(),
            "final_output": None
        }

    def on_agent_finish(self, agent: Agent, task: Task, output: Any):
        """Called when an agent finishes executing a task.

        Args:
            agent: The agent that finished execution
            task: The task that was executed
            output: The final output from the agent
        """
        trace_key = f"{agent.id}_{task.id}"
        if trace_key in self.traces:
            self.traces[trace_key]["final_output"] = output
            self.traces[trace_key]["end_time"] = datetime.now()

        self.current_agent_id = None
        self.current_task_id = None

    def on_step_start(self, step_type: str, content: Any = None):
        """Called when a step starts.

        Args:
            step_type: Type of step (e.g., "thinking", "action")
            content: Content of the step
        """
        if not self.current_agent_id or not self.current_task_id:
            return

        trace_key = f"{self.current_agent_id}_{self.current_task_id}"
        if trace_key in self.traces:
            step = {
                "type": step_type,
                "content": content,
                "start_time": datetime.now()
            }
            self.traces[trace_key]["steps"].append(step)

    def on_step_end(self, output: Any = None):
        """Called when a step ends.

        Args:
            output: Output of the step
        """
        if not self.current_agent_id or not self.current_task_id:
            return

        trace_key = f"{self.current_agent_id}_{self.current_task_id}"
        if trace_key in self.traces and self.traces[trace_key]["steps"]:
            current_step = self.traces[trace_key]["steps"][-1]
            current_step["end_time"] = datetime.now()
            if output is not None:
                current_step["output"] = output

    def on_tool_use(self, tool_name: str, tool_args: Dict[str, Any], result: Any):
        """Called when a tool is used.

        Args:
            tool_name: The name of the tool that was used
            tool_args: The arguments passed to the tool
            result: The result from the tool
        """
        if not self.current_agent_id or not self.current_task_id:
            return

        trace_key = f"{self.current_agent_id}_{self.current_task_id}"
        if trace_key in self.traces:
            tool_use = {
                "tool": tool_name,
                "args": tool_args,
                "result": result,
                "timestamp": datetime.now()
            }
            self.traces[trace_key]["tool_uses"].append(tool_use)

    def on_knowledge_retrieval(self, query: str, documents: List[str]):
        """Called when a knowledge retrieval operation is completed.

        Args:
            query: The query that was used for retrieval
            documents: The documents that were retrieved
        """
        if not self.current_agent_id or not self.current_task_id:
            return

        trace_key = f"{self.current_agent_id}_{self.current_task_id}"
        if trace_key not in self.traces:
            return

        self.traces[trace_key]["knowledge_retrievals"].append({
            "query": query,
            "documents": documents,
            "timestamp": datetime.now()
        })

    def on_llm_call_start(self, messages: Union[str, List[Dict[str, Any]]], tools: Optional[List[Dict]] = None):
        """Called when an LLM call is started.

        Args:
            messages: The messages being sent to the LLM
            tools: Any tools provided to the LLM
        """
        if not self.current_agent_id or not self.current_task_id:
            return

        trace_key = f"{self.current_agent_id}_{self.current_task_id}"
        if trace_key not in self.traces:
            return

        # Store the start information in the current step for later completion
        self.current_llm_call = {
            "messages": messages,
            "tools": tools,
            "start_time": datetime.now(),
            "response": None,
            "end_time": None
        }

    def on_llm_call_end(self, messages: Union[str, List[Dict[str, Any]]], response: Any):
        """Called when an LLM call is completed.

        Args:
            messages: The messages that were sent to the LLM
            response: The response from the LLM
        """
        if not self.current_agent_id or not self.current_task_id:
            return

        trace_key = f"{self.current_agent_id}_{self.current_task_id}"
        if trace_key not in self.traces:
            return

        # Calculate token usage if possible (simplified estimation)
        total_tokens = 0
        if hasattr(response, "usage") and hasattr(response.usage, "total_tokens"):
            total_tokens = response.usage.total_tokens

        # Use the stored start time if available, otherwise use the current time
        current_time = datetime.now()
        start_time = None
        if hasattr(self, "current_llm_call") and self.current_llm_call:
            start_time = self.current_llm_call.get("start_time")
        
        # If we don't have a start time, just use the current time for both
        # This means we won't have duration info, but we'll still record the call
        if not start_time:
            start_time = current_time

        # Complete the LLM call record
        llm_call = {
            "messages": messages,
            "response": response,
            "start_time": start_time,
            "end_time": current_time,
            "total_tokens": total_tokens
        }

        # Store the completed LLM call
        self.traces[trace_key]["llm_calls"].append(llm_call)

        # Clear the current LLM call record
        if hasattr(self, "current_llm_call"):
            self.current_llm_call = None

    def get_trace(self, agent_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the execution trace for a specific agent-task pair.

        Args:
            agent_id: ID of the agent
            task_id: ID of the task

        Returns:
            Dict: The execution trace, or None if not found
        """
        trace_key = f"{agent_id}_{task_id}"
        return self.traces.get(trace_key)


def create_evaluation_callbacks() -> EvaluationTraceCallback:
    """Create and register evaluation callbacks with the CrewAI event system.

    Returns:
        EvaluationTraceCallback: The configured evaluation trace listener
    """
    # Get the singleton instance
    callbacks = EvaluationTraceCallback()

    # It will be automatically attached to the event bus when used
    # No need to manually register it as it's a BaseEventListener
    return callbacks


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from crewai.evaluation.base_evaluator import AgentEvaluator

def evaluate_agent(
    agent: Agent,
    task: Task,
    trace: Dict[str, Any],
    evaluator: "AgentEvaluator"
) -> AgentEvaluationResult:
    """Evaluate an agent's performance on a task using collected trace data.

    Args:
        agent: The agent to evaluate
        task: The task that was executed
        trace: The execution trace
        evaluator: The agent evaluator to use

    Returns:
        AgentEvaluationResult: The evaluation results
    """
    return evaluator.evaluate(
        agent=agent,
        task=task,
        execution_trace=trace,
        final_output=trace.get("final_output")
    )


def evaluate_crew(
    crew: Crew,
    traces: Dict[str, Dict[str, Any]],
    create_evaluator: Callable[[], "AgentEvaluator"]
) -> Dict[str, AgentEvaluationResult]:
    """Evaluate all agents in a crew using collected trace data.

    Args:
        crew: The crew to evaluate
        traces: The execution traces for all agents
        create_evaluator: Function to create an agent evaluator

    Returns:
        Dict[str, AgentEvaluationResult]: Mapping of agent IDs to evaluation results
    """
    results = {}

    for agent in crew.agents:
        for task in crew.tasks:
            trace_key = f"{agent.id}_{task.id}"
            if trace_key in traces:
                # Create a fresh evaluator for each agent-task pair
                evaluator = create_evaluator()

                results[trace_key] = evaluate_agent(
                    agent=agent,
                    task=task,
                    trace=traces[trace_key],
                    evaluator=evaluator
                )

    return results


def create_default_evaluator(llm=None, with_meta_evaluator=True): # -> 'AgentEvaluator'
    """Create a default evaluator with standard metrics.

    Args:
        llm: The LLM to use for evaluations (optional)
        with_meta_evaluator: Whether to include a meta-evaluator

    Returns:
        List[BaseEvaluator]: A list of configured evaluators
    """
    from crewai.evaluation.evaluators import GoalAlignmentEvaluator, KnowledgeRetrievalEvaluator, SemanticQualityEvaluator
    from crewai.evaluation.evaluators_tools import ToolUsageEvaluator, StepEfficiencyEvaluator
    from crewai.evaluation.evaluators_reasoning import ReasoningEfficiencyEvaluator
    from crewai.evaluation.meta_evaluator import MetaEvaluator

    evaluators = [
        GoalAlignmentEvaluator(llm=llm),
        SemanticQualityEvaluator(llm=llm),
        ToolUsageEvaluator(llm=llm),
        StepEfficiencyEvaluator(llm=llm),
        KnowledgeRetrievalEvaluator(llm=llm),
        ReasoningEfficiencyEvaluator(llm=llm),
    ]

    meta_evaluator = MetaEvaluator(llm=llm) if with_meta_evaluator else None

    # Import here to avoid circular imports
    from crewai.evaluation.base_evaluator import AgentEvaluator
    return AgentEvaluator(evaluators=evaluators, meta_evaluator=meta_evaluator)


def evaluate_execution_from_callbacks(
    agent: Agent,
    task: Task,
    callbacks: EvaluationTraceCallback,
    evaluator: Optional["AgentEvaluator"] = None
) -> AgentEvaluationResult:
    """Evaluate an agent execution from collected callback data.

    Args:
        agent: The agent to evaluate
        task: The task that was executed
        callbacks: The callback object with collected traces
        evaluator: The evaluator to use (or a default one will be created)

    Returns:
        AgentEvaluationResult: The evaluation results
    """
    trace = callbacks.get_trace(agent.id, task.id)

    if not trace:
        raise ValueError(f"No trace found for agent {agent.id} and task {task.id}")

    evaluator = evaluator or create_default_evaluator()

    return evaluate_agent(
        agent=agent,
        task=task,
        trace=trace,
        evaluator=evaluator
    )