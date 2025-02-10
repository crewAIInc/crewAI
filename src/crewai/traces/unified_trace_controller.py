import inspect
import os
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

from crewai.traces.context import TraceContext
from crewai.traces.enums import CrewType, RunType, TraceType
from crewai.traces.models import (
    CrewTrace,
    FlowStepIO,
    LLMRequest,
    LLMResponse,
    ToolCall,
)


class UnifiedTraceController:
    """Controls and manages trace execution and recording.

    This class handles the lifecycle of traces including creation, execution tracking,
    and recording of results for various types of operations (LLM calls, tool calls, flow steps).
    """

    def __init__(
        self,
        trace_type: TraceType,
        run_type: RunType,
        crew_type: CrewType,
        run_id: str,
        deployment_instance_id: str = os.environ.get(
            "CREWAI_DEPLOYMENT_INSTANCE_ID", ""
        ),
        parent_trace_id: Optional[str] = None,
        agent_role: Optional[str] = "unknown",
        task_name: Optional[str] = None,
        task_description: Optional[str] = None,
        task_id: Optional[str] = None,
        flow_step: Dict[str, Any] = {},
        tool_calls: List[ToolCall] = [],
        **context: Any,
    ) -> None:
        """Initialize a new trace controller.

        Args:
            trace_type: Type of trace being recorded.
            run_type: Type of run being executed.
            crew_type: Type of crew executing the trace.
            run_id: Unique identifier for the run.
            deployment_instance_id: Optional deployment instance identifier.
            parent_trace_id: Optional parent trace identifier for nested traces.
            agent_role: Role of the agent executing the trace.
            task_name: Optional name of the task being executed.
            task_description: Optional description of the task.
            task_id: Optional unique identifier for the task.
            flow_step: Optional flow step information.
            tool_calls: Optional list of tool calls made during execution.
            **context: Additional context parameters.
        """
        self.trace_id = str(uuid4())
        self.run_id = run_id
        self.parent_trace_id = parent_trace_id
        self.trace_type = trace_type
        self.run_type = run_type
        self.crew_type = crew_type
        self.context = context
        self.agent_role = agent_role
        self.task_name = task_name
        self.task_description = task_description
        self.task_id = task_id
        self.deployment_instance_id = deployment_instance_id
        self.children: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error: Optional[str] = None
        self.tool_calls = tool_calls
        self.flow_step = flow_step
        self.status: str = "running"

    @classmethod
    def get_task_traces(cls, task_id: str) -> List["UnifiedTraceController"]:
        """Get all traces for a specific task.

        Args:
            task_id: The ID of the task to get traces for

        Returns:
            List of traces associated with the task
        """
        if not hasattr(cls, "_task_traces"):
            cls._task_traces = {}
        return cls._task_traces.get(task_id, [])

    @classmethod
    def clear_task_traces(cls, task_id: str) -> None:
        """Clear traces for a specific task.

        Args:
            task_id: The ID of the task to clear traces for
        """
        if hasattr(cls, "_task_traces") and task_id in cls._task_traces:
            del cls._task_traces[task_id]

    def _get_current_trace(self) -> "UnifiedTraceController":
        return TraceContext.get_current()

    def start_trace(self) -> "UnifiedTraceController":
        """Start the trace execution.

        Returns:
            UnifiedTraceController: Self for method chaining.
        """
        self.start_time = datetime.now(UTC)
        return self

    def end_trace(self, result: Any = None, error: Optional[str] = None) -> None:
        """End the trace execution and record results.

        Args:
            result: Optional result from the trace execution.
            error: Optional error message if the trace failed.
        """
        self.end_time = datetime.now(UTC)
        self.status = "error" if error else "completed"
        self.error = error
        self._record_trace(result)

    def add_child_trace(self, child_trace: Dict[str, Any]) -> None:
        """Add a child trace to this trace's execution history.

        Args:
            child_trace: The child trace information to add.
        """
        self.children.append(child_trace)

    def to_crew_trace(self) -> CrewTrace:
        """Convert to CrewTrace format for storage.

        Returns:
            CrewTrace: The trace data in CrewTrace format.
        """
        latency_ms = None

        if self.tool_calls and hasattr(self.tool_calls[0], "start_time"):
            self.start_time = self.tool_calls[0].start_time

        if self.start_time and self.end_time:
            latency_ms = int((self.end_time - self.start_time).total_seconds() * 1000)

        request = None
        response = None
        flow_step_obj = None

        if self.trace_type in [TraceType.LLM_CALL, TraceType.TOOL_CALL]:
            request = LLMRequest(
                model=self.context.get("model", "unknown"),
                messages=self.context.get("messages", []),
                temperature=self.context.get("temperature"),
                max_tokens=self.context.get("max_tokens"),
                stop_sequences=self.context.get("stop_sequences"),
            )
            if "response" in self.context:
                response = LLMResponse(
                    content=self.context["response"].get("content", ""),
                    finish_reason=self.context["response"].get("finish_reason"),
                )

        elif self.trace_type == TraceType.FLOW_STEP:
            flow_step_obj = FlowStepIO(
                function_name=self.flow_step.get("function_name", "unknown"),
                inputs=self.flow_step.get("inputs", {}),
                outputs={"result": self.context.get("response")},
                metadata=self.flow_step.get("metadata", {}),
            )

        return CrewTrace(
            deployment_instance_id=self.deployment_instance_id,
            trace_id=self.trace_id,
            task_id=self.task_id,
            run_id=self.run_id,
            agent_role=self.agent_role,
            task_name=self.task_name,
            task_description=self.task_description,
            trace_type=self.trace_type.value,
            crew_type=self.crew_type.value,
            run_type=self.run_type.value,
            start_time=self.start_time,
            end_time=self.end_time,
            latency_ms=latency_ms,
            request=request,
            response=response,
            flow_step=flow_step_obj,
            tool_calls=self.tool_calls,
            tokens_used=self.context.get("tokens_used"),
            prompt_tokens=self.context.get("prompt_tokens"),
            completion_tokens=self.context.get("completion_tokens"),
            status=self.status,
            error=self.error,
        )

    def _record_trace(self, result: Any = None) -> None:
        """Record the trace using PlusClient.

        Args:
            result: Optional result to include in the trace.
        """
        if result:
            self.context["response"] = result

        # TODO: Add trace to record_task_finished


def should_trace() -> bool:
    """Check if tracing is enabled via environment variable."""
    return os.getenv("CREWAI_ENABLE_TRACING", "false").lower() == "true"


# Crew main trace
def init_crew_main_trace(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to initialize and track the main crew execution trace.

    This decorator sets up the trace context for the main crew execution,
    handling both synchronous and asynchronous crew operations.

    Args:
        func: The crew function to be traced.

    Returns:
        Wrapped function that creates and manages the main crew trace context.
    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not should_trace():
            return func(self, *args, **kwargs)

        trace = build_crew_main_trace(self)
        with TraceContext.set_current(trace):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                trace.end_trace(error=str(e))
                raise

    return wrapper


def build_crew_main_trace(self: Any) -> "UnifiedTraceController":
    """Build the main trace controller for a crew execution.

    This function creates a trace controller configured for the main crew execution,
    handling different run types (kickoff, test, train) and maintaining context.

    Args:
        self: The crew instance.

    Returns:
        UnifiedTraceController: The configured trace controller for the crew.
    """
    run_type = RunType.KICKOFF
    if hasattr(self, "_test") and self._test:
        run_type = RunType.TEST
    elif hasattr(self, "_train") and self._train:
        run_type = RunType.TRAIN

    current_trace = TraceContext.get_current()

    trace = UnifiedTraceController(
        trace_type=TraceType.LLM_CALL,
        run_type=run_type,
        crew_type=current_trace.crew_type if current_trace else CrewType.CREW,
        run_id=current_trace.run_id if current_trace else str(self.id),
        parent_trace_id=current_trace.trace_id if current_trace else None,
    )
    return trace


# Flow main trace
def init_flow_main_trace(
    func: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Decorator to initialize and track the main flow execution trace.

    Args:
        func: The async flow function to be traced.

    Returns:
        Wrapped async function that creates and manages the main flow trace context.
    """

    @wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not should_trace():
            return await func(self, *args, **kwargs)

        trace = build_flow_main_trace(self, *args, **kwargs)
        with TraceContext.set_current(trace):
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                trace.end_trace(error=str(e))
                raise

    return wrapper


def build_flow_main_trace(
    self: Any, *args: Any, **kwargs: Any
) -> "UnifiedTraceController":
    """Build the main trace controller for a flow execution.

    Args:
        self: The flow instance.
        *args: Variable positional arguments.
        **kwargs: Variable keyword arguments.

    Returns:
        UnifiedTraceController: The configured trace controller for the flow.
    """
    current_trace = TraceContext.get_current()
    trace = UnifiedTraceController(
        trace_type=TraceType.FLOW_STEP,
        run_id=current_trace.run_id if current_trace else str(self.flow_id),
        parent_trace_id=current_trace.trace_id if current_trace else None,
        crew_type=CrewType.FLOW,
        run_type=RunType.KICKOFF,
        context={
            "crew_name": self.__class__.__name__,
            "inputs": kwargs.get("inputs", {}),
            "agents": [],
            "tasks": [],
        },
    )
    return trace


# Flow step trace
def trace_flow_step(
    func: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Decorator to trace individual flow step executions.

    Args:
        func: The async flow step function to be traced.

    Returns:
        Wrapped async function that creates and manages the flow step trace context.
    """

    @wraps(func)
    async def wrapper(
        self: Any,
        method_name: str,
        method: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not should_trace():
            return await func(self, method_name, method, *args, **kwargs)

        trace = build_flow_step_trace(self, method_name, method, *args, **kwargs)
        with TraceContext.set_current(trace):
            trace.start_trace()
            try:
                result = await func(self, method_name, method, *args, **kwargs)
                trace.end_trace(result=result)
                return result
            except Exception as e:
                trace.end_trace(error=str(e))
                raise

    return wrapper


def build_flow_step_trace(
    self: Any, method_name: str, method: Callable[..., Any], *args: Any, **kwargs: Any
) -> "UnifiedTraceController":
    """Build a trace controller for an individual flow step.

    Args:
        self: The flow instance.
        method_name: Name of the method being executed.
        method: The actual method being executed.
        *args: Variable positional arguments.
        **kwargs: Variable keyword arguments.

    Returns:
        UnifiedTraceController: The configured trace controller for the flow step.
    """
    current_trace = TraceContext.get_current()

    # Get method signature
    sig = inspect.signature(method)
    params = list(sig.parameters.values())

    # Create inputs dictionary mapping parameter names to values
    method_params = [p for p in params if p.name != "self"]
    inputs: Dict[str, Any] = {}

    # Map positional args to their parameter names
    for i, param in enumerate(method_params):
        if i < len(args):
            inputs[param.name] = args[i]

    # Add keyword arguments
    inputs.update(kwargs)

    trace = UnifiedTraceController(
        trace_type=TraceType.FLOW_STEP,
        run_type=current_trace.run_type if current_trace else RunType.KICKOFF,
        crew_type=current_trace.crew_type if current_trace else CrewType.FLOW,
        run_id=current_trace.run_id if current_trace else str(self.flow_id),
        parent_trace_id=current_trace.trace_id if current_trace else None,
        flow_step={
            "function_name": method_name,
            "inputs": inputs,
            "metadata": {
                "crew_name": self.__class__.__name__,
            },
        },
    )
    return trace


# LLM trace
def trace_llm_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to trace LLM calls.

    Args:
        func: The function to trace.

    Returns:
        Wrapped function that creates and manages the LLM call trace context.
    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not should_trace():
            return func(self, *args, **kwargs)

        trace = build_llm_trace(self, *args, **kwargs)
        with TraceContext.set_current(trace):
            trace.start_trace()
            try:
                response = func(self, *args, **kwargs)
                # Extract relevant data from response
                trace_response = {
                    "content": response["choices"][0]["message"]["content"],
                    "finish_reason": response["choices"][0].get("finish_reason"),
                }

                # Add usage metrics to context
                if "usage" in response:
                    trace.context["tokens_used"] = response["usage"].get(
                        "total_tokens", 0
                    )
                    trace.context["prompt_tokens"] = response["usage"].get(
                        "prompt_tokens", 0
                    )
                    trace.context["completion_tokens"] = response["usage"].get(
                        "completion_tokens", 0
                    )

                trace.end_trace(trace_response)
                return response
            except Exception as e:
                trace.end_trace(error=str(e))
                raise

    return wrapper


def build_llm_trace(
    self: Any, params: Dict[str, Any], *args: Any, **kwargs: Any
) -> Any:
    """Build a trace controller for an LLM call.

    Args:
        self: The LLM instance.
        params: The parameters for the LLM call.
        *args: Variable positional arguments.
        **kwargs: Variable keyword arguments.

    Returns:
        UnifiedTraceController: The configured trace controller for the LLM call.
    """
    current_trace = TraceContext.get_current()
    agent, task = self._get_execution_context()

    # Get new messages and tool results
    new_messages = self._get_new_messages(params.get("messages", []))
    new_tool_results = self._get_new_tool_results(agent)

    # Create trace context
    trace = UnifiedTraceController(
        trace_type=TraceType.TOOL_CALL if new_tool_results else TraceType.LLM_CALL,
        crew_type=current_trace.crew_type if current_trace else CrewType.CREW,
        run_type=current_trace.run_type if current_trace else RunType.KICKOFF,
        run_id=current_trace.run_id if current_trace else str(uuid4()),
        parent_trace_id=current_trace.trace_id if current_trace else None,
        agent_role=agent.role if agent else "unknown",
        task_id=str(task.id) if task else None,
        task_name=task.name if task else None,
        task_description=task.description if task else None,
        model=self.model,
        messages=new_messages,
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        stop_sequences=self.stop,
        tool_calls=[
            ToolCall(
                name=result["tool_name"],
                arguments=result["tool_args"],
                output=str(result["result"]),
                start_time=result.get("start_time", ""),
                end_time=datetime.now(UTC),
            )
            for result in new_tool_results
        ],
    )
    return trace
