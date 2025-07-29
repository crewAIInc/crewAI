import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict


from .execution_context_tracker import ExecutionContextTracker, PrivacyFilter


@dataclass
class TraceEvent:
    """Individual trace event payload"""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat() + "Z"
    )
    type: str = ""
    source_fingerprint: Optional[str] = None
    source_type: Optional[str] = None
    fingerprint_metadata: Optional[Dict[str, Any]] = None
    correlation_ids: Dict[str, str] = field(default_factory=dict)
    event_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TraceEventFactory:
    """Single responsibility: Create trace events with context and privacy filtering"""

    def __init__(
        self, context_tracker: ExecutionContextTracker, privacy_filter: PrivacyFilter
    ):
        self.context_tracker = context_tracker
        self.privacy_filter = privacy_filter

    def create_event(
        self, event_type: str, source: Any, event: Any, trace_id: str
    ) -> TraceEvent:
        """Create a standardized trace event with full context"""
        trace_event = TraceEvent(
            type=event_type,
            source_fingerprint=getattr(event, "source_fingerprint", None),
            source_type=getattr(event, "source_type", None),
            fingerprint_metadata=getattr(event, "fingerprint_metadata", None),
        )

        trace_event.correlation_ids["trace_id"] = trace_id

        context_correlations = self.context_tracker.get_context_correlations()
        trace_event.correlation_ids.update(context_correlations)

        # Add source-specific correlations
        if hasattr(source, "id"):
            trace_event.correlation_ids["source_id"] = str(source.id)
        if hasattr(source, "crew") and hasattr(source.crew, "id"):
            trace_event.correlation_ids["crew_id"] = str(source.crew.id)
        if hasattr(source, "flow") and hasattr(source.flow, "id"):
            trace_event.correlation_ids["flow_id"] = str(source.flow.id)

        # Build event data based on event type
        try:
            trace_event.event_data = self._build_event_data(event_type, source, event)
        except Exception as e:
            print(f"Error building event data: {e} for event type: {event_type}")
            trace_event.event_data = {}

        return trace_event

    def _build_event_data(
        self, event_type: str, source: Any, event: Any
    ) -> Dict[str, Any]:
        """Build event data based on event type"""
        event_data = {}

        if event_type == "crew_kickoff_started":
            event_data = {
                "crew_name": getattr(event, "crew_name", "Unknown Crew"),
                "inputs": self.privacy_filter.filter_content(
                    str(getattr(event, "inputs", ""))
                )
                if hasattr(event, "inputs") and event.inputs
                else None,
            }

        elif event_type == "crew_kickoff_completed":
            event_data = {
                "crew_name": getattr(event, "crew_name", "Unknown Crew"),
                "output": self.privacy_filter.filter_content(str(event.output.raw))
                if hasattr(event, "output") and event.output
                else None,
                "token_usage": source.token_usage.model_dump()
                if hasattr(source, "token_usage") and source.token_usage
                else None,
            }

        elif event_type == "task_started":
            print("event type:", type(event))
            event_data = {
                # Quick access fields (for dashboards/alerts)
                "task_description": self.privacy_filter.filter_content(
                    str(getattr(source, "description", ""))
                ),
                "agent_role": getattr(source.agent, "role", None)
                if hasattr(source, "agent")
                else None,
                "expected_output": self.privacy_filter.filter_content(
                    str(getattr(source, "expected_output", ""))
                ),
            }

        elif event_type == "task_completed":
            event_data = {
                "task_id": str(getattr(source, "id", "unknown")),
                "success": True,
                "output_summary": self.privacy_filter.filter_content(
                    str(event.output.raw)
                )
                if hasattr(event, "output") and event.output
                else None,
            }

        elif event_type == "agent_execution_started":
            event_data = {
                "agent_role": getattr(event.agent, "role", "Unknown Agent")
                if hasattr(event, "agent")
                else "Unknown Agent",
                "task_description": self.privacy_filter.filter_content(
                    str(getattr(event, "task_prompt", ""))
                )
                if hasattr(event, "task_prompt")
                else None,
                "tools_available": [tool.name for tool in event.tools]
                if hasattr(event, "tools") and event.tools
                else [],
            }

        elif event_type == "agent_execution_completed":
            event_data = {
                "agent_role": getattr(event.agent, "role", "Unknown Agent")
                if hasattr(event, "agent")
                else "Unknown Agent",
                "success": True,
                "output_summary": self.privacy_filter.filter_content(
                    str(getattr(event, "output", ""))
                )
                if hasattr(event, "output")
                else None,
            }

        elif event_type == "llm_call_started":
            model = getattr(event, "model", "unknown")
            if model == "unknown" and hasattr(source, "llm"):
                model = getattr(source.llm, "model", "unknown")

            event_data = {
                "event": self.serialize_for_tracing(event),
            }

        elif event_type == "llm_call_completed":
            model = getattr(event, "model", "unknown")
            if model == "unknown" and hasattr(source, "llm"):
                model = getattr(source.llm, "model", "unknown")
            event_data = {
                "model": model,
                "response": self.privacy_filter.filter_content(str(event.response))
                if hasattr(event, "response")
                else None,
                "response_cost": getattr(event, "response_cost", None),
            }

        elif event_type == "tool_usage_started":
            event_data = {
                "tool_name": getattr(event, "tool_name", "unknown"),
                "tool_executor": getattr(event, "agent_role", None),
                "tool_args": self.privacy_filter.filter_content(
                    str(getattr(event, "tool_args", {}))
                )
                if hasattr(event, "tool_args")
                else None,
                "args_count": len(getattr(event, "tool_args", {}))
                if hasattr(event, "tool_args")
                else 0,
            }

        elif event_type == "tool_usage_finished":
            event_data = {
                "tool_name": getattr(event, "tool_name", "unknown"),
                "success": True,
                "result": self.privacy_filter.filter_content(
                    str(getattr(event, "output", ""))
                )
                if hasattr(event, "output")
                else None,
                "result_length": len(str(getattr(event, "output", "")))
                if hasattr(event, "output")
                else 0,
                "duration_ms": getattr(event, "duration_ms", None),
                "from_cache": getattr(event, "from_cache", False),
            }

        elif event_type == "flow_kickoff_started":
            event_data = {
                "flow_name": getattr(source, "name", "Unknown Flow"),
                "flow_class": source.__class__.__name__,
                "inputs": self.privacy_filter.filter_content(
                    str(getattr(event, "inputs", ""))
                ),
                "method_count": len(
                    [
                        m
                        for m in dir(source)
                        if hasattr(getattr(source, m), "_flow_method_type")
                    ]
                ),
            }

        elif event_type == "flow_method_started":
            event_data = {
                "method_name": getattr(event, "method_name", "unknown"),
                "method_type": getattr(event, "method_type", "unknown"),
                "flow_name": getattr(source, "name", "Unknown Flow"),
                "dependencies": getattr(event, "dependencies", []),
            }

        # # Add timing data if available
        # if hasattr(event, "duration_ms"):
        #     event_data["duration_ms"] = event.duration_ms

        return event_data

    def serialize_for_tracing(
        self, obj, exclude_patterns=None, max_depth=5, current_depth=0
    ):
        """Comprehensive serialization for tracing - capture everything safely"""
        exclude_patterns = exclude_patterns or {
            "crew",
            "agent_executor",
            "cache_handler",
            "tools_handler",
            "llm",
            "_crew",
            "_agent",
            # LLM-related exclusions
            "client",
            "_client",
            "session",
            "_session",
            "callbacks",
            "callback_manager",
            # Agent-related exclusions
            "memory",
            "_memory",
            "step_callback",
            "execution_logs",
            # Tool-related exclusions
            "func",
            "_func",
            "coroutine",
            "_coroutine",
            # Common circular reference patterns
        }

        if current_depth > max_depth:
            return f"<max_depth_reached: {type(obj).__name__}>"

        try:
            if hasattr(obj, "model_dump"):
                # Pydantic model - get everything except circular refs
                return obj.model_dump(
                    mode="json",
                    exclude=exclude_patterns,
                    exclude_unset=False,  # Keep unset fields for full context
                    exclude_none=False,  # Keep None values for completeness
                    by_alias=True,
                )
            elif isinstance(obj, dict):
                return {
                    k: self.serialize_for_tracing(
                        v, exclude_patterns, max_depth, current_depth + 1
                    )
                    for k, v in obj.items()
                    if k not in exclude_patterns
                }
            elif isinstance(obj, (list, tuple)):
                return [
                    self.serialize_for_tracing(
                        item, exclude_patterns, max_depth, current_depth + 1
                    )
                    for item in obj
                ]
            elif hasattr(obj, "__dict__"):
                # Custom objects - serialize their attributes
                return {
                    f"__{type(obj).__name__}__": {
                        k: self.serialize_for_tracing(
                            v, exclude_patterns, max_depth, current_depth + 1
                        )
                        for k, v in obj.__dict__.items()
                        if k not in exclude_patterns and not k.startswith("_")
                    }
                }
            else:
                return obj

        except Exception as e:
            return f"<serialization_error: {type(obj).__name__}: {str(e)}>"
