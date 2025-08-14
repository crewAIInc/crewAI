from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.crew_events import CrewKickoffStartedEvent, CrewKickoffCompletedEvent
from crewai.utilities.events.task_events import TaskStartedEvent, TaskCompletedEvent
from crewai.utilities.events.agent_events import AgentExecutionStartedEvent, AgentExecutionCompletedEvent
from crewai.utilities.events.llm_events import LLMCallStartedEvent, LLMCallCompletedEvent
from crewai.utilities.events.tool_events import ToolUsageEvent

from crewai.utilities.events.generic_workflow_events import (
    WorkflowStartedEvent,
    WorkflowCompletedEvent,
    TaskStartedEvent as GenericTaskStartedEvent, # Alias to avoid name collision
    TaskCompletedEvent as GenericTaskCompletedEvent, # Alias to avoid name collision
    AgentActionOccurredEvent,
)

from typing import Callable, Any

class CrewAIEventAdapter:
    """
    Adapts CrewAI's native events into generic workflow events.
    """
    def __init__(self, generic_event_publisher: Callable[[Any], None]):
        self.generic_event_publisher = generic_event_publisher
        self.setup_listeners()
        print("CrewAI Event Adapter initialized and listening.")

    def setup_listeners(self):
        """Register listeners for CrewAI native events."""
        crewai_event_bus.on(CrewKickoffStartedEvent)(self._handle_crew_started)
        crewai_event_bus.on(CrewKickoffCompletedEvent)(self._handle_crew_completed)
        crewai_event_bus.on(TaskStartedEvent)(self._handle_task_started)
        crewai_event_bus.on(TaskCompletedEvent)(self._handle_task_completed)
        crewai_event_bus.on(AgentExecutionStartedEvent)(self._handle_agent_execution_started)
        crewai_event_bus.on(AgentExecutionCompletedEvent)(self._handle_agent_execution_completed)
        crewai_event_bus.on(LLMCallStartedEvent)(self._handle_llm_call_started)
        crewai_event_bus.on(LLMCallCompletedEvent)(self._handle_llm_call_completed)
        crewai_event_bus.on(ToolUsageEvent)(self._handle_tool_usage)

    def _handle_crew_started(self, source, event):
        workflow_id = getattr(source, 'id', str(source)) # Assuming crew has an ID or can be represented as string
        workflow_name = getattr(source, 'name', 'Unknown Workflow')
        generic_event = WorkflowStartedEvent(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            metadata={"crew_source_type": type(source).__name__}
        )
        self.generic_event_publisher(generic_event)

    def _handle_crew_completed(self, source, event):
        workflow_id = getattr(source, 'id', str(source))
        workflow_name = getattr(source, 'name', 'Unknown Workflow')
        success = True # CrewAI doesn't explicitly pass success/failure in this event, assume success for now
        generic_event = WorkflowCompletedEvent(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            success=success,
            metadata={"crew_source_type": type(source).__name__}
        )
        self.generic_event_publisher(generic_event)

    def _handle_task_started(self, source, event):
        task = getattr(event, 'task', None)
        if not task: return
        agent = getattr(task, 'agent', None)
        if not agent: return

        workflow_id = getattr(source, 'id', 'unknown_workflow') # Need to get workflow_id from context
        task_id = getattr(task, 'id', str(task))
        task_description = getattr(task, 'description', 'No description')
        assigned_agent_id = getattr(agent, 'id', str(agent))
        assigned_agent_role = getattr(agent, 'role', 'Unknown Role')

        generic_event = GenericTaskStartedEvent(
            workflow_id=workflow_id,
            task_id=task_id,
            task_description=task_description,
            assigned_agent_id=assigned_agent_id,
            assigned_agent_role=assigned_agent_role,
            metadata={"task_source_type": type(task).__name__}
        )
        self.generic_event_publisher(generic_event)

    def _handle_task_completed(self, source, event):
        task = getattr(event, 'task', None)
        if not task: return
        agent = getattr(task, 'agent', None)
        if not agent: return

        workflow_id = getattr(source, 'id', 'unknown_workflow') # Need to get workflow_id from context
        task_id = getattr(task, 'id', str(task))
        task_description = getattr(task, 'description', 'No description')
        assigned_agent_id = getattr(agent, 'id', str(agent))
        assigned_agent_role = getattr(agent, 'role', 'Unknown Role')
        output = getattr(event, 'output', None)
        success = True # CrewAI doesn't explicitly pass success/failure in this event, assume success for now

        generic_event = GenericTaskCompletedEvent(
            workflow_id=workflow_id,
            task_id=task_id,
            task_description=task_description,
            assigned_agent_id=assigned_agent_id,
            assigned_agent_role=assigned_agent_role,
            output=output,
            success=success,
            metadata={"task_source_type": type(task).__name__}
        )
        self.generic_event_publisher(generic_event)

    def _handle_agent_execution_started(self, source, event):
        agent = getattr(event, 'agent', None)
        if not agent: return
        workflow_id = getattr(source, 'id', 'unknown_workflow') # Need to get workflow_id from context
        agent_id = getattr(agent, 'id', str(agent))
        
        generic_event = AgentActionOccurredEvent(
            workflow_id=workflow_id,
            agent_id=agent_id,
            action_type="agent_execution_started",
            action_details={"agent_role": getattr(agent, 'role', 'Unknown Role')},
            metadata={"agent_source_type": type(agent).__name__}
        )
        self.generic_event_publisher(generic_event)

    def _handle_agent_execution_completed(self, source, event):
        agent = getattr(event, 'agent', None)
        if not agent: return
        workflow_id = getattr(source, 'id', 'unknown_workflow') # Need to get workflow_id from context
        agent_id = getattr(agent, 'id', str(agent))
        
        generic_event = AgentActionOccurredEvent(
            workflow_id=workflow_id,
            agent_id=agent_id,
            action_type="agent_execution_completed",
            action_details={"agent_role": getattr(agent, 'role', 'Unknown Role'), "output": getattr(event, 'output', None)},
            metadata={"agent_source_type": type(agent).__name__}
        )
        self.generic_event_publisher(generic_event)

    def _handle_llm_call_started(self, source, event):
        llm = getattr(event, 'llm', None)
        if not llm: return
        workflow_id = getattr(source, 'id', 'unknown_workflow') # Need to get workflow_id from context
        
        generic_event = AgentActionOccurredEvent(
            workflow_id=workflow_id,
            agent_id=getattr(source, 'id', 'unknown_agent'), # Assuming source is agent
            action_type="llm_call_started",
            action_details={"llm_model": getattr(llm, 'model_name', 'Unknown LLM')},
            metadata={"llm_source_type": type(llm).__name__}
        )
        self.generic_event_publisher(generic_event)

    def _handle_llm_call_completed(self, source, event):
        llm = getattr(event, 'llm', None)
        if not llm: return
        workflow_id = getattr(source, 'id', 'unknown_workflow') # Need to get workflow_id from context
        
        generic_event = AgentActionOccurredEvent(
            workflow_id=workflow_id,
            agent_id=getattr(source, 'id', 'unknown_agent'), # Assuming source is agent
            action_type="llm_call_completed",
            action_details={"llm_model": getattr(llm, 'model_name', 'Unknown LLM'), "response": getattr(event, 'response', None)},
            metadata={"llm_source_type": type(llm).__name__}
        )
        self.generic_event_publisher(generic_event)

    def _handle_tool_usage(self, source, event):
        tool = getattr(event, 'tool', None)
        if not tool: return
        workflow_id = getattr(source, 'id', 'unknown_workflow') # Need to get workflow_id from context
        
        generic_event = AgentActionOccurredEvent(
            workflow_id=workflow_id,
            agent_id=getattr(source, 'id', 'unknown_agent'), # Assuming source is agent
            action_type="tool_usage",
            action_details={"tool_name": getattr(tool, 'name', 'Unknown Tool'), "input": getattr(event, 'input', None), "output": getattr(event, 'output', None)},
            metadata={"tool_source_type": type(tool).__name__}
        )
        self.generic_event_publisher(generic_event)
