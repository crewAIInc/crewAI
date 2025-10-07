#!/usr/bin/env python3
"""
CrewAI Execution Tracer
=======================

Provides workflow transparency by capturing crew execution steps and interactions.
Enables developers to see what agents did and how they collaborated.

Solution for CrewAI Issue #3268: Access to sequence of actions and detailed
interaction logs including HumanMessage, AIMessage, ToolCall, ToolMessage.

Usage:
    from crewai.utilities.execution_tracer import ExecutionTracer
    
    tracer = ExecutionTracer()
    crew = Crew(
        agents=[agent],
        tasks=[task],
        step_callback=tracer.on_step_complete,
        after_kickoff_callback=tracer.on_crew_complete
    )
    
    result = crew.kickoff()
    print(result.execution_steps)    # Sequence of actions
    print(result.interaction_logs)   # Categorized logs
"""

import json
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# Content length limit for execution steps
MAX_CONTENT_LENGTH = 500


class StepType(str, Enum):
    """Step types for categorizing crew execution events"""
    HUMAN_MESSAGE = "HumanMessage"
    AI_MESSAGE = "AIMessage"
    TOOL_CALL = "ToolCall"
    TOOL_MESSAGE = "ToolMessage"
    TASK_COMPLETE = "TaskComplete"
    CREW_START = "CrewStart"
    CREW_COMPLETE = "CrewComplete"


class ExecutionStep:
    """Single execution step in crew workflow"""
    
    def __init__(self, step_type: StepType, agent_name: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.step_type = step_type
        self.agent_name = agent_name
        self.content = content
        self.timestamp = time.time()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization"""
        # Ensure metadata is JSON-serializable
        json_safe_metadata = {}
        for key, value in self.metadata.items():
            try:
                json.dumps(value)
                json_safe_metadata[key] = value
            except (TypeError, ValueError):
                json_safe_metadata[key] = str(value)
        
        return {
            "step_type": self.step_type.value,
            "agent": self.agent_name,
            "content": self.content,
            "timestamp": self.timestamp,
            "time": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3],
            "metadata": json_safe_metadata
        }


class ExecutionTrace:
    """Collection of execution steps with analysis methods"""
    
    def __init__(self):
        self.steps: List[ExecutionStep] = []
        self.start_time = time.time()
    
    def add_step(self, step_type: StepType, agent_name: str, content: str, **metadata: Any) -> ExecutionStep:
        """Add execution step to trace"""
        step = ExecutionStep(step_type, agent_name, content, metadata)
        self.steps.append(step)
        return step
    
    def actions(self) -> List[Dict[str, Any]]:
        """Get chronological sequence of all actions"""
        return [step.to_dict() for step in self.steps]
    
    def logs(self) -> Dict[str, Any]:
        """Get detailed interaction logs categorized by type"""
        logs_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for step in self.steps:
            step_type = step.step_type.value
            if step_type not in logs_by_type:
                logs_by_type[step_type] = []
            logs_by_type[step_type].append(step.to_dict())
        
        return {
            "total_steps": len(self.steps),
            "execution_time": time.time() - self.start_time,
            "step_types": list(logs_by_type.keys()),
            "logs_by_type": logs_by_type,
            "chronological_sequence": self.actions()
        }


class ExecutionTracer:
    """CrewAI execution tracer for workflow transparency"""
    
    def __init__(self):
        self.trace = ExecutionTrace()
        self.current_task = None
        self.current_agent = None
    
    def on_step_complete(self, agent_output: Any) -> Any:
        """Callback for agent step completion"""
        
        # Extract agent information
        agent_name = self._extract_agent_name(agent_output)
        content = self._extract_content(agent_output)
        step_type = self._categorize_step(content, agent_output)
        
        # Collect metadata
        metadata = self._extract_metadata(agent_output)
        
        # Record step
        self.trace.add_step(
            step_type=step_type,
            agent_name=agent_name,
            content=content[:MAX_CONTENT_LENGTH],  # Limit content length
            raw_output_type=type(agent_output).__name__,
            task=self.current_task.description if self.current_task else None,
            **self._json_safe(metadata)
        )
        
        return agent_output
    
    def on_task_complete(self, task_output: Any) -> Any:
        """Callback for task completion"""
        
        agent_name = getattr(task_output, 'agent', 'Unknown')
        task_description = getattr(task_output, 'description', 'Unknown Task')
        task_result = getattr(task_output, 'raw', str(task_output))
        
        self.trace.add_step(
            step_type=StepType.TASK_COMPLETE,
            agent_name=agent_name,
            content=f"Task: {task_description}\nResult: {task_result}"[:MAX_CONTENT_LENGTH],
            task_description=task_description,
            task_name=getattr(task_output, 'name', None),
            raw_result=task_result
        )
        
        return task_output
    
    def on_crew_start(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Callback for crew execution start"""
        
        self.trace.add_step(
            step_type=StepType.CREW_START,
            agent_name="System",
            content="Crew execution started with tracing enabled"
        )
        
        return inputs or {}
    
    def on_crew_complete(self, crew_output: Any) -> Any:
        """Callback for crew execution completion - adds trace to output"""
        
        # Record completion step
        self.trace.add_step(
            step_type=StepType.CREW_COMPLETE,
            agent_name="System", 
            content=f"Crew execution completed with {len(self.trace.steps)} traced steps"
        )
        
        # Extend CrewOutput with execution trace
        if hasattr(crew_output, '__dict__'):
            crew_output.__dict__['execution_steps'] = self.trace.actions()
            crew_output.__dict__['interaction_logs'] = self.trace.logs()
            crew_output.__dict__['tracing_enabled'] = True
        
        return crew_output
    
    def _extract_agent_name(self, agent_output: Any) -> str:
        """Extract agent name from output object"""
        if hasattr(agent_output, 'agent'):
            return getattr(agent_output.agent, 'role', 'Unknown')
        return self.current_agent.role if self.current_agent else "Unknown"
    
    def _extract_content(self, agent_output: Any) -> str:
        """Extract content from agent output"""
        # ToolResult objects
        if hasattr(agent_output, 'result'):
            return f"Tool returned: {str(agent_output.result)[:200]}"
        
        # AgentAction objects
        if hasattr(agent_output, 'text'):
            return agent_output.text
        if hasattr(agent_output, 'thought'):
            return agent_output.thought
            
        return str(agent_output)
    
    def _extract_metadata(self, agent_output: Any) -> Dict[str, Any]:
        """Extract metadata from agent output"""
        metadata = {}
        
        # Tool-related metadata
        if hasattr(agent_output, 'tool'):
            metadata['tool'] = agent_output.tool
        if hasattr(agent_output, 'tool_input'):
            metadata['tool_input'] = agent_output.tool_input
        if hasattr(agent_output, 'result_as_answer'):
            metadata['result_as_answer'] = agent_output.result_as_answer
            
        return metadata
    
    def _categorize_step(self, content: str, agent_output: Any) -> StepType:
        """Categorize step type based on agent output analysis"""
        
        # ToolResult objects -> ToolMessage
        if hasattr(agent_output, 'result'):
            return StepType.TOOL_MESSAGE
        
        # AgentAction objects
        if hasattr(agent_output, 'thought') and hasattr(agent_output, 'tool') and hasattr(agent_output, 'tool_input'):
            # Has tool = ToolCall, no tool = AIMessage
            if agent_output.tool and agent_output.tool.strip():
                return StepType.TOOL_CALL
            else:
                return StepType.AI_MESSAGE
        
        # Content-based fallback categorization
        content_lower = content.lower()
        if any(keyword in content_lower for keyword in ['task:', 'user input', 'human:', 'instruction:']):
            return StepType.HUMAN_MESSAGE
        else:
            return StepType.AI_MESSAGE
    
    def _json_safe(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure metadata values are JSON-serializable"""
        safe_metadata = {}
        for key, value in metadata.items():
            try:
                json.dumps(value)  # Test if serializable
                safe_metadata[key] = value
            except (TypeError, ValueError):
                # Convert to string if not serializable
                safe_metadata[key] = str(value)
        return safe_metadata