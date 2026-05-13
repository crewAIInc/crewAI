"""NewAgent — standalone, conversational, self-improving agent."""

from crewai.new_agent.cli_provider import CLIProvider
from crewai.new_agent.coworker_tools import MultiDelegateTool
from crewai.new_agent.dreaming import DreamingEngine
from crewai.new_agent.knowledge_discovery import KnowledgeDiscovery
from crewai.new_agent.models import (
    AgentSettings,
    AgentStatus,
    MemoryScope,
    MemorySlice,
    Message,
    MessageAction,
    PromptLayer,
    PromptStack,
    ProvenanceEntry,
    TokenUsage,
)
from crewai.new_agent.new_agent import NewAgent, clear_amp_cache
from crewai.new_agent.planning import PlanningEngine
from crewai.new_agent.provider import (
    ConversationStorage,
    ConversationalProvider,
    DirectProvider,
    SQLiteConversationStorage,
)
from crewai.new_agent.scheduler import ScheduleTaskTool, ScheduledTask, TaskScheduler
from crewai.new_agent.skill_builder import SkillBuilder
from crewai.new_agent.spawn_tools import SpawnSubtaskArgs, SpawnSubtaskTool


__all__ = [
    "AgentSettings",
    "AgentStatus",
    "CLIProvider",
    "ConversationStorage",
    "ConversationalProvider",
    "DirectProvider",
    "DreamingEngine",
    "KnowledgeDiscovery",
    "MemoryScope",
    "MemorySlice",
    "Message",
    "MessageAction",
    "MultiDelegateTool",
    "NewAgent",
    "PlanningEngine",
    "PromptLayer",
    "PromptStack",
    "ProvenanceEntry",
    "SQLiteConversationStorage",
    "ScheduleTaskTool",
    "ScheduledTask",
    "SkillBuilder",
    "SpawnSubtaskArgs",
    "SpawnSubtaskTool",
    "TaskScheduler",
    "TokenUsage",
    "clear_amp_cache",
]

try:
    from crewai.new_agent.event_listener import register_new_agent_listeners

    register_new_agent_listeners()
except Exception:
    pass
