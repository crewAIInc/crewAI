from typing import Dict, List, Optional, Any, TypedDict, Union

class AgentConfig(TypedDict, total=False):
    """TypedDict for agent configuration loaded from YAML."""
    role: str
    goal: str
    backstory: str
    verbose: bool

class TaskConfig(TypedDict, total=False):
    """TypedDict for task configuration loaded from YAML."""
    description: str
    expected_output: str
    agent: str  # Role of the agent to execute this task
