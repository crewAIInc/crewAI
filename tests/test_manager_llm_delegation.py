import pytest

from crewai import Agent
from crewai.tools.agent_tools.base_agent_tools import BaseAgentTool


class InternalAgentTool(BaseAgentTool):
    """Concrete implementation of BaseAgentTool for testing."""

    def _run(self, *args, **kwargs):
        """Implement required _run method."""
        return "Test response"


def test_agent_tool_role_matching():
    test_cases = [
        ("Futel Official Infopoint", True),  # exact match
        ('  "Futel Official Infopoint"  ', True),  # extra quotes and spaces
        ("Futel Official Infopoint\n", True),  # trailing newline
        ('"Futel Official Infopoint"', True),  # embedded quotes
        (" FUTEL\nOFFICIAL   INFOPOINT ", True),  # multiple whitespace and newline
        ("futel official infopoint", True),  # lowercase
        ("FUTEL OFFICIAL INFOPOINT", True),  # uppercase
        ("Non Existent Agent", False),  # non-existent agent
        (None, False),  # None agent name
    ]

    for role_name, should_match in test_cases:
        # Setup your test agent and agent tool
        test_agent = Agent(
            role="Futel Official Infopoint",
            goal="Answer questions about Futel",
            backstory="Futel Football Club info",
            allow_delegation=False,
        )
        agent_tool = InternalAgentTool(
            name="test_tool", description="Test tool", agents=[test_agent]
        )

        # Execute the test using _execute
        result = agent_tool._execute(
            agent_name=role_name, task="Test task", context=None
        )

        if should_match:
            assert (
                "coworker mentioned not found" not in result.lower()
            ), f"Should find agent with role name: {role_name}"
        else:
            assert (
                "coworker mentioned not found" in result.lower()
            ), f"Should not find agent with role name: {role_name}"
