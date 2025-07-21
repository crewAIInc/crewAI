"""
Tests for issue #3197: Custom logger conflicts with Crew AI logging
"""
import logging
import io
import sys
from unittest.mock import patch
import pytest
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class TestInput(BaseModel):
    message: str = Field(description="Message to log")


class CustomLoggingTool(BaseTool):
    name: str = "custom_logging_tool"
    description: str = "A tool that uses Python's logging module"
    args_schema: type[BaseModel] = TestInput

    def _run(self, message: str) -> str:
        logger = logging.getLogger("test_custom_logger")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('CUSTOM_LOG: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.info(f"Custom logger message: {message}")
        print(f"Print message: {message}")
        
        return f"Logged: {message}"


def test_custom_logger_with_verbose_false():
    """Test that custom loggers work when verbose=False"""
    agent = Agent(
        role="Test Agent",
        goal="Test logging",
        backstory="Testing agent",
        tools=[CustomLoggingTool()],
        verbose=False
    )
    
    task = Task(
        description="Log a test message",
        expected_output="Confirmation of logging",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )
    
    with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        result = crew.kickoff()
        output = mock_stdout.getvalue()
        
        assert "Custom logger message" in output or "Print message" in output
        assert result is not None


def test_custom_logger_with_verbose_true():
    """Test that custom loggers work when verbose=True after the fix"""
    agent = Agent(
        role="Test Agent", 
        goal="Test logging",
        backstory="Testing agent",
        tools=[CustomLoggingTool()],
        verbose=True
    )
    
    task = Task(
        description="Log a test message",
        expected_output="Confirmation of logging", 
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        result = crew.kickoff()
        output = mock_stdout.getvalue()
        
        assert "Custom logger message" in output or "Print message" in output
        assert result is not None


def test_console_formatter_pause_resume():
    """Test that ConsoleFormatter properly pauses and resumes Live sessions"""
    from crewai.utilities.events.utils.console_formatter import ConsoleFormatter
    from rich.tree import Tree
    
    formatter = ConsoleFormatter(verbose=True)
    
    tree = Tree("Test Tree")
    formatter.print(tree)
    
    assert formatter._live is not None
    assert not formatter._live_paused
    
    formatter.pause_live_updates()
    assert formatter._live_paused
    assert formatter._live is None
    assert formatter._paused_tree is not None
    
    formatter.resume_live_updates()
    assert not formatter._live_paused
    assert formatter._live is not None


def test_console_formatter_non_tree_printing():
    """Test that non-Tree content properly pauses/resumes Live sessions"""
    from crewai.utilities.events.utils.console_formatter import ConsoleFormatter
    from rich.tree import Tree
    
    formatter = ConsoleFormatter(verbose=True)
    
    tree = Tree("Test Tree")
    formatter.print(tree)
    
    assert formatter._live is not None
    
    with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        formatter.print("Non-tree content")
        output = mock_stdout.getvalue()
        
        assert "Non-tree content" in output
        assert formatter._live is not None
