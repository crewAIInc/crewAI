"""
Reproduction test for issue #3197: Custom logger conflicts with Crew AI logging
This script demonstrates the problem where custom Python loggers don't work 
when CrewAI's verbose=True is enabled.
"""
import logging
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class TestInput(BaseModel):
    message: str = Field(description="Message to log")


class CustomLoggingTool(BaseTool):
    name: str = "custom_logging_tool"
    description: str = "A tool that uses Python's logging module to demonstrate the conflict"
    args_schema: type[BaseModel] = TestInput

    def _run(self, message: str) -> str:
        logger = logging.getLogger("custom_tool_logger")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.info(f"CUSTOM LOGGER MESSAGE: {message}")
        print(f"PRINT MESSAGE: {message}")
        
        return f"Logged message: {message}"


def test_logging_with_verbose_true():
    """Test case that reproduces the logging conflict when verbose=True"""
    print("=== Testing with verbose=True (should show logging conflict) ===")
    
    agent = Agent(
        role="Test Agent",
        goal="Test custom logging functionality",
        backstory="An agent that tests logging",
        tools=[CustomLoggingTool()],
        verbose=True
    )
    
    task = Task(
        description="Use the custom logging tool to log a test message",
        expected_output="A confirmation that the message was logged",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    result = crew.kickoff()
    print(f"Result: {result}")


def test_logging_with_verbose_false():
    """Test case that shows logging works when verbose=False"""
    print("\n=== Testing with verbose=False (logging should work) ===")
    
    agent = Agent(
        role="Test Agent",
        goal="Test custom logging functionality", 
        backstory="An agent that tests logging",
        tools=[CustomLoggingTool()],
        verbose=False
    )
    
    task = Task(
        description="Use the custom logging tool to log a test message",
        expected_output="A confirmation that the message was logged",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )
    
    result = crew.kickoff()
    print(f"Result: {result}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_logging_with_verbose_false()
    test_logging_with_verbose_true()
