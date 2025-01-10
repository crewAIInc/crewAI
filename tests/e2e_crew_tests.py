import asyncio
import os
import tempfile

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.process import Process
from crewai.task import Task
from crewai.tasks.conditional_task import ConditionalTask


def test_basic_crew_execution(default_agent):
    """Test basic crew execution using the default agent fixture."""

    # Initialize agents by copying the default agent fixture
    researcher = default_agent.copy()
    researcher.role = "Researcher"
    researcher.goal = "Research the latest advancements in AI."
    researcher.backstory = "An expert in AI technologies."

    writer = default_agent.copy()
    writer.role = "Writer"
    writer.goal = "Write an article based on research findings."
    writer.backstory = "A professional writer specializing in technology topics."

    # Define tasks
    research_task = Task(
        description="Provide a summary of the latest advancements in AI.",
        expected_output="A detailed summary of recent AI advancements.",
        agent=researcher,
    )

    writing_task = Task(
        description="Write an article based on the research summary.",
        expected_output="An engaging article on AI advancements.",
        agent=writer,
    )

    # Create the crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
    )

    # Execute the crew
    result = crew.kickoff()

    # Assertions to verify the result
    assert result is not None, "Crew execution did not return a result."
    assert isinstance(result, CrewOutput), "Result is not an instance of CrewOutput."
    assert (
        "AI advancements" in result.raw
        or "artificial intelligence" in result.raw.lower()
    ), "Result does not contain expected content."


def test_hierarchical_crew_with_manager(default_llm_config):
    """Test hierarchical crew execution with a manager agent."""

    # Initialize agents using the default LLM config fixture
    ceo = Agent(
        role="CEO",
        goal="Oversee the project and ensure quality deliverables.",
        backstory="A seasoned executive with a keen eye for detail.",
        llm=default_llm_config,
    )

    developer = Agent(
        role="Developer",
        goal="Implement software features as per requirements.",
        backstory="An experienced software developer.",
        llm=default_llm_config,
    )

    tester = Agent(
        role="Tester",
        goal="Test software features and report bugs.",
        backstory="A meticulous QA engineer.",
        llm=default_llm_config,
    )

    # Define tasks
    development_task = Task(
        description="Develop the new authentication feature.",
        expected_output="Code implementation of the authentication feature.",
        agent=developer,
    )

    testing_task = Task(
        description="Test the authentication feature for vulnerabilities.",
        expected_output="A report on any found bugs or vulnerabilities.",
        agent=tester,
    )

    # Create the crew with hierarchical process
    crew = Crew(
        agents=[ceo, developer, tester],
        tasks=[development_task, testing_task],
        process=Process.hierarchical,
        manager_agent=ceo,
    )

    # Execute the crew
    result = crew.kickoff()

    # Assertions to verify the result
    assert result is not None, "Crew execution did not return a result."
    assert isinstance(result, CrewOutput), "Result is not an instance of CrewOutput."
    assert (
        "authentication" in result.raw.lower()
    ), "Result does not contain expected content."


@pytest.mark.asyncio
async def test_asynchronous_task_execution(default_llm_config):
    """Test crew execution with asynchronous tasks."""

    # Initialize agent
    data_processor = Agent(
        role="Data Processor",
        goal="Process large datasets efficiently.",
        backstory="An expert in data processing and analysis.",
        llm=default_llm_config,
    )

    # Define tasks with async_execution=True
    async_task1 = Task(
        description="Process dataset A asynchronously.",
        expected_output="Processed results of dataset A.",
        agent=data_processor,
        async_execution=True,
    )

    async_task2 = Task(
        description="Process dataset B asynchronously.",
        expected_output="Processed results of dataset B.",
        agent=data_processor,
        async_execution=True,
    )

    # Create the crew
    crew = Crew(
        agents=[data_processor],
        tasks=[async_task1, async_task2],
        process=Process.sequential,
    )

    # Execute the crew asynchronously
    result = await crew.kickoff_async()

    # Assertions to verify the result
    assert result is not None, "Crew execution did not return a result."
    assert isinstance(result, CrewOutput), "Result is not an instance of CrewOutput."
    assert (
        "dataset a" in result.raw.lower() or "dataset b" in result.raw.lower()
    ), "Result does not contain expected content."


def test_crew_with_conditional_task(default_llm_config):
    """Test crew execution that includes a conditional task."""

    # Initialize agents
    analyst = Agent(
        role="Analyst",
        goal="Analyze data and make decisions based on insights.",
        backstory="A data analyst with experience in predictive modeling.",
        llm=default_llm_config,
    )

    decision_maker = Agent(
        role="Decision Maker",
        goal="Make decisions based on analysis.",
        backstory="An executive responsible for strategic decisions.",
        llm=default_llm_config,
    )

    # Define tasks
    analysis_task = Task(
        description="Analyze the quarterly financial data.",
        expected_output="A report highlighting key financial insights.",
        agent=analyst,
    )

    decision_task = ConditionalTask(
        description="If the profit margin is below 10%, recommend cost-cutting measures.",
        expected_output="Recommendations for reducing costs.",
        agent=decision_maker,
        condition=lambda output: "profit margin below 10%" in output.lower(),
    )

    # Create the crew
    crew = Crew(
        agents=[analyst, decision_maker],
        tasks=[analysis_task, decision_task],
        process=Process.sequential,
    )

    # Execute the crew
    result = crew.kickoff()

    # Assertions to verify the result
    assert result is not None, "Crew execution did not return a result."
    assert isinstance(result, CrewOutput), "Result is not an instance of CrewOutput."
    assert len(result.tasks_output) >= 1, "No tasks were executed."


def test_crew_with_output_file():
    """Test crew execution that writes output to a file."""

    # Access the API key from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    assert openai_api_key, "OPENAI_API_KEY environment variable is not set."

    # Create a temporary directory for output files
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Initialize agent
        content_creator = Agent(
            role="Content Creator",
            goal="Generate engaging blog content.",
            backstory="A creative writer with a passion for storytelling.",
            llm={"provider": "openai", "model": "gpt-4", "api_key": openai_api_key},
        )

        # Define task with output file
        output_file_path = f"{tmpdirname}/blog_post.txt"
        blog_task = Task(
            description="Write a blog post about the benefits of remote work.",
            expected_output="An informative and engaging blog post.",
            agent=content_creator,
            output_file=output_file_path,
        )

        # Create the crew
        crew = Crew(
            agents=[content_creator],
            tasks=[blog_task],
            process=Process.sequential,
        )

        # Execute the crew
        crew.kickoff()

        # Assertions to verify the result
        assert os.path.exists(output_file_path), "Output file was not created."

        # Read the content from the file and perform assertions
        with open(output_file_path, "r") as file:
            content = file.read()
            assert (
                "remote work" in content.lower()
            ), "Output file does not contain expected content."


def test_invalid_hierarchical_process():
    """Test that an error is raised when using hierarchical process without a manager agent or manager_llm."""
    with pytest.raises(ValueError) as exc_info:
        Crew(
            agents=[],
            tasks=[],
            process=Process.hierarchical,  # Hierarchical process without a manager
        )
    assert "manager_llm or manager_agent is required" in str(exc_info.value)


def test_crew_with_memory(memory_agent, memory_tasks):
    """Test crew execution utilizing memory."""

    # Enable memory in the crew
    crew = Crew(
        agents=[memory_agent],
        tasks=memory_tasks,
        process=Process.sequential,
        memory=True,  # Enable memory
    )

    # Execute the crew
    result = crew.kickoff()

    # Assertions to verify the result
    assert result is not None, "Crew execution did not return a result."
    assert isinstance(result, CrewOutput), "Result is not an instance of CrewOutput."
    assert (
        "history of ai" in result.raw.lower() and "future of ai" in result.raw.lower()
    ), "Result does not contain expected content."
