"""Test Agent creation and execution basic functionality."""

import os
import threading
from unittest import mock
from unittest.mock import MagicMock, patch

from crewai.agents.crew_agent_executor import AgentFinish, CrewAgentExecutor
from crewai.cli.constants import DEFAULT_LLM_MODEL
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.knowledge_config import KnowledgeConfig
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.process import Process
from crewai.tools.tool_calling import InstructorToolCalling
from crewai.tools.tool_usage import ToolUsage
from crewai.utilities.errors import AgentRepositoryError
import pytest

from crewai import Agent, Crew, Task
from crewai.agents.cache import CacheHandler
from crewai.tools import tool
from crewai.utilities import RPMController


def test_agent_llm_creation_with_env_vars():
    # Store original environment variables
    original_api_key = os.environ.get("OPENAI_API_KEY")
    original_api_base = os.environ.get("OPENAI_API_BASE")
    original_model_name = os.environ.get("OPENAI_MODEL_NAME")

    # Set up environment variables
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    os.environ["OPENAI_API_BASE"] = "https://test-api-base.com"
    os.environ["OPENAI_MODEL_NAME"] = "gpt-4-turbo"

    # Create an agent without specifying LLM
    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    # Check if LLM is created correctly
    assert isinstance(agent.llm, BaseLLM)
    assert agent.llm.model == "gpt-4-turbo"
    assert agent.llm.api_key == "test_api_key"
    assert agent.llm.base_url == "https://test-api-base.com"

    # Clean up environment variables
    del os.environ["OPENAI_API_KEY"]
    del os.environ["OPENAI_API_BASE"]
    del os.environ["OPENAI_MODEL_NAME"]

    if original_api_key:
        os.environ["OPENAI_API_KEY"] = original_api_key
    if original_api_base:
        os.environ["OPENAI_API_BASE"] = original_api_base
    if original_model_name:
        os.environ["OPENAI_MODEL_NAME"] = original_model_name

    # Create an agent without specifying LLM
    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    # Check if LLM is created correctly
    assert isinstance(agent.llm, BaseLLM)
    assert agent.llm.model != "gpt-4-turbo"
    assert agent.llm.api_key != "test_api_key"
    assert agent.llm.base_url != "https://test-api-base.com"

    # Restore original environment variables
    if original_api_key:
        os.environ["OPENAI_API_KEY"] = original_api_key
    if original_api_base:
        os.environ["OPENAI_API_BASE"] = original_api_base
    if original_model_name:
        os.environ["OPENAI_MODEL_NAME"] = original_model_name


def test_agent_creation():
    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"


def test_agent_with_only_system_template():
    """Test that an agent with only system_template works without errors."""
    agent = Agent(
        role="Test Role",
        goal="Test Goal",
        backstory="Test Backstory",
        allow_delegation=False,
        system_template="You are a test agent...",
        # prompt_template is intentionally missing
    )

    assert agent.role == "Test Role"
    assert agent.goal == "Test Goal"
    assert agent.backstory == "Test Backstory"


def test_agent_with_only_prompt_template():
    """Test that an agent with only system_template works without errors."""
    agent = Agent(
        role="Test Role",
        goal="Test Goal",
        backstory="Test Backstory",
        allow_delegation=False,
        prompt_template="You are a test agent...",
        # prompt_template is intentionally missing
    )

    assert agent.role == "Test Role"
    assert agent.goal == "Test Goal"
    assert agent.backstory == "Test Backstory"


def test_agent_with_missing_response_template():
    """Test that an agent with system_template and prompt_template but no response_template works without errors."""
    agent = Agent(
        role="Test Role",
        goal="Test Goal",
        backstory="Test Backstory",
        allow_delegation=False,
        system_template="You are a test agent...",
        prompt_template="This is a test prompt...",
        # response_template is intentionally missing
    )

    assert agent.role == "Test Role"
    assert agent.goal == "Test Goal"
    assert agent.backstory == "Test Backstory"


def test_agent_default_values():
    agent = Agent(role="test role", goal="test goal", backstory="test backstory")
    assert agent.llm.model == DEFAULT_LLM_MODEL
    assert agent.allow_delegation is False


def test_custom_llm():
    agent = Agent(
        role="test role", goal="test goal", backstory="test backstory", llm="gpt-4"
    )
    assert agent.llm.model == "gpt-4"


@pytest.mark.vcr()
def test_agent_execution():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        allow_delegation=False,
    )

    task = Task(
        description="How much is 1 + 1?",
        agent=agent,
        expected_output="the result of the math operation.",
    )

    output = agent.execute_task(task)
    assert output == "The result of the math operation 1 + 1 is 2."


@pytest.mark.vcr()
def test_agent_execution_with_tools():
    @tool
    def multiplier(first_number: int, second_number: int) -> float:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        allow_delegation=False,
    )

    task = Task(
        description="What is 3 times 4?",
        agent=agent,
        expected_output="The result of the multiplication.",
    )
    received_events = []
    condition = threading.Condition()
    event_handled = False

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def handle_tool_end(source, event):
        nonlocal event_handled
        received_events.append(event)
        with condition:
            event_handled = True
            condition.notify()

    output = agent.execute_task(task)
    assert output == "12"

    with condition:
        if not event_handled:
            condition.wait(timeout=5)
    assert event_handled, "Timeout waiting for tool usage event"
    assert len(received_events) == 1
    assert isinstance(received_events[0], ToolUsageFinishedEvent)
    assert received_events[0].tool_name == "multiplier"
    assert received_events[0].tool_args == {"first_number": 3, "second_number": 4}


@pytest.mark.vcr()
def test_logging_tool_usage():
    @tool
    def multiplier(first_number: int, second_number: int) -> float:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        verbose=True,
    )

    assert agent.llm.model == DEFAULT_LLM_MODEL
    assert agent.tools_handler.last_used_tool is None
    task = Task(
        description="What is 3 times 4?",
        agent=agent,
        expected_output="The result of the multiplication.",
    )
    # force cleaning cache
    agent.tools_handler.cache = CacheHandler()
    output = agent.execute_task(task)
    tool_usage = InstructorToolCalling(
        tool_name=multiplier.name, arguments={"first_number": 3, "second_number": 4}
    )

    assert output == "12"
    assert agent.tools_handler.last_used_tool.tool_name == tool_usage.tool_name
    assert agent.tools_handler.last_used_tool.arguments == tool_usage.arguments


@pytest.mark.vcr()
def test_cache_hitting():
    @tool
    def multiplier(first_number: int, second_number: int) -> float:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    cache_handler = CacheHandler()

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        allow_delegation=False,
        cache_handler=cache_handler,
        verbose=True,
    )

    task1 = Task(
        description="What is 2 times 6?",
        agent=agent,
        expected_output="The result of the multiplication.",
    )
    task2 = Task(
        description="What is 3 times 3?",
        agent=agent,
        expected_output="The result of the multiplication.",
    )

    output = agent.execute_task(task1)
    output = agent.execute_task(task2)
    assert cache_handler._cache == {
        'multiplier-{"first_number": 2, "second_number": 6}': 12,
        'multiplier-{"first_number": 3, "second_number": 3}': 9,
    }

    task = Task(
        description="What is 2 times 6 times 3? Return only the number",
        agent=agent,
        expected_output="The result of the multiplication.",
    )
    output = agent.execute_task(task)
    assert output == "36"

    assert cache_handler._cache == {
        'multiplier-{"first_number": 2, "second_number": 6}': 12,
        'multiplier-{"first_number": 3, "second_number": 3}': 9,
        'multiplier-{"first_number": 12, "second_number": 3}': 36,
    }
    received_events = []
    condition = threading.Condition()
    event_handled = False

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def handle_tool_end(source, event):
        nonlocal event_handled
        received_events.append(event)
        with condition:
            event_handled = True
            condition.notify()

    task = Task(
        description="What is 2 times 6? Return only the result of the multiplication.",
        agent=agent,
        expected_output="The result of the multiplication.",
    )
    output = agent.execute_task(task)
    assert output == "12"

    with condition:
        if not event_handled:
            condition.wait(timeout=5)
    assert event_handled, "Timeout waiting for tool usage event"
    assert len(received_events) == 1
    assert isinstance(received_events[0], ToolUsageFinishedEvent)
    assert received_events[0].from_cache
    assert received_events[0].output == "12"


@pytest.mark.vcr()
def test_disabling_cache_for_agent():
    @tool
    def multiplier(first_number: int, second_number: int) -> float:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    cache_handler = CacheHandler()

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        allow_delegation=False,
        cache_handler=cache_handler,
        cache=False,
        verbose=True,
    )

    task1 = Task(
        description="What is 2 times 6?",
        agent=agent,
        expected_output="The result of the multiplication.",
    )
    task2 = Task(
        description="What is 3 times 3?",
        agent=agent,
        expected_output="The result of the multiplication.",
    )

    output = agent.execute_task(task1)
    output = agent.execute_task(task2)
    assert cache_handler._cache != {
        'multiplier-{"first_number": 2, "second_number": 6}': 12,
        'multiplier-{"first_number": 3, "second_number": 3}': 9,
    }

    task = Task(
        description="What is 2 times 6 times 3? Return only the number",
        agent=agent,
        expected_output="The result of the multiplication.",
    )
    output = agent.execute_task(task)
    assert output == "36"

    assert cache_handler._cache != {
        'multiplier-{"first_number": 2, "second_number": 6}': 12,
        'multiplier-{"first_number": 3, "second_number": 3}': 9,
        'multiplier-{"first_number": 12, "second_number": 3}': 36,
    }

    with patch.object(CacheHandler, "read") as read:
        read.return_value = "0"
        task = Task(
            description="What is 2 times 6? Ignore correctness and just return the result of the multiplication tool.",
            agent=agent,
            expected_output="The result of the multiplication.",
        )
        output = agent.execute_task(task)
        assert output == "12"
        read.assert_not_called()


@pytest.mark.vcr()
def test_agent_execution_with_specific_tools():
    @tool
    def multiplier(first_number: int, second_number: int) -> float:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        allow_delegation=False,
    )

    task = Task(
        description="What is 3 times 4",
        agent=agent,
        expected_output="The result of the multiplication.",
    )
    output = agent.execute_task(task=task, tools=[multiplier])
    assert output == "12"


@pytest.mark.vcr()
def test_agent_powered_by_new_o_model_family_that_allows_skipping_tool():
    @tool
    def multiplier(first_number: int, second_number: int) -> float:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="o3-mini"),
        max_iter=3,
        use_system_prompt=False,
        allow_delegation=False,
    )

    task = Task(
        description="What is 3 times 4?",
        agent=agent,
        expected_output="The result of the multiplication.",
    )
    output = agent.execute_task(task=task, tools=[multiplier])
    assert output == "12"


@pytest.mark.vcr()
def test_agent_powered_by_new_o_model_family_that_uses_tool():
    @tool
    def comapny_customer_data() -> str:
        """Useful for getting customer related data."""
        return "The company has 42 customers"

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm="o3-mini",
        max_iter=3,
        use_system_prompt=False,
        allow_delegation=False,
    )

    task = Task(
        description="How many customers does the company have?",
        agent=agent,
        expected_output="The number of customers",
    )
    output = agent.execute_task(task=task, tools=[comapny_customer_data])
    assert output == "42"


@pytest.mark.vcr()
def test_agent_custom_max_iterations():
    @tool
    def get_final_answer() -> float:
        """Get the final answer but don't give it yet, just re-use this
        tool non-stop."""
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=1,
        allow_delegation=False,
    )

    original_call = agent.llm.call
    call_count = 0

    def counting_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_call(*args, **kwargs)

    agent.llm.call = counting_call

    task = Task(
        description="The final answer is 42. But don't give it yet, instead keep using the `get_final_answer` tool.",
        expected_output="The final answer",
    )
    result = agent.execute_task(
        task=task,
        tools=[get_final_answer],
    )

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert call_count > 0
    # With max_iter=1, expect 2 calls:
    # - Call 1: iteration 0
    # - Call 2: iteration 1 (max reached, handle_max_iterations_exceeded called, then loop breaks)
    assert call_count == 2


@pytest.mark.vcr()
@pytest.mark.timeout(30)
def test_agent_max_iterations_stops_loop():
    """Test that agent execution terminates when max_iter is reached."""

    @tool
    def get_data(step: str) -> str:
        """Get data for a step. Always returns data requiring more steps."""
        return f"Data for {step}: incomplete, need to query more steps."

    agent = Agent(
        role="data collector",
        goal="collect data using the get_data tool",
        backstory="You must use the get_data tool extensively",
        max_iter=2,
        allow_delegation=False,
    )

    task = Task(
        description="Use get_data tool for step1, step2, step3, step4, step5, step6, step7, step8, step9, and step10. Do NOT stop until you've called it for ALL steps.",
        expected_output="A summary of all data collected",
    )

    result = agent.execute_task(
        task=task,
        tools=[get_data],
    )

    assert result is not None
    assert isinstance(result, str)

    assert agent.agent_executor.iterations <= agent.max_iter + 2, (
        f"Agent ran {agent.agent_executor.iterations} iterations "
        f"but should stop around {agent.max_iter + 1}. "
    )


@pytest.mark.vcr()
def test_agent_repeated_tool_usage(capsys):
    """Test that agents handle repeated tool usage appropriately.

    Notes:
        Investigate whether to pin down the specific execution flow by examining
        src/crewai/agents/crew_agent_executor.py:177-186 (max iterations check)
        and src/crewai/tools/tool_usage.py:152-157 (repeated usage detection)
        to ensure deterministic behavior.
    """

    @tool
    def get_final_answer() -> float:
        """Get the final answer but don't give it yet, just re-use this tool non-stop."""
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=4,
        llm="gpt-4",
        allow_delegation=False,
        verbose=True,
    )

    task = Task(
        description="The final answer is 42. But don't give it until I tell you so, instead keep using the `get_final_answer` tool.",
        expected_output="The final answer, don't give it until I tell you so",
    )
    # force cleaning cache
    agent.tools_handler.cache = CacheHandler()
    agent.execute_task(
        task=task,
        tools=[get_final_answer],
    )

    captured = capsys.readouterr()
    output_lower = captured.out.lower()

    has_repeated_usage_message = "tried reusing the same input" in output_lower
    has_max_iterations = "maximum iterations reached" in output_lower
    has_final_answer = "final answer" in output_lower or "42" in captured.out

    assert has_repeated_usage_message or (has_max_iterations and has_final_answer), (
        f"Expected repeated tool usage handling or proper max iteration handling. Output was: {captured.out[:500]}..."
    )


@pytest.mark.vcr()
def test_agent_repeated_tool_usage_check_even_with_disabled_cache(capsys):
    @tool
    def get_final_answer(anything: str) -> float:
        """Get the final answer but don't give it yet, just re-use this
        tool non-stop."""
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=4,
        llm="gpt-4",
        allow_delegation=False,
        verbose=True,
        cache=False,
    )

    task = Task(
        description="The final answer is 42. But don't give it until I tell you so, instead keep using the `get_final_answer` tool.",
        expected_output="The final answer, don't give it until I tell you so",
    )

    agent.execute_task(
        task=task,
        tools=[get_final_answer],
    )

    captured = capsys.readouterr()

    # More flexible check, look for either the repeated usage message or verification that max iterations was reached
    output_lower = captured.out.lower()

    has_repeated_usage_message = "tried reusing the same input" in output_lower
    has_max_iterations = "maximum iterations reached" in output_lower
    has_final_answer = "final answer" in output_lower or "42" in captured.out

    assert has_repeated_usage_message or (has_max_iterations and has_final_answer), (
        f"Expected repeated tool usage handling or proper max iteration handling. Output was: {captured.out[:500]}..."
    )


@pytest.mark.vcr()
def test_agent_moved_on_after_max_iterations():
    @tool
    def get_final_answer() -> float:
        """Get the final answer but don't give it yet, just re-use this
        tool non-stop."""
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=5,
        allow_delegation=False,
    )

    task = Task(
        description="The final answer is 42. But don't give it yet, instead keep using the `get_final_answer` tool over and over until you're told you can give your final answer.",
        expected_output="The final answer",
    )
    output = agent.execute_task(
        task=task,
        tools=[get_final_answer],
    )
    assert output == "42"


@pytest.mark.vcr()
def test_agent_respect_the_max_rpm_set(capsys):
    @tool
    def get_final_answer() -> float:
        """Get the final answer but don't give it yet, just re-use this
        tool non-stop."""
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=5,
        max_rpm=1,
        verbose=True,
        allow_delegation=False,
    )

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        task = Task(
            description="Use tool logic for `get_final_answer` but fon't give you final answer yet, instead keep using it unless you're told to give your final answer",
            expected_output="The final answer",
        )
        output = agent.execute_task(
            task=task,
            tools=[get_final_answer],
        )
        assert "42" in output or "final answer" in output.lower()
        captured = capsys.readouterr()
        assert "Max RPM reached, waiting for next minute to start." in captured.out
        moveon.assert_called()


@pytest.mark.vcr()
def test_agent_respect_the_max_rpm_set_over_crew_rpm(capsys):
    from unittest.mock import patch

    from crewai.tools import tool

    @tool
    def get_final_answer() -> float:
        """Get the final answer but don't give it yet, just re-use this
        tool non-stop."""
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=4,
        max_rpm=10,
        verbose=True,
    )

    task = Task(
        description="Use tool logic for `get_final_answer` but fon't give you final answer yet, instead keep using it unless you're told to give your final answer",
        expected_output="The final answer",
        tools=[get_final_answer],
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], max_rpm=1, verbose=True)

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        crew.kickoff()
        captured = capsys.readouterr()
        assert "Max RPM reached, waiting for next minute to start." not in captured.out
        moveon.assert_not_called()


@pytest.mark.vcr()
def test_agent_without_max_rpm_respects_crew_rpm(capsys):
    from unittest.mock import patch

    from crewai.tools import tool

    @tool
    def get_final_answer() -> float:
        """Get the final answer but don't give it yet, just re-use this tool non-stop."""
        return 42

    agent1 = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_rpm=10,
        max_iter=2,
        verbose=True,
        allow_delegation=False,
    )

    agent2 = Agent(
        role="test role2",
        goal="test goal2",
        backstory="test backstory2",
        max_iter=5,
        verbose=True,
        allow_delegation=False,
    )

    tasks = [
        Task(
            description="Just say hi.",
            agent=agent1,
            expected_output="Your greeting.",
        ),
        Task(
            description=(
                "NEVER give a Final Answer, unless you are told otherwise, "
                "instead keep using the `get_final_answer` tool non-stop, "
                "until you must give your best final answer"
            ),
            expected_output="The final answer",
            tools=[get_final_answer],
            agent=agent2,
        ),
    ]

    # Set crew's max_rpm to 1 to trigger RPM limit
    crew = Crew(agents=[agent1, agent2], tasks=tasks, max_rpm=1, verbose=True)

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        result = crew.kickoff()
        # Verify the crew executed and RPM limit was triggered
        assert result is not None
        assert moveon.called


@pytest.mark.vcr()
def test_agent_error_on_parsing_tool(capsys):
    from unittest.mock import patch

    from crewai.tools import tool

    @tool
    def get_final_answer() -> float:
        """Get the final answer but don't give it yet, just re-use this
        tool non-stop."""
        return 42

    agent1 = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=1,
        verbose=True,
    )
    tasks = [
        Task(
            description="Use the get_final_answer tool.",
            expected_output="The final answer",
            agent=agent1,
            tools=[get_final_answer],
        )
    ]

    crew = Crew(
        agents=[agent1],
        tasks=tasks,
        verbose=True,
        function_calling_llm="gpt-4o",
    )
    with patch.object(ToolUsage, "_original_tool_calling") as force_exception_1:
        force_exception_1.side_effect = Exception("Error on parsing tool.")
        with patch.object(ToolUsage, "_render") as force_exception_2:
            force_exception_2.side_effect = Exception("Error on parsing tool.")
            crew.kickoff()
    captured = capsys.readouterr()
    assert "Error on parsing tool." in captured.out


@pytest.mark.vcr()
def test_agent_remembers_output_format_after_using_tools_too_many_times():
    from unittest.mock import patch

    from crewai.tools import tool

    @tool
    def get_final_answer() -> float:
        """Get the final answer but don't give it yet, just re-use this
        tool non-stop."""
        return 42

    agent1 = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=6,
        verbose=True,
    )
    tasks = [
        Task(
            description="Use tool logic for `get_final_answer` but fon't give you final answer yet, instead keep using it unless you're told to give your final answer",
            expected_output="The final answer",
            agent=agent1,
            tools=[get_final_answer],
        )
    ]

    crew = Crew(agents=[agent1], tasks=tasks, verbose=True)

    with patch.object(ToolUsage, "_remember_format") as remember_format:
        crew.kickoff()
        remember_format.assert_called()


@pytest.mark.vcr()
def test_agent_use_specific_tasks_output_as_context(capsys):
    agent1 = Agent(role="test role", goal="test goal", backstory="test backstory")
    agent2 = Agent(role="test role2", goal="test goal2", backstory="test backstory2")

    say_hi_task = Task(
        description="Just say hi.", agent=agent1, expected_output="Your greeting."
    )
    say_bye_task = Task(
        description="Just say bye.", agent=agent1, expected_output="Your farewell."
    )
    answer_task = Task(
        description="Answer accordingly to the context you got.",
        expected_output="Your answer.",
        context=[say_hi_task],
        agent=agent2,
    )

    tasks = [say_hi_task, say_bye_task, answer_task]

    crew = Crew(agents=[agent1, agent2], tasks=tasks)
    result = crew.kickoff()

    assert "bye" not in result.raw.lower()
    assert "hi" in result.raw.lower() or "hello" in result.raw.lower()


@pytest.mark.vcr()
def test_agent_step_callback():
    class StepCallback:
        def callback(self, step):
            pass

    with patch.object(StepCallback, "callback") as callback:

        @tool
        def learn_about_ai() -> str:
            """Useful for when you need to learn about AI to write an paragraph about it."""
            return "AI is a very broad field."

        agent1 = Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            tools=[learn_about_ai],
            step_callback=StepCallback().callback,
        )

        essay = Task(
            description="Write and then review an small paragraph on AI until it's AMAZING",
            expected_output="The final paragraph.",
            agent=agent1,
        )
        tasks = [essay]
        crew = Crew(agents=[agent1], tasks=tasks)

        callback.return_value = "ok"
        crew.kickoff()
        callback.assert_called()


@pytest.mark.vcr()
def test_agent_function_calling_llm():
    from crewai.llm import LLM
    llm = LLM(model="gpt-4o", is_litellm=True)

    @tool
    def learn_about_ai() -> str:
        """Useful for when you need to learn about AI to write an paragraph about it."""
        return "AI is a very broad field."

    agent1 = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[learn_about_ai],
        llm="gpt-4o",
        max_iter=2,
        function_calling_llm=llm,
    )

    essay = Task(
        description="Write and then review an small paragraph on AI until it's AMAZING",
        expected_output="The final paragraph.",
        agent=agent1,
    )
    tasks = [essay]
    crew = Crew(agents=[agent1], tasks=tasks)
    from unittest.mock import patch

    from crewai.tools.tool_usage import ToolUsage
    import instructor

    with (
        patch.object(
            instructor, "from_litellm", wraps=instructor.from_litellm
        ) as mock_from_litellm,
        patch.object(
            ToolUsage,
            "_original_tool_calling",
            side_effect=Exception("Forced exception"),
        ) as mock_original_tool_calling,
    ):
        crew.kickoff()
        mock_from_litellm.assert_called()
        mock_original_tool_calling.assert_called()


@pytest.mark.vcr()
def test_tool_result_as_answer_is_the_final_answer_for_the_agent():
    from crewai.tools import BaseTool

    class MyCustomTool(BaseTool):
        name: str = "Get Greetings"
        description: str = "Get a random greeting back"

        def _run(self) -> str:
            return "Howdy!"

    agent1 = Agent(
        role="Data Scientist",
        goal="Product amazing resports on AI",
        backstory="You work with data and AI",
        tools=[MyCustomTool(result_as_answer=True)],
    )

    essay = Task(
        description="Write and then review an small paragraph on AI until it's AMAZING. But first use the `Get Greetings` tool to get a greeting.",
        expected_output="The final paragraph with the full review on AI and no greeting.",
        agent=agent1,
    )
    tasks = [essay]
    crew = Crew(agents=[agent1], tasks=tasks)

    result = crew.kickoff()
    assert result.raw == "Howdy!"


@pytest.mark.vcr()
def test_tool_usage_information_is_appended_to_agent():
    from crewai.tools import BaseTool

    class MyCustomTool(BaseTool):
        name: str = "Decide Greetings"
        description: str = "Decide what is the appropriate greeting to use"

        def _run(self) -> str:
            return "Howdy!"

    agent1 = Agent(
        role="Friendly Neighbor",
        goal="Make everyone feel welcome",
        backstory="You are the friendly neighbor",
        tools=[MyCustomTool(result_as_answer=True)],
    )

    greeting = Task(
        description="Say an appropriate greeting.",
        expected_output="The greeting.",
        agent=agent1,
    )
    tasks = [greeting]
    crew = Crew(agents=[agent1], tasks=tasks)

    crew.kickoff()
    assert agent1.tools_results == [
        {
            "result": "Howdy!",
            "tool_name": "Decide Greetings",
            "tool_args": {},
            "result_as_answer": True,
        }
    ]


def test_agent_definition_based_on_dict():
    config = {
        "role": "test role",
        "goal": "test goal",
        "backstory": "test backstory",
        "verbose": True,
    }

    agent = Agent(**config)

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"
    assert agent.verbose is True
    assert agent.tools == []


# test for human input
@pytest.mark.vcr()
def test_agent_human_input():
    # Agent configuration
    config = {
        "role": "test role",
        "goal": "test goal",
        "backstory": "test backstory",
    }

    agent = Agent(**config)

    # Task configuration with human input enabled
    task = Task(
        agent=agent,
        description="Say the word: Hi",
        expected_output="The word: Hi",
        human_input=True,
    )

    # Side effect function for _ask_human_input to simulate multiple feedback iterations
    feedback_responses = iter(
        [
            "Don't say hi, say Hello instead!",  # First feedback: instruct change
            "",  # Second feedback: empty string signals acceptance
        ]
    )

    def ask_human_input_side_effect(*args, **kwargs):
        return next(feedback_responses)

    # Patch both _ask_human_input and _invoke_loop to avoid real API/network calls.
    with (
        patch.object(
            CrewAgentExecutor,
            "_ask_human_input",
            side_effect=ask_human_input_side_effect,
        ) as mock_human_input,
        patch.object(
            CrewAgentExecutor,
            "_invoke_loop",
            return_value=AgentFinish(output="Hello", thought="", text=""),
        ),
    ):
        # Execute the task
        output = agent.execute_task(task)

        # Assertions to ensure the agent behaves correctly.
        # It should have requested feedback twice.
        assert mock_human_input.call_count == 2
        # The final result should be processed to "Hello"
        assert output.strip().lower() == "hello"


def test_interpolate_inputs():
    agent = Agent(
        role="{topic} specialist",
        goal="Figure {goal} out",
        backstory="I am the master of {role}",
    )

    agent.interpolate_inputs({"topic": "AI", "goal": "life", "role": "all things"})
    assert agent.role == "AI specialist"
    assert agent.goal == "Figure life out"
    assert agent.backstory == "I am the master of all things"

    agent.interpolate_inputs({"topic": "Sales", "goal": "stuff", "role": "nothing"})
    assert agent.role == "Sales specialist"
    assert agent.goal == "Figure stuff out"
    assert agent.backstory == "I am the master of nothing"


def test_not_using_system_prompt():
    agent = Agent(
        role="{topic} specialist",
        goal="Figure {goal} out",
        backstory="I am the master of {role}",
        use_system_prompt=False,
    )

    agent.create_agent_executor()
    assert not agent.agent_executor.prompt.get("user")
    assert not agent.agent_executor.prompt.get("system")


def test_using_system_prompt():
    agent = Agent(
        role="{topic} specialist",
        goal="Figure {goal} out",
        backstory="I am the master of {role}",
    )

    agent.create_agent_executor()
    assert agent.agent_executor.prompt.get("user")
    assert agent.agent_executor.prompt.get("system")


def test_system_and_prompt_template():
    agent = Agent(
        role="{topic} specialist",
        goal="Figure {goal} out",
        backstory="I am the master of {role}",
        system_template="""<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>""",
        prompt_template="""<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>""",
        response_template="""<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>""",
    )

    expected_prompt = """<|start_header_id|>system<|end_header_id|>

You are {role}. {backstory}
Your personal goal is: {goal}
To give my best complete final answer to the task use the exact following format:

Thought: I now can give a great answer
Final Answer: my best complete final answer to the task.
Your final answer must be the great and the most complete as possible, it must be outcome described.

I MUST use these formats, my job depends on it!<|eot_id|>
<|start_header_id|>user<|end_header_id|>


Current Task: {input}

Begin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it!

Thought:<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""

    with patch.object(CrewAgentExecutor, "_format_prompt") as mock_format_prompt:
        mock_format_prompt.return_value = expected_prompt

        # Trigger the _format_prompt method
        agent.agent_executor._format_prompt("dummy_prompt", {})

        # Assert that _format_prompt was called
        mock_format_prompt.assert_called_once()

        # Assert that the returned prompt matches the expected prompt
        assert mock_format_prompt.return_value == expected_prompt


@pytest.mark.vcr()
def test_task_allow_crewai_trigger_context():
    from crewai import Crew

    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    task = Task(
        description="Analyze the data",
        expected_output="Analysis report",
        agent=agent,
        allow_crewai_trigger_context=True,
    )
    crew = Crew(agents=[agent], tasks=[task])
    crew.kickoff({"crewai_trigger_payload": "Important context data"})

    prompt = task.prompt()

    assert "Analyze the data" in prompt
    assert "Trigger Payload: Important context data" in prompt


@pytest.mark.vcr()
def test_task_without_allow_crewai_trigger_context():
    from crewai import Crew

    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    task = Task(
        description="Analyze the data",
        expected_output="Analysis report",
        agent=agent,
        allow_crewai_trigger_context=False,
    )

    crew = Crew(agents=[agent], tasks=[task])
    crew.kickoff({"crewai_trigger_payload": "Important context data"})

    prompt = task.prompt()

    assert "Analyze the data" in prompt
    assert "Trigger Payload:" not in prompt
    assert "Important context data" not in prompt


@pytest.mark.vcr()
def test_task_allow_crewai_trigger_context_no_payload():
    from crewai import Crew

    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    task = Task(
        description="Analyze the data",
        expected_output="Analysis report",
        agent=agent,
        allow_crewai_trigger_context=True,
    )

    crew = Crew(agents=[agent], tasks=[task])
    crew.kickoff({"other_input": "other data"})

    prompt = task.prompt()

    assert "Analyze the data" in prompt
    assert "Trigger Payload:" not in prompt


@pytest.mark.vcr()
def test_do_not_allow_crewai_trigger_context_for_first_task_hierarchical():
    from crewai import Crew

    agent1 = Agent(role="First Agent", goal="First goal", backstory="First backstory")
    agent2 = Agent(
        role="Second Agent", goal="Second goal", backstory="Second backstory"
    )

    first_task = Task(
        description="Process initial data",
        expected_output="Initial analysis",
        agent=agent1,
    )

    crew = Crew(
        agents=[agent1, agent2],
        tasks=[first_task],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
    )

    crew.kickoff({"crewai_trigger_payload": "Initial context data"})

    first_prompt = first_task.prompt()
    assert "Process initial data" in first_prompt
    assert "Trigger Payload: Initial context data" not in first_prompt


@pytest.mark.vcr()
def test_first_task_auto_inject_trigger():
    from crewai import Crew

    agent1 = Agent(role="First Agent", goal="First goal", backstory="First backstory")
    agent2 = Agent(
        role="Second Agent", goal="Second goal", backstory="Second backstory"
    )

    first_task = Task(
        description="Process initial data",
        expected_output="Initial analysis",
        agent=agent1,
    )

    second_task = Task(
        description="Process secondary data",
        expected_output="Secondary analysis",
        agent=agent2,
    )

    crew = Crew(agents=[agent1, agent2], tasks=[first_task, second_task])
    crew.kickoff({"crewai_trigger_payload": "Initial context data"})

    first_prompt = first_task.prompt()
    assert "Process initial data" in first_prompt
    assert "Trigger Payload: Initial context data" in first_prompt

    second_prompt = second_task.prompt()
    assert "Process secondary data" in second_prompt
    assert "Trigger Payload:" not in second_prompt


@pytest.mark.vcr()
def test_ensure_first_task_allow_crewai_trigger_context_is_false_does_not_inject():
    from crewai import Crew

    agent1 = Agent(role="First Agent", goal="First goal", backstory="First backstory")
    agent2 = Agent(
        role="Second Agent", goal="Second goal", backstory="Second backstory"
    )

    first_task = Task(
        description="Process initial data",
        expected_output="Initial analysis",
        agent=agent1,
        allow_crewai_trigger_context=False,
    )

    second_task = Task(
        description="Process secondary data",
        expected_output="Secondary analysis",
        agent=agent2,
        allow_crewai_trigger_context=True,
    )

    crew = Crew(agents=[agent1, agent2], tasks=[first_task, second_task])
    crew.kickoff({"crewai_trigger_payload": "Context data"})

    first_prompt = first_task.prompt()
    assert "Trigger Payload: Context data" not in first_prompt

    second_prompt = second_task.prompt()
    assert "Trigger Payload: Context data" in second_prompt


@patch("crewai.agent.core.CrewTrainingHandler")
def test_agent_training_handler(crew_training_handler):
    task_prompt = "What is 1 + 1?"
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        verbose=True,
    )
    crew_training_handler.return_value.load.return_value = {
        f"{agent.id!s}": {"0": {"human_feedback": "good"}}
    }

    result = agent._training_handler(task_prompt=task_prompt)

    assert result == "What is 1 + 1?\n\nYou MUST follow these instructions: \n good"

    crew_training_handler.assert_has_calls(
        [mock.call("training_data.pkl"), mock.call().load()]
    )


@patch("crewai.agent.core.CrewTrainingHandler")
def test_agent_use_trained_data(crew_training_handler):
    task_prompt = "What is 1 + 1?"
    agent = Agent(
        role="researcher",
        goal="test goal",
        backstory="test backstory",
        verbose=True,
    )
    crew_training_handler.return_value.load.return_value = {
        agent.role: {
            "suggestions": [
                "The result of the math operation must be right.",
                "Result must be better than 1.",
            ]
        }
    }

    result = agent._use_trained_data(task_prompt=task_prompt)

    assert (
        result == "What is 1 + 1?\n\nYou MUST follow these instructions: \n"
        " - The result of the math operation must be right.\n - Result must be better than 1."
    )
    crew_training_handler.assert_has_calls(
        [mock.call("trained_agents_data.pkl"), mock.call().load()]
    )


def test_agent_max_retry_limit():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_retry_limit=1,
    )

    task = Task(
        agent=agent,
        description="Say the word: Hi",
        expected_output="The word: Hi",
        human_input=True,
    )

    error_message = "Error happening while sending prompt to model."
    with patch.object(
        CrewAgentExecutor, "invoke", wraps=agent.agent_executor.invoke
    ) as invoke_mock:
        invoke_mock.side_effect = Exception(error_message)

        assert agent._times_executed == 0
        assert agent.max_retry_limit == 1

        with pytest.raises(Exception) as e:
            agent.execute_task(
                task=task,
            )
        assert e.value.args[0] == error_message
        assert agent._times_executed == 2

        invoke_mock.assert_has_calls(
            [
                mock.call(
                    {
                        "input": "Say the word: Hi\n\nThis is the expected criteria for your final answer: The word: Hi\nyou MUST return the actual complete content as the final answer, not a summary.",
                        "tool_names": "",
                        "tools": "",
                        "ask_for_human_input": True,
                    }
                ),
                mock.call(
                    {
                        "input": "Say the word: Hi\n\nThis is the expected criteria for your final answer: The word: Hi\nyou MUST return the actual complete content as the final answer, not a summary.",
                        "tool_names": "",
                        "tools": "",
                        "ask_for_human_input": True,
                    }
                ),
            ]
        )


def test_agent_with_llm():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="gpt-3.5-turbo", temperature=0.7),
    )

    assert isinstance(agent.llm, BaseLLM)
    assert agent.llm.model == "gpt-3.5-turbo"
    assert agent.llm.temperature == 0.7


def test_agent_with_custom_stop_words():
    stop_words = ["STOP", "END"]
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="gpt-3.5-turbo", stop=stop_words),
    )

    assert isinstance(agent.llm, BaseLLM)
    assert set(agent.llm.stop) == set([*stop_words, "\nObservation:"])
    assert all(word in agent.llm.stop for word in stop_words)
    assert "\nObservation:" in agent.llm.stop


def test_agent_with_callbacks():
    def dummy_callback(response):
        pass

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="gpt-3.5-turbo", callbacks=[dummy_callback], is_litellm=True),
    )

    assert isinstance(agent.llm, BaseLLM)
    # All LLM implementations now support callbacks consistently
    assert hasattr(agent.llm, "callbacks")
    assert len(agent.llm.callbacks) == 1
    assert agent.llm.callbacks[0] == dummy_callback


def test_agent_with_additional_kwargs():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(
            model="gpt-3.5-turbo",
            temperature=0.8,
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.1,
        ),
    )

    assert isinstance(agent.llm, BaseLLM)
    assert agent.llm.model == "gpt-3.5-turbo"
    assert agent.llm.temperature == 0.8
    assert agent.llm.top_p == 0.9
    assert agent.llm.presence_penalty == 0.1
    assert agent.llm.frequency_penalty == 0.1


@pytest.mark.vcr()
def test_llm_call():
    llm = LLM(model="gpt-3.5-turbo")
    messages = [{"role": "user", "content": "Say 'Hello, World!'"}]

    response = llm.call(messages)
    assert "Hello, World!" in response


@pytest.mark.vcr()
def test_llm_call_with_error():
    llm = LLM(model="non-existent-model")
    messages = [{"role": "user", "content": "This should fail"}]

    with pytest.raises(Exception):  # noqa: B017
        llm.call(messages)


@pytest.mark.vcr()
def test_handle_context_length_exceeds_limit():
    # Import necessary modules
    from crewai.utilities.agent_utils import handle_context_length
    from crewai.utilities.i18n import I18N
    from crewai.utilities.printer import Printer

    # Create mocks for dependencies
    printer = Printer()
    i18n = I18N()

    # Create an agent just for its LLM
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        respect_context_window=True,
    )

    llm = agent.llm

    # Create test messages
    messages = [
        {
            "role": "user",
            "content": "This is a test message that would exceed context length",
        }
    ]

    # Set up test parameters
    respect_context_window = True
    callbacks = []

    # Apply our patch to summarize_messages to force an error
    with patch("crewai.utilities.agent_utils.summarize_messages") as mock_summarize:
        mock_summarize.side_effect = ValueError("Context length limit exceeded")

        # Directly call handle_context_length with our parameters
        with pytest.raises(ValueError) as excinfo:
            handle_context_length(
                respect_context_window=respect_context_window,
                printer=printer,
                messages=messages,
                llm=llm,
                callbacks=callbacks,
                i18n=i18n,
            )

        # Verify our patch was called and raised the correct error
        assert "Context length limit exceeded" in str(excinfo.value)
        mock_summarize.assert_called_once()


@pytest.mark.vcr()
def test_handle_context_length_exceeds_limit_cli_no():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        respect_context_window=False,
    )
    task = Task(description="test task", agent=agent, expected_output="test output")

    with patch.object(
        CrewAgentExecutor, "invoke", wraps=agent.agent_executor.invoke
    ) as private_mock:
        task = Task(
            description="The final answer is 42. But don't give it yet, instead keep using the `get_final_answer` tool.",
            expected_output="The final answer",
        )
        agent.execute_task(
            task=task,
        )
        private_mock.assert_called_once()
        pytest.raises(SystemExit)
        with patch(
            "crewai.utilities.agent_utils.handle_context_length"
        ) as mock_handle_context:
            mock_handle_context.assert_not_called()


def test_agent_with_all_llm_attributes():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(
            model="gpt-3.5-turbo",
            timeout=10,
            temperature=0.7,
            top_p=0.9,
            # n=1,
            stop=["STOP", "END"],
            max_tokens=100,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            # logit_bias={50256: -100},  # Example: bias against the EOT token
            response_format={"type": "json_object"},
            seed=42,
            logprobs=True,
            top_logprobs=5,
            base_url="https://api.openai.com/v1",
            # api_version="2023-05-15",
            api_key="sk-your-api-key-here",
        ),
    )

    assert isinstance(agent.llm, BaseLLM)
    assert agent.llm.model == "gpt-3.5-turbo"
    assert agent.llm.timeout == 10
    assert agent.llm.temperature == 0.7
    assert agent.llm.top_p == 0.9
    # assert agent.llm.n == 1
    assert set(agent.llm.stop) == set(["STOP", "END", "\nObservation:"])
    assert all(word in agent.llm.stop for word in ["STOP", "END", "\nObservation:"])
    assert agent.llm.max_tokens == 100
    assert agent.llm.presence_penalty == 0.1
    assert agent.llm.frequency_penalty == 0.1
    # assert agent.llm.logit_bias == {50256: -100}
    assert agent.llm.response_format == {"type": "json_object"}
    assert agent.llm.seed == 42
    assert agent.llm.logprobs
    assert agent.llm.top_logprobs == 5
    assert agent.llm.base_url == "https://api.openai.com/v1"
    # assert agent.llm.api_version == "2023-05-15"
    assert agent.llm.api_key == "sk-your-api-key-here"


@pytest.mark.vcr()
def test_llm_call_with_all_attributes():
    llm = LLM(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=50,
        stop=["STOP"],
        presence_penalty=0.1,
        frequency_penalty=0.1,
    )
    messages = [{"role": "user", "content": "Say 'Hello, World!' and then say STOP"}]

    response = llm.call(messages)
    assert "Hello, World!" in response
    assert "STOP" not in response


@pytest.mark.vcr()
@pytest.mark.skip(reason="Requires local Ollama instance")
def test_agent_with_ollama_llama3():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="ollama/llama3.2:3b", base_url="http://localhost:11434"),
    )

    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "ollama/llama3.2:3b"
    assert agent.llm.base_url == "http://localhost:11434"

    task = "Respond in 20 words. Which model are you?"
    response = agent.llm.call([{"role": "user", "content": task}])

    assert response
    assert len(response.split()) <= 25  # Allow a little flexibility in word count
    assert "Llama3" in response or "AI" in response or "language model" in response


@pytest.mark.vcr()
@pytest.mark.skip(reason="Requires local Ollama instance")
def test_llm_call_with_ollama_llama3():
    llm = LLM(
        model="ollama/llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=30,
    )
    messages = [
        {"role": "user", "content": "Respond in 20 words. Which model are you?"}
    ]

    response = llm.call(messages)

    assert response
    assert len(response.split()) <= 25  # Allow a little flexibility in word count
    assert "Llama3" in response or "AI" in response or "language model" in response


@pytest.mark.vcr()
def test_agent_execute_task_basic():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm="gpt-4o-mini",
    )

    task = Task(
        description="Calculate 2 + 2",
        expected_output="The result of the calculation",
        agent=agent,
    )

    result = agent.execute_task(task)
    assert "4" in result


@pytest.mark.vcr()
def test_agent_execute_task_with_context():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="gpt-3.5-turbo"),
    )

    task = Task(
        description="Summarize the given context in one sentence",
        expected_output="A one-sentence summary",
        agent=agent,
    )

    context = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."

    result = agent.execute_task(task, context=context)
    assert len(result.split(".")) == 3
    assert "fox" in result.lower() and "dog" in result.lower()


@pytest.mark.vcr()
def test_agent_execute_task_with_tool():
    @tool
    def dummy_tool(query: str) -> str:
        """Useful for when you need to get a dummy result for a query."""
        return f"Dummy result for: {query}"

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="gpt-3.5-turbo"),
        tools=[dummy_tool],
    )

    task = Task(
        description="Use the dummy tool to get a result for 'test query'",
        expected_output="The result from the dummy tool",
        agent=agent,
    )

    result = agent.execute_task(task)
    assert "you should always think about what to do" in result


@pytest.mark.vcr()
def test_agent_execute_task_with_custom_llm():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="gpt-3.5-turbo", temperature=0.7, max_tokens=50),
    )

    task = Task(
        description="Write a haiku about AI",
        expected_output="A haiku (3 lines, 5-7-5 syllable pattern) about AI",
        agent=agent,
    )

    result = agent.execute_task(task)
    assert "In circuits they thrive" in result
    assert "Artificial minds awake" in result
    assert "Future's coded drive" in result


@pytest.mark.vcr()
@pytest.mark.skip(reason="Requires local Ollama instance")
def test_agent_execute_task_with_ollama():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="ollama/llama3.2:3b", base_url="http://localhost:11434"),
    )

    task = Task(
        description="Explain what AI is in one sentence",
        expected_output="A one-sentence explanation of AI",
        agent=agent,
    )

    result = agent.execute_task(task)
    assert len(result.split(".")) == 2
    assert "AI" in result or "artificial intelligence" in result.lower()


@pytest.mark.vcr()
def test_agent_with_knowledge_sources():
    content = "Brandon's favorite color is red and he likes Mexican food."
    string_source = StringKnowledgeSource(content=content)
    with patch("crewai.knowledge") as mock_knowledge:
        mock_knowledge_instance = mock_knowledge.return_value
        mock_knowledge_instance.sources = [string_source]
        mock_knowledge_instance.search.return_value = [{"content": content}]
        mock_knowledge.add_sources.return_value = [string_source]

        agent = Agent(
            role="Information Agent",
            goal="Provide information based on knowledge sources",
            backstory="You have access to specific knowledge sources.",
            llm=LLM(model="gpt-4o-mini"),
            knowledge_sources=[string_source],
        )

        task = Task(
            description="What is Brandon's favorite color?",
            expected_output="Brandon's favorite color.",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        with patch.object(Knowledge, "add_sources") as mock_add_sources:
            result = crew.kickoff()
            assert mock_add_sources.called, "add_sources() should have been called"
            mock_add_sources.assert_called_once()
            assert "red" in result.raw.lower()


@pytest.mark.vcr()
def test_agent_with_knowledge_sources_with_query_limit_and_score_threshold():
    content = "Brandon's favorite color is red and he likes Mexican food."
    string_source = StringKnowledgeSource(content=content)
    knowledge_config = KnowledgeConfig(results_limit=10, score_threshold=0.5)
    with (
        patch(
            "crewai.knowledge.storage.knowledge_storage.KnowledgeStorage"
        ) as mock_knowledge_storage,
        patch(
            "crewai.knowledge.source.base_knowledge_source.KnowledgeStorage"
        ) as mock_base_knowledge_storage,
        patch("crewai.rag.chromadb.client.ChromaDBClient") as mock_chromadb,
    ):
        mock_storage_instance = mock_knowledge_storage.return_value
        mock_storage_instance.sources = [string_source]
        mock_storage_instance.query.return_value = [{"content": content}]
        mock_storage_instance.save.return_value = None

        mock_chromadb_instance = mock_chromadb.return_value
        mock_chromadb_instance.add_documents.return_value = None

        mock_base_knowledge_storage.return_value = mock_storage_instance

        with patch.object(Knowledge, "query") as mock_knowledge_query:
            agent = Agent(
                role="Information Agent",
                goal="Provide information based on knowledge sources",
                backstory="You have access to specific knowledge sources.",
                llm=LLM(model="gpt-4o-mini"),
                knowledge_sources=[string_source],
                knowledge_config=knowledge_config,
            )
            task = Task(
                description="What is Brandon's favorite color?",
                expected_output="Brandon's favorite color.",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task])
            crew.kickoff()

            assert agent.knowledge is not None
            mock_knowledge_query.assert_called_once_with(
                ["Brandon's favorite color"],
                **knowledge_config.model_dump(),
            )


@pytest.mark.vcr()
def test_agent_with_knowledge_sources_with_query_limit_and_score_threshold_default():
    content = "Brandon's favorite color is red and he likes Mexican food."
    string_source = StringKnowledgeSource(content=content)
    knowledge_config = KnowledgeConfig()

    with (
        patch(
            "crewai.knowledge.storage.knowledge_storage.KnowledgeStorage"
        ) as mock_knowledge_storage,
        patch(
            "crewai.knowledge.source.base_knowledge_source.KnowledgeStorage"
        ) as mock_base_knowledge_storage,
        patch("crewai.rag.chromadb.client.ChromaDBClient") as mock_chromadb,
    ):
        mock_storage_instance = mock_knowledge_storage.return_value
        mock_storage_instance.sources = [string_source]
        mock_storage_instance.query.return_value = [{"content": content}]
        mock_storage_instance.save.return_value = None

        mock_chromadb_instance = mock_chromadb.return_value
        mock_chromadb_instance.add_documents.return_value = None

        mock_base_knowledge_storage.return_value = mock_storage_instance

        with patch.object(Knowledge, "query") as mock_knowledge_query:
            agent = Agent(
                role="Information Agent",
                goal="Provide information based on knowledge sources",
                backstory="You have access to specific knowledge sources.",
                llm=LLM(model="gpt-4o-mini"),
                knowledge_sources=[string_source],
                knowledge_config=knowledge_config,
            )
            task = Task(
                description="What is Brandon's favorite color?",
                expected_output="Brandon's favorite color.",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task])
            crew.kickoff()

            assert agent.knowledge is not None
            mock_knowledge_query.assert_called_once_with(
                ["Brandon's favorite color"],
                **knowledge_config.model_dump(),
            )


@pytest.mark.vcr()
def test_agent_with_knowledge_sources_extensive_role():
    content = "Brandon's favorite color is red and he likes Mexican food."
    string_source = StringKnowledgeSource(content=content)

    with (
        patch("crewai.knowledge") as mock_knowledge,
        patch(
            "crewai.knowledge.storage.knowledge_storage.KnowledgeStorage.save"
        ) as mock_save,
    ):
        mock_knowledge_instance = mock_knowledge.return_value
        mock_knowledge_instance.sources = [string_source]
        mock_knowledge_instance.query.return_value = [{"content": content}]
        mock_save.return_value = None

        agent = Agent(
            role="Information Agent with extensive role description that is longer than 80 characters",
            goal="Provide information based on knowledge sources",
            backstory="You have access to specific knowledge sources.",
            llm=LLM(model="gpt-4o-mini"),
            knowledge_sources=[string_source],
        )

        task = Task(
            description="What is Brandon's favorite color?",
            expected_output="Brandon's favorite color.",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        assert "red" in result.raw.lower()


@pytest.mark.vcr()
def test_agent_with_knowledge_sources_works_with_copy():
    content = "Brandon's favorite color is red and he likes Mexican food."
    string_source = StringKnowledgeSource(content=content)

    with patch(
        "crewai.knowledge.source.base_knowledge_source.BaseKnowledgeSource",
        autospec=True,
    ) as mock_knowledge_source:
        mock_knowledge_source_instance = mock_knowledge_source.return_value
        mock_knowledge_source_instance.__class__ = BaseKnowledgeSource
        mock_knowledge_source_instance.sources = [string_source]

        agent = Agent(
            role="Information Agent",
            goal="Provide information based on knowledge sources",
            backstory="You have access to specific knowledge sources.",
            llm=LLM(model="gpt-4o-mini"),
            knowledge_sources=[string_source],
        )

        with patch(
            "crewai.knowledge.storage.knowledge_storage.KnowledgeStorage"
        ) as mock_knowledge_storage:
            mock_knowledge_storage_instance = mock_knowledge_storage.return_value
            agent.knowledge_storage = mock_knowledge_storage_instance

            agent_copy = agent.copy()

            assert agent_copy.role == agent.role
            assert agent_copy.goal == agent.goal
            assert agent_copy.backstory == agent.backstory
            assert agent_copy.knowledge_sources is not None
            assert len(agent_copy.knowledge_sources) == 1
            assert isinstance(agent_copy.knowledge_sources[0], StringKnowledgeSource)
            assert agent_copy.knowledge_sources[0].content == content
            assert isinstance(agent_copy.llm, BaseLLM)


@pytest.mark.vcr()
def test_agent_with_knowledge_sources_generate_search_query():
    content = "Brandon's favorite color is red and he likes Mexican food."
    string_source = StringKnowledgeSource(content=content)

    with (
        patch("crewai.knowledge") as mock_knowledge,
        patch(
            "crewai.knowledge.storage.knowledge_storage.KnowledgeStorage"
        ) as mock_knowledge_storage,
        patch(
            "crewai.knowledge.source.base_knowledge_source.KnowledgeStorage"
        ) as mock_base_knowledge_storage,
        patch("crewai.rag.chromadb.client.ChromaDBClient") as mock_chromadb,
    ):
        mock_knowledge_instance = mock_knowledge.return_value
        mock_knowledge_instance.sources = [string_source]
        mock_knowledge_instance.query.return_value = [{"content": content}]

        mock_storage_instance = mock_knowledge_storage.return_value
        mock_storage_instance.sources = [string_source]
        mock_storage_instance.query.return_value = [{"content": content}]
        mock_storage_instance.save.return_value = None

        mock_chromadb_instance = mock_chromadb.return_value
        mock_chromadb_instance.add_documents.return_value = None

        mock_base_knowledge_storage.return_value = mock_storage_instance

        agent = Agent(
            role="Information Agent with extensive role description that is longer than 80 characters",
            goal="Provide information based on knowledge sources",
            backstory="You have access to specific knowledge sources.",
            llm=LLM(model="gpt-4o-mini"),
            knowledge_sources=[string_source],
        )

        task = Task(
            description="What is Brandon's favorite color?",
            expected_output="The answer to the question, in a format like this: `{{name: str, favorite_color: str}}`",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        # Updated assertion to check the JSON content
        assert "Brandon" in str(agent.knowledge_search_query)
        assert "favorite color" in str(agent.knowledge_search_query)

        assert "red" in result.raw.lower()


@pytest.mark.vcr()
@pytest.mark.skip(reason="Requires OpenRouter API key")
def test_agent_with_knowledge_with_no_crewai_knowledge():
    mock_knowledge = MagicMock(spec=Knowledge)

    agent = Agent(
        role="Information Agent",
        goal="Provide information based on knowledge sources",
        backstory="You have access to specific knowledge sources.",
        llm=LLM(
            model="openrouter/openai/gpt-4o-mini",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        ),
        knowledge=mock_knowledge,
    )

    # Create a task that requires the agent to use the knowledge
    task = Task(
        description="What is Vidit's favorite color?",
        expected_output="Vidit's favorclearite color.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    crew.kickoff()
    mock_knowledge.query.assert_called_once()


@pytest.mark.vcr()
def test_agent_with_only_crewai_knowledge():
    mock_knowledge = MagicMock(spec=Knowledge)

    agent = Agent(
        role="Information Agent",
        goal="Provide information based on knowledge sources",
        backstory="You have access to specific knowledge sources.",
        llm=LLM(
            model="gpt-4o-mini",
        ),
    )

    # Create a task that requires the agent to use the knowledge
    task = Task(
        description="What is Vidit's favorite color?",
        expected_output="Vidit's favorite color.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], knowledge=mock_knowledge)
    crew.kickoff()
    mock_knowledge.query.assert_called_once()


@pytest.mark.vcr()
@pytest.mark.skip(reason="Requires OpenRouter API key")
def test_agent_knowledege_with_crewai_knowledge():
    crew_knowledge = MagicMock(spec=Knowledge)
    agent_knowledge = MagicMock(spec=Knowledge)

    agent = Agent(
        role="Information Agent",
        goal="Provide information based on knowledge sources",
        backstory="You have access to specific knowledge sources.",
        llm=LLM(
            model="openrouter/openai/gpt-4o-mini",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        ),
        knowledge=agent_knowledge,
    )

    # Create a task that requires the agent to use the knowledge
    task = Task(
        description="What is Vidit's favorite color?",
        expected_output="Vidit's favorclearite color.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], knowledge=crew_knowledge)
    crew.kickoff()
    agent_knowledge.query.assert_called_once()
    crew_knowledge.query.assert_called_once()


@pytest.mark.vcr()
def test_litellm_auth_error_handling():
    """Test that LiteLLM authentication errors are handled correctly and not retried."""
    from litellm import AuthenticationError as LiteLLMAuthenticationError

    # Create an agent with a mocked LLM and max_retry_limit=0
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="gpt-4", is_litellm=True),
        max_retry_limit=0,  # Disable retries for authentication errors
    )

    # Create a task
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )

    # Mock the LLM call to raise AuthenticationError
    with (
        patch.object(LLM, "call") as mock_llm_call,
        pytest.raises(LiteLLMAuthenticationError, match="Invalid API key"),
    ):
        mock_llm_call.side_effect = LiteLLMAuthenticationError(
            message="Invalid API key", llm_provider="openai", model="gpt-4"
        )
        agent.execute_task(task)

    # Verify the call was only made once (no retries)
    mock_llm_call.assert_called_once()


def test_crew_agent_executor_litellm_auth_error():
    """Test that CrewAgentExecutor handles LiteLLM authentication errors by raising them."""
    from crewai.agents.tools_handler import ToolsHandler
    from litellm.exceptions import AuthenticationError

    # Create an agent and executor
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="gpt-4", api_key="invalid_api_key", is_litellm=True),
    )
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )

    # Create executor with all required parameters
    executor = CrewAgentExecutor(
        agent=agent,
        task=task,
        llm=agent.llm,
        crew=None,
        prompt={"system": "You are a test agent", "user": "Execute the task: {input}"},
        max_iter=5,
        tools=[],
        tools_names="",
        stop_words=[],
        tools_description="",
        tools_handler=ToolsHandler(),
    )

    # Mock the LLM call to raise AuthenticationError
    with (
        patch.object(LLM, "call") as mock_llm_call,
        pytest.raises(AuthenticationError) as exc_info,
    ):
        mock_llm_call.side_effect = AuthenticationError(
            message="Invalid API key", llm_provider="openai", model="gpt-4"
        )
        executor.invoke(
            {
                "input": "test input",
                "tool_names": "",
                "tools": "",
            }
        )

    # Verify the call was only made once (no retries)
    mock_llm_call.assert_called_once()

    # Assert that the exception was raised and has the expected attributes
    assert exc_info.type is AuthenticationError
    assert "Invalid API key".lower() in exc_info.value.message.lower()
    assert exc_info.value.llm_provider == "openai"
    assert exc_info.value.model == "gpt-4"


def test_litellm_anthropic_error_handling():
    """Test that AnthropicError from LiteLLM is handled correctly and not retried."""
    from litellm.llms.anthropic.common_utils import AnthropicError

    # Create an agent with a mocked LLM that uses an Anthropic model
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="claude-3.5-sonnet-20240620", is_litellm=True),
        max_retry_limit=0,
    )

    # Create a task
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )

    # Mock the LLM call to raise AnthropicError
    with (
        patch.object(LLM, "call") as mock_llm_call,
        pytest.raises(AnthropicError, match="Test Anthropic error"),
    ):
        mock_llm_call.side_effect = AnthropicError(
            status_code=500,
            message="Test Anthropic error",
        )
        agent.execute_task(task)

    # Verify the LLM call was only made once (no retries)
    mock_llm_call.assert_called_once()


@pytest.mark.vcr()
def test_get_knowledge_search_query():
    """Test that _get_knowledge_search_query calls the LLM with the correct prompts."""
    from crewai.utilities.i18n import I18N

    content = "The capital of France is Paris."
    string_source = StringKnowledgeSource(content=content)

    agent = Agent(
        role="Information Agent",
        goal="Provide information based on knowledge sources",
        backstory="I have access to knowledge sources",
        llm=LLM(model="gpt-4"),
        knowledge_sources=[string_source],
    )

    task = Task(
        description="What is the capital of France?",
        expected_output="The capital of France is Paris.",
        agent=agent,
    )

    i18n = I18N()
    task_prompt = task.prompt()

    with (
        patch(
            "crewai.knowledge.storage.knowledge_storage.KnowledgeStorage"
        ) as mock_knowledge_storage,
        patch(
            "crewai.knowledge.source.base_knowledge_source.KnowledgeStorage"
        ) as mock_base_knowledge_storage,
        patch("crewai.rag.chromadb.client.ChromaDBClient") as mock_chromadb,
        patch.object(agent, "_get_knowledge_search_query") as mock_get_query,
    ):
        mock_storage_instance = mock_knowledge_storage.return_value
        mock_storage_instance.sources = [string_source]
        mock_storage_instance.query.return_value = [{"content": content}]
        mock_storage_instance.save.return_value = None

        mock_chromadb_instance = mock_chromadb.return_value
        mock_chromadb_instance.add_documents.return_value = None

        mock_base_knowledge_storage.return_value = mock_storage_instance

        mock_get_query.return_value = "Capital of France"

        crew = Crew(agents=[agent], tasks=[task])
        crew.kickoff()

        mock_get_query.assert_called_once_with(task_prompt, task)

    with patch.object(agent.llm, "call") as mock_llm_call:
        agent._get_knowledge_search_query(task_prompt, task)

        mock_llm_call.assert_called_once_with(
            [
                {
                    "role": "system",
                    "content": i18n.slice(
                        "knowledge_search_query_system_prompt"
                    ).format(task_prompt=task.description),
                },
                {
                    "role": "user",
                    "content": i18n.slice("knowledge_search_query").format(
                        task_prompt=task_prompt
                    ),
                },
            ]
        )


@pytest.fixture
def mock_get_auth_token():
    with patch(
        "crewai.cli.authentication.token.get_auth_token", return_value="test_token"
    ):
        yield


@patch("crewai.cli.plus_api.PlusAPI.get_agent")
def test_agent_from_repository(mock_get_agent, mock_get_auth_token):
    from crewai_tools import (
        FileReadTool,
        SerperDevTool,
    )

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = {
        "role": "test role",
        "goal": "test goal",
        "backstory": "test backstory",
        "tools": [
            {
                "module": "crewai_tools",
                "name": "SerperDevTool",
                "init_params": {"n_results": "30"},
            },
            {
                "module": "crewai_tools",
                "name": "FileReadTool",
                "init_params": {"file_path": "test.txt"},
            },
        ],
    }
    mock_get_agent.return_value = mock_get_response

    agent = Agent(from_repository="test_agent")

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"
    assert len(agent.tools) == 2

    assert isinstance(agent.tools[0], SerperDevTool)
    assert agent.tools[0].n_results == 30
    assert isinstance(agent.tools[1], FileReadTool)
    assert agent.tools[1].file_path == "test.txt"


@patch("crewai.cli.plus_api.PlusAPI.get_agent")
def test_agent_from_repository_override_attributes(mock_get_agent, mock_get_auth_token):
    from crewai_tools import SerperDevTool

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = {
        "role": "test role",
        "goal": "test goal",
        "backstory": "test backstory",
        "tools": [
            {"name": "SerperDevTool", "module": "crewai_tools", "init_params": {}}
        ],
    }
    mock_get_agent.return_value = mock_get_response
    agent = Agent(from_repository="test_agent", role="Custom Role")

    assert agent.role == "Custom Role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"
    assert len(agent.tools) == 1
    assert isinstance(agent.tools[0], SerperDevTool)


@patch("crewai.cli.plus_api.PlusAPI.get_agent")
def test_agent_from_repository_with_invalid_tools(mock_get_agent, mock_get_auth_token):
    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = {
        "role": "test role",
        "goal": "test goal",
        "backstory": "test backstory",
        "tools": [
            {
                "name": "DoesNotExist",
                "module": "crewai_tools",
            }
        ],
    }
    mock_get_agent.return_value = mock_get_response
    with pytest.raises(
        AgentRepositoryError,
        match="Tool DoesNotExist could not be loaded: module 'crewai_tools' has no attribute 'DoesNotExist'",
    ):
        Agent(from_repository="test_agent")


@patch("crewai.cli.plus_api.PlusAPI.get_agent")
def test_agent_from_repository_internal_error(mock_get_agent, mock_get_auth_token):
    mock_get_response = MagicMock()
    mock_get_response.status_code = 500
    mock_get_response.text = "Internal server error"
    mock_get_agent.return_value = mock_get_response
    with pytest.raises(
        AgentRepositoryError,
        match="Agent test_agent could not be loaded: Internal server error",
    ):
        Agent(from_repository="test_agent")


@patch("crewai.cli.plus_api.PlusAPI.get_agent")
def test_agent_from_repository_agent_not_found(mock_get_agent, mock_get_auth_token):
    mock_get_response = MagicMock()
    mock_get_response.status_code = 404
    mock_get_response.text = "Agent not found"
    mock_get_agent.return_value = mock_get_response
    with pytest.raises(
        AgentRepositoryError,
        match="Agent test_agent does not exist, make sure the name is correct or the agent is available on your organization",
    ):
        Agent(from_repository="test_agent")


@patch("crewai.cli.plus_api.PlusAPI.get_agent")
@patch("crewai.utilities.agent_utils.Settings")
@patch("crewai.utilities.agent_utils.console")
def test_agent_from_repository_displays_org_info(
    mock_console, mock_settings, mock_get_agent, mock_get_auth_token
):
    mock_settings_instance = MagicMock()
    mock_settings_instance.org_uuid = "test-org-uuid"
    mock_settings_instance.org_name = "Test Organization"
    mock_settings.return_value = mock_settings_instance

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = {
        "role": "test role",
        "goal": "test goal",
        "backstory": "test backstory",
        "tools": [],
    }
    mock_get_agent.return_value = mock_get_response

    agent = Agent(from_repository="test_agent")

    mock_console.print.assert_any_call(
        "Fetching agent from organization: Test Organization (test-org-uuid)",
        style="bold blue",
    )

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"


@patch("crewai.cli.plus_api.PlusAPI.get_agent")
@patch("crewai.utilities.agent_utils.Settings")
@patch("crewai.utilities.agent_utils.console")
def test_agent_from_repository_without_org_set(
    mock_console, mock_settings, mock_get_agent, mock_get_auth_token
):
    mock_settings_instance = MagicMock()
    mock_settings_instance.org_uuid = None
    mock_settings_instance.org_name = None
    mock_settings.return_value = mock_settings_instance

    mock_get_response = MagicMock()
    mock_get_response.status_code = 401
    mock_get_response.text = "Unauthorized access"
    mock_get_agent.return_value = mock_get_response

    with pytest.raises(
        AgentRepositoryError,
        match="Agent test_agent could not be loaded: Unauthorized access",
    ):
        Agent(from_repository="test_agent")

    mock_console.print.assert_any_call(
        "No organization currently set. We recommend setting one before using: `crewai org switch <org_id>` command.",
        style="yellow",
    )

def test_agent_apps_consolidated_functionality():
    agent = Agent(
        role="Platform Agent",
        goal="Use platform tools",
        backstory="Platform specialist",
        apps=["gmail/create_task", "slack/update_status", "hubspot"]
    )
    expected = {"gmail/create_task", "slack/update_status", "hubspot"}
    assert set(agent.apps) == expected

    agent_apps_only = Agent(
        role="App Agent",
        goal="Use apps",
        backstory="App specialist",
        apps=["gmail", "slack"]
    )
    assert set(agent_apps_only.apps) == {"gmail", "slack"}

    agent_default = Agent(
        role="Regular Agent",
        goal="Regular tasks",
        backstory="Regular agent"
    )
    assert agent_default.apps is None


def test_agent_apps_validation():
    agent = Agent(
        role="Custom Agent",
        goal="Test validation",
        backstory="Test agent",
        apps=["custom_app", "another_app/action"]
    )
    assert set(agent.apps) == {"custom_app", "another_app/action"}

    with pytest.raises(ValueError, match=r"Invalid app format.*Apps can only have one '/' for app/action format"):
        Agent(
            role="Invalid Agent",
            goal="Test validation",
            backstory="Test agent",
            apps=["app/action/invalid"]
        )


@patch.object(Agent, 'get_platform_tools')
def test_app_actions_propagated_to_platform_tools(mock_get_platform_tools):
    from crewai.tools import tool

    @tool
    def action_tool() -> str:
        """Mock action platform tool."""
        return "action tool result"

    mock_get_platform_tools.return_value = [action_tool]

    agent = Agent(
        role="Action Agent",
        goal="Execute actions",
        backstory="Action specialist",
        apps=["gmail/send_email", "slack/update_status"]
    )

    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )

    crew = Crew(agents=[agent], tasks=[task])
    tools = crew._prepare_tools(agent, task, [])

    mock_get_platform_tools.assert_called_once()
    call_args = mock_get_platform_tools.call_args[1]
    assert set(call_args["apps"]) == {"gmail/send_email", "slack/update_status"}
    assert len(tools) >= 1


@patch.object(Agent, 'get_platform_tools')
def test_mixed_apps_and_actions_propagated(mock_get_platform_tools):
    from crewai.tools import tool

    @tool
    def combined_tool() -> str:
        """Mock combined platform tool."""
        return "combined tool result"

    mock_get_platform_tools.return_value = [combined_tool]

    agent = Agent(
        role="Combined Agent",
        goal="Use apps and actions",
        backstory="Platform specialist",
        apps=["gmail", "slack", "gmail/create_task", "slack/update_status"]
    )

    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )

    crew = Crew(agents=[agent], tasks=[task])
    tools = crew._prepare_tools(agent, task, [])

    mock_get_platform_tools.assert_called_once()
    call_args = mock_get_platform_tools.call_args[1]
    expected_apps = {"gmail", "slack", "gmail/create_task", "slack/update_status"}
    assert set(call_args["apps"]) == expected_apps
    assert len(tools) >= 1

def test_agent_without_apps_no_platform_tools():
    """Test that agents without apps don't trigger platform tools integration."""
    agent = Agent(
        role="Regular Agent",
        goal="Regular tasks",
        backstory="Regular agent"
    )

    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )

    crew = Crew(agents=[agent], tasks=[task])

    tools = crew._prepare_tools(agent, task, [])
    assert tools == []
