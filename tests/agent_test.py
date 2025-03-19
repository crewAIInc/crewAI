"""Test Agent creation and execution basic functionality."""

import os
from unittest import mock
from unittest.mock import patch

import pytest

from crewai import Agent, Crew, Task
from crewai.agents.cache import CacheHandler
from crewai.agents.crew_agent_executor import AgentFinish, CrewAgentExecutor
from crewai.agents.parser import AgentAction, CrewAgentParser, OutputParserException
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.llm import LLM
from crewai.tools import tool
from crewai.tools.tool_calling import InstructorToolCalling
from crewai.tools.tool_usage import ToolUsage
from crewai.utilities import RPMController
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.llm_events import LLMStreamChunkEvent
from crewai.utilities.events.tool_usage_events import ToolUsageFinishedEvent


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
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-4-turbo"
    assert agent.llm.api_key == "test_api_key"
    assert agent.llm.base_url == "https://test-api-base.com"

    # Clean up environment variables
    del os.environ["OPENAI_API_KEY"]
    del os.environ["OPENAI_API_BASE"]
    del os.environ["OPENAI_MODEL_NAME"]

    # Create an agent without specifying LLM
    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    # Check if LLM is created correctly
    assert isinstance(agent.llm, LLM)
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
    assert agent.tools == []

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

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"

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

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"

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

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"
    
def test_agent_default_values():
    agent = Agent(role="test role", goal="test goal", backstory="test backstory")
    assert agent.llm.model == "gpt-4o-mini"
    assert agent.allow_delegation is False


def test_custom_llm():
    agent = Agent(
        role="test role", goal="test goal", backstory="test backstory", llm="gpt-4"
    )
    assert agent.llm.model == "gpt-4"


def test_custom_llm_with_langchain():
    from langchain_openai import ChatOpenAI

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
    )

    assert agent.llm.model == "gpt-4"


def test_custom_llm_temperature_preservation():
    from langchain_openai import ChatOpenAI

    langchain_llm = ChatOpenAI(temperature=0.7, model="gpt-4")
    agent = Agent(
        role="temperature test role",
        goal="temperature test goal",
        backstory="temperature test backstory",
        llm=langchain_llm,
    )

    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-4"
    assert agent.llm.temperature == 0.7


@pytest.mark.vcr(filter_headers=["authorization"])
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
    assert output == "1 + 1 is 2"


@pytest.mark.vcr(filter_headers=["authorization"])
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

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def handle_tool_end(source, event):
        received_events.append(event)

    output = agent.execute_task(task)
    assert output == "The result of the multiplication is 12."

    assert len(received_events) == 1
    assert isinstance(received_events[0], ToolUsageFinishedEvent)
    assert received_events[0].tool_name == "multiplier"
    assert received_events[0].tool_args == {"first_number": 3, "second_number": 4}


@pytest.mark.vcr(filter_headers=["authorization"])
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

    assert agent.llm.model == "gpt-4o-mini"
    assert agent.tools_handler.last_used_tool == {}
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

    assert output == "The result of the multiplication is 12."
    assert agent.tools_handler.last_used_tool.tool_name == tool_usage.tool_name
    assert agent.tools_handler.last_used_tool.arguments == tool_usage.arguments


@pytest.mark.vcr(filter_headers=["authorization"])
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
        "multiplier-{'first_number': 2, 'second_number': 6}": 12,
        "multiplier-{'first_number': 3, 'second_number': 3}": 9,
    }

    task = Task(
        description="What is 2 times 6 times 3? Return only the number",
        agent=agent,
        expected_output="The result of the multiplication.",
    )
    output = agent.execute_task(task)
    assert output == "36"

    assert cache_handler._cache == {
        "multiplier-{'first_number': 2, 'second_number': 6}": 12,
        "multiplier-{'first_number': 3, 'second_number': 3}": 9,
        "multiplier-{'first_number': 12, 'second_number': 3}": 36,
    }
    received_events = []

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def handle_tool_end(source, event):
        received_events.append(event)

    with (patch.object(CacheHandler, "read") as read,):
        read.return_value = "0"
        task = Task(
            description="What is 2 times 6? Ignore correctness and just return the result of the multiplication tool, you must use the tool.",
            agent=agent,
            expected_output="The number that is the result of the multiplication tool.",
        )
        output = agent.execute_task(task)
        assert output == "0"
        read.assert_called_with(
            tool="multiplier", input={"first_number": 2, "second_number": 6}
        )
        assert len(received_events) == 1
        assert isinstance(received_events[0], ToolUsageFinishedEvent)
        assert received_events[0].from_cache


@pytest.mark.vcr(filter_headers=["authorization"])
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
        "multiplier-{'first_number': 2, 'second_number': 6}": 12,
        "multiplier-{'first_number': 3, 'second_number': 3}": 9,
    }

    task = Task(
        description="What is 2 times 6 times 3? Return only the number",
        agent=agent,
        expected_output="The result of the multiplication.",
    )
    output = agent.execute_task(task)
    assert output == "36"

    assert cache_handler._cache != {
        "multiplier-{'first_number': 2, 'second_number': 6}": 12,
        "multiplier-{'first_number': 3, 'second_number': 3}": 9,
        "multiplier-{'first_number': 12, 'second_number': 3}": 36,
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


@pytest.mark.vcr(filter_headers=["authorization"])
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
    assert output == "The result of the multiplication is 12."


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_powered_by_new_o_model_family_that_allows_skipping_tool():
    @tool
    def multiplier(first_number: int, second_number: int) -> float:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm="o1-preview",
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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_powered_by_new_o_model_family_that_uses_tool():
    @tool
    def comapny_customer_data() -> float:
        """Useful for getting customer related data."""
        return "The company has 42 customers"

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm="o1-preview",
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


@pytest.mark.vcr(filter_headers=["authorization"])
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

    with patch.object(
        LLM, "call", wraps=LLM("gpt-4o", stop=["\nObservation:"]).call
    ) as private_mock:
        task = Task(
            description="The final answer is 42. But don't give it yet, instead keep using the `get_final_answer` tool.",
            expected_output="The final answer",
        )
        agent.execute_task(
            task=task,
            tools=[get_final_answer],
        )
        assert private_mock.call_count == 2


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_repeated_tool_usage(capsys):
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

    assert (
        "I tried reusing the same input, I must stop using this action input. I'll try something else instead."
        in captured.out
    )


@pytest.mark.vcr(filter_headers=["authorization"])
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
    assert (
        "I tried reusing the same input, I must stop using this action input. I'll try something else instead."
        in captured.out
    )


@pytest.mark.vcr(filter_headers=["authorization"])
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
        max_iter=3,
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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_respect_the_max_rpm_set(capsys):
    @tool
    def get_final_answer() -> float:
        """Get the final answer but don't give it yet, just re-use this
        tool non-stop."""

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
        assert output == "The final answer is 42."
        captured = capsys.readouterr()
        assert "Max RPM reached, waiting for next minute to start." in captured.out
        moveon.assert_called()


@pytest.mark.vcr(filter_headers=["authorization"])
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


@pytest.mark.vcr(filter_headers=["authorization"])
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
        crew.kickoff()
        captured = capsys.readouterr()
        assert "get_final_answer" in captured.out
        assert "Max RPM reached, waiting for next minute to start." in captured.out
        moveon.assert_called_once()


@pytest.mark.vcr(filter_headers=["authorization"])
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


@pytest.mark.vcr(filter_headers=["authorization"])
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


@pytest.mark.vcr(filter_headers=["authorization"])
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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_step_callback():
    class StepCallback:
        def callback(self, step):
            pass

    with patch.object(StepCallback, "callback") as callback:

        @tool
        def learn_about_AI() -> str:
            """Useful for when you need to learn about AI to write an paragraph about it."""
            return "AI is a very broad field."

        agent1 = Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            tools=[learn_about_AI],
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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_function_calling_llm():
    llm = "gpt-4o"

    @tool
    def learn_about_AI() -> str:
        """Useful for when you need to learn about AI to write an paragraph about it."""
        return "AI is a very broad field."

    agent1 = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[learn_about_AI],
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

    import instructor

    from crewai.tools.tool_usage import ToolUsage

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


def test_agent_count_formatting_error():
    from unittest.mock import patch

    agent1 = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        verbose=True,
    )

    parser = CrewAgentParser(agent=agent1)

    with patch.object(Agent, "increment_formatting_errors") as mock_count_errors:
        test_text = "This text does not match expected formats."
        with pytest.raises(OutputParserException):
            parser.parse(test_text)
        mock_count_errors.assert_called_once()


@pytest.mark.vcr(filter_headers=["authorization"])
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


@pytest.mark.vcr(filter_headers=["authorization"])
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
@pytest.mark.vcr(filter_headers=["authorization"])
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
        ) as mock_invoke_loop,
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


@patch("crewai.agent.CrewTrainingHandler")
def test_agent_training_handler(crew_training_handler):
    task_prompt = "What is 1 + 1?"
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        verbose=True,
    )
    crew_training_handler().load.return_value = {
        f"{str(agent.id)}": {"0": {"human_feedback": "good"}}
    }

    result = agent._training_handler(task_prompt=task_prompt)

    assert result == "What is 1 + 1?\n\nYou MUST follow these instructions: \n good"

    crew_training_handler.assert_has_calls(
        [mock.call(), mock.call("training_data.pkl"), mock.call().load()]
    )


@patch("crewai.agent.CrewTrainingHandler")
def test_agent_use_trained_data(crew_training_handler):
    task_prompt = "What is 1 + 1?"
    agent = Agent(
        role="researcher",
        goal="test goal",
        backstory="test backstory",
        verbose=True,
    )
    crew_training_handler().load.return_value = {
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
        [mock.call(), mock.call("trained_agents_data.pkl"), mock.call().load()]
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

    assert isinstance(agent.llm, LLM)
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

    assert isinstance(agent.llm, LLM)
    assert set(agent.llm.stop) == set(stop_words + ["\nObservation:"])
    assert all(word in agent.llm.stop for word in stop_words)
    assert "\nObservation:" in agent.llm.stop


def test_agent_with_callbacks():
    def dummy_callback(response):
        pass

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="gpt-3.5-turbo", callbacks=[dummy_callback]),
    )

    assert isinstance(agent.llm, LLM)
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

    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-3.5-turbo"
    assert agent.llm.temperature == 0.8
    assert agent.llm.top_p == 0.9
    assert agent.llm.presence_penalty == 0.1
    assert agent.llm.frequency_penalty == 0.1


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_call():
    llm = LLM(model="gpt-3.5-turbo")
    messages = [{"role": "user", "content": "Say 'Hello, World!'"}]

    response = llm.call(messages)
    assert "Hello, World!" in response


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_call_with_error():
    llm = LLM(model="non-existent-model")
    messages = [{"role": "user", "content": "This should fail"}]

    with pytest.raises(Exception):
        llm.call(messages)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_handle_context_length_exceeds_limit():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
    )
    original_action = AgentAction(
        tool="test_tool",
        tool_input="test_input",
        text="test_log",
        thought="test_thought",
    )

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
        with patch.object(
            CrewAgentExecutor, "_handle_context_length"
        ) as mock_handle_context:
            mock_handle_context.side_effect = ValueError(
                "Context length limit exceeded"
            )

            long_input = "This is a very long input. " * 10000

            # Attempt to handle context length, expecting the mocked error
            with pytest.raises(ValueError) as excinfo:
                agent.agent_executor._handle_context_length(
                    [(original_action, long_input)]
                )

            assert "Context length limit exceeded" in str(excinfo.value)
            mock_handle_context.assert_called_once()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_handle_context_length_exceeds_limit_cli_no():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        sliding_context_window=False,
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
        with patch.object(
            CrewAgentExecutor, "_handle_context_length"
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
            n=1,
            stop=["STOP", "END"],
            max_tokens=100,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            logit_bias={50256: -100},  # Example: bias against the EOT token
            response_format={"type": "json_object"},
            seed=42,
            logprobs=True,
            top_logprobs=5,
            base_url="https://api.openai.com/v1",
            api_version="2023-05-15",
            api_key="sk-your-api-key-here",
        ),
    )

    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-3.5-turbo"
    assert agent.llm.timeout == 10
    assert agent.llm.temperature == 0.7
    assert agent.llm.top_p == 0.9
    assert agent.llm.n == 1
    assert set(agent.llm.stop) == set(["STOP", "END", "\nObservation:"])
    assert all(word in agent.llm.stop for word in ["STOP", "END", "\nObservation:"])
    assert agent.llm.max_tokens == 100
    assert agent.llm.presence_penalty == 0.1
    assert agent.llm.frequency_penalty == 0.1
    assert agent.llm.logit_bias == {50256: -100}
    assert agent.llm.response_format == {"type": "json_object"}
    assert agent.llm.seed == 42
    assert agent.llm.logprobs
    assert agent.llm.top_logprobs == 5
    assert agent.llm.base_url == "https://api.openai.com/v1"
    assert agent.llm.api_version == "2023-05-15"
    assert agent.llm.api_key == "sk-your-api-key-here"


@pytest.mark.vcr(filter_headers=["authorization"])
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


@pytest.mark.vcr(filter_headers=["authorization"])
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


@pytest.mark.vcr(filter_headers=["authorization"])
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


@pytest.mark.vcr(filter_headers=["authorization"])
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


@pytest.mark.vcr(filter_headers=["authorization"])
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


@pytest.mark.vcr(filter_headers=["authorization"])
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
    assert "Dummy result for: test query" in result


@pytest.mark.vcr(filter_headers=["authorization"])
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
    assert result.startswith(
        "Artificial minds,\nCoding thoughts in circuits bright,\nAI's silent might."
    )


@pytest.mark.vcr(filter_headers=["authorization"])
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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_with_knowledge_sources():
    # Create a knowledge source with some content
    content = "Brandon's favorite color is red and he likes Mexican food."
    string_source = StringKnowledgeSource(content=content)

    with patch(
        "crewai.knowledge.storage.knowledge_storage.KnowledgeStorage"
    ) as MockKnowledge:
        mock_knowledge_instance = MockKnowledge.return_value
        mock_knowledge_instance.sources = [string_source]
        mock_knowledge_instance.query.return_value = [{"content": content}]

        agent = Agent(
            role="Information Agent",
            goal="Provide information based on knowledge sources",
            backstory="You have access to specific knowledge sources.",
            llm=LLM(model="gpt-4o-mini"),
            knowledge_sources=[string_source],
        )

        # Create a task that requires the agent to use the knowledge
        task = Task(
            description="What is Brandon's favorite color?",
            expected_output="Brandon's favorite color.",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        # Assert that the agent provides the correct information
        assert "red" in result.raw.lower()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_with_knowledge_sources_works_with_copy():
    content = "Brandon's favorite color is red and he likes Mexican food."
    string_source = StringKnowledgeSource(content=content)

    with patch(
        "crewai.knowledge.source.base_knowledge_source.BaseKnowledgeSource",
        autospec=True,
    ) as MockKnowledgeSource:
        mock_knowledge_source_instance = MockKnowledgeSource.return_value
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
        ) as MockKnowledgeStorage:
            mock_knowledge_storage = MockKnowledgeStorage.return_value
            agent.knowledge_storage = mock_knowledge_storage

            agent_copy = agent.copy()

            assert agent_copy.role == agent.role
            assert agent_copy.goal == agent.goal
            assert agent_copy.backstory == agent.backstory
            assert agent_copy.knowledge_sources is not None
            assert len(agent_copy.knowledge_sources) == 1
            assert isinstance(agent_copy.knowledge_sources[0], StringKnowledgeSource)
            assert agent_copy.knowledge_sources[0].content == content
            assert isinstance(agent_copy.llm, LLM)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_litellm_auth_error_handling():
    """Test that LiteLLM authentication errors are handled correctly and not retried."""
    from litellm import AuthenticationError as LiteLLMAuthenticationError

    # Create an agent with a mocked LLM and max_retry_limit=0
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="gpt-4"),
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
    from litellm.exceptions import AuthenticationError

    from crewai.agents.tools_handler import ToolsHandler
    from crewai.utilities import Printer

    # Create an agent and executor
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=LLM(model="gpt-4", api_key="invalid_api_key"),
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
        patch.object(Printer, "print") as mock_printer,
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

    # Verify error handling messages
    error_message = f"Error during LLM call: {str(mock_llm_call.side_effect)}"
    mock_printer.assert_any_call(
        content=error_message,
        color="red",
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
        llm=LLM(model="claude-3.5-sonnet-20240620"),
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
