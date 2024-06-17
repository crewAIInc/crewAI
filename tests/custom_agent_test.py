"""Test Custom CustomAgent creation and execution basic functionality."""

from unittest.mock import patch

import pytest
from langchain.tools import tool
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI

from crewai import Crew, Task, CustomAgent, Agent
from crewai.agents.cache import CacheHandler

from crewai.agents.parser import CrewAgentParser
from crewai.utilities import RPMController

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)

# example langchain custom agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


@tool
def multiplier(first_number: int, second_number: int) -> float:
    """Useful for when you need to multiply two numbers together."""
    return first_number * second_number


tools = [get_word_length, multiplier]
llm_with_tools = llm.bind_tools(tools)
custom_langchain_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)


def test_custom_agent_creation():
    agent_executor = AgentExecutor(
        agent=custom_langchain_agent, tools=tools, verbose=True
    ).invoke
    agent = CustomAgent(
        agent_executor=agent_executor,
        role="test role",
        goal="test goal",
        backstory="test backstory",
    )

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"
    assert agent.tools == []
    assert agent.agent_executor == agent_executor


def test_custom_agent_default_values():
    agent_executor = AgentExecutor(
        agent=custom_langchain_agent, tools=tools, verbose=True
    ).invoke
    agent = CustomAgent(
        agent_executor=agent_executor,
        role="test role",
        goal="test goal",
        backstory="test backstory",
    )

    assert isinstance(agent.llm, ChatOpenAI)
    assert agent.llm.model_name == "gpt-4o"
    assert agent.llm.temperature == 0.7
    assert agent.llm.verbose is False
    assert agent.allow_delegation is True


@pytest.mark.vcr(filter_headers=["authorization"])
def test_custom_agent_execution():
    agent_executor = AgentExecutor(
        agent=custom_langchain_agent, tools=tools, verbose=True
    ).invoke
    agent = CustomAgent(
        agent_executor=agent_executor,
        output_key="output",  # if we are using langchain custom agents then we need an output key otherwise we don't
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
    assert output == "1 + 1 = 2"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_custom_agent_execution_with_tools():
    @tool
    def multiplier(first_number: int, second_number: int) -> float:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    agent = CustomAgent(
        agent_executor=AgentExecutor(
            agent=custom_langchain_agent, tools=[multiplier], verbose=True
        ).invoke,
        output_key="output",
        role="test role",
        goal="test goal",
        backstory="test backstory",
        allow_delegation=False,
    )

    multiply_task = Task(
        description="multiply 4 and 4",
        agent=agent,
        expected_output="The result of the multiplication.",
    )

    result = agent.execute_task(multiply_task)
    assert result == "The result of the multiplication is 16."


@pytest.mark.vcr(filter_headers=["authorization"])
def test_disabling_cache_for_custom_agent():
    @tool
    def multiplier(first_number: int, second_number: int) -> float:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    cache_handler = CacheHandler()

    agent = CustomAgent(
        agent_executor=AgentExecutor(
            agent=custom_langchain_agent, tools=[multiplier], verbose=True
        ).invoke,
        output_key="output",
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
def test_agent_without_max_rpm_respet_crew_rpm(capsys):
    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def get_final_answer(numbers) -> float:
        """Get the final answer but don't give it yet, just re-use this
        tool non-stop."""
        return 42

    agent1 = CustomAgent(
        agent_executor=AgentExecutor(
            agent=custom_langchain_agent, tools=[get_final_answer], verbose=True
        ).invoke,
        output_key="output",
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_rpm=10,
        verbose=True,
        allow_delegation=False,
    )

    agent2 = Agent(
        role="test role2",
        goal="test goal2",
        backstory="test backstory2",
        max_iter=2,
        verbose=True,
        allow_delegation=False,
    )

    tasks = [
        Task(
            description="Just say hi.", agent=agent1, expected_output="Your greeting."
        ),
        Task(
            description="NEVER give a Final Answer, instead keep using the `get_final_answer` tool non-stop",
            expected_output="The final answer",
            tools=[get_final_answer],
            agent=agent2,
        ),
    ]

    crew = Crew(agents=[agent1, agent2], tasks=tasks, max_rpm=1, verbose=2)

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        crew.kickoff()
        captured = capsys.readouterr()
        assert "get_final_answer" in captured.out
        assert "Max RPM reached, waiting for next minute to start." in captured.out
        moveon.assert_called_once()


def test_agent_count_formatting_error():
    from unittest.mock import patch

    agent1 = CustomAgent(
        agent_executor=AgentExecutor(
            agent=custom_langchain_agent, tools=[multiplier], verbose=True
        ).invoke,
        output_key="output",
        role="test role",
        goal="test goal",
        backstory="test backstory",
        verbose=True,
    )

    parser = CrewAgentParser()
    parser.agent = agent1

    with patch.object(CustomAgent, "increment_formatting_errors") as mock_count_errors:
        test_text = "This text does not match expected formats."
        with pytest.raises(OutputParserException):
            parser.parse(test_text)
        mock_count_errors.assert_called_once()


def test_interpolate_inputs():
    agent = CustomAgent(
        agent_executor=AgentExecutor(
            agent=custom_langchain_agent, tools=[multiplier], verbose=True
        ).invoke,
        output_key="output",
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
