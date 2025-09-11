import json
from typing import Any, ClassVar
from unittest.mock import Mock, patch

from crewai.agent import Agent
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.crew import Crew
from crewai.project import CrewBase, agent, crew, task
from crewai.task import Task
from crewai.tools import tool


@tool
def mock_bigquery_single_row():
    """Mock BigQuery tool that returns a single row"""
    return {"id": 1, "name": "John", "age": 30}


@tool
def mock_bigquery_multiple_rows():
    """Mock BigQuery tool that returns multiple rows"""
    return [
        {"id": 1, "name": "John", "age": 30},
        {"id": 2, "name": "Jane", "age": 25},
        {"id": 3, "name": "Bob", "age": 35},
        {"id": 4, "name": "Alice", "age": 28},
    ]


@tool
def mock_bigquery_large_dataset():
    """Mock BigQuery tool that returns a large dataset"""
    return [{"id": i, "name": f"User{i}", "value": f"data_{i}"} for i in range(100)]


@tool
def mock_bigquery_nested_data():
    """Mock BigQuery tool that returns nested data structures"""
    return [
        {
            "id": 1,
            "user": {"name": "John", "email": "john@example.com"},
            "orders": [
                {"order_id": 101, "amount": 50.0},
                {"order_id": 102, "amount": 75.0},
            ],
        },
        {
            "id": 2,
            "user": {"name": "Jane", "email": "jane@example.com"},
            "orders": [{"order_id": 103, "amount": 100.0}],
        },
    ]


@CrewBase
class MCPTestCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    mcp_server_params: ClassVar[dict[str, Any]] = {"host": "localhost", "port": 8000}
    mcp_connect_timeout = 120

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def data_analyst(self):
        return Agent(
            role="Data Analyst",
            goal="Analyze data from various sources",
            backstory="Expert in data analysis and BigQuery",
            tools=[mock_bigquery_single_row, mock_bigquery_multiple_rows],
        )

    @agent
    def mcp_agent(self):
        return Agent(
            role="MCP Agent",
            goal="Use MCP tools to fetch data",
            backstory="Agent that uses MCP tools",
            tools=self.get_mcp_tools(),
        )

    @task
    def analyze_single_row(self):
        return Task(
            description="Use mock_bigquery_single_row tool to get data",
            expected_output="Single row of data",
            agent=self.data_analyst(),
        )

    @task
    def analyze_multiple_rows(self):
        return Task(
            description="Use mock_bigquery_multiple_rows tool to get data",
            expected_output="Multiple rows of data",
            agent=self.data_analyst(),
        )

    @crew
    def crew(self):
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


def test_single_row_tool_output():
    """Test that single row tool output works correctly"""
    result = mock_bigquery_single_row.invoke({})
    assert isinstance(result, dict)
    assert result["id"] == 1
    assert result["name"] == "John"
    assert result["age"] == 30


def test_multiple_rows_tool_output():
    """Test that multiple rows tool output is preserved"""
    result = mock_bigquery_multiple_rows.invoke({})
    assert isinstance(result, list)
    assert len(result) == 4
    assert result[0]["id"] == 1
    assert result[1]["id"] == 2
    assert result[2]["id"] == 3
    assert result[3]["id"] == 4


def test_large_dataset_tool_output():
    """Test that large datasets are handled correctly"""
    result = mock_bigquery_large_dataset.invoke({})
    assert isinstance(result, list)
    assert len(result) == 100
    assert result[0]["id"] == 0
    assert result[99]["id"] == 99


def test_nested_data_tool_output():
    """Test that nested data structures are preserved"""
    result = mock_bigquery_nested_data.invoke({})
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["user"]["name"] == "John"
    assert len(result[0]["orders"]) == 2
    assert result[1]["user"]["name"] == "Jane"
    assert len(result[1]["orders"]) == 1


def test_tool_result_formatting():
    """Test that tool results are properly formatted as strings"""
    from crewai.tools.tool_usage import ToolUsage

    tool_usage = ToolUsage()

    single_result = mock_bigquery_single_row.invoke({})
    formatted_single = tool_usage._format_result(single_result)
    assert isinstance(formatted_single, str)
    parsed_single = json.loads(formatted_single)
    assert parsed_single["id"] == 1

    multi_result = mock_bigquery_multiple_rows.invoke({})
    formatted_multi = tool_usage._format_result(multi_result)
    assert isinstance(formatted_multi, str)
    parsed_multi = json.loads(formatted_multi)
    assert len(parsed_multi) == 4
    assert parsed_multi[0]["id"] == 1
    assert parsed_multi[3]["id"] == 4


def test_mcp_crew_with_mock_tools():
    """Test MCP crew integration with mock tools"""
    with patch("embedchain.client.Client.setup"):
        from crewai_tools import MCPServerAdapter
        from crewai_tools.adapters.mcp_adapter import ToolCollection

    mock_adapter = Mock(spec=MCPServerAdapter)
    mock_adapter.tools = ToolCollection([mock_bigquery_multiple_rows])

    with patch("crewai_tools.MCPServerAdapter", return_value=mock_adapter):
        crew = MCPTestCrew()
        mcp_agent = crew.mcp_agent()
        assert mock_bigquery_multiple_rows in mcp_agent.tools


def test_tool_output_preserves_structure():
    """Test that tool output preserves data structure through the processing pipeline"""
    from crewai.tools.tool_usage import ToolUsage

    tool_usage = ToolUsage()

    bigquery_result = [
        {"id": 1, "name": "John", "revenue": 1000.50},
        {"id": 2, "name": "Jane", "revenue": 2500.75},
        {"id": 3, "name": "Bob", "revenue": 1750.25},
    ]

    formatted_result = tool_usage._format_result(bigquery_result)

    assert isinstance(formatted_result, str)

    parsed_result = json.loads(formatted_result)
    assert len(parsed_result) == 3
    assert parsed_result[0]["id"] == 1
    assert parsed_result[1]["name"] == "Jane"
    assert parsed_result[2]["revenue"] == 1750.25


def test_tool_output_backward_compatibility():
    """Test that simple string/number outputs still work"""
    from crewai.tools.tool_usage import ToolUsage

    tool_usage = ToolUsage()

    string_result = "Simple string result"
    formatted_string = tool_usage._format_result(string_result)
    assert formatted_string == "Simple string result"

    number_result = 42
    formatted_number = tool_usage._format_result(number_result)
    assert formatted_number == "42"

    bool_result = True
    formatted_bool = tool_usage._format_result(bool_result)
    assert formatted_bool == "True"


def test_malformed_data_handling():
    """Test that malformed data is handled gracefully"""
    from crewai.tools.tool_usage import ToolUsage

    tool_usage = ToolUsage()

    class NonSerializable:
        def __str__(self):
            return "NonSerializable object"

    non_serializable = NonSerializable()
    formatted_result = tool_usage._format_result(non_serializable)
    assert formatted_result == "NonSerializable object"
