from textwrap import dedent
from typing import Callable

import mcp

from crewai.tools import BaseTool, ToolCollection, tool


def test_creating_a_tool_using_annotation():
    @tool("Name of my tool")
    def my_tool(question: str) -> str:
        """Clear description for what this tool is useful for, your agent will need this information to use it."""
        return question

    # Assert all the right attributes were defined
    assert my_tool.name == "Name of my tool"
    assert (
        my_tool.description
        == "Tool Name: Name of my tool\nTool Arguments: {'question': {'description': None, 'type': 'str'}}\nTool Description: Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    assert my_tool.args_schema.model_json_schema()["properties"] == {
        "question": {"title": "Question", "type": "string"}
    }
    assert (
        my_tool.func("What is the meaning of life?") == "What is the meaning of life?"
    )

    converted_tool = my_tool.to_structured_tool()
    assert converted_tool.name == "Name of my tool"

    assert (
        converted_tool.description
        == "Tool Name: Name of my tool\nTool Arguments: {'question': {'description': None, 'type': 'str'}}\nTool Description: Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    assert converted_tool.args_schema.model_json_schema()["properties"] == {
        "question": {"title": "Question", "type": "string"}
    }
    assert (
        converted_tool.func("What is the meaning of life?")
        == "What is the meaning of life?"
    )


def test_creating_a_tool_using_baseclass():
    class MyCustomTool(BaseTool):
        name: str = "Name of my tool"
        description: str = "Clear description for what this tool is useful for, your agent will need this information to use it."

        def _run(self, question: str) -> str:
            return question

    my_tool = MyCustomTool()
    # Assert all the right attributes were defined
    assert my_tool.name == "Name of my tool"

    assert (
        my_tool.description
        == "Tool Name: Name of my tool\nTool Arguments: {'question': {'description': None, 'type': 'str'}}\nTool Description: Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    assert my_tool.args_schema.model_json_schema()["properties"] == {
        "question": {"title": "Question", "type": "string"}
    }
    assert my_tool.run("What is the meaning of life?") == "What is the meaning of life?"

    converted_tool = my_tool.to_structured_tool()
    assert converted_tool.name == "Name of my tool"

    assert (
        converted_tool.description
        == "Tool Name: Name of my tool\nTool Arguments: {'question': {'description': None, 'type': 'str'}}\nTool Description: Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    assert converted_tool.args_schema.model_json_schema()["properties"] == {
        "question": {"title": "Question", "type": "string"}
    }
    assert (
        converted_tool._run("What is the meaning of life?")
        == "What is the meaning of life?"
    )


def test_setting_cache_function():
    class MyCustomTool(BaseTool):
        name: str = "Name of my tool"
        description: str = "Clear description for what this tool is useful for, your agent will need this information to use it."
        cache_function: Callable = lambda: False

        def _run(self, question: str) -> str:
            return question

    my_tool = MyCustomTool()
    # Assert all the right attributes were defined
    assert not my_tool.cache_function()


def test_default_cache_function_is_true():
    class MyCustomTool(BaseTool):
        name: str = "Name of my tool"
        description: str = "Clear description for what this tool is useful for, your agent will need this information to use it."

        def _run(self, question: str) -> str:
            return question

    my_tool = MyCustomTool()
    # Assert all the right attributes were defined
    assert my_tool.cache_function()


def test_tool_collection_from_mcp():
    # define the most simple mcp server with one tool that echoes the input text
    mcp_server_script = dedent("""\
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Echo Server")

        @mcp.tool()
        def echo_tool(text: str) -> str:
            return text

        mcp.run()
    """).strip()

    mcp_server_params = mcp.StdioServerParameters(
        command="python",
        args=["-c", mcp_server_script],
    )

    with ToolCollection.from_mcp(mcp_server_params) as tool_collection:
        assert len(tool_collection.tools) == 1, "Expected 1 tool"
        assert tool_collection.tools[0].name == "echo_tool", (
            "Expected tool name to be 'echo_tool'"
        )
        assert tool_collection.tools[0].run(text="Hello") == "Hello", (
            "Expected tool to echo the input text"
        )


def test_tool_collection_from_mcp_sse():
    import subprocess
    import time

    # define the most simple mcp server with one tool that echoes the input text
    mcp_server_script = dedent("""\
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Echo Server", host="127.0.0.1", port=8000)

        @mcp.tool()
        def echo_tool(text: str) -> str:
            return text

        mcp.run("sse")
    """).strip()

    # start the SSE mcp server in a subprocess
    server_process = subprocess.Popen(
        ["python", "-c", mcp_server_script],
    )

    # wait for the server to start
    time.sleep(1)

    try:
        with ToolCollection.from_mcp(
            {"url": "http://127.0.0.1:8000/sse"}
        ) as tool_collection:
            assert len(tool_collection.tools) == 1, "Expected 1 tool"
            assert tool_collection.tools[0].name == "echo_tool", (
                "Expected tool name to be 'echo_tool'"
            )
            assert tool_collection.tools[0].run(text="Hello") == "Hello", (
                "Expected tool to echo the input text"
            )
    finally:
        # clean up the process when test is done
        server_process.kill()
        server_process.wait()
