from textwrap import dedent

import pytest
from mcp import StdioServerParameters

from crewai_tools import MCPServerAdapter


@pytest.fixture
def echo_server_script():
    return dedent(
        '''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Echo Server")

        @mcp.tool()
        def echo_tool(text: str) -> str:
            """Echo the input text"""
            return f"Echo: {text}"
        
        mcp.run()
        '''
    )


@pytest.fixture
def echo_server_sse_script():
    return dedent(
        '''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Echo Server", host="127.0.0.1", port=8000)

        @mcp.tool()
        def echo_tool(text: str) -> str:
            """Echo the input text"""
            return f"Echo: {text}"

        mcp.run("sse")
        '''
    )


@pytest.fixture
def echo_sse_server(echo_server_sse_script):
    import subprocess
    import time

    # Start the SSE server process with its own process group
    process = subprocess.Popen(
        ["python", "-c", echo_server_sse_script],
    )

    # Give the server a moment to start up
    time.sleep(1)

    try:
        yield {"url": "http://127.0.0.1:8000/sse"}
    finally:
        # Clean up the process when test is done
        process.kill()
        process.wait()


def test_context_manager_syntax(echo_server_script):
    serverparams = StdioServerParameters(
        command="uv", args=["run", "python", "-c", echo_server_script]
    )
    with MCPServerAdapter(serverparams) as tools:
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"
        assert tools[0].run(text="hello") == "Echo: hello"

def test_context_manager_syntax_sse(echo_sse_server):
    sse_serverparams = echo_sse_server
    with MCPServerAdapter(sse_serverparams) as tools:
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"
        assert tools[0].run(text="hello") == "Echo: hello"

def test_try_finally_syntax(echo_server_script):
    serverparams = StdioServerParameters(
        command="uv", args=["run", "python", "-c", echo_server_script]
    )
    try:
        mcp_server_adapter = MCPServerAdapter(serverparams)
        tools = mcp_server_adapter.tools
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"
        assert tools[0].run(text="hello") == "Echo: hello"
    finally:
        mcp_server_adapter.stop()
        
def test_try_finally_syntax_sse(echo_sse_server):
    sse_serverparams = echo_sse_server
    mcp_server_adapter = MCPServerAdapter(sse_serverparams)
    try:
        tools = mcp_server_adapter.tools
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"
        assert tools[0].run(text="hello") == "Echo: hello"
    finally:
        mcp_server_adapter.stop()
