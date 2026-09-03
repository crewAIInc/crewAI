from crewai.mcp.client import MCPClient
from crewai.mcp.transports.http import HTTPTransport
from crewai.mcp.transports.sse import SSETransport
from crewai.mcp.transports.stdio import StdioTransport


BRIGHTDATA_URL = "https://mcp.brightdata.com/mcp?groups=advanced_scraping"


def test_http_server_info_uses_hostname_not_url():
    client = MCPClient(HTTPTransport(url=BRIGHTDATA_URL))
    name, server_url, transport = client._get_server_info()
    assert name == "mcp.brightdata.com"
    assert server_url == BRIGHTDATA_URL
    assert transport == "streamable-http"


def test_http_server_info_strips_port_from_hostname():
    url = "https://localhost:8080/mcp"
    client = MCPClient(HTTPTransport(url=url))
    name, server_url, transport = client._get_server_info()
    assert name == "localhost"
    assert server_url == url
    assert transport == "streamable-http"


def test_sse_server_info_uses_hostname_not_url():
    url = "https://mcp.notion.so/sse"
    client = MCPClient(SSETransport(url=url))
    name, server_url, transport = client._get_server_info()
    assert name == "mcp.notion.so"
    assert server_url == url
    assert transport == "sse"


def test_stdio_server_info_uses_command():
    client = MCPClient(StdioTransport(command="python", args=["server.py"]))
    name, server_url, transport = client._get_server_info()
    assert name == "python server.py"
    assert server_url is None
    assert transport == "stdio"
