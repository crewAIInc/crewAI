# Baizhi Cloud tools

`BaizhiSearchTool`, `BaizhiScrapeTool`, and `BaizhiExtractTool` use the hosted
[Baizhi Cloud Agent Toolkit](https://github.com/chaitin/baizhi-agent-toolkit)
through Streamable HTTP MCP at `https://agent-toolkit.app.baizhi.cloud/mcp`.
The wire tools are `websearch_search`, `web_scrape`, and `web_extract`.

A Baizhi Cloud account and your own API key are required. Requests send search
terms, URLs, and extraction instructions to Baizhi Cloud and may consume paid
credits. Check your account's current pricing and balance before calling the
tools. These client tools are MIT-licensed as part of CrewAI; the hosted backend
is a separate service and is not included in this source code.

## Setup

```bash
uv add 'crewai-tools[mcp]'
```

Set `BAIZHI_API_KEY` in your process environment or secret manager. The tools also
accept `api_key` when constructed; do not put the key in agent prompts or tool
arguments. It is excluded from the tool's serialized configuration and repr.
A deserialized tool needs its key supplied again, for example from the environment.

```python
from crewai_tools import BaizhiExtractTool, BaizhiScrapeTool, BaizhiSearchTool

search = BaizhiSearchTool(timeout=60)
print(search.run(query="MCP protocol documentation", count=5,
                 filter={"domains": ["modelcontextprotocol.io"]}))

scrape = BaizhiScrapeTool()
print(scrape.run(url="https://example.com", return_format="markdown"))

extract = BaizhiExtractTool()
print(extract.run(url="https://example.com", fields={"title": "string"}))
# Alternatively: extract.run(url="https://example.com", instruction="Extract the title")

# Pass these instances to Agent(tools=[search, scrape, extract], ...).
# In async code use: await search.arun(query="MCP protocol documentation")
```

Search accepts 1–50 results, `time_range` (`day`, `week`, `month`, `year`), and
optional `need_summary`. Put site restrictions in `filter.domains` or
`filter.exclude_domains` as bare domains/IPs, rather than adding `site:` to the
query. Extraction requires nonempty `fields` or `instruction`; field types are
`string`, `number`, `boolean`, and `array`. Scraping and extraction accept
`accept_language`; `download` defaults to `False` and should be enabled only if
the user requests an export.

Results are JSON strings when the server returns MCP `structuredContent`, or
joined text blocks otherwise. MCP tool errors and transport failures raise
sanitized exceptions; non-text blocks without structured data produce an error. Literal key echoes in successful results are redacted. SDK diagnostics inside a Baizhi call remove key echoes and raw exception tracebacks; unrelated SDK logging is unchanged. Avoid HTTP debug logging when using secrets. Redirects and proxy environment variables are disabled. Page URLs with embedded credentials and malformed domain filters are rejected, but URL validation is not a network-level guarantee that a page is public. Do not submit private or signed URLs.

`timeout` is a total deadline covering initialization and the tool call.
There are no automatic tool-call retries. Async cancellation propagates, but a
timeout or cancellation cannot guarantee that the remote operation stopped or
that credits were not consumed. Application or agent retries are separate and
can incur another charge. Enable only the tools needed for the task and treat
returned web content as untrusted input.
