# BrowserUseTool

## Description

Runs a task in a cloud browser with the [Browser Use](https://browser-use.com) Cloud agent (API v4) and returns the agent's final answer. The agent navigates, clicks, fills forms and extracts data.

## Installation

```shell
uv add "crewai-tools[browser-use]"
```

Set `BROWSER_USE_API_KEY` (create a key at https://cloud.browser-use.com/settings?tab=api-keys).

## Example

```python
from crewai_tools import BrowserUseTool

tool = BrowserUseTool()
tool.run(task="Find the title of the top story on news.ycombinator.com")
```

The result is JSON with `result`, `session_id` and `run_id`. Pass `session_id` back to continue in the same browser.

## Arguments

- `api_key` Optional. Default is the `BROWSER_USE_API_KEY` env variable.
- `model` Optional. Browser Use agent model. Default is the API default.
- `timeout` Optional. Seconds to wait before the run is cancelled. Default is `900`.
