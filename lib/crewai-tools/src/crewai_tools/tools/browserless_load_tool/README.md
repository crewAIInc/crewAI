# BrowserlessLoadTool

## Description

The BrowserlessLoadTool enables CrewAI agents to load and scrape webpages using the [Browserless](https://browserless.io) smart scrape API. It automatically handles JavaScript rendering, anti-bot measures, and captchas through a cascading strategy pipeline.

Supported output formats: `markdown`, `html`, `screenshot`, `pdf`, `links`.

## Requirements

- A Browserless API token (get one at [browserless.io](https://browserless.io))
- Set the `BROWSERLESS_API_TOKEN` environment variable

## Installation

The tool uses `requests` which is already included with CrewAI. No additional packages needed.

```bash
pip install crewai-tools
```

## Example

```python
from crewai_tools import BrowserlessLoadTool

# Basic usage - returns markdown by default
tool = BrowserlessLoadTool()

# With HTML format
tool = BrowserlessLoadTool(formats=["html"])

# With multiple formats (returns markdown by priority)
tool = BrowserlessLoadTool(formats=["markdown", "links"])

# With custom timeout (milliseconds)
tool = BrowserlessLoadTool(timeout=120000)

# Self-hosted instance
tool = BrowserlessLoadTool(base_url="http://localhost:3000")
```

## Arguments

| Argument    | Type         | Default                                       | Description                          |
|-------------|--------------|-----------------------------------------------|--------------------------------------|
| `api_token` | `str`        | `BROWSERLESS_API_TOKEN` env var               | Browserless API token                |
| `base_url`  | `str`        | `BROWSERLESS_BASE_URL` env var or cloud URL   | Browserless instance URL             |
| `formats`   | `list[str]`  | `["markdown"]`                                | Output formats to request            |
| `timeout`   | `int`        | `60000`                                       | Request timeout in milliseconds      |

## Environment Variables

| Variable                 | Required | Description                        |
|--------------------------|----------|------------------------------------|
| `BROWSERLESS_API_TOKEN`  | Yes      | API token for Browserless services |
| `BROWSERLESS_BASE_URL`   | No       | Base URL for Browserless instance  |
