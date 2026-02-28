# TzafonLoadTool

## Description

[Tzafon](https://tzafon.ai) is a platform for programmatic control of browsers and desktops in seconds.

Key Features:

- Lightweight performance - 75x less CPU and 74% less memory than standard headless runtimes
- Resilient under load - 99.99% reliability and 4.8x faster cold starts, built for agents that never crash
- Isolated by design - Each environment runs in a Firecracker micro-VM for consistent performance
- Unified API surface - Launch and scale Browser, Sandbox, and Linux from a single, unified API

For more information about Tzafon, please visit the [Tzafon website](https://tzafon.ai) or if you want to check out the docs, you can visit the [Tzafon docs](https://docs.tzafon.ai).

## Installation

- Head to [Tzafon](https://tzafon.ai/dashboard) to sign up and generate an API key. Once you've done this set the `TZAFON_API_KEY` environment variable or you can pass it to the `TzafonLoadTool` constructor.
- Install the [Tzafon SDK](https://docs.tzafon.ai/quickstart) along with `crewai[tools]` package:

```
pip install tzafon playwright 'crewai[tools]'
```

## Example

Utilize the TzafonLoadTool as follows to allow your agent to load websites:

```python
from crewai_tools import TzafonLoadTool

tool = TzafonLoadTool()
```

## Arguments

`__init__` arguments:

- `api_key`: Optional. Specifies Tzafon API key. Defaults to the `TZAFON_API_KEY` environment variable.
- `text_content` Retrieve only text content. Default is `True`.

`run` arguments:

- `url`: The website url to load.
