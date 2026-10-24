# AntibrowLoadTool

Load a page in a persistent [AntiBrow](https://antibrow.com) profile and return its
visible text.

Other scraping tools hand the agent a fresh browser per call. Here the unit is a
profile: cookies, storage, an engine-level fingerprint and the profile's own proxy
persist between runs, so a page behind a login opens already signed in.

## Installation

```shell
uv add antibrow
```

Set `ANTIBROW_API_KEY`, or pass `api_key` to the constructor. The browser engine
downloads on the first launch and is cached.

## Usage

```python
from crewai_tools import AntibrowLoadTool

tool = AntibrowLoadTool(profile="research-01")
try:
    text = tool.run(url="https://example.com", selector="main")
finally:
    tool.close()
```

`keep_open` (default `True`) reuses one browser across calls, so a login survives
between steps of a crew. Call `close()` when the crew is done - an abandoned browser
is a whole browser still running.

Two agents that must not share an identity need two profile names, not two tabs.

A persistent identity removes the tells that come from starting over every run - a
fresh profile, a stock automation fingerprint, your own IP. It does not promise that
a given site will accept an automated session.
