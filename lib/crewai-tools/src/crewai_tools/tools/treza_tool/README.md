# Treza Tools

## Description

Make AI pictures and video clips (Veo, Kling, Seedance and more) with [Treza](https://www.trezalabs.com). Treza prices every render first; with `max_cost_usd` set, a render that costs more is refused and nothing is spent.

- `TrezaImageTool`: a picture from a description.
- `TrezaVideoTool`: a video clip from a description (one to five minutes).
- `TrezaGetRunTool`: looks up a render that was still going when its tool stopped waiting.

Each returns JSON with the finished file's `url`, `price_usd` and `model`.

## Installation

```shell
pip install 'crewai[tools]'
export TREZA_API_KEY="your-key"  # https://www.trezalabs.com/platform/api-keys
```

## Example

```python
from crewai_tools import TrezaGetRunTool, TrezaImageTool, TrezaVideoTool

tools = [
    TrezaImageTool(max_cost_usd=0.10),
    TrezaVideoTool(max_cost_usd=1.00),
    TrezaGetRunTool(),
]
```

## Arguments

Set when the tool is built (the agent cannot change them):

- `api_key`: the Treza API key. Defaults to `TREZA_API_KEY`.
- `max_cost_usd`: the most one render may cost. `None` means no cap.
- `wait_seconds`: how long to wait for a render (default 300). A render still going comes back with `"status": "running"` and its `run_id` and `pipeline_id`.
