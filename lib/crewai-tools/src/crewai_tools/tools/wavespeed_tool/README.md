# WaveSpeed Tools

## Description
`WaveSpeedImageTool` and `WaveSpeedVideoTool` give the Agent the ability to generate images and videos with models hosted on [WaveSpeed AI](https://wavespeed.ai).

- `WaveSpeedImageTool` generates an image from a text prompt. When reference image URLs are given, it edits or combines them according to the prompt.
- `WaveSpeedVideoTool` generates a video from a text prompt. When an image URL is given, it animates that image.

Each call submits one prediction, waits for it to finish, and returns the URLs of the generated files as JSON.

## Installation
Install the crewai_tools package and set your API key:
```shell
pip install 'crewai[tools]'
export WAVESPEED_API_KEY="your-api-key"
```

## Example
```python
from crewai_tools import WaveSpeedImageTool, WaveSpeedVideoTool

Agent(
    ...
    tools=[WaveSpeedImageTool(), WaveSpeedVideoTool()],
)
```

Default models are `bytedance/seedream-v5.5` (with `bytedance/seedream-v5.5/edit` when reference images are given) and `alibaba/wan-3.0/text-to-video` (with `alibaba/wan-3.0/image-to-video` when an image is given). Any other model id can be passed as `model`:

```python
image_tool = WaveSpeedImageTool(aspect_ratio="16:9", resolution="2k")
video_tool = WaveSpeedVideoTool(duration=5, resolution="720p")
other_tool = WaveSpeedImageTool(model="vendor/model-id", extra_params={"seed": 42})
```

Optional settings such as `aspect_ratio`, `resolution` or `duration` are sent only when set; otherwise the model defaults apply. `timeout` (300 s for images, 900 s for videos) bounds how long a call waits for its result.
