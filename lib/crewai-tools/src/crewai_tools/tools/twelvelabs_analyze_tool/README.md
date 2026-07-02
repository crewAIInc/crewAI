# TwelveLabsAnalyzeTool

## Description
A tool that uses [TwelveLabs'](https://twelvelabs.io) Pegasus video-understanding
model to analyze video content and answer natural-language prompts about it.
Pegasus reasons over a video's visuals, speech, and on-screen text, making the
tool useful for summarization, question answering over video, content
moderation, and metadata extraction.

You can grab a free API key at https://twelvelabs.io — there's a generous free tier.

## Installation
Install the required packages:
```shell
pip install 'crewai[tools]' twelvelabs
```

Set your API key:
```shell
export TWELVELABS_API_KEY="your_api_key"
```

## Example Usage

### Analyze a public video URL
```python
from crewai_tools import TwelveLabsAnalyzeTool

tool = TwelveLabsAnalyzeTool()
result = tool.run(
    prompt="Summarize this video in three bullet points.",
    video_url="https://example.com/my-video.mp4",
)
print(result)
```

### Analyze a video already indexed in TwelveLabs
```python
tool = TwelveLabsAnalyzeTool()
result = tool.run(
    prompt="List every product shown in this video.",
    video_id="your_indexed_video_id",
)
```

### Customize the model and generation
```python
tool = TwelveLabsAnalyzeTool(
    api_key="your_api_key",   # or rely on TWELVELABS_API_KEY
    model_name="pegasus1.5",
    max_tokens=4096,
)
```

## Arguments
- `prompt` (str, required): What to extract from the video.
- `video_url` (str): Public URL of the video to analyze. Provide this or `video_id`.
- `video_id` (str): ID of a video already indexed in TwelveLabs. Provide this or `video_url`.

## Constructor Options
- `api_key` (str): TwelveLabs API key. Falls back to `TWELVELABS_API_KEY`.
- `model_name` (str): Pegasus model to use. Defaults to `pegasus1.5`.
- `max_tokens` (int): Maximum tokens in the answer (must be at least 512). Defaults to `2048`.
- `temperature` (float): Optional sampling temperature.
