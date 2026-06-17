# MuAPI Tools

`MuApiImageTool` and `MuApiVideoTool` — generate images and videos using [muapi.ai](https://muapi.ai), a unified API with 400+ generative media models.

## Setup

```bash
export MUAPI_API_KEY=your_key_here
```

Get an API key at [muapi.ai/dashboard/api-keys](https://muapi.ai/dashboard/api-keys).

## Usage

```python
from crewai import Agent, Task, Crew
from crewai_tools import MuApiImageTool, MuApiVideoTool

image_tool = MuApiImageTool()
video_tool = MuApiVideoTool()

designer = Agent(
    role="Creative Director",
    goal="Create visual assets for marketing campaigns",
    tools=[image_tool, video_tool],
    backstory="You are a creative director who generates images and videos.",
)

task = Task(
    description="Generate a photorealistic product photo of a coffee mug on a marble table.",
    expected_output="URL of the generated image",
    agent=designer,
)

crew = Crew(agents=[designer], tasks=[task])
result = crew.kickoff()
```

## Supported Image Models

`flux-schnell`, `flux-dev`, `flux-kontext-dev/pro/max`, `hidream-fast/dev/full`,
`midjourney`, `gpt4o`, `gpt-image-2`, `imagen4`, `imagen4-fast`, `seedream`,
`reve`, `ideogram`, `hunyuan`, `wan2.1`, `qwen`

## Supported Video Models

`veo3`, `veo3-fast`, `kling-master`, `wan2.1`, `wan2.2`, `seedance-pro`,
`seedance-pro-fast`, `runway`, `pixverse`, `sora`, `minimax-hailuo-02-pro`
