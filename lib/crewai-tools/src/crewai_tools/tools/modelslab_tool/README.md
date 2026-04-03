# ModelsLab Image Generation Tool

Generates images from text prompts using [ModelsLab's](https://modelslab.com) text-to-image API.

ModelsLab provides access to 200+ AI models including Flux, SDXL, Stable Diffusion, and thousands of community fine-tunes via a single unified API.

## Installation

```bash
pip install 'crewai-tools'
pip install requests
```

## Prerequisites

Set your ModelsLab API key as an environment variable:

```bash
export MODELSLAB_API_KEY="your-api-key"
```

Get your API key at https://modelslab.com/dashboard/api-keys

## Usage

```python
from crewai import Agent, Task, Crew
from crewai_tools import ModelsLabImageGenerationTool

# Basic usage with default Flux model
image_tool = ModelsLabImageGenerationTool()

# Custom model and dimensions
image_tool = ModelsLabImageGenerationTool(
    model="flux",          # or "sdxl", "realistic-vision-v51", etc.
    width=1024,
    height=1024,
    negative_prompt="blurry, low quality",
)

designer = Agent(
    role="Visual Designer",
    goal="Create stunning images for the project",
    backstory="An experienced visual designer with a great eye for aesthetics.",
    tools=[image_tool],
    verbose=True,
)

task = Task(
    description="Create an image of a futuristic city at night with neon lights.",
    agent=designer,
    expected_output="A URL to the generated image.",
)

crew = Crew(agents=[designer], tasks=[task])
result = crew.kickoff()
print(result)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"flux"` | Model ID (e.g., `"flux"`, `"sdxl"`, `"realistic-vision-v51"`) |
| `width` | int | `512` | Image width in pixels (divisible by 8) |
| `height` | int | `512` | Image height in pixels (divisible by 8) |
| `samples` | int | `1` | Number of images to generate (1–4) |
| `num_inference_steps` | int | `30` | Denoising steps (1–50) |
| `guidance_scale` | float | `7.5` | Prompt adherence (1–20) |
| `negative_prompt` | str | `None` | Content to avoid in the image |
| `seed` | int | `None` | Random seed for reproducibility |

## API Reference

- Docs: https://docs.modelslab.com/image-generation/community-models/text2img
- Available models: https://modelslab.com/models
