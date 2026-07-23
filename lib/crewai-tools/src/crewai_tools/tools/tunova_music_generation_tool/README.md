# TunovaMusicGenerationTool Documentation

## Description
The TunovaMusicGenerationTool generates complete songs from text prompts using the [Tunova](https://tunova.ai) API, a hosted Suno-quality music generation service. The tool submits a generation job, waits for the render to finish, and returns the track title and audio URL. Songs are billed only on successful renders, so failed generations can be retried at no extra cost.

## Features
- Full song generation from a single text prompt
- Optional instrumental-only mode (no vocals)
- Configurable maximum wait time for the render
- Billed only on successful renders — failed renders are automatically refunded
- Free tier available (no card required)

## Installation
```shell
pip install 'crewai[tools]'
```

## Usage
```python
from crewai_tools import TunovaMusicGenerationTool

# Initialize the tool
tool = TunovaMusicGenerationTool()

# Generate a song
result = tool.run(
    prompt="an upbeat synthwave track about summer nights",
    make_instrumental=False,  # Optional: generate without vocals (default: False)
    wait_seconds=360,  # Optional: max seconds to wait for the render (default: 360)
)
```

## Configuration
1. **API Key Setup**:
   - Sign up for an account at [tunova.ai](https://tunova.ai)
   - Obtain your API key
   - Set the environment variable: `TUNOVA_API_KEY`

## Response Format
The tool returns a short string result:
- On success: the track title, duration, and audio URL for each generated clip
- On failure: an explanatory error message (failed renders are automatically refunded, so retrying is free)
- On timeout: the job id and a note that only successful renders are billed
