# OCR Tool

## Description

This tool performs Optical Character Recognition (OCR) on images using supported LLMs. It can extract text from both local image files and images available via URLs. The tool leverages the LLM's vision capabilities to provide accurate text extraction from images.

## Installation
Install the crewai_tools package
```shell
pip install 'crewai[tools]'
```

## Supported LLMs

Any LLM that supports the `vision` feature should work. It must accept image_url as a user message.
The tool has been tested with:
- OpenAI's `gpt-4o`
- Gemini's `gemini/gemini-1.5-pro`

## Usage

In order to use the OCRTool, make sure your LLM supports the `vision` feature and the appropriate API key is set in the environment (e.g., `OPENAI_API_KEY` for OpenAI).

```python
from crewai_tools import OCRTool

selected_llm = LLM(model="gpt-4o") # select your LLM, the tool has been tested with gpt-4o and gemini/gemini-1.5-pro

ocr_tool = OCRTool(llm=selected_llm)

@agent
def researcher(self) -> Agent:
    return Agent(
        config=self.agents_config["researcher"],
        allow_delegation=False,
        tools=[ocr_tool]
    )
```

The tool accepts either a local file path or a URL to the image:
- For local files, provide the absolute or relative path
- For remote images, provide the complete URL starting with 'http' or 'https'
