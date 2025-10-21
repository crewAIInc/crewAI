# Vision Tool

## Description

This tool is used to extract text from images. When passed to the agent it will extract the text from the image and then use it to generate a response, report or any other output. The URL or the PATH of the image should be passed to the Agent.


## Installation
Install the crewai_tools package
```shell
pip install 'crewai[tools]'
```

## Usage

In order to use the VisionTool, the OpenAI API key should be set in the environment variable `OPENAI_API_KEY`.

```python
from crewai_tools import VisionTool

vision_tool = VisionTool()

@agent
def researcher(self) -> Agent:
    return Agent(
        config=self.agents_config["researcher"],
        allow_delegation=False,
        tools=[vision_tool]
    )
```
