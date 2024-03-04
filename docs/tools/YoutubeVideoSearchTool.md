# YoutubeVideoSearchTool

!!! note "Depend on OpenAI"
    All RAG tools at the moment can only use openAI to generate embeddings, we are working on adding support for other providers.

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description

This tool is part of the `crewai_tools` package and is designed to perform semantic searches within Youtube video content, utilizing Retrieval-Augmented Generation (RAG) techniques. It is one of several "Search" tools in the package that leverage RAG for different sources. The YoutubeVideoSearchTool allows for flexibility in searches; users can search across any Youtube video content without specifying a video URL, or they can target their search to a specific Youtube video by providing its URL.

## Installation

To utilize the YoutubeVideoSearchTool, you must first install the `crewai_tools` package. This package contains the YoutubeVideoSearchTool among other utilities designed to enhance your data analysis and processing tasks. Install the package by executing the following command in your terminal:

```
pip install 'crewai[tools]'
```

## Example

To integrate the YoutubeVideoSearchTool into your Python projects, follow the example below. This demonstrates how to use the tool both for general Youtube content searches and for targeted searches within a specific video's content.

```python
from crewai_tools import YoutubeVideoSearchTool

# General search across Youtube content without specifying a video URL, so the agent can search within any Youtube video content it learns about irs url during its operation
tool = YoutubeVideoSearchTool()

# Targeted search within a specific Youtube video's content
tool = YoutubeVideoSearchTool(youtube_video_url='https://youtube.com/watch?v=example')
```
## Arguments

The YoutubeVideoSearchTool accepts the following initialization arguments:

- `youtube_video_url`: An optional argument at initialization but required if targeting a specific Youtube video. It specifies the Youtube video URL path you want to search within.