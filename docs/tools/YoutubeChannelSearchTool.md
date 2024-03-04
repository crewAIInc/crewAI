# YoutubeChannelSearchTool

!!! note "Depend on OpenAI"
    All RAG tools at the moment can only use openAI to generate embeddings, we are working on adding support for other providers.

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
This tool is designed to perform semantic searches within a specific Youtube channel's content. Leveraging the RAG (Retrieval-Augmented Generation) methodology, it provides relevant search results, making it invaluable for extracting information or finding specific content without the need to manually sift through videos. It streamlines the search process within Youtube channels, catering to researchers, content creators, and viewers seeking specific information or topics.

## Installation
To utilize the YoutubeChannelSearchTool, the `crewai_tools` package must be installed. Execute the following command in your shell to install:

```shell
pip install 'crewai[tools]'
```

## Example
To begin using the YoutubeChannelSearchTool, follow the example below. This demonstrates initializing the tool with a specific Youtube channel handle and conducting a search within that channel's content.

```python
from crewai_tools import YoutubeChannelSearchTool

# Initialize the tool to search within any Youtube channel's content the agent learns about during its execution
tool = YoutubeChannelSearchTool()

# OR

# Initialize the tool with a specific Youtube channel handle to target your search
tool = YoutubeChannelSearchTool(youtube_channel_handle='@exampleChannel')
```

## Arguments
- `youtube_channel_handle` : A mandatory string representing the Youtube channel handle. This parameter is crucial for initializing the tool to specify the channel you want to search within. The tool is designed to only search within the content of the provided channel handle.
