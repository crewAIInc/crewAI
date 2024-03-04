# MDXSearchTool

## Description
The MDX Search Tool, a key component of the `crewai_tools` package, is designed for advanced market data extraction, offering invaluable support to researchers and analysts requiring immediate market insights in the AI sector. With its ability to interface with various data sources and tools, it streamlines the process of acquiring, reading, and organizing market data efficiently.

## Installation
To utilize the MDX Search Tool, ensure the `crewai_tools` package is installed. If not already present, install it using the following command:

```shell
pip install 'crewai[tools]'
```

## Example
Configuring and using the MDX Search Tool involves setting up environment variables and utilizing the tool within a crewAI project for market research. Here's a simple example:

```python
from crewai_tools import MDXSearchTool

# Initialize the tool so the agent can search any MDX content if it learns about during its execution
tool = MDXSearchTool()

# OR

# Initialize the tool with a specific MDX file path for exclusive search within that document
tool = MDXSearchTool(mdx='path/to/your/document.mdx')
```

## Arguments
- mdx: **Optional** The MDX path for the search. Can be provided at initialization