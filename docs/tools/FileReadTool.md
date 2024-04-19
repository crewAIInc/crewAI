# FileReadTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
The FileReadTool conceptually represents a suite of functionalities within the crewai_tools package aimed at facilitating file reading and content retrieval. This suite includes tools for processing batch text files, reading runtime configuration files, and importing data for analytics. It supports a variety of text-based file formats such as `.txt`, `.csv`, `.json`, and more. Depending on the file type, the suite offers specialized functionality, such as converting JSON content into a Python dictionary for ease of use.

## Installation
To utilize the functionalities previously attributed to the FileReadTool, install the crewai_tools package:

```shell
pip install 'crewai[tools]'
```

## Usage Example
To get started with the FileReadTool:

```python
from crewai_tools import FileReadTool

# Initialize the tool to read any files the agents knows or lean the path for
file_read_tool = FileReadTool()

# OR

# Initialize the tool with a specific file path, so the agent can only read the content of the specified file
file_read_tool = FileReadTool(file_path='path/to/your/file.txt')
```

## Arguments
- `file_path`: The path to the file you want to read. It accepts both absolute and relative paths. Ensure the file exists and you have the necessary permissions to access it.