# DirectoryReadTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
The DirectoryReadTool is a highly efficient utility designed for the comprehensive listing of directory contents. It recursively navigates through the specified directory, providing users with a detailed enumeration of all files, including those nested within subdirectories. This tool is indispensable for tasks requiring a thorough inventory of directory structures or for validating the organization of files within directories.

## Installation
Install the `crewai_tools` package to use the DirectoryReadTool in your project. If you haven't added this package to your environment, you can easily install it with pip using the following command:

```shell
pip install 'crewai[tools]'
```

This installs the latest version of the `crewai_tools` package, allowing access to the DirectoryReadTool and other utilities.

## Example
The DirectoryReadTool is simple to use. The code snippet below shows how to set up and use the tool to list the contents of a specified directory:

```python
from crewai_tools import DirectoryReadTool

# Initialize the tool so the agent can read any directory's content it learns about during execution
tool = DirectoryReadTool()

# OR

# Initialize the tool with a specific directory, so the agent can only read the content of the specified directory
tool = DirectoryReadTool(directory='/path/to/your/directory')
```

## Arguments
The DirectoryReadTool requires minimal configuration for use. The essential argument for this tool is as follows:

- `directory`: **Optional** A argument that specifies the path to the directory whose contents you wish to list. It accepts both absolute and relative paths, guiding the tool to the desired directory for content listing.
