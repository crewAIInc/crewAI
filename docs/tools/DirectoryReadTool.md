```markdown
# DirectoryReadTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
The DirectoryReadTool is a powerful utility designed to provide a comprehensive listing of directory contents. It can recursively navigate through the specified directory, offering users a detailed enumeration of all files, including those within subdirectories. This tool is crucial for tasks that require a thorough inventory of directory structures or for validating the organization of files within directories.

## Installation
To utilize the DirectoryReadTool in your project, install the `crewai_tools` package. If this package is not yet part of your environment, you can install it using pip with the command below:

```shell
pip install 'crewai[tools]'
```

This command installs the latest version of the `crewai_tools` package, granting access to the DirectoryReadTool among other utilities.

## Example
Employing the DirectoryReadTool is straightforward. The following code snippet demonstrates how to set it up and use the tool to list the contents of a specified directory:

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

- `directory`: **Optional**. An argument that specifies the path to the directory whose contents you wish to list. It accepts both absolute and relative paths, guiding the tool to the desired directory for content listing.