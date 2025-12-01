```markdown
# DirectoryReadTool

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

# Initialize the tool with the directory you want to explore
tool = DirectoryReadTool(directory='/path/to/your/directory')

# Use the tool to list the contents of the specified directory
directory_contents = tool.run()
print(directory_contents)
```

This example demonstrates the essential steps to utilize the DirectoryReadTool effectively, highlighting its simplicity and user-friendly design.

## Arguments
The DirectoryReadTool requires minimal configuration for use. The essential argument for this tool is as follows:

- `directory`: A mandatory argument that specifies the path to the directory whose contents you wish to list. It accepts both absolute and relative paths, guiding the tool to the desired directory for content listing.

The DirectoryReadTool provides a user-friendly and efficient way to list directory contents, making it an invaluable tool for managing and inspecting directory structures.
```

This revised documentation for the DirectoryReadTool maintains the structure and content requirements as outlined, with adjustments made for clarity, consistency, and adherence to the high-quality standards exemplified in the provided documentation example.
