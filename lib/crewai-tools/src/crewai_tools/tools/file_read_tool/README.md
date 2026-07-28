# FileReadTool

## Description

The FileReadTool is a versatile component of the crewai_tools package, designed to streamline the process of reading and retrieving content from files. It is particularly useful in scenarios such as batch text file processing, runtime configuration file reading, and data importation for analytics. This tool supports any text-based file format, including `.txt`, `.csv`, `.json`, and `.md`. Content is always returned as plain text — parsing it (for example, `json.loads` on a `.json` file) is up to the agent or your own code.

The tool also supports reading specific chunks of a file by specifying a starting line and the number of lines to read, which is helpful when working with large files that don't need to be loaded entirely into memory. Reading a window stops as soon as the requested lines have been collected, so it does not scan the rest of the file.

## Installation

Install the crewai_tools package to use the FileReadTool in your projects:

```shell
pip install 'crewai[tools]'
```

## Example

To get started with the FileReadTool:

```python
from crewai_tools import FileReadTool

# Initialize the tool to read any file the agent knows or learns the path for
file_read_tool = FileReadTool()

# OR

# Initialize the tool with a specific file path, so the agent reads that file by default
file_read_tool = FileReadTool(file_path='path/to/your/file.txt')

# Read a specific chunk of the file (lines 100-149)
partial_content = file_read_tool.run(file_path='path/to/your/file.txt', start_line=100, line_count=50)
```

## Arguments

The agent supplies these at runtime:

- `file_path`: (Optional) The path to the file you want to read. It accepts both absolute and relative paths. Ensure the file exists and you have the necessary permissions to access it. Omit it to read the default file configured at construction; if there is no default, the tool reports that no path was provided.
- `start_line`: (Optional) The line number to start reading from (1-indexed). Defaults to 1 (the first line).
- `line_count`: (Optional) The number of lines to read. If not provided, reads from the start_line to the end of the file.

You set these when constructing the tool:

- `file_path`: (Optional) A default file to read when the agent calls the tool with no arguments.
- `base_dir`: (Optional) The directory that runtime paths must stay inside. Defaults to the current working directory.
- `encoding`: (Optional) Text encoding used to decode the file. Defaults to `utf-8`.

## Allowed paths

Because the file path is usually chosen by an LLM at runtime, reads are confined to a sandbox:

- Paths supplied at runtime must resolve inside `base_dir` (the current working directory by default). `..` segments and symlinks are resolved before the check, so they cannot be used to escape.
- A `file_path` passed to the constructor is developer-declared intent, so it is always allowed past the containment check — even outside `base_dir`. The read itself can still fail if the file is missing, is a directory, or is not permitted. It is pinned when the tool is built, so a later change of working directory cannot repoint it, and the agent can address it either by omitting `file_path` or by using the name shown in the tool's description. Declaring one file does not expose its siblings.

To let an agent read a directory tree outside the working directory, point `base_dir` at it:

```python
# The agent may read anything under /data, and nothing outside it
file_read_tool = FileReadTool(base_dir='/data')
```

Setting `CREWAI_TOOLS_ALLOW_UNSAFE_PATHS=true` disables path validation, but it applies process-wide to every crewai-tools tool, including the SSRF protections on URL-fetching tools, so prefer `base_dir`.
