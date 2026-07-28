# FileWriterTool Documentation

## Description
The `FileWriterTool` is a component of the crewai_tools package, designed to simplify the process of writing content to files. It is particularly useful in scenarios such as generating reports, saving logs, creating configuration files, and more. This tool supports creating new directories if they don't exist, making it easier to organize your output, and writes UTF-8 by default rather than the platform's locale encoding.

## Installation
Install the crewai_tools package to use the `FileWriterTool` in your projects:

```shell
pip install 'crewai[tools]'
```

## Example
To get started with the `FileWriterTool`:

```python
from crewai_tools import FileWriterTool

# Initialize the tool
file_writer_tool = FileWriterTool()

# Write content to a file in a specified directory
result = file_writer_tool.run(
    filename='example.txt',
    content='This is a test content.',
    directory='test_directory',
)
print(result)
```

## Arguments
The agent supplies these at runtime:

- `filename`: The name of the file to write, relative to `directory`. May include subdirectories, which are created if they don't exist.
- `content`: The text content to write into the file.
- `directory` (optional): The path to the directory where the file will be created. A relative path resolves inside the tool's allowed directory — `base_dir` when set, the current working directory otherwise — and defaults to its root. If the directory does not exist, it will be created.
- `overwrite` (optional): Whether to replace the file when it already exists. Accepts `true`/`false` (also `yes`/`no`, `on`/`off`, `1`/`0`). Defaults to `false`, which reports an error instead of replacing existing content.

You set these when constructing the tool:

- `base_dir` (optional): The directory that writes must stay inside. Defaults to the current working directory.
- `encoding` (optional): Text encoding used to write the file. Defaults to `utf-8`.

## Allowed paths
Because both the directory and the filename are usually chosen by an LLM at runtime, writes are confined to a sandbox: the resolved `directory` must be inside `base_dir` (the current working directory by default), and the resolved file must be inside that `directory`. `..` segments, absolute paths, and symlinks are resolved before both checks, so they cannot be used to escape.

To let an agent write outside the working directory, point `base_dir` at the target tree:

```python
# The agent may write anywhere under /var/output, and nowhere outside it
file_writer_tool = FileWriterTool(base_dir='/var/output')
```

Setting `CREWAI_TOOLS_ALLOW_UNSAFE_PATHS=true` disables path validation, but it applies process-wide to every crewai-tools tool, including the SSRF protections on URL-fetching tools, so prefer `base_dir`.

## Conclusion
By integrating the `FileWriterTool` into your crews, the agents can execute the process of writing content to files and creating directories. This tool is essential for tasks that require saving output data, creating structured file systems, and more. By adhering to the setup and usage guidelines provided, incorporating this tool into projects is straightforward and efficient.
