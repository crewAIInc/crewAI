# WarpGrep Codebase Search Tool

An API-based codebase search tool powered by [Morph's WarpGrep](https://morphllm.com) agent. WarpGrep is an RL-trained search subagent that performs multi-turn, grep-based codebase exploration and returns precise file spans with line numbers.

## Prerequisites

- **MORPH_API_KEY**: Get your API key at [morphllm.com](https://morphllm.com)
- **ripgrep (rg)**: Must be installed on the host machine
  - macOS: `brew install ripgrep`
  - Debian/Ubuntu: `apt install ripgrep`
  - Windows: `choco install ripgrep`

## Usage

```python
from crewai_tools import WarpGrepTool

# Uses MORPH_API_KEY from environment
tool = WarpGrepTool()

# Or with a specific directory
tool = WarpGrepTool(directory="/path/to/your/project")

# Use in a crew
from crewai import Agent

researcher = Agent(
    role="Code Researcher",
    goal="Find relevant code in the repository",
    tools=[tool],
)
```

## Configuration

| Parameter   | Type       | Default                                          | Description                              |
|-------------|------------|--------------------------------------------------|------------------------------------------|
| `api_key`   | `str`      | `$MORPH_API_KEY`                                 | Morph API key                            |
| `directory` | `str`      | Current working directory                        | Root directory to search                 |
| `max_turns` | `int`      | `4`                                              | Maximum search turns (1-8)               |
| `model`     | `str`      | `morph-warp-grep-v2`                             | WarpGrep model to use                    |
| `api_url`   | `str`      | `https://api.morphllm.com/v1/chat/completions`   | API endpoint                             |

## How It Works

1. Generates a file tree of the target directory
2. Sends the tree and your search query to the WarpGrep model
3. The model responds with tool calls (ripgrep, read, list_directory)
4. Tool calls are executed locally on your machine
5. Results are sent back for the next turn
6. When the model calls `finish`, the relevant code spans are read and returned
