# GitHubSearchTool

## Description
The GitHubSearchTool is a Read, Append, and Generate (RAG) tool specifically designed for conducting semantic searches within GitHub repositories. Utilizing advanced semantic search capabilities, it sifts through code, pull requests, issues, and repositories, making it an essential tool for developers, researchers, or anyone in need of precise information from GitHub.

## Installation
To use the GitHubSearchTool, first ensure the crewai_tools package is installed in your Python environment:

```shell
pip install 'crewai[tools]'
```

This command installs the necessary package to run the GitHubSearchTool along with any other tools included in the crewai_tools package.

## Example
Here’s how you can use the GitHubSearchTool to perform semantic searches within a GitHub repository:
```python
from crewai_tools import GitHubSearchTool

# Initialize the tool for semantic searches within a specific GitHub repository
tool = GitHubSearchTool(
	github_repo='https://github.com/example/repo',
	content_types=['code', 'issue'] # Options: code, repo, pr, issue
)

# OR

# Initialize the tool for semantic searches within a specific GitHub repository, so the agent can search any repository if it learns about during its execution
tool = GitHubSearchTool(
	content_types=['code', 'issue'] # Options: code, repo, pr, issue
)
```

## Arguments
- `github_repo` : The URL of the GitHub repository where the search will be conducted. This is a mandatory field and specifies the target repository for your search.
- `content_types` : Specifies the types of content to include in your search. You must provide a list of content types from the following options: `code` for searching within the code, `repo` for searching within the repository's general information, `pr` for searching within pull requests, and `issue` for searching within issues. This field is mandatory and allows tailoring the search to specific content types within the GitHub repository.
