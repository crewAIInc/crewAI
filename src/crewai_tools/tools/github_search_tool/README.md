# GithubSearchTool

## Description
The GithubSearchTool is a Read, Append, and Generate (RAG) tool specifically designed for conducting semantic searches within GitHub repositories. Utilizing advanced semantic search capabilities, it sifts through code, pull requests, issues, and repositories, making it an essential tool for developers, researchers, or anyone in need of precise information from GitHub.

## Installation
To use the GithubSearchTool, first ensure the crewai_tools package is installed in your Python environment:

```shell
pip install 'crewai[tools]'
```

This command installs the necessary package to run the GithubSearchTool along with any other tools included in the crewai_tools package.

## Example
Here’s how you can use the GithubSearchTool to perform semantic searches within a GitHub repository:
```python
from crewai_tools import GithubSearchTool

# Initialize the tool for semantic searches within a specific GitHub repository
tool = GithubSearchTool(
    gh_token='...',
	github_repo='https://github.com/example/repo',
	content_types=['code', 'issue'] # Options: code, repo, pr, issue
)

# OR

# Initialize the tool for semantic searches within a specific GitHub repository, so the agent can search any repository if it learns about during its execution
tool = GithubSearchTool(
    gh_token='...',
	content_types=['code', 'issue'] # Options: code, repo, pr, issue
)
```

## Arguments
- `gh_token` : The GitHub token used to authenticate the search. This is a mandatory field and allows the tool to access the GitHub API for conducting searches.
- `github_repo` : The URL of the GitHub repository where the search will be conducted. This is a mandatory field and specifies the target repository for your search.
- `content_types` : Specifies the types of content to include in your search. You must provide a list of content types from the following options: `code` for searching within the code, `repo` for searching within the repository's general information, `pr` for searching within pull requests, and `issue` for searching within issues. This field is mandatory and allows tailoring the search to specific content types within the GitHub repository.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = GithubSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```
