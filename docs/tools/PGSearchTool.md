# PGSearchTool

!!! note "Under Development"
    The PGSearchTool is currently under development. This document outlines the intended functionality and interface. As development progresses, please be aware that some features may not be available or could change.

## Description
The PGSearchTool is envisioned as a powerful tool for facilitating semantic searches within PostgreSQL database tables. By leveraging advanced Retrieve and Generate (RAG) technology, it aims to provide an efficient means for querying database table content, specifically tailored for PostgreSQL databases. The tool's goal is to simplify the process of finding relevant data through semantic search queries, offering a valuable resource for users needing to conduct advanced queries on extensive datasets within a PostgreSQL environment.

## Installation
The `crewai_tools` package, which will include the PGSearchTool upon its release, can be installed using the following command:

```shell
pip install 'crewai[tools]'
```

(Note: The PGSearchTool is not yet available in the current version of the `crewai_tools` package. This installation command will be updated once the tool is released.)

## Example Usage
Below is a proposed example showcasing how to use the PGSearchTool for conducting a semantic search on a table within a PostgreSQL database:

```python
rom crewai_tools import PGSearchTool

# Initialize the tool with the database URI and the target table name
tool = PGSearchTool(db_uri='postgresql://user:password@localhost:5432/mydatabase', table_name='employees')
```

## Arguments
The PGSearchTool is designed to require the following arguments for its operation:

- `db_uri`: A string representing the URI of the PostgreSQL database to be queried. This argument will be mandatory and must include the necessary authentication details and the location of the database.
- `table_name`: A string specifying the name of the table within the database on which the semantic search will be performed. This argument will also be mandatory.

## Custom Model and Embeddings

The tool intends to use OpenAI for both embeddings and summarization by default. Users will have the option to customize the model using a config dictionary as follows:

```python
tool = PGSearchTool(
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