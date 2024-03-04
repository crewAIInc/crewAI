# PGSearchTool

!!! note "Depend on OpenAI"
    All RAG tools at the moment can only use openAI to generate embeddings, we are working on adding support for other providers.

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
This tool is designed to facilitate semantic searches within PostgreSQL database tables. Leveraging the RAG (Retrieve and Generate) technology, the PGSearchTool provides users with an efficient means of querying database table content, specifically tailored for PostgreSQL databases. It simplifies the process of finding relevant data through semantic search queries, making it an invaluable resource for users needing to perform advanced queries on extensive datasets within a PostgreSQL database.

## Installation
To install the `crewai_tools` package and utilize the PGSearchTool, execute the following command in your terminal:

```shell
pip install 'crewai[tools]'
```

## Example
Below is an example showcasing how to use the PGSearchTool to conduct a semantic search on a table within a PostgreSQL database:

```python
from crewai_tools import PGSearchTool

# Initialize the tool with the database URI and the target table name
tool = PGSearchTool(db_uri='postgresql://user:password@localhost:5432/mydatabase', table_name='employees')

```

## Arguments
The PGSearchTool requires the following arguments for its operation:

- `db_uri`: A string representing the URI of the PostgreSQL database to be queried. This argument is mandatory and must include the necessary authentication details and the location of the database.
- `table_name`: A string specifying the name of the table within the database on which the semantic search will be performed. This argument is mandatory.