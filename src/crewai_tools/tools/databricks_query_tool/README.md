# Databricks Query Tool

## Description

This tool allows AI agents to execute SQL queries against Databricks workspace tables and retrieve the results. It provides a simple interface for querying data from Databricks tables using SQL, making it easy for agents to access and analyze data stored in Databricks.

## Installation

Install the crewai_tools package with the databricks extra:

```shell
pip install 'crewai[tools]' 'databricks-sdk'
```

## Authentication

The tool requires Databricks authentication credentials. You can provide these in two ways:

1. **Using Databricks CLI profile**:
   - Set the `DATABRICKS_CONFIG_PROFILE` environment variable to your profile name.

2. **Using direct credentials**:
   - Set both `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables.

Example:
```shell
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi1234567890abcdef"
```

## Usage

```python
from crewai_tools import DatabricksQueryTool

# Basic usage
databricks_tool = DatabricksQueryTool()

# With default parameters for catalog, schema, and warehouse
databricks_tool = DatabricksQueryTool(
    default_catalog="my_catalog",
    default_schema="my_schema",
    default_warehouse_id="warehouse_id"
)

# Example in a CrewAI agent
@agent
def data_analyst(self) -> Agent:
    return Agent(
        config=self.agents_config["data_analyst"],
        allow_delegation=False,
        tools=[databricks_tool]
    )
```

## Parameters

When executing queries, you can provide the following parameters:

- `query` (required): SQL query to execute against the Databricks workspace
- `catalog` (optional): Databricks catalog name
- `schema` (optional): Databricks schema name
- `warehouse_id` (optional): Databricks SQL warehouse ID
- `row_limit` (optional): Maximum number of rows to return (default: 1000)

If not provided, the tool will use the default values set during initialization.