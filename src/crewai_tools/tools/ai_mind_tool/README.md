# AIMind Tool

## Description

[Minds](https://mindsdb.com/minds) are AI systems provided by [MindsDB](https://mindsdb.com/) that work similarly to large language models (LLMs) but go beyond by answering any question from any data.

This is accomplished by selecting the most relevant data for an answer using parametric search, understanding the meaning and providing responses within the correct context through semantic search, and finally, delivering precise answers by analyzing data and using machine learning (ML) models.

The `AIMindTool` can be used to query data sources in natural language by simply configuring their connection parameters.

## Installation

1. Install the `crewai[tools]` package:

```shell
pip install 'crewai[tools]'
```

2. Install the Minds SDK:

```shell
pip install minds-sdk
```

3. Sign for a Minds account [here](https://mdb.ai/register), and obtain an API key.

4. Set the Minds API key in an environment variable named `MINDS_API_KEY`.

## Usage

```python
from crewai_tools import AIMindTool


# Initialize the AIMindTool.
aimind_tool = AIMindTool(
    datasources=[
        {
            "description": "house sales data",
            "engine": "postgres",
            "connection_data": {
                "user": "demo_user",
                "password": "demo_password",
                "host": "samples.mindsdb.com",
                "port": 5432,
                "database": "demo",
                "schema": "demo_data"
            },
            "tables": ["house_sales"]
        }
    ]
)

aimind_tool.run("How many 3 bedroom houses were sold in 2008?")
```

The `datasources` parameter is a list of dictionaries, each containing the following keys:

- `description`: A description of the data contained in the datasource.
- `engine`: The engine (or type) of the datasource. Find a list of supported engines in the link below.
- `connection_data`: A dictionary containing the connection parameters for the datasource. Find a list of connection parameters for each engine in the link below.
- `tables`: A list of tables that the data source will use. This is optional and can be omitted if all tables in the data source are to be used.

A list of supported data sources and their connection parameters can be found [here](https://docs.mdb.ai/docs/data_sources).

```python
from crewai import Agent
from crewai.project import agent


# Define an agent with the AIMindTool.
@agent
def researcher(self) -> Agent:
    return Agent(
        config=self.agents_config["researcher"],
        allow_delegation=False,
        tools=[aimind_tool]
    )
```
