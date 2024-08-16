# NL2SQL Tool

## Description

This tool is used to convert natural language to SQL queries. When passsed to the agent it will generate queries and then use them to interact with the database.

This enables multiple workflows like having an Agent to access the database fetch information based on the goal and then use the information to generate a response, report or any other output. Along with that proivdes the ability for the Agent to update the database based on its goal.

**Attention**: Make sure that the Agent has access to a Read-Replica or that is okay for the Agent to run insert/update queries on the database.

## Requirements

- SqlAlchemy
- Any DB compatible library (e.g. psycopg2, mysql-connector-python)

## Installation
Install the crewai_tools package
```shell
pip install 'crewai[tools]'
```

## Usage

In order to use the NL2SQLTool, you need to pass the database URI to the tool. The URI should be in the format `dialect+driver://username:password@host:port/database`.


```python
from crewai_tools import NL2SQLTool

# psycopg2 was installed to run this example with PostgreSQL
nl2sql = NL2SQLTool(db_uri="postgresql://example@localhost:5432/test_db")

@agent
def researcher(self) -> Agent:
    return Agent(
        config=self.agents_config["researcher"],
        allow_delegation=False,
        tools=[nl2sql]
    )
```

## Example

The primary task goal was:

"Retrieve the average, maximum, and minimum monthly revenue for each city, but only include cities that have more than one user. Also, count the number of user in each city and sort the results by the average monthly revenue in descending order"

So the Agent tried to get information from the DB, the first one is wrong so the Agent tries again and gets the correct information and passes to the next agent.

![alt text](https://github.com/crewAIInc/crewAI-tools/blob/main/crewai_tools/tools/nl2sql/images/image-2.png?raw=true)
![alt text](https://github.com/crewAIInc/crewAI-tools/raw/main/crewai_tools/tools/nl2sql/images/image-3.png)


The second task goal was:

"Review the data and create a detailed report, and then create the table on the database with the fields based on the data provided.
Include information on the average, maximum, and minimum monthly revenue for each city, but only include cities that have more than one user. Also, count the number of users in each city and sort the results by the average monthly revenue in descending order."

Now things start to get interesting, the Agent generates the SQL query to not only create the table but also insert the data into the table. And in the end the Agent still returns the final report which is exactly what was in the database.

![alt text](https://github.com/crewAIInc/crewAI-tools/raw/main/crewai_tools/tools/nl2sql/images/image-4.png)
![alt text](https://github.com/crewAIInc/crewAI-tools/raw/main/crewai_tools/tools/nl2sql/images/image-5.png)

![alt text](https://github.com/crewAIInc/crewAI-tools/raw/main/crewai_tools/tools/nl2sql/images/image-9.png)
![alt text](https://github.com/crewAIInc/crewAI-tools/raw/main/crewai_tools/tools/nl2sql/images/image-7.png)


This is a simple example of how the NL2SQLTool can be used to interact with the database and generate reports based on the data in the database.

The Tool provides endless possibilities on the logic of the Agent and how it can interact with the database.

```
 DB -> Agent -> ... -> Agent -> DB
```
