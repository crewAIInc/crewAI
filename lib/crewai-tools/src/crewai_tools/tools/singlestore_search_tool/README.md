# SingleStoreSearchTool

## Description
The SingleStoreSearchTool is designed to facilitate semantic searches and SQL queries within SingleStore database tables. This tool provides a secure interface for executing SELECT and SHOW queries against SingleStore databases, with built-in connection pooling for optimal performance. It supports various connection methods and allows you to work with specific table subsets within your database.

## Installation
To install the `crewai_tools` package with SingleStore support, execute the following command:

```shell
pip install 'crewai[tools]'
```

Or install with the SingleStore extra for the latest dependencies:

```shell
uv sync --extra singlestore
```

Or install the required dependencies manually:

```shell
pip install singlestoredb>=1.12.4 SQLAlchemy>=2.0.40
```

## Features

- 🔒 **Secure Query Execution**: Only SELECT and SHOW queries are allowed for security
- 🚀 **Connection Pooling**: Built-in connection pooling for optimal performance
- 📊 **Table Subset Support**: Work with specific tables or all tables in the database
- 🔧 **Flexible Configuration**: Multiple connection methods supported
- 🛡️ **SSL/TLS Support**: Comprehensive SSL configuration options
- ⚡ **Efficient Resource Management**: Automatic connection lifecycle management

## Basic Usage

### Simple Connection

```python
from crewai_tools import SingleStoreSearchTool

# Basic connection using host/user/password
tool = SingleStoreSearchTool(
    host='localhost',
    user='your_username',
    password='your_password',
    database='your_database',
    port=3306
)

# Execute a search query
result = tool._run("SELECT * FROM employees WHERE department = 'Engineering' LIMIT 10")
print(result)
```

### Working with Specific Tables

```python
# Initialize tool for specific tables only
tool = SingleStoreSearchTool(
    tables=['employees', 'departments'],  # Only work with these tables
    host='your_host',
    user='your_username',
    password='your_password',
    database='your_database'
)
```

## Complete CrewAI Integration Example

Here's a complete example showing how to use the SingleStoreSearchTool with CrewAI agents and tasks:

```python
from crewai import Agent, Task, Crew
from crewai_tools import SingleStoreSearchTool

# Initialize the SingleStore search tool
singlestore_tool = SingleStoreSearchTool(
    tables=["products", "sales", "customers"],  # Specify the tables you want to search
    host="localhost",
    port=3306,
    user="root",
    password="pass",
    database="crewai",
)

# Create an agent that uses this tool
data_analyst = Agent(
    role="Business Analyst",
    goal="Analyze and answer business questions using SQL data",
    backstory="Expert in interpreting business needs and transforming them into data queries.",
    tools=[singlestore_tool],
    verbose=True,
    embedder={
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
        },
    },
)

# Define a task
task = Task(
    description="List the top 2 customers by total sales amount.",
    agent=data_analyst,
    expected_output="A ranked list of top 2 customers that have the highest total sales amount, including their names and total sales figures.",
)

# Run the crew
crew = Crew(tasks=[task], verbose=True)
result = crew.kickoff()
```

### Advanced CrewAI Example with Multiple Agents

```python
from crewai import Agent, Task, Crew
from crewai_tools import SingleStoreSearchTool

# Initialize the tool with connection URL
singlestore_tool = SingleStoreSearchTool(
    host="user:password@localhost:3306/ecommerce_db",
    tables=["orders", "products", "customers", "order_items"]
)

# Data Analyst Agent
data_analyst = Agent(
    role="Senior Data Analyst",
    goal="Extract insights from database queries and provide data-driven recommendations",
    backstory="You are an experienced data analyst with expertise in SQL and business intelligence.",
    tools=[singlestore_tool],
    verbose=True
)

# Business Intelligence Agent
bi_specialist = Agent(
    role="Business Intelligence Specialist",
    goal="Transform data insights into actionable business recommendations",
    backstory="You specialize in translating complex data analysis into clear business strategies.",
    verbose=True
)

# Define multiple tasks
data_extraction_task = Task(
    description="""
    Analyze the sales data to find:
    1. Top 5 best-selling products by quantity
    2. Monthly sales trends for the last 6 months
    3. Customer segments by purchase frequency
    """,
    agent=data_analyst,
    expected_output="Detailed SQL query results with sales analysis including product rankings, trends, and customer segments."
)

insights_task = Task(
    description="""
    Based on the sales data analysis, provide business recommendations for:
    1. Inventory management for top products
    2. Marketing strategies for different customer segments
    3. Sales forecasting insights
    """,
    agent=bi_specialist,
    expected_output="Strategic business recommendations with actionable insights based on the data analysis.",
    context=[data_extraction_task]
)

# Create and run the crew
analytics_crew = Crew(
    agents=[data_analyst, bi_specialist],
    tasks=[data_extraction_task, insights_task],
    verbose=True
)

result = analytics_crew.kickoff()
```

## Connection Methods

SingleStore supports multiple connection methods. Choose the one that best fits your environment:

### 1. Standard Connection

```python
tool = SingleStoreSearchTool(
    host='your_host',
    user='your_username',
    password='your_password',
    database='your_database',
    port=3306
)
```

### 2. Connection URL (Recommended)

You can use a complete connection URL in the `host` parameter for simplified configuration:

```python
# Using connection URL in host parameter
tool = SingleStoreSearchTool(
    host='user:password@localhost:3306/database_name'
)

# Or for SingleStore Cloud
tool = SingleStoreSearchTool(
    host='user:password@your_cloud_host:3333/database_name?ssl_disabled=false'
)
```

### 3. Environment Variable Configuration

Set the `SINGLESTOREDB_URL` environment variable and initialize the tool without any connection arguments:

```bash
# Set the environment variable
export SINGLESTOREDB_URL="singlestoredb://user:password@localhost:3306/database_name"

# Or for cloud connections
export SINGLESTOREDB_URL="singlestoredb://user:password@your_cloud_host:3333/database_name?ssl_disabled=false"
```

```python
# No connection arguments needed when using environment variable
tool = SingleStoreSearchTool()

# Or specify only table subset
tool = SingleStoreSearchTool(tables=['employees', 'departments'])
```

### 4. Connection with SSL

```python
tool = SingleStoreSearchTool(
    host='your_host',
    user='your_username',
    password='your_password',
    database='your_database',
    ssl_ca='/path/to/ca-cert.pem',
    ssl_cert='/path/to/client-cert.pem',
    ssl_key='/path/to/client-key.pem'
)
```

### 5. Advanced Configuration

```python
tool = SingleStoreSearchTool(
    host='your_host',
    user='your_username',
    password='your_password',
    database='your_database',
    # Connection pool settings
    pool_size=10,
    max_overflow=20,
    timeout=60,
    # Advanced options
    charset='utf8mb4',
    autocommit=True,
    connect_timeout=30,
    results_format='tuple',
    # Custom connection attributes
    conn_attrs={
        'program_name': 'MyApp',
        'custom_attr': 'value'
    }
)
```

## Configuration Parameters

### Basic Connection Parameters
- `host`: Database host address or complete connection URL
- `user`: Database username
- `password`: Database password
- `port`: Database port (default: 3306)
- `database`: Database name
- `tables`: List of specific tables to work with (optional)

### Connection Pool Parameters
- `pool_size`: Maximum number of connections in the pool (default: 5)
- `max_overflow`: Maximum overflow connections beyond pool_size (default: 10)
- `timeout`: Connection timeout in seconds (default: 30)

### SSL/TLS Parameters
- `ssl_key`: Path to client private key file
- `ssl_cert`: Path to client certificate file
- `ssl_ca`: Path to certificate authority file
- `ssl_disabled`: Disable SSL (default: None)
- `ssl_verify_cert`: Verify server certificate
- `ssl_verify_identity`: Verify server identity

### Advanced Parameters
- `charset`: Character set for the connection
- `autocommit`: Enable autocommit mode
- `connect_timeout`: Connection timeout in seconds
- `results_format`: Format for query results ('tuple', 'dict', etc.)
- `vector_data_format`: Format for vector data ('binary', 'json')
- `parse_json`: Parse JSON columns automatically


For more detailed connection options and advanced configurations, refer to the [SingleStore Python SDK documentation](https://singlestoredb-python.labs.singlestore.com/getting-started.html).
