# Snowflake Search Tool

A tool for executing queries on Snowflake data warehouse with built-in connection pooling, retry logic, and async execution support.

## Installation

```bash
uv sync --extra snowflake

OR 
uv pip install snowflake-connector-python>=3.5.0 snowflake-sqlalchemy>=1.5.0 cryptography>=41.0.0

OR 
pip install snowflake-connector-python>=3.5.0 snowflake-sqlalchemy>=1.5.0 cryptography>=41.0.0
```

## Quick Start

```python
import asyncio
from crewai_tools import SnowflakeSearchTool, SnowflakeConfig

# Create configuration
config = SnowflakeConfig(
    account="your_account",
    user="your_username", 
    password="your_password",
    warehouse="COMPUTE_WH",
    database="your_database",
    snowflake_schema="your_schema"  # Note: Uses snowflake_schema instead of schema
)

# Initialize tool
tool = SnowflakeSearchTool(
    config=config,
    pool_size=5,
    max_retries=3,
    enable_caching=True
)

# Execute query
async def main():
    results = await tool._run(
        query="SELECT * FROM your_table LIMIT 10",
        timeout=300
    )
    print(f"Retrieved {len(results)} rows")

if __name__ == "__main__":
    asyncio.run(main())
```

## Features

- ✨ Asynchronous query execution
- 🚀 Connection pooling for better performance
- 🔄 Automatic retries for transient failures
- 💾 Query result caching (optional)
- 🔒 Support for both password and key-pair authentication
- 📝 Comprehensive error handling and logging

## Configuration Options

### SnowflakeConfig Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| account | Yes | Snowflake account identifier |
| user | Yes | Snowflake username |
| password | Yes* | Snowflake password |
| private_key_path | No* | Path to private key file (alternative to password) |
| warehouse | Yes | Snowflake warehouse name |
| database | Yes | Default database |
| snowflake_schema | Yes | Default schema |
| role | No | Snowflake role |
| session_parameters | No | Custom session parameters dict |

\* Either password or private_key_path must be provided

### Tool Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| pool_size | 5 | Number of connections in the pool |
| max_retries | 3 | Maximum retry attempts for failed queries |
| retry_delay | 1.0 | Delay between retries in seconds |
| enable_caching | True | Enable/disable query result caching |

## Advanced Usage

### Using Key-Pair Authentication

```python
config = SnowflakeConfig(
    account="your_account",
    user="your_username",
    private_key_path="/path/to/private_key.p8",
    warehouse="your_warehouse",
    database="your_database",
    snowflake_schema="your_schema"
)
```

### Custom Session Parameters

```python
config = SnowflakeConfig(
    # ... other config parameters ...
    session_parameters={
        "QUERY_TAG": "my_app",
        "TIMEZONE": "America/Los_Angeles"
    }
)
```

## Best Practices

1. **Error Handling**: Always wrap query execution in try-except blocks
2. **Logging**: Enable logging to track query execution and errors
3. **Connection Management**: Use appropriate pool sizes for your workload
4. **Timeouts**: Set reasonable query timeouts to prevent hanging
5. **Security**: Use key-pair auth in production and never hardcode credentials

## Example with Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    try:
        # ... tool initialization ...
        results = await tool._run(query="SELECT * FROM table LIMIT 10")
        logger.info(f"Query completed successfully. Retrieved {len(results)} rows")
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise
```

## Error Handling

The tool automatically handles common Snowflake errors:
- DatabaseError
- OperationalError
- ProgrammingError
- Network timeouts
- Connection issues

Errors are logged and retried based on your retry configuration. 