# ShopSavvy Product Search Tool

## Description

The `ShopSavvyProductSearchTool` lets CrewAI agents search for products and get real-time pricing across thousands of retailers via the [ShopSavvy Data API](https://shopsavvy.com/data). Queries can be product names, barcodes (UPC/EAN/ISBN), ASINs, or URLs.

## Installation

```shell
pip install 'crewai[tools]'
```

## Environment Variables

A ShopSavvy API key is required. Sign up at [shopsavvy.com/data](https://shopsavvy.com/data) to get one.

```bash
export SHOPSAVVY_API_KEY='your_api_key'
```

## Example

```python
from crewai import Agent, Task, Crew
from crewai_tools import ShopSavvyProductSearchTool

tool = ShopSavvyProductSearchTool()

researcher = Agent(
    role='Shopping Researcher',
    goal='Find the best prices on products',
    backstory='An expert at comparing prices across retailers.',
    tools=[tool],
    verbose=True,
)

task = Task(
    description='Find the best current prices for the Sony WH-1000XM5 headphones.',
    expected_output='A summary of available prices from different retailers.',
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
print(result)
```

## Arguments

- `query` (str): **Required**. Product name, barcode, UPC, EAN, ISBN, ASIN, or URL.
- `api_key` (str, optional): Your ShopSavvy API key. Defaults to the `SHOPSAVVY_API_KEY` environment variable.
- `base_url` (str, optional): API base URL. Defaults to `https://api.shopsavvy.com/v1`.

## Documentation

Full API documentation: [shopsavvy.com/data/documentation](https://shopsavvy.com/data/documentation)
