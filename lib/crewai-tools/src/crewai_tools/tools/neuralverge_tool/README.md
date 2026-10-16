# NeuralVerge Tools

## Description

[NeuralVerge](https://neuralverge.ai) is a company and person business-intelligence API. These tools give
CrewAI agents reverse email and phone lookup, email finding and verification, company funding data
(Crunchbase), LinkedIn and Amazon data, web search, AI extraction from any URL and multi-step AI research.

Every tool returns the NeuralVerge envelope as a JSON string:
`{"session_id", "kind", "human" (Markdown summary), "machine" (structured result), "total_points"}`.
API errors (bad key, points limit, upstream unavailable) come back as a `ToolFailure`, so the agent's
`tool_failure_policy` applies.

| Tool | Endpoint | Points |
|---|---|---|
| `NeuralVergePersonByEmailTool` | `run-email-enrichment` | 10 |
| `NeuralVergeEmailFinderTool` | `run-email-finder` | 10 |
| `NeuralVergeEmailValidationTool` | `run-email-validation` | 1 |
| `NeuralVergePhoneLookupTool` | `run-phone-enrichment` | 10 |
| `NeuralVergeUSPhoneLookupTool` | `run-phone-enrichment-us` | 100 |
| `NeuralVergeCompanyFundingTool` | `run-crunchbase-company` | 15 |
| `NeuralVergeWebSearchTool` | `run-search` | 5 |
| `NeuralVergeExtractTool` | `run-extract` | ~5 |
| `NeuralVergeResearchTool` | `run-research`, then polls `get-session-status` every 3 s | 20–400 |
| `NeuralVergeLinkedInProfileEmailTool` | `run-linkedin-email` | 10 |
| `NeuralVergeLinkedInProfileFinderTool` | `run-linkedin-domain` | 10 |
| `NeuralVergeLinkedInCompanySearchTool` | `run-linkedin-company-search` | 5 / company |
| `NeuralVergeLinkedInPeopleSearchTool` | `run-linkedin-people-search` | 100 / 25 results |
| `NeuralVergeLinkedInCompanyEmployeesTool` | `run-linkedin-company-employee` | 30 / run + 5 / profile |
| `NeuralVergeAmazonProductSearchTool` | `run-amazon-product-search` | 1 / product |
| `NeuralVergeAmazonProductTool` | `run-amazon-product` | 5 |
| `NeuralVergeAmazonOfferTool` | `run-amazon-product-offers` | 5 / offer |
| `NeuralVergeAmazonSellerTool` | `run-amazon-seller` | 5 |
| `NeuralVergeAmazonSellerProductsTool` | `run-amazon-seller-products` | 1 / product |

1 point = $0.001; every response carries the `total_points` actually charged.

## Installation

No extra package is needed; the tools call the REST API with `requests`.

```shell
uv add 'crewai[tools]'
```

## Environment Variables

```bash
export NEURALVERGE_API_KEY='your_neuralverge_api_key'   # app.neuralverge.ai → Settings → API
```

The key is sent in the `x-api-key` header. You can also pass `api_key=` to any tool.

## Example

```python
from crewai import Agent, Crew, Task
from crewai_tools import (
    NeuralVergeCompanyFundingTool,
    NeuralVergeEmailFinderTool,
    NeuralVergeEmailValidationTool,
    NeuralVergePersonByEmailTool,
)

researcher = Agent(
    role="Account Researcher",
    goal="Build accurate briefs on B2B prospects",
    backstory="You research companies and their decision makers before sales calls.",
    tools=[
        NeuralVergePersonByEmailTool(),
        NeuralVergeEmailFinderTool(),
        NeuralVergeEmailValidationTool(),
        NeuralVergeCompanyFundingTool(),
    ],
)

task = Task(
    description=(
        "Find out who is behind jane.doe@example.com, check that the address is "
        "deliverable, and summarise their company's funding."
    ),
    expected_output="A short Markdown brief on the person and the company.",
    agent=researcher,
)

print(Crew(agents=[researcher], tasks=[task]).kickoff())
```

## Arguments

Common initialization arguments (all tools):

- `api_key`: NeuralVerge API key. Defaults to `NEURALVERGE_API_KEY`.
- `base_url`: API base URL. Defaults to `https://api.neuralverge.ai`.
- `timeout`: HTTP timeout per request in seconds. Defaults to `120`.

`NeuralVergeResearchTool` also takes:

- `poll_interval`: seconds between status polls. Defaults to `3`.
- `max_wait`: seconds to wait for the task before returning a `timeout` failure. Defaults to `600`.

Run-time arguments are defined by each tool's `args_schema` (for example `email`, `first_name` /
`last_name` / `domain`, `crunchbase_url`, `asin` + `domain`).
