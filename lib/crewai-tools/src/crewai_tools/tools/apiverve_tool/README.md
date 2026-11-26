# APIVerve Tools

Give agents access to the [APIVerve](https://apiverve.com) catalog of about 380 REST APIs behind one key: email and phone validation, IP, DNS and SSL lookups, exchange and VAT rates, VIN decoding, document and image conversion, and more.

- `ApiverveFindApisTool` searches the catalog and returns matching API ids with their parameters. It needs no key.
- `ApiverveRunApiTool` calls any API by id and returns the `data` of its response as JSON.

## Environment

- `APIVERVE_API_KEY` (required for `ApiverveRunApiTool`). Get one at https://dashboard.apiverve.com. A free plan is available.

## Example

```python
from crewai import Agent, Crew, Task
from crewai_tools import ApiverveFindApisTool, ApiverveRunApiTool

analyst = Agent(
    role="Data checker",
    goal="Verify contact details before they reach the CRM",
    backstory="You look up the right APIVerve API, then call it.",
    tools=[ApiverveFindApisTool(), ApiverveRunApiTool()],
)

task = Task(
    description="Is jane@example.com a deliverable, non-disposable address?",
    expected_output="A short verdict with the evidence",
    agent=analyst,
)

Crew(agents=[analyst], tasks=[task]).kickoff()
```

## Direct usage

```python
from crewai_tools import ApiverveFindApisTool, ApiverveRunApiTool

print(ApiverveFindApisTool().run(query="vat rates"))
print(ApiverveRunApiTool().run(api="emailvalidator", parameters={"email": "jane@example.com"}))
```

## Parameters

`ApiverveFindApisTool.run`:

- `query` (str): what you need, for example `"validate email"`.

`ApiverveRunApiTool.run`:

- `api` (str): the API id, for example `emailvalidator`, `iplookup` or `vindecoder`.
- `parameters` (dict): the API's parameters. Each one is sent as a query parameter or in the JSON body, as the catalog declares.

## Security

Requests go only to `https://api.apiverve.com`, with the key in the `x-api-key` header. Redirects are not followed, and the key is removed from any text returned to the agent. Errors come back as strings that start with `Error:`.
