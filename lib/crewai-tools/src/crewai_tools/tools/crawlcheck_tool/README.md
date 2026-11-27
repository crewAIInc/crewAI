# CrawlCheck tools

## Description
Two tools that let a crew ask before it acts on a website. [CrawlCheck](https://crawlcheck.io) reads websites the way AI crawlers do and answers with a signed decision.

- `CrawlCheckSiteCheckTool`: before reading, citing, connecting to or buying from a site, get a decision (`allow`, `warn`, `require_confirmation` or `block`) with the reasons. The `proceed` field is the answer.
- `CrawlCheckVerifyFactTool`: check a phone number, address, hours or service area you hold against what the business publishes itself.

No API key is needed for decisions. A CrawlCheck licence key (`CRAWLCHECK_API_KEY`) unlocks the full findings for licensed domains.

## Installation
```shell
pip install 'crewai[tools]' crawlcheck
```

## Example
```python
from crewai import Agent, Crew, Task
from crewai_tools import CrawlCheckSiteCheckTool, CrawlCheckVerifyFactTool

researcher = Agent(
    role="Careful researcher",
    goal="Recommend only businesses whose websites CrawlCheck says can be cited",
    backstory="You check before you cite.",
    tools=[CrawlCheckSiteCheckTool(), CrawlCheckVerifyFactTool()],
)
task = Task(
    description="Can we recommend example.com for a plumbing job? Check it first.",
    expected_output="A yes or no with the CrawlCheck decision quoted.",
    agent=researcher,
)
print(Crew(agents=[researcher], tasks=[task]).kickoff())
```

## Arguments
- `CrawlCheckSiteCheckTool(api_key=None, allow_warn=False)`: `allow_warn=True` lets a `warn` decision proceed.
- `CrawlCheckVerifyFactTool(api_key=None)`.

## Output
Both tools return a JSON string. A site check includes `proceed`, `decision`, `reasons`, `reason_codes` and `receipt_sha256` (the signed decision's hash, verifiable offline with the `crawlcheck` package). A fact check includes `ok`, `verified` and the published value.
