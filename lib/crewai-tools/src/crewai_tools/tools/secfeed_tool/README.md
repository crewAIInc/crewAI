# SecFeed SEC filing tools

## Description

Tools that give agents SEC EDGAR data through the [SecFeed API](https://secfeedapi.com):

- `SecFeedFilingSearchTool`: reads the right section (Business, Risk Factors, MD&A, Market Risk) of a company's latest 10-K or 10-Q and returns the passages most relevant to a question, with the filing link. The section is picked from the question, or set with `section`.
- `SecFeedFinancialStatementsTool`: income statement, balance sheet and cash flow from the latest 10-K or 10-Q XBRL data.
- `SecFeedInsiderTradesTool`: latest Form 4 insider transactions (role, buy or sell, shares, price, value, 10b5-1 flag).
- `SecFeed8KReportsTool`: latest 8-K current reports and the items each one covers.

Every tool returns compact plain text with a source link, so agents can cite the filing. No embeddings or vector store are needed: the API returns the exact filing section.

## Installation

No extra package is needed beyond `crewai-tools`. Get an API key at https://secfeedapi.com/signup/ and set it:

```shell
export SECFEED_API_KEY=sf_...
```

## Example

```python
from crewai import Agent
from crewai_tools import (
    SecFeed8KReportsTool,
    SecFeedFilingSearchTool,
    SecFeedFinancialStatementsTool,
    SecFeedInsiderTradesTool,
)

analyst = Agent(
    role="Equity analyst",
    goal="Explain a company's risks and financial health from its SEC filings",
    backstory="You only trust primary sources and cite every filing you use.",
    tools=[
        SecFeedFilingSearchTool(),
        SecFeedFinancialStatementsTool(),
        SecFeedInsiderTradesTool(),
        SecFeed8KReportsTool(),
    ],
)
```

The agent calls the tools with arguments such as:

```python
SecFeedFilingSearchTool().run(ticker="NVDA", question="What are the risks from export controls?", form="10-K")
SecFeedInsiderTradesTool().run(ticker="MSFT", code="S", limit=5)
```

## Arguments

| Tool | Arguments |
|---|---|
| `SecFeedFilingSearchTool` | `ticker`, `question`, `form` (`10-K` or `10-Q`, default `10-K`), `section` (optional: `business`, `risk`, `mdna`, `market_risk`) |
| `SecFeedFinancialStatementsTool` | `ticker`, `form` (`10-K` or `10-Q`) |
| `SecFeedInsiderTradesTool` | `ticker`, `code` (`P` buys, `S` sales, empty for all), `limit` (up to 50) |
| `SecFeed8KReportsTool` | `ticker`, `limit` (up to 50) |

Each tool also accepts `api_key` at construction instead of the environment variable. Filing lookups and sections are cached per tool instance, so repeated questions about the same filing cost no extra API calls.
