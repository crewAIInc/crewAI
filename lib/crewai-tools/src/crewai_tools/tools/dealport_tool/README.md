# DealportTool

## Description

`DealportTool` connects CrewAI agents to [Dealport](https://dealport.com), the SMB knowledge layer for agents and humans covering 28M+ US private company Records, NYC commercial and multifamily building Records, and the human principals behind them with primary public-filing citations.

Create a workspace API key (`dp_live_...`) at [app.dealport.com/connect?tab=keys](https://app.dealport.com/connect?tab=keys) and export `DP_API_KEY`. New accounts start with a **$5.00 free USD balance** on pay-as-you-go pricing:

- **Free (`$0.00`)**: `preview_companies` market sizing, `match_company_list` CSV/JSON matching, `balance`, Project annotations, and 30-day repeat reads
- **`$0.01`**: company `Record` (`GET /api/v1/companies`, `GET /api/v1/records/{id}`), NYC building `Record` (`GET /api/v1/buildings`), or fresh company website pages
- **`$0.03`**: 50-state `lookup_property` parcel and owner deed lookup
- **`$0.05`**: `Playbook` check, monthly `Project` watch, or NYC building owner lookup
- **`$0.09`**: verified human owner/contact hit (`POST /api/v1/owners`, `$0.00` on miss or 30-day repeat)
- **`$0.50`**: deep company research
- **`$0.02` / `$25.00`**: outreach email sent by **Avery** (Dealport's AI outreach specialist, who always discloses she is an AI) / accepted double-opt-in Warm Yes introduction

## Usage

```python
from crewai import Agent, Crew, Task
from crewai_tools import DealportTool

dealport = DealportTool()

# 1. Size a US company buy-box for $0.00 before spending balance
preview = dealport.run(
    action="preview_companies",
    state="OH",
    naics=["238220"],
    min_employees=15,
    independent=True,
)

# 2. Attach to a CrewAI agent
analyst = Agent(
    role="Origination Analyst",
    goal="Size target US SMB markets and inspect cited company Records",
    backstory="You verify every private company fact against primary public filings.",
    tools=[dealport],
)
```

## Supported Actions

| `action` | Endpoint | Description |
| --- | --- | --- |
| `preview_companies` | `POST /api/v1/companies/preview` | Free (`$0.00`) US company market count and up to 5 sample companies |
| `search_companies` | `GET /api/v1/companies` | Search 28M+ cited US private company Records (`$0.01`/company, `$0.00` 30-day repeat) |
| `search_buildings` | `GET /api/v1/buildings` | Search NYC commercial and multifamily building Records by borough (`MN`, `BK`, `QN`, `BX`, `SI`), unit count, and ACRIS loan maturity (`$0.01`/building) |
| `get_record` | `GET /api/v1/records/{id}` | Open a cited company Record (UUID) or NYC building Record (10-digit BBL) (`$0.01`, `$0.00` 30-day repeat) |
| `find_owner` | `POST /api/v1/owners` | Resolve the verified human owner behind a company UUID (`$0.09` hit, `$0.00` miss) |

## Remote MCP Alternative

To expose Dealport's full 30-tool surface (including `Projects`, `Playbooks`, 50-state `lookup_property`, and Avery introductions), connect `MCPServerAdapter` to `https://app.dealport.com/api/mcp` or read `https://app.dealport.com/llms.txt`.
