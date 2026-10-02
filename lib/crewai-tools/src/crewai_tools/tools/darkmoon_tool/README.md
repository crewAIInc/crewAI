# Darkmoon Tools

Give a CrewAI agent the ability to start an autonomous penetration test with [Darkmoon](https://github.com/ASCIT31/Dark-Moon), read back the findings, and list campaigns.

Darkmoon is a self-hosted, GPL-3.0 autonomous AI pentest platform: an LLM orchestrates specialist agents and offensive tools and proves each finding with a real exploit. These tools call the **Darkmoon Dashboard API of an instance you operate yourself**. There is no public hosted endpoint.

> Open source versus Pro: the Darkmoon engine and CLI are open source. The web dashboard and its API, which these tools use, are part of the paid Pro edition. Darkmoon's remediation to pull request feature is also Pro and is intentionally not exposed by these tools.

> Only run assessments against systems you own or are explicitly authorised to test. Findings can include false positives and must be reviewed by a qualified human.

## Tools

| Tool | Purpose |
| --- | --- |
| `DarkmoonRunPentestTool` | Start a campaign against an authorised target. Optionally wait for it to finish and return its findings and severity statistics. |
| `DarkmoonGetFindingsTool` | Return the findings and severity statistics for a campaign id. |
| `DarkmoonListCampaignsTool` | List the campaigns visible to the authenticated dashboard user. |

All tools return JSON strings. Findings carry the fields Darkmoon records, for example `title`, `severity`, `cvss_score`, `category`, `status`, `description`, `endpoint` and `remediation`.

## Environment

- `DARKMOON_BASE_URL` (required): base URL of your Darkmoon Dashboard API, e.g. `http://localhost:8000`
- `DARKMOON_USERNAME` (required): dashboard username
- `DARKMOON_PASSWORD` (required): dashboard password

`base_url`, `username` and `password` can also be passed to the tool constructor. The tools log in with `POST /api/v1/auth/login` once and reuse the JWT.

## Usage

```python
from crewai import Agent, Crew, Task
from crewai_tools import (
    DarkmoonGetFindingsTool,
    DarkmoonListCampaignsTool,
    DarkmoonRunPentestTool,
)

analyst = Agent(
    role="Application security analyst",
    goal="Assess an authorised staging target and summarise the findings",
    backstory="You triage Darkmoon findings and flag likely false positives.",
    tools=[
        DarkmoonRunPentestTool(),
        DarkmoonGetFindingsTool(),
        DarkmoonListCampaignsTool(),
    ],
)

task = Task(
    description="Run a Darkmoon assessment against staging.example.com and summarise the high severity findings.",
    expected_output="A short list of confirmed findings ordered by severity.",
    agent=analyst,
)

print(Crew(agents=[analyst], tasks=[task]).kickoff())
```

### Run options

`DarkmoonRunPentestTool` accepts `target` (required), `wait_for_completion` (default `true`), `program`, `focus` (comma separated focus areas), `severity` (minimum severity), `timeout_seconds` (default 1800) and `poll_interval_seconds` (default 5). With `wait_for_completion=false` it returns `{"status": "started", "run_id": ...}` immediately; use `DarkmoonListCampaignsTool` and `DarkmoonGetFindingsTool` later. When waiting, the result reports `timed_out: true` if the run did not finish in time.

## Links

- Darkmoon: https://github.com/ASCIT31/Dark-Moon
