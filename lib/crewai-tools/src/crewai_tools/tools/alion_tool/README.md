# Alion tools

Company, startup, investor and job-posting data from [Alion](https://alion.io/developers), a live graph of companies worldwide.

| Tool | What it does | Needs a key |
|---|---|---|
| `AlionCompanySearchTool` | Companies by industry, country, size, startup status, funding stage, recent rounds, investor, technology and hiring | No |
| `AlionCompanyProfileTool` | One company in depth: funding, hiring, stack, offices, visa sponsorship (identity only without a key) | For the full profile |
| `AlionSimilarCompaniesTool` | Lookalikes of a company | No |
| `AlionJobSearchTool` | Verified job postings by role, company, country, work mode, seniority, technology and pay | No |

```python
from crewai_tools import AlionCompanySearchTool, AlionJobSearchTool

tools = [AlionCompanySearchTool(), AlionJobSearchTool()]
```

Without `ALION_API_KEY` the tools use Alion's free open address (300 calls a day per network, compact rows). Create a key at https://alion.io/developers/keys for full company profiles.
