# OpticParse & PhishVision Tools for CrewAI

Enterprise AI Vision web scraping and zero-day threat intelligence tools for autonomous CrewAI agents.

## Overview

- **OpticParseTool**: Scrapes and extracts structured, token-optimized data from live websites using Multimodal Vision. Bypasses bot protections, Turnstile challenges, and dynamic JavaScript rendering without brittle CSS selectors.
- **PhishVisionTool**: Inspects URLs and domains for zero-day phishing campaigns, brand impersonation, and smart contract wallet drainers in under 1.6 seconds.

## Installation

Both tools are included natively in `crewai-tools`:

```bash
pip install crewai-tools
```

No external binary drivers or headless browser packages required.

## Setup & Environment

An API key is optional. Public starter trial mode is enabled by default:

```bash
export OPTICPARSE_API_KEY="op_live_your_key_here"
```

## Quickstart

### 1. Vision Web Scraping with OpticParseTool

```python
from crewai import Agent, Task, Crew
from crewai_tools import OpticParseTool

# Initialize tool
scraper_tool = OpticParseTool()

# Create research agent
researcher = Agent(
    role="Senior Market Analyst",
    goal="Extract current product pricing and specifications from target website",
    backstory="Specialized in accurate competitive intelligence extraction.",
    tools=[scraper_tool],
    verbose=True,
)

task = Task(
    description="Extract the pricing table and feature list from https://example.com/pricing",
    expected_output="Structured JSON summary of tiers, monthly pricing, and features.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
print(result)
```

### 2. Cybersecurity Inspection with PhishVisionTool

```python
from crewai import Agent, Task, Crew
from crewai_tools import PhishVisionTool

security_tool = PhishVisionTool()

guard_agent = Agent(
    role="Cybersecurity Sentinel",
    goal="Verify that user-supplied URLs are safe before visiting or interacting",
    backstory="Zero-trust security auditor preventing access to credential harvesters.",
    tools=[security_tool],
    verbose=True,
)

task = Task(
    description="Audit https://suspicious-claim-drop.xyz for phishing or crypto wallet drainer threats.",
    expected_output="Security verdict, threat score, and flagged threat vectors.",
    agent=guard_agent,
)

crew = Crew(agents=[guard_agent], tasks=[task])
result = crew.kickoff()
print(result)
```
