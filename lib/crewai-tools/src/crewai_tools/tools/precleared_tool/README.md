# PreclearedTool Documentation

## Description
Precleared answers one question per recipient: is it legal to send this person a cold email? The tool checks one address before the first email to anyone who is not already a customer. Precleared looks the recipient up live (for the UK, Companies House and the company's website) and applies the rules of the recipient's country, in the UK, US, Canada, Germany, France, Australia, Ireland, the Netherlands, Spain, Italy, Latvia, the UAE, Egypt and Qatar. It returns a verdict (`ok_to_email`, `consent_needed`, `blocked` or `unknown`), the rule that decided it with its source, what the email's footer must include, a next step and a signed receipt id that can be shown to a client or regulator later. Nothing is sent to the recipient.

## Installation
```shell
pip install 'crewai[tools]'
```
No extra package is needed. A key is optional: without one a small daily allowance shared by everyone on your network applies; a free key from https://precleared.dev/get-key gives your agent its own.

## Example
```python
import os
from crewai import Agent, Crew, Task
from crewai_tools import PreclearedTool

os.environ["PRECLEARED_API_KEY"] = "pc_live_..."  # optional

precleared = PreclearedTool()

outreach = Agent(
    role="Outbound researcher",
    goal="Only email people we are allowed to email",
    backstory="You prepare cold emails for a UK company and never send to an address Precleared has not cleared.",
    tools=[precleared],
)

task = Task(
    description=(
        "Check whether we may cold email jane@acme.co.uk from the UK. "
        "The address is on their website and our pitch is bookkeeping software for small firms."
    ),
    expected_output="The verdict, the rule, the footer requirements and the receipt id.",
    agent=outreach,
)

Crew(agents=[outreach], tasks=[task]).kickoff()
```

## Arguments
- `recipient_email` (required): the address to check.
- `sender_country`: two-letter code of the sender, default `GB`. Only Canada adds sender-side rules.
- `address_source`: `published`, `bought`, `scraped`, `given` or `unknown` (default).
- `pitch`: one line on what you offer. Some countries need it to be relevant to the recipient's role.
- `company_name`, `recipient_name`: help find a UK company on the register. The name is not stored.
- `recipient_country`: two-letter code, only if you are sure. Otherwise Precleared works it out.
- `existing_customer`: `true` only for people or companies that are already your customers.

## What the agent gets back
A short text with the verdict, the recipient's country, the rule and its source, what the footer must include, the next step, the receipt id with its public page, and a one-line sentence for the agent's own report. Send only on `ok_to_email`, with a footer that meets the footer requirements. Treat `unknown` as `consent_needed`. When the daily allowance is used up, the tool returns the message to pass to the person running the agent instead of raising.

## Steps to Get Started
1. **Package Installation**: Confirm that the `crewai[tools]` package is installed in your Python environment.
2. **Key (optional)**: Get a free key at https://precleared.dev/get-key and store it in an environment variable named `PRECLEARED_API_KEY`.
3. **Use it before the first email**: give the tool to the agent that prepares outreach and tell it to check each address before the first email and keep the receipt id with the lead.

## Conclusion
Precleared gives outbound agents the one check that is about the law rather than deliverability: whether a cold email to this person is allowed where they are, with the rule quoted and a receipt to show later. Precleared's answers are decision support with sources, not legal advice. Docs: https://precleared.dev/docs
