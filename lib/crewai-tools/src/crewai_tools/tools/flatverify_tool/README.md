# Flatverify Tools

## Description

Two tools that let an agent check whether email addresses can receive mail with [Flatverify](https://flatverify.com):

- `FlatverifyVerifyEmailTool` checks one address.
- `FlatverifyBulkVerifyTool` checks a list as one job, waits for it and returns one row per address.

Each address comes back `valid`, `invalid`, `risky` or `unknown`, with a `sub_status` reason such as `accepted`, `mailbox_not_found`, `catch_all`, `role_account` or `disposable`, and `disposable`, `role`, `free` and `catch_all` flags.

## Installation

```shell
pip install 'crewai[tools]'
```

Create an API key in the [Flatverify app](https://app.flatverify.com) under API keys and set it:

```shell
export FLATVERIFY_API_KEY="fv_..."
```

## Example

```python
from crewai import Agent, Crew, Task
from crewai_tools import FlatverifyBulkVerifyTool, FlatverifyVerifyEmailTool

checker = Agent(
    role="List hygiene specialist",
    goal="Make sure every address on a lead list can receive mail",
    backstory="You clean lead lists before they reach the sending tool.",
    tools=[FlatverifyVerifyEmailTool(), FlatverifyBulkVerifyTool()],
)

task = Task(
    description=(
        "Check these addresses and list the ones that are safe to email: "
        "maya.chen@amazon.com, j.ortiz@salesforce.com, info@shopify.com"
    ),
    expected_output="The valid addresses, and the reason for each address you dropped.",
    agent=checker,
)

Crew(agents=[checker], tasks=[task]).kickoff()
```

Direct use:

```python
FlatverifyVerifyEmailTool().run(email="maya.chen@amazon.com")
FlatverifyBulkVerifyTool().run(emails=["j.ortiz@salesforce.com", "sam.lee@stripe.com"])
```

## Arguments

Both tools accept these settings.

- `api_key` (optional) defaults to the `FLATVERIFY_API_KEY` environment variable.
- `timeout` is the number of seconds before one HTTP request times out. Default `30`.
- `max_retries` sets how often rate limited (429) and busy (503) answers are retried after the `Retry-After` delay. Default `3`.

`FlatverifyBulkVerifyTool` also accepts these.

- `wait_timeout` is how many seconds the tool waits for the job. Default `300`. If the job is not done by then, the tool returns its `job_id`, and the agent calls the tool again with only that `job_id`.
- `max_results` caps the result rows returned per call. Default `1000`.

Run arguments: `email` for the single check; `emails`, optional `name` and optional `job_id` for the bulk check. API errors are returned as a `ToolFailure`, so the run records them as failed calls. The API key is never logged.

Every Flatverify plan is unlimited; plans differ only in speed. New accounts get 100 free checks. API docs: https://flatverify.com/docs
