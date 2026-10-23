# Jev Noul Guardrail

`create_jev_noul_guardrail` builds a callable task guardrail that asks TypeSafe's
Jev model one [Noul](https://docs.typesafe.ai/primitives/noul) question about a
task's output. Jev answers with the probability that the question is true, and the
output passes when that probability is greater than or equal to your threshold.

Phrase the question so that "true" means the output is acceptable.

## Installation

The guardrail is opt-in. Install the TypeSafe SDK and set `TYPESAFE_API_KEY`:

```bash
uv add typesafe-sdk
```

## Example

```python
from crewai import Agent, Task
from crewai_tools.tools.jev_noul_guardrail import create_jev_noul_guardrail

cites_a_source = create_jev_noul_guardrail(
    "Does this answer cite at least one source for its main claim?",
    threshold=0.8,
)

researcher = Agent(
    role="Researcher",
    goal="Answer questions with sources",
    backstory="You always back up claims with a citation.",
)

task = Task(
    description="Explain why the sky is blue.",
    expected_output="A short explanation that cites a source.",
    agent=researcher,
    guardrail=cites_a_source,
    guardrail_max_retries=2,
)
```

When Jev's probability is below 0.8, the guardrail returns `(False, reason)` and
CrewAI's normal guardrail retry asks the agent to try again, up to
`guardrail_max_retries` times. When it passes, the output is kept unchanged.

## Behavior

- **Threshold:** a finite number from 0 to 1. A probability equal to the threshold
  passes.
- **Missing or malformed answers:** if the response has no Noul answer under the
  question name, or the probability is not a finite number from 0 to 1, the output
  is rejected.
- **SDK errors:** authentication, network, API and response-validation errors from
  the TypeSafe SDK are raised as they are. They never count as a pass and are not
  turned into a retry.
- **Client:** by default a `TypeSafeClient` is created for each check from the
  `TYPESAFE_*` environment variables and closed afterwards. Pass `client=` to reuse
  your own client, and `model=` to choose a Jev model.

The guardrail only reads Jev's Noul probability. It does not use Choice
confidences, so a low-confidence Choice answer is not treated as a Noul rejection.
