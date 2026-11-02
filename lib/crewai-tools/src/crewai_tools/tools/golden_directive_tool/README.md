# GoldenDirectiveTool

## Description

The GoldenDirectiveTool calculates the relationship score from The Golden Directive (https://thegoldendirective.pages.dev/), a candidate training reward proposed for sustainable agents:

Y = 2(A x E) / (A + E)

A is an agent's net contribution-health score, floored at 0, designed to be grounded in resource ledgers (compute, energy, water, hardware, economics, and CO2 later), not self-report. E is the health of everything not-A, against a healthy baseline. If either A or E reaches 0, Y reaches 0.

The tool performs the calculation only. It does not measure A or E. Given a cropped environment score as well, it reports full Y, cropped Y, and the gap between them, as the Directive's reporting rule requires.

The Golden Directive is a proposal, not a deployed solution, and A and E measurement is not yet operational for real training in the source proposal.

## Installation

Install the `crewai_tools` package to use the GoldenDirectiveTool in your project:

```shell
pip install 'crewai[tools]'
```

## Example

```python
from crewai_tools import GoldenDirectiveTool

tool = GoldenDirectiveTool()

# Healthy: A = 8, E = 8
print(tool.run(agent_score=8, environment_score=8))
# A = 8.0
# E (full) = 8.0
# Y (full) = 8.0000

# With a cropped frame: full E = 3, cropped E = 7
print(tool.run(agent_score=8, environment_score=3, environment_score_cropped=7))
# A = 8.0
# E (full) = 3.0
# Y (full) = 4.3636
# E (cropped) = 7.0
# Y (cropped) = 7.4667
# Gap (cropped Y - full Y) = 3.1030
```

## Arguments

- `agent_score` (float, mandatory): the agent score A.
- `environment_score` (float, mandatory): the environment health score E, full frame.
- `environment_score_cropped` (float, optional): the environment health score E computed on a cropped frame.
