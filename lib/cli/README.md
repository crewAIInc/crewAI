# crewai-cli

CLI for CrewAI — scaffold, run, deploy and manage AI agent crews without
installing the full framework.

## Installation

```bash
pip install crewai-cli
```

This pulls in `crewai-core` (shared utilities) but not the `crewai` framework
itself, so commands that don't need a crew loaded — `crewai version`,
`crewai login`, `crewai org list`, `crewai config *`, `crewai traces *`,
`crewai create`, `crewai template *` — work standalone.

Commands that load a user's crew or flow (`crewai run`, `crewai train`,
`crewai test`, `crewai chat`, `crewai replay`, `crewai reset-memories`,
`crewai deploy push`, `crewai tool publish`) require `crewai` to be installed
in the project's environment. They print a clear error if it is missing.

To install both at once:

```bash
pip install crewai[cli]
```
