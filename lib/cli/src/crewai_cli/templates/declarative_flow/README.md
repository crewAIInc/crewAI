# {{name}} Flow

This project defines a CrewAI Flow in `src/{{folder_name}}/flow.yaml`.

## Install

```bash
crewai install
```

## Run

```bash
crewai flow kickoff
```

Edit `src/{{folder_name}}/flow.yaml` to change the flow. Add reusable crews under `src/{{folder_name}}/crews/`, custom Python tools under `src/{{folder_name}}/tools/`, and shared knowledge files under `src/{{folder_name}}/knowledge/`.
