# CrewAI + OpenUI

This example connects a CrewAI Flow to OpenUI over AG-UI. CrewAI owns the
conversation and model call; OpenUI owns the component library, prompt,
streaming parser, renderer, theme, and browser interactions.

```text
AgentInterface -> AG-UI request -> CrewAI Flow -> LLM
      ^                                  |
      +---- streamed OpenUI Lang --------+
```

The OpenUI system prompt is generated from the same `library` exported by
`frontend/src/library.ts`. Generated prompt artifacts are intentionally ignored
by git.

## Requirements

- Python 3.10-3.13 and [uv](https://docs.astral.sh/uv/)
- Node.js 20+
- An OpenAI API key

The example pins CrewAI 1.15.16, `ag-ui-crewai` 0.3.0,
`@openuidev/react-ui` 0.13.6, and `@openuidev/react-lang` 0.2.11.

## Run it

Install the frontend first so its predev step can generate the prompt used by
the backend:

```bash
cd frontend
npm install
npm run generate:prompt
npm run dev
```

In a second terminal, start the CrewAI server:

```bash
cd backend
cp .env.example .env
# Add OPENAI_API_KEY to .env
uv sync
uv run uvicorn openui_crewai_example.server:app --reload --port 8000
```

Open [http://localhost:5173](http://localhost:5173). The frontend sends AG-UI
requests directly to `http://localhost:8000/openui` by default. Override this
with `VITE_CREWAI_URL` when the server runs elsewhere.

Try each starter:

1. **Chart** renders a bar chart from model-produced OpenUI Lang.
2. **Compare queues** renders a comparison with clickable follow-ups. Clicking
   one sends a new user turn through the same CrewAI Flow.
3. **Estimate project** renders a validated form. Submitting it sends both the
   action and form values through the same CrewAI Flow and renders the reply.

## Verify it

```bash
cd frontend
npm test
npm run build

cd ../backend
uv run pytest
```

Do not commit `frontend/generated/system-prompt.txt` or
`frontend/generated/system-prompt.spec.json`; regenerate them whenever the
component library or prompt options change.
