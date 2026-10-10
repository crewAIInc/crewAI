# crewai-pyodide

Run CrewAI in the browser and in Node.js on [Pyodide](https://pyodide.org), the
CPython interpreter compiled to WebAssembly.

crewai and crewai-core are pure Python, so the same wheels that run on CPython
run under Pyodide. This package adapts the runtime around them:

- **HTTP over `fetch`.** WebAssembly has no sockets. httpx clients (and so
  the `openai` SDK) send requests through the JavaScript `fetch`
  API, with streaming responses.
- **No threads.** Thread pools and `Thread.start()` run their work inline; the
  event bus schedules async handlers on the host's event loop.
- **Native dependencies.** Dependencies are installed from Pyodide's package
  index, or from PyPI for pure-Python wheels, skipping packages that have no
  WebAssembly build. `chromadb` is replaced by import-only placeholders.
- **Background work off.** Telemetry and the PyPI update check are disabled.

## Requirements

A runtime with WebAssembly JavaScript Promise Integration (JSPI): Chrome or
Edge 137+, or Node.js with `--experimental-wasm-jspi`. crewai's async path
still calls synchronous helpers that block on async work, which only JSPI makes
possible under Pyodide.

Pyodide 0.29 (Python 3.13).

## Usage

Build the wheels from this checkout and serve the example app:

```bash
uv run python lib/crewai-pyodide/build.py
python -m http.server -d lib/crewai-pyodide/dist
```

Open <http://localhost:8000>, enter an OpenAI-compatible endpoint and API key,
and run a two-agent crew in the page.

From your own page, load `crewai-pyodide.mjs` and point it at the built wheels:

```js
import { loadCrewAI } from "./crewai-pyodide.mjs";

const pyodide = await loadCrewAI({ wheelBaseURL: "./wheels/" });
pyodide.globals.set("api_key", apiKey);

const answer = await pyodide.runPythonAsync(`
import crewai_pyodide
from crewai import LLM, Agent, Crew, Task

llm = LLM(model="gpt-4o-mini", api_key=api_key)
agent = Agent(role="Writer", goal="Write haiku", backstory="A poet", llm=llm)
task = Task(description="A haiku about WebAssembly", expected_output="A haiku", agent=agent)

result = await crewai_pyodide.run(Crew(agents=[agent], tasks=[task]).akickoff())
result.raw
`);
```

Inside Pyodide, always use the async entry points (`crew.akickoff()`,
`flow.kickoff_async()`, `llm.acall()`) and wrap them in `crewai_pyodide.run()`:
Pyodide's event loop is always running, so the synchronous `kickoff()` refuses
to start, and top-level `await` does not run inside the asyncio task that parts
of crewai require.

To install into a Pyodide instance you already have:

```python
import micropip

await micropip.install("https://example.com/wheels/crewai_pyodide-1.15.27-py3-none-any.whl")

import crewai_pyodide

await crewai_pyodide.install(
    ["https://example.com/wheels/crewai-1.15.27-py3-none-any.whl",
     "https://example.com/wheels/crewai_core-1.15.27-py3-none-any.whl"]
)
```

## What works

- Agents, Tasks, Crews and Flows
- The native OpenAI provider and OpenAI-compatible endpoints via `base_url`,
  including streaming and native tool calling. Other providers built on httpx
  SDKs (e.g. `extras=["anthropic"]`) should work but are not yet tested here.
- Custom tools (`@tool` and `BaseTool`) that do not need native code; `requests`
  works through Pyodide's urllib3 support, `urllib.request` does not

## What does not

- Memory, Knowledge and RAG: their default storage (LanceDB, ChromaDB) has no
  WebAssembly build. Flows run without their automatic memory.
- MCP servers over stdio, code execution in subprocesses, and most packages in
  `crewai-tools`
- Parallel execution: async tasks and tool calls run one at a time
- Telemetry and tracing

## Snapshots

Importing crewai takes several seconds. Isolate hosts avoid paying that on
every start by snapshotting the interpreter after its imports and restoring it
in each new isolate. Such hosts forbid reading randomness during the imports,
since it would be shared by every restored isolate, so `bootstrap()` makes the
import entropy-free:

- `opentelemetry.context` gets counter-based UUIDs for its context keys, which
  only need to be unique within the process.
- `openai.lib.bedrock` gets a placeholder HMAC key while it is imported.

In each restored isolate, call `crewai_pyodide.reseed()` before doing work to
reseed `random` and replace the placeholder; `crewai_pyodide.run()` does this on
first use. `crewai_pyodide.audit_entropy()` records any Python-level entropy
read inside a block, and the smoke test uses it to keep `import crewai` clean.

Pyodide's own `makeMemorySnapshot()` cannot restore native extension modules
(pydantic_core, numpy, cryptography), so it does not work for crewai yet.
workerd's snapshots do.

## Browser security

API keys used in a page are visible to anyone who can open it. Use this for
local tools, demos, or keys the user enters themselves; otherwise send requests
through a proxy that holds the key. The provider must allow cross-origin
requests: OpenAI does; Anthropic only does for requests that carry the
`anthropic-dangerous-direct-browser-access: true` header, so reach it through
a proxy.

## Testing

```bash
uv run pytest lib/crewai-pyodide/tests          # unit tests on CPython
cd lib/crewai-pyodide && npm install && npm run build && npm run test:smoke
```

The smoke test loads the built wheels into Pyodide under Node.js and runs a
crew, a streamed crew, a tool call and a flow against a mock OpenAI API.
