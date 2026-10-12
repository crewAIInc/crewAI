# crewai-pyodide

Run CrewAI in the browser and in Node.js on [Pyodide](https://pyodide.org), the
CPython interpreter compiled to WebAssembly.

Requirements, usage, and what works are documented in
[Running in the Browser](https://docs.crewai.com/en/guides/advanced/running-in-the-browser).
This file covers how the package works and how to develop it.

## How it works

crewai and crewai-core are pure Python, so the same wheels that run on CPython
run under Pyodide. This package adapts the runtime around them:

- **HTTP over `fetch`.** WebAssembly has no sockets. httpx clients (and so
  the `openai` SDK) send requests through the JavaScript `fetch`
  API, with streaming responses.
- **No threads.** Thread pools and `Thread.start()` run their work inline; the
  event bus schedules async handlers on the host's event loop.
- **Native dependencies.** Dependencies are installed from Pyodide's package
  index, or from PyPI for pure-Python wheels, skipping packages that have no
  WebAssembly build. `chromadb` and `portalocker` are replaced by import-only
  placeholders, and `crewai_core.lock_store` uses in-process locks.
- **Background work off.** Telemetry and the PyPI update check are disabled.

Providers built on other httpx SDKs (e.g. `extras=["anthropic"]`) should work
but are not yet tested here. Anthropic only allows cross-origin requests that
carry the `anthropic-dangerous-direct-browser-access: true` header, so reach it
through a proxy.

## Installing into an existing Pyodide instance

```python
import micropip

await micropip.install("https://example.com/wheels/crewai_pyodide-1.15.27-py3-none-any.whl")

import crewai_pyodide

await crewai_pyodide.install(
    ["https://example.com/wheels/crewai-1.15.27-py3-none-any.whl",
     "https://example.com/wheels/crewai_core-1.15.27-py3-none-any.whl"]
)
```

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

## Testing

```bash
uv run pytest lib/crewai-pyodide/tests          # unit tests on CPython
cd lib/crewai-pyodide && npm install && npm run build && npm run test:smoke
```

The smoke test loads the built wheels into Pyodide under Node.js and runs a
crew, a streamed crew, a tool call and a flow against a mock OpenAI API.
