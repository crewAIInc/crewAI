## Building CrewAI Tools

This guide shows you how to build high‑quality CrewAI tools that match the patterns in this repository and are ready to be merged. It focuses on: architecture, conventions, environment variables, dependencies, testing, documentation, and a complete example.

### Who this is for
- Contributors creating new tools under `crewai_tools/tools/*`
- Maintainers reviewing PRs for consistency and DX

---

## Quick‑start checklist
1. Create a new folder under `crewai_tools/tools/<your_tool_name>/` with a `README.md` and a `<your_tool_name>.py`.
2. Implement a class that ends with `Tool` and subclasses `BaseTool` (or `RagTool` when appropriate).
3. Define a Pydantic `args_schema` with explicit field descriptions and validation.
4. Declare `env_vars` and `package_dependencies` in the class when needed.
5. Lazily initialize clients in `__init__` or `_run` and handle missing credentials with clear errors.
6. Implement `_run(...) -> str | dict` and, if needed, `_arun(...)`.
7. Add tests under `tests/tools/` (unit, no real network calls; mock or record safely).
8. Add a concise tool `README.md` with usage and required env vars.
9. If you add optional dependencies, register them in `pyproject.toml` under `[project.optional-dependencies]` and reference that extra in your tool docs.
10. Run `uv run pytest` and `pre-commit run -a` locally; ensure green.

---

## Tool anatomy and conventions

### BaseTool pattern
All tools follow this structure:

```python
from typing import Any, List, Optional, Type

import os
from pydantic import BaseModel, Field
from crewai.tools import BaseTool, EnvVar


class MyToolInput(BaseModel):
    """Input schema for MyTool."""
    query: str = Field(..., description="Your input description here")
    limit: int = Field(5, ge=1, le=50, description="Max items to return")


class MyTool(BaseTool):
    name: str = "My Tool"
    description: str = "Explain succinctly what this tool does and when to use it."
    args_schema: Type[BaseModel] = MyToolInput

    # Only include when applicable
    env_vars: List[EnvVar] = [
        EnvVar(name="MY_API_KEY", description="API key for My service", required=True),
    ]
    package_dependencies: List[str] = ["my-sdk"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Lazy import to keep base install light
        try:
            import my_sdk  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "Missing optional dependency 'my-sdk'. Install with: \n"
                "  uv add crewai-tools --extra my-sdk\n"
                "or\n"
                "  pip install my-sdk\n"
            ) from exc

        if "MY_API_KEY" not in os.environ:
            raise ValueError("Environment variable MY_API_KEY is required for MyTool")

    def _run(self, query: str, limit: int = 5, **_: Any) -> str:
        """Synchronous execution. Return a concise string or JSON string."""
        # Implement your logic here; do not print. Return the content.
        # Handle errors gracefully, return clear messages.
        return f"Processed {query} with limit={limit}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Optional async counterpart if your client supports it."""
        # Prefer delegating to _run when the client is thread-safe
        return self._run(*args, **kwargs)
```

Key points:
- Class name must end with `Tool` to be auto‑discovered by our tooling.
- Use `args_schema` for inputs; always include `description` and validation.
- Validate env vars early and fail with actionable errors.
- Keep outputs deterministic and compact; favor `str` (possibly JSON‑encoded) or small dicts converted to strings.
- Avoid printing; return the final string.

### Error handling
- Wrap network and I/O with try/except and return a helpful message. See `BraveSearchTool` and others for patterns.
- Validate required inputs and environment configuration with clear messages.
- Keep exceptions user‑friendly; do not leak stack traces.

### Rate limiting and retries
- If the upstream API enforces request pacing, implement minimal rate limiting (see `BraveSearchTool`).
- Consider idempotency and backoff for transient errors where appropriate.

### Async support
- Implement `_arun` only if your library has a true async client or your sync calls are thread‑safe.
- Otherwise, delegate `_arun` to `_run` as in multiple existing tools.

### Returning values
- Return a string (or JSON string) that’s ready to display in an agent transcript.
- If returning structured data, keep it small and human‑readable. Use stable keys and ordering.

---

## RAG tools and adapters

If your tool is a knowledge source, consider extending `RagTool` and/or creating an adapter.

- `RagTool` exposes `add(...)` and a `query(question: str) -> str` contract through an `Adapter`.
- See `crewai_tools/tools/rag/rag_tool.py` and adapters like `embedchain_adapter.py` and `lancedb_adapter.py`.

Minimal adapter example:

```python
from typing import Any
from pydantic import BaseModel
from crewai_tools.tools.rag.rag_tool import Adapter, RagTool


class MemoryAdapter(Adapter):
    store: list[str] = []

    def add(self, text: str, **_: Any) -> None:
        self.store.append(text)

    def query(self, question: str) -> str:
        # naive demo: return all text containing any word from the question
        tokens = set(question.lower().split())
        hits = [t for t in self.store if tokens & set(t.lower().split())]
        return "\n".join(hits) if hits else "No relevant content found."


class MemoryRagTool(RagTool):
    name: str = "In‑memory RAG"
    description: str = "Toy RAG that stores text in memory and returns matches."
    adapter: Adapter = MemoryAdapter()
```

When using external vector DBs (MongoDB, Qdrant, Weaviate), study the existing tools to follow indexing, embedding, and query configuration patterns closely.

---

## Toolkits (multiple related tools)

Some integrations expose a toolkit (a group of tools) rather than a single class. See Bedrock `browser_toolkit.py` and `code_interpreter_toolkit.py`.

Guidelines:
- Provide small, focused `BaseTool` classes for each operation (e.g., `navigate`, `click`, `extract_text`).
- Offer a helper `create_<name>_toolkit(...) -> Tuple[ToolkitClass, List[BaseTool]]` to create tools and manage resources.
- If you open external resources (browsers, interpreters), support cleanup methods and optionally context manager usage.

---

## Environment variables and dependencies

### env_vars
- Declare as `env_vars: List[EnvVar]` with `name`, `description`, `required`, and optional `default`.
- Validate presence in `__init__` or on first `_run` call.

### Dependencies
- List runtime packages in `package_dependencies` on the class.
- If they are genuinely optional, add an extra under `[project.optional-dependencies]` in `pyproject.toml` (e.g., `tavily-python`, `serpapi`, `scrapfly-sdk`).
- Use lazy imports to avoid hard deps for users who don’t need the tool.

---

## Testing

Place tests under `tests/tools/` and follow these rules:
- Do not hit real external services in CI. Use mocks, fakes, or recorded fixtures where allowed.
- Validate input validation, env var handling, error messages, and happy path output formatting.
- Keep tests fast and deterministic.

Example skeleton (`tests/tools/my_tool_test.py`):

```python
import os
import pytest
from crewai_tools.tools.my_tool.my_tool import MyTool


def test_requires_env_var(monkeypatch):
    monkeypatch.delenv("MY_API_KEY", raising=False)
    with pytest.raises(ValueError):
        MyTool()


def test_happy_path(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "test")
    tool = MyTool()
    result = tool.run(query="hello", limit=2)
    assert "hello" in result
```

Run locally:

```bash
uv run pytest
pre-commit run -a
```

---

## Documentation

Each tool must include a `README.md` in its folder with:
- What it does and when to use it
- Required env vars and optional extras (with install snippet)
- Minimal usage example

Update the root `README.md` only if the tool introduces a new category or notable capability.

---

## Discovery and specs

Our internal tooling discovers classes whose names end with `Tool`. Keep your class exported from the module path under `crewai_tools/tools/...` to be picked up by scripts like `crewai_tools.generate_tool_specs.py`.

---

## Full example: “Weather Search Tool”

This example demonstrates: `args_schema`, `env_vars`, `package_dependencies`, lazy imports, validation, and robust error handling.

```python
# file: crewai_tools/tools/weather_tool/weather_tool.py
from typing import Any, List, Optional, Type
import os
import requests
from pydantic import BaseModel, Field
from crewai.tools import BaseTool, EnvVar


class WeatherToolInput(BaseModel):
    """Input schema for WeatherTool."""
    city: str = Field(..., description="City name, e.g., 'Berlin'")
    country: Optional[str] = Field(None, description="ISO country code, e.g., 'DE'")
    units: str = Field(
        default="metric",
        description="Units system: 'metric' or 'imperial'",
        pattern=r"^(metric|imperial)$",
    )


class WeatherTool(BaseTool):
    name: str = "Weather Search"
    description: str = (
        "Look up current weather for a city using a public weather API."
    )
    args_schema: Type[BaseModel] = WeatherToolInput

    env_vars: List[EnvVar] = [
        EnvVar(
            name="WEATHER_API_KEY",
            description="API key for the weather service",
            required=True,
        ),
    ]
    package_dependencies: List[str] = ["requests"]

    base_url: str = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if "WEATHER_API_KEY" not in os.environ:
            raise ValueError("WEATHER_API_KEY is required for WeatherTool")

    def _run(self, city: str, country: Optional[str] = None, units: str = "metric") -> str:
        try:
            q = f"{city},{country}" if country else city
            params = {
                "q": q,
                "units": units,
                "appid": os.environ["WEATHER_API_KEY"],
            }
            resp = requests.get(self.base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            main = data.get("weather", [{}])[0].get("main", "Unknown")
            desc = data.get("weather", [{}])[0].get("description", "")
            temp = data.get("main", {}).get("temp")
            feels = data.get("main", {}).get("feels_like")
            city_name = data.get("name", city)

            return (
                f"Weather in {city_name}: {main} ({desc}). "
                f"Temperature: {temp}°, feels like {feels}°."
            )
        except requests.Timeout:
            return "Weather service timed out. Please try again later."
        except requests.HTTPError as e:
            return f"Weather service error: {e.response.status_code} {e.response.text[:120]}"
        except Exception as e:
            return f"Unexpected error fetching weather: {e}"
```

Folder layout:

```
crewai_tools/tools/weather_tool/
  ├─ weather_tool.py
  └─ README.md
```

And `README.md` should document env vars and usage.

---

## PR checklist
- [ ] Tool lives under `crewai_tools/tools/<name>/`
- [ ] Class ends with `Tool` and subclasses `BaseTool` (or `RagTool`)
- [ ] Precise `args_schema` with descriptions and validation
- [ ] `env_vars` declared (if any) and validated
- [ ] `package_dependencies` and optional extras added in `pyproject.toml` (if any)
- [ ] Clear error handling; no prints
- [ ] Unit tests added (`tests/tools/`), fast and deterministic
- [ ] Tool `README.md` with usage and env vars
- [ ] `pre-commit` and `pytest` pass locally

---

## Tips for great DX
- Keep responses short and useful—agents quote your tool output directly.
- Validate early; fail fast with actionable guidance.
- Prefer lazy imports; minimize default install surface.
- Mirror patterns from similar tools in this repo for a consistent developer experience.

Happy building!


