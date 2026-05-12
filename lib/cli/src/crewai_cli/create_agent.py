"""Create agent definitions via interactive prompts."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from crewai_cli.constants import ENV_VARS, MODELS
from crewai_cli.utils import load_env_vars, write_env_file


AGENT_TEMPLATE = """\
{{
  // Agent identity — defines the agent's persona and expertise
  // These three fields shape how the agent thinks and communicates
  "name": "{name}",

  // What this agent does (any role you want)
  "role": "{role}",

  // The agent's primary objective
  "goal": "{goal}",

  // Background context that shapes personality and approach
  "backstory": "{backstory}",

  // Which LLM powers this agent
  // Format: "provider/model" — e.g., "openai/gpt-4o", "anthropic/claude-sonnet-4-20250514"
  "llm": "{llm}",

  // Separate LLM for tool/function calls (optional, defaults to main LLM)
  // Useful for using a cheaper model for tool routing
  // "function_calling_llm": "openai/gpt-4o-mini",

  // Tools this agent can use — referenced by name from the crewai-tools package
  // See: https://docs.crewai.com/tools for available tools
  // Use "custom:tool_name" for custom tools defined in your tools/ directory
  "tools": [],

  // MCP servers — external tool servers following the Model Context Protocol
  // Can be URLs ("https://mcp.example.com") or platform slugs ("notion")
  "mcps": [],

  // Platform app integrations — managed by CrewAI Platform
  // App name ("gmail") or app/action ("gmail/send_email")
  "apps": [],

  // Coworkers — other agents this agent can delegate work to
  // {{"ref": "name"}} for local agents in agents/ directory
  // {{"amp": "handle"}} for agents from the CrewAI AMP repository (your org)
  // {{"amp": "handle", "llm": "..."}} for AMP agents with LLM override
  // {{"a2a": "url"}} for remote agents via A2A protocol
  "coworkers": [],

  // Knowledge sources — files/directories the agent can search for context
  // Supports: PDF, CSV, JSON, TXT, Excel, and directories
  "knowledge_sources": [],

  // Output guardrail — validates agent responses before sending to user
  // "type": "llm" uses an LLM to check the response against instructions
  // Remove this block to disable guardrails
  // "guardrail": {{
  //   "type": "llm",
  //   "instructions": "Never reveal internal pricing information.",
  //   "llm": "openai/gpt-4o-mini"
  // }},

  // Settings — all have sensible defaults, only override what you need
  "settings": {{
    // Agent remembers across conversations
    "memory": true,

    // Enable extended thinking / chain-of-thought
    "reasoning": true,

    // Dreaming: consolidate memories over time into canonical insights
    "self_improving": true,

    // Agent plans before complex tasks
    "planning": true,

    // Agent decides at runtime whether to plan (default: true)
    // "auto_plan": true,

    // Allow agent to spawn parallel copies for subtasks (default: true)
    // "can_spawn_copies": true,

    // How deep spawned copies can nest (default: 1)
    // "max_spawn_depth": 1,

    // Max parallel copies running at once (default: 4)
    // "max_concurrent_spawns": 4,

    // Messages sent to LLM per turn, null = all (default: null)
    // "max_history_messages": null,

    // Detect claimed-but-not-done actions (default: false)
    // "narration_guard": false,

    // Hours between dreaming cycles (default: 24)
    // "dreaming_interval_hours": 24,

    // New memories before dreaming triggers (default: 10)
    // "dreaming_trigger_threshold": 10,

    // Separate LLM for dreaming (default: uses agent's LLM)
    // "dreaming_llm": "openai/gpt-4o-mini",

    // Provenance detail level: "minimal", "standard", or "detailed"
    // "provenance_detail": "standard"
  }}
}}
"""

PROJECT_CONFIG_TEMPLATE = """\
{
  // Project configuration for crewai agents
  // Rooms define how agents collaborate in the TUI

  "rooms": {
    "common": {
      // Which agents participate in this room
      "agents": [],

      // Engagement mode:
      //   "dm" — chat with one agent at a time (default)
      //   "tagged" — @mention to direct messages
      //   "organic" — all agents see messages, respond if relevant
      "engagement": "dm"
    }
  }
}
"""


_STARTER_CASES = """\
[
  {
    "input": "Hello, what can you help me with?",
    "criteria": "The agent should clearly describe its role and capabilities."
  }
]
"""


_PROVIDER_TO_EXTRA: dict[str, str] = {
    # Native providers with dedicated SDK extras
    "anthropic": "anthropic",
    "gemini": "google-genai",
    "google": "google-genai",
    "azure": "azure-ai-inference",
    "azure_openai": "azure-ai-inference",
    "bedrock": "bedrock",
    "aws": "aws",
    # Providers that route through litellm
    "watsonx": "litellm",
    "groq": "litellm",
    "nvidia_nim": "litellm",
    "huggingface": "litellm",
    "sambanova": "litellm",
    # OpenAI-compatible providers — no extra needed:
    # openai, ollama, cerebras, deepseek, openrouter, hosted_vllm, dashscope
}

_PROVIDER_BONUS_EXTRAS: dict[str, list[str]] = {
    "watsonx": ["watson"],
}


_GITIGNORE_TEMPLATE = """\
.env
__pycache__/
.DS_Store
.crewai/
"""


def _build_pyproject(project_name: str, crewai_version: str, llm_provider: str) -> str:
    """Build pyproject.toml content with the right LLM provider extra."""
    extras = ["tools"]
    provider_extra = _PROVIDER_TO_EXTRA.get(llm_provider, "")
    if provider_extra and provider_extra not in extras:
        extras.append(provider_extra)
    for bonus in _PROVIDER_BONUS_EXTRAS.get(llm_provider, []):
        if bonus not in extras:
            extras.append(bonus)

    extras_str = ",".join(extras)

    lines = [
        "[project]",
        f'name = "{project_name}"',
        'version = "0.1.0"',
        'description = "CrewAI agent project"',
        'requires-python = ">=3.10,<3.14"',
        "dependencies = [",
        f'    "crewai[{extras_str}]>={crewai_version}",',
        f'    "crewai-cli>={crewai_version}",',
        "]",
        "",
        "[tool.uv]",
        'prerelease = "allow"',
        "constraint-dependencies = [",
        '    "onnxruntime<=1.25.1",',
        "]",
        "",
        "[tool.crewai]",
        'type = "agent"',
        "",
    ]
    return "\n".join(lines)


def _bootstrap_project(base: Path, llm_model: str = "") -> None:
    """Create project structure if it doesn't exist yet."""
    agents_dir = base / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    tools_dir = base / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    tests_dir = base / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    config_path = base / "config.json"
    if not config_path.exists():
        config_path.write_text(PROJECT_CONFIG_TEMPLATE, encoding="utf-8")

    provider = llm_model.split("/")[0].lower() if "/" in llm_model else ""
    pyproject_path = base / "pyproject.toml"
    if not pyproject_path.exists():
        crewai_version = _get_crewai_version()
        pyproject_path.write_text(
            _build_pyproject(base.name, crewai_version, provider),
            encoding="utf-8",
        )
    else:
        _maybe_add_provider_extra(pyproject_path, provider)

    gitignore_path = base / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(_GITIGNORE_TEMPLATE, encoding="utf-8")


def _maybe_add_provider_extra(pyproject_path: Path, provider: str) -> None:
    """If the pyproject.toml exists but doesn't include the provider extra, add it."""
    all_extras = []
    primary = _PROVIDER_TO_EXTRA.get(provider, "")
    if primary:
        all_extras.append(primary)
    all_extras.extend(_PROVIDER_BONUS_EXTRAS.get(provider, []))
    if not all_extras:
        return
    try:
        content = pyproject_path.read_text(encoding="utf-8")
        missing = [
            e for e in all_extras
            if f"[{e}]" not in content and f",{e}]" not in content and f",{e}," not in content
        ]
        if not missing:
            return
        import re as _re
        suffix = "," + ",".join(missing)
        def _add_extras(m: _re.Match) -> str:
            bracket = m.group(0)
            return bracket[:-1] + suffix + "]"
        updated = _re.sub(r'crewai\[[^\]]+\]', _add_extras, content, count=1)
        if updated != content:
            pyproject_path.write_text(updated, encoding="utf-8")
    except Exception:
        pass


def _get_crewai_version() -> str:
    """Get the installed crewai version for the dependency pin."""
    try:
        from crewai_cli.version import get_crewai_version
        return get_crewai_version()
    except Exception:
        return "1.14.5"


def _run_uv_sync(base: Path) -> None:
    """Run uv sync to install dependencies."""
    click.echo()
    click.secho("Installing dependencies...", fg="cyan")
    try:
        result = subprocess.run(
            ["uv", "sync"],
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            click.secho("Dependencies installed successfully.", fg="green")
        else:
            click.secho("Failed to install dependencies:", fg="red")
            if result.stderr:
                click.echo(result.stderr)
            click.echo("Try running: uv sync")
    except FileNotFoundError:
        click.secho(
            "uv not found. Install it (https://docs.astral.sh/uv/) then run: uv sync",
            fg="yellow",
        )
    except subprocess.TimeoutExpired:
        click.secho("uv sync timed out. Run manually: uv sync", fg="yellow")
    except Exception as e:
        click.secho(f"Could not run uv sync: {e}", fg="yellow")
        click.echo("Run manually: uv sync")


def _create_benchmark_cases(base: Path, agent_name: str) -> None:
    """Create a starter benchmark cases file for the agent."""
    cases_path = base / "tests" / f"{agent_name}_cases.json"
    if cases_path.exists():
        return
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(_STARTER_CASES, encoding="utf-8")


_PROVIDERS: list[tuple[str, str]] = [
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("gemini", "Google Gemini"),
    ("groq", "Groq (fast inference)"),
    ("ollama", "Ollama (local)"),
]

_PROVIDER_MODELS: dict[str, list[tuple[str, str]]] = {
    "openai": [
        ("gpt-5.5", "GPT-5.5"),
        ("gpt-5.5-pro", "GPT-5.5 Pro"),
        ("o4-mini", "o4-mini (reasoning, fast)"),
        ("o3", "o3 (reasoning)"),
        ("gpt-4.1-mini", "GPT-4.1 Mini (budget)"),
    ],
    "anthropic": [
        ("claude-opus-4-6", "Claude Opus 4.6"),
        ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
        ("claude-haiku-4-5-20251001", "Claude Haiku 4.5 (fast)"),
        ("claude-3-7-sonnet-20250219", "Claude 3.7 Sonnet"),
        ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
    ],
    "gemini": [
        ("gemini-3-pro-preview", "Gemini 3 Pro (preview)"),
        ("gemini-2.5-pro-exp-03-25", "Gemini 2.5 Pro"),
        ("gemini-2.5-flash-preview-04-17", "Gemini 2.5 Flash"),
        ("gemini-2.0-flash-001", "Gemini 2.0 Flash"),
        ("gemini-1.5-pro", "Gemini 1.5 Pro"),
    ],
    "groq": [
        ("llama-3.3-70b-versatile", "Llama 3.3 70B"),
        ("llama-3.1-70b-versatile", "Llama 3.1 70B"),
        ("llama-3.1-8b-instant", "Llama 3.1 8B (fast)"),
        ("deepseek-r1-distill-llama-70b", "DeepSeek R1 70B"),
        ("mixtral-8x7b-32768", "Mixtral 8x7B"),
    ],
    "ollama": [
        ("llama3.3", "Llama 3.3"),
        ("llama3.1", "Llama 3.1"),
        ("deepseek-r1", "DeepSeek R1"),
        ("qwen2.5", "Qwen 2.5"),
        ("mistral", "Mistral"),
    ],
}


_POPULAR_TOOLS: list[tuple[str, str]] = [
    ("SerperDevTool", "Web search via Serper API"),
    ("ScrapeWebsiteTool", "Scrape and extract content from URLs"),
    ("FileReadTool", "Read local files"),
    ("FileWriterTool", "Write content to local files"),
    ("DirectoryReadTool", "List directory contents"),
    ("CodeInterpreterTool", "Execute Python code in a sandbox"),
    ("CSVSearchTool", "Search within CSV files"),
    ("PDFSearchTool", "Search within PDF documents"),
    ("JSONSearchTool", "Search within JSON files"),
    ("GithubSearchTool", "Search GitHub repositories"),
    ("YoutubeVideoSearchTool", "Search YouTube video transcripts"),
    ("TavilySearchTool", "Web search via Tavily API"),
    ("BraveSearchTool", "Web search via Brave API"),
    ("RagTool", "RAG over custom knowledge sources"),
    ("DallETool", "Generate images with DALL-E"),
    ("VisionTool", "Analyze images with vision models"),
]


_AGENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


# ── Arrow-key selection helpers ──────────────────────────────────


_CYAN = "\033[36m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _is_interactive() -> bool:
    """Check if stdin/stdout are real terminals (not piped or in tests)."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _read_key() -> str:
    """Read a single keypress. Returns 'up', 'down', 'enter', 'space', or the char."""
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            return {"H": "up", "P": "down"}.get(ch2, "")
        if ch == "\r":
            return "enter"
        if ch == " ":
            return "space"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch

    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            return "esc"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _draw_single(labels: list[str], cursor: int, *, clear: bool = False) -> None:
    """Draw single-select menu. If clear=True, move cursor up first."""
    total = len(labels)
    if clear:
        sys.stdout.write(f"\033[{total}A")
    for i, label in enumerate(labels):
        if i == cursor:
            sys.stdout.write(f"\033[2K  {_CYAN}→{_RESET} {_BOLD}{label}{_RESET}\n")
        else:
            sys.stdout.write(f"\033[2K    {label}\n")
    sys.stdout.flush()


def _draw_multi(labels: list[str], cursor: int, selected: set[int], *, clear: bool = False) -> None:
    """Draw multi-select menu with checkboxes."""
    hint = f"  {_DIM}↑↓ navigate, space toggle, enter confirm{_RESET}"
    total = len(labels) + 1  # +1 for hint line
    if clear:
        sys.stdout.write(f"\033[{total}A")
    sys.stdout.write(f"\033[2K{hint}\n")
    for i, label in enumerate(labels):
        check = f"{_CYAN}[×]{_RESET}" if i in selected else "[ ]"
        arrow = f"{_CYAN}→{_RESET} " if i == cursor else "  "
        bold = f"{_BOLD}{label}{_RESET}" if i == cursor else label
        sys.stdout.write(f"\033[2K    {arrow}{check} {bold}\n")
    sys.stdout.flush()


def _clear_lines(n: int) -> None:
    """Clear n lines above and position cursor at the top."""
    sys.stdout.write(f"\033[{n}A")
    for _ in range(n):
        sys.stdout.write("\033[2K\n")
    sys.stdout.write(f"\033[{n}A")
    sys.stdout.flush()


def create_agent(name: str | None = None) -> None:
    """Create an agent definition interactively.

    Both paths (with and without a name) ask the same structured
    questions and produce the same annotated JSONC output.
    """
    click.secho("\nCrewAI Agent Creator\n", fg="cyan", bold=True)

    if name is None:
        name = _prompt_agent_name()

    base = Path.cwd()
    # Directories are bootstrapped now, pyproject written after model selection
    for d in ("agents", "tools", "tests"):
        (base / d).mkdir(parents=True, exist_ok=True)

    dest = base / "agents" / f"{name}.jsonc"
    if dest.exists():
        if not click.confirm(f"File {dest} already exists. Overwrite?"):
            click.secho("Operation cancelled.", fg="yellow")
            return

    click.secho(f"Configuring agent: {name}\n", fg="cyan")

    role = click.prompt("  Role (what this agent does)", type=str)
    goal = click.prompt("  Goal (the agent's objective)", type=str)
    backstory = click.prompt(
        "  Backstory (context that shapes personality, optional)",
        type=str, default="", show_default=False,
    )

    llm = _select_model()

    tools = _select_tools()

    content = AGENT_TEMPLATE.format(
        name=name,
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
    )

    if tools:
        tools_json = json.dumps(tools)
        content = content.replace('"tools": []', f'"tools": {tools_json}')

    dest.write_text(content, encoding="utf-8")
    _bootstrap_project(base, llm)
    _add_agent_to_config(base, name)
    _create_benchmark_cases(base, name)
    _setup_env(base, llm)
    _run_uv_sync(base)

    click.echo()
    click.secho(f"Agent created: {dest}", fg="green", bold=True)
    click.echo("Run: crewai run")


def _select_model() -> str:
    """Two-step selection: provider first, then model."""
    # Step 1: Pick provider
    provider_labels = [label for _, label in _PROVIDERS]
    provider_labels.append("Other (enter manually)")

    click.echo()
    click.secho("  LLM Provider:", fg="cyan")
    p_idx = _arrow_or_fallback(provider_labels)

    if p_idx == len(_PROVIDERS):
        custom = click.prompt("  Enter model (provider/model)", type=str)
        return custom.strip()

    provider_key, provider_name = _PROVIDERS[p_idx]
    click.secho(f"  → {provider_name}", fg="green")

    # Step 2: Pick model from that provider
    models = _PROVIDER_MODELS.get(provider_key, [])
    model_labels = [f"{label}  ({model_id})" for model_id, label in models]
    model_labels.append("Other (enter model name)")

    click.echo()
    click.secho(f"  {provider_name} Model:", fg="cyan")
    m_idx = _arrow_or_fallback(model_labels)

    if m_idx == len(models):
        custom = click.prompt(f"  Enter model name for {provider_key}/", type=str)
        result = f"{provider_key}/{custom.strip()}"
    else:
        model_id = models[m_idx][0]
        result = f"{provider_key}/{model_id}"

    click.secho(f"  → {result}", fg="green")
    return result


def _arrow_or_fallback(labels: list[str]) -> int:
    """Arrow-key select if interactive, numbered fallback otherwise."""
    if _is_interactive():
        try:
            return _arrow_select_one(labels)
        except Exception:
            pass
    return _numbered_select(labels)


def _arrow_select_one(labels: list[str]) -> int:
    """Arrow-key single-select. Returns selected index."""
    cursor = 0
    total = len(labels)
    _draw_single(labels, cursor)
    while True:
        key = _read_key()
        if key == "up" and cursor > 0:
            cursor -= 1
            _draw_single(labels, cursor, clear=True)
        elif key == "down" and cursor < total - 1:
            cursor += 1
            _draw_single(labels, cursor, clear=True)
        elif key == "enter":
            _clear_lines(total)
            return cursor


def _numbered_select(labels: list[str]) -> int:
    """Numbered fallback for non-TTY environments."""
    for idx, label in enumerate(labels, 1):
        click.echo(f"    {idx}. {label}")
    click.echo()
    while True:
        choice = click.prompt("  Select", type=str, default="1")
        try:
            num = int(choice)
            if 1 <= num <= len(labels):
                return num - 1
        except ValueError:
            pass
        click.secho(f"  Invalid choice. Enter 1-{len(labels)}.", fg="red")


def _select_tools() -> list[str]:
    """Let the user pick tools from popular options and/or add custom ones."""
    labels = [f"{cls_name:<28s} {desc}" for cls_name, desc in _POPULAR_TOOLS]
    labels.append("Add custom tool class names")

    click.echo()
    click.secho("  Tools (space to select, enter to confirm):", fg="cyan")

    if _is_interactive():
        try:
            indices = _select_tools_interactive(labels)
        except Exception:
            indices = _select_tools_fallback(labels)
    else:
        indices = _select_tools_fallback(labels)

    selected: list[str] = []
    has_custom = False
    for idx in indices:
        if idx == len(_POPULAR_TOOLS):
            has_custom = True
        elif 0 <= idx < len(_POPULAR_TOOLS):
            cls_name = _POPULAR_TOOLS[idx][0]
            if cls_name not in selected:
                selected.append(cls_name)

    if has_custom:
        custom = click.prompt(
            "  Custom tool class names (comma-separated)",
            type=str, default="", show_default=False,
        )
        for name in custom.split(","):
            name = name.strip()
            if name and name not in selected:
                selected.append(name)

    if selected:
        click.secho(f"  → {', '.join(selected)}", fg="green")
    return selected


def _select_tools_interactive(labels: list[str]) -> list[int]:
    """Arrow-key multi-select for tools."""
    cursor = 0
    chosen: set[int] = set()
    total_lines = len(labels) + 1  # +1 for hint line

    _draw_multi(labels, cursor, chosen)

    while True:
        key = _read_key()
        if key == "up" and cursor > 0:
            cursor -= 1
            _draw_multi(labels, cursor, chosen, clear=True)
        elif key == "down" and cursor < len(labels) - 1:
            cursor += 1
            _draw_multi(labels, cursor, chosen, clear=True)
        elif key == "space":
            if cursor in chosen:
                chosen.discard(cursor)
            else:
                chosen.add(cursor)
            _draw_multi(labels, cursor, chosen, clear=True)
        elif key == "enter":
            _clear_lines(total_lines)
            return sorted(chosen)


def _select_tools_fallback(labels: list[str]) -> list[int]:
    """Numbered fallback for non-TTY environments."""
    for idx, label in enumerate(labels, 1):
        click.echo(f"    {idx:2d}. {label}")
    click.echo()

    raw = click.prompt(
        "  Select tools (e.g. 1 3 5)", type=str, default="", show_default=False,
    )
    if not raw.strip():
        return []

    indices: list[int] = []
    for token in raw.split():
        try:
            num = int(token)
            if 1 <= num <= len(labels):
                indices.append(num - 1)
        except ValueError:
            pass
    return indices


def _setup_env(base: Path, llm_model: str) -> None:
    """Prompt for API keys based on the selected LLM provider and write .env."""
    env_vars = load_env_vars(base)

    provider = llm_model.split("/")[0].lower() if "/" in llm_model else ""
    if not provider:
        return

    env_vars["MODEL"] = llm_model

    already_set = all(
        details.get("key_name", "") in env_vars
        for details in ENV_VARS.get(provider, [])
        if "key_name" in details
    )
    if already_set and env_vars.get("MODEL"):
        return

    if provider in ENV_VARS:
        click.echo()
        for details in ENV_VARS[provider]:
            key_name = details.get("key_name")
            if not key_name or key_name in env_vars:
                continue
            if details.get("default"):
                env_vars[key_name] = details.get("API_BASE", "")
                continue
            value = click.prompt(
                f"  {details.get('prompt', f'Enter {key_name}')}",
                default="", show_default=False,
            )
            if value.strip():
                env_vars[key_name] = value.strip()

    if env_vars:
        write_env_file(base, env_vars)
        click.secho("API keys saved to .env", fg="green")
    else:
        click.secho(
            "No API keys provided. Create a .env file manually before running.",
            fg="yellow",
        )


def _prompt_agent_name() -> str:
    """Prompt for a valid agent identifier."""
    while True:
        name = click.prompt(
            "  Agent identifier (lowercase, hyphens/underscores, no spaces)",
            type=str,
        )
        name = name.strip().lower()
        if _AGENT_NAME_RE.match(name):
            return name
        click.secho(
            "  Invalid name — use lowercase letters, numbers, hyphens, or underscores.",
            fg="red",
        )


def _strip_comments(text: str) -> str:
    """Strip // and /* */ comments from JSONC text, then fix trailing commas."""
    result = re.sub(r'(?<!:)//.*?$', '', text, flags=re.MULTILINE)
    result = re.sub(r'/\*.*?\*/', '', result, flags=re.DOTALL)
    result = re.sub(r',\s*([}\]])', r'\1', result)
    return result


def _add_agent_to_config(base: Path, agent_name: str) -> None:
    """Add the agent to the common room in config.json."""
    config_path = base / "config.json"
    if not config_path.exists():
        return

    try:
        raw = config_path.read_text(encoding="utf-8")
        clean = _strip_comments(raw)
        config = json.loads(clean)

        rooms = config.get("rooms", {})
        common = rooms.get("common", {"agents": [], "engagement": "dm"})
        agents = common.get("agents", [])
        if agent_name not in agents:
            agents.append(agent_name)
            common["agents"] = agents
            rooms["common"] = common
            config["rooms"] = rooms
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except Exception as e:
        click.echo(f"Warning: Could not update config.json: {e}", err=True)
