"""Scaffold a new JSON-first crew project."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any

import click
from rich.console import Console
from rich.text import Text

from crewai_cli.constants import ENV_VARS
from crewai_cli.tui_picker import pick_many, pick_one
from crewai_cli.utils import (
    enable_prompt_line_editing,
    is_dmn_mode_enabled,
    load_env_vars,
    render_template,
    write_env_file,
)
from crewai_cli.version import get_crewai_tools_dependency


# ── Provider / model data ───────────────────────────────────────

_PROVIDERS: list[tuple[str, str]] = [
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("gemini", "Google Gemini"),
    ("groq", "Groq"),
    ("ollama", "Ollama"),
    ("bedrock", "AWS Bedrock"),
    ("azure", "Azure OpenAI"),
    ("nvidia_nim", "NVIDIA NIM"),
    ("huggingface", "Hugging Face"),
    ("cerebras", "Cerebras"),
    ("sambanova", "SambaNova"),
    ("watson", "IBM watsonx"),
]

_PROVIDER_MODELS: dict[str, list[tuple[str, str]]] = {
    "openai": [
        ("gpt-5.5", "GPT-5.5"),
        ("gpt-5.5-pro", "GPT-5.5 Pro"),
        ("gpt-5.4", "GPT-5.4"),
        ("o4-mini", "o4-mini"),
        ("gpt-4.1", "GPT-4.1"),
        ("gpt-4.1-mini", "GPT-4.1 Mini"),
    ],
    "anthropic": [
        ("claude-opus-4-6", "Claude Opus 4.6"),
        ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
        ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
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
        ("llama-3.1-8b-instant", "Llama 3.1 8B"),
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

_TEMPLATES_DIR = Path(__file__).parent / "templates" / "json_crew"


# ── Common tools for picker ────────────────────────────────────

_TOOL_CATEGORIES: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Search & Research",
        [
            ("SerperDevTool", "Google search via Serper API"),
            ("BraveSearchTool", "Web search via Brave Search"),
            ("BraveWebSearchTool", "Focused Brave web search"),
            ("BraveNewsSearchTool", "Search current news with Brave"),
            ("BraveImageSearchTool", "Search images with Brave"),
            ("BraveVideoSearchTool", "Search videos with Brave"),
            ("BraveLocalPOIsTool", "Find local places with Brave"),
            ("BraveLocalPOIsDescriptionTool", "Describe local places with Brave"),
            ("BraveLLMContextTool", "Fetch Brave search context"),
            ("TavilySearchTool", "Web search via Tavily"),
            ("TavilyResearchTool", "Run Tavily research"),
            ("TavilyGetResearchTool", "Retrieve Tavily research results"),
            ("TavilyExtractorTool", "Extract content with Tavily"),
            ("EXASearchTool", "Semantic web search via Exa"),
            ("ExaSearchTool", "Semantic web search via Exa"),
            ("LinkupSearchTool", "Web search via Linkup"),
            ("SerpApiGoogleSearchTool", "Google search via SerpApi"),
            ("SerpApiGoogleShoppingTool", "Google Shopping via SerpApi"),
            ("SerplyWebSearchTool", "Web search via Serply"),
            ("SerplyNewsSearchTool", "News search via Serply"),
            ("SerplyScholarSearchTool", "Scholar search via Serply"),
            ("SerplyJobSearchTool", "Job search via Serply"),
            ("SerplyWebpageToMarkdownTool", "Convert webpages with Serply"),
            ("ParallelSearchTool", "Run parallel web searches"),
            ("BrightDataSearchTool", "Search with Bright Data"),
            ("GithubSearchTool", "Search GitHub repositories"),
            ("ArxivPaperTool", "Search arXiv academic papers"),
        ],
    ),
    (
        "Web Scraping",
        [
            ("ScrapeWebsiteTool", "Extract content from a URL"),
            ("ScrapeElementFromWebsiteTool", "Extract page elements from a URL"),
            ("FirecrawlScrapeWebsiteTool", "Scrape with Firecrawl"),
            ("FirecrawlCrawlWebsiteTool", "Crawl a website with Firecrawl"),
            ("FirecrawlSearchTool", "Search with Firecrawl"),
            ("SeleniumScrapingTool", "Browser-based scraping"),
            ("JinaScrapeWebsiteTool", "Scrape with Jina"),
            ("ScrapegraphScrapeTool", "AI-powered page scraping"),
            ("SerperScrapeWebsiteTool", "Scrape pages with Serper"),
            ("BrowserbaseLoadTool", "Load web pages with Browserbase"),
            ("HyperbrowserLoadTool", "Load web pages with Hyperbrowser"),
            ("MultiOnTool", "Control web workflows with MultiOn"),
            ("SpiderTool", "Crawl websites with Spider"),
            ("StagehandTool", "Browser automation with Stagehand"),
            ("BrightDataWebUnlockerTool", "Unlock websites with Bright Data"),
            ("BrightDataDatasetTool", "Fetch Bright Data datasets"),
            ("WebsiteSearchTool", "RAG search on a website"),
        ],
    ),
    (
        "File & Document",
        [
            ("DirectoryReadTool", "List directory contents"),
            ("DirectorySearchTool", "Search directory contents"),
            ("FileReadTool", "Read local files"),
            ("FileWriterTool", "Write to local files"),
            ("FileCompressorTool", "Compress local files"),
            ("CSVSearchTool", "Search within CSV files"),
            ("PDFSearchTool", "Search within PDF files"),
            ("DOCXSearchTool", "Search within DOCX files"),
            ("MDXSearchTool", "Search within MDX files"),
            ("JSONSearchTool", "Search within JSON files"),
            ("TXTSearchTool", "Search within text files"),
            ("XMLSearchTool", "Search within XML files"),
            ("OCRTool", "Extract text with OCR"),
            ("YoutubeVideoSearchTool", "Search within YouTube videos"),
            ("YoutubeChannelSearchTool", "Search within YouTube channels"),
        ],
    ),
    (
        "Code & Data",
        [
            ("CodeDocsSearchTool", "Search code documentation"),
            ("RagTool", "RAG over custom data sources"),
            ("NL2SQLTool", "Natural language to SQL queries"),
            ("DatabricksQueryTool", "Query Databricks data"),
            ("SingleStoreSearchTool", "Search SingleStore data"),
        ],
    ),
    (
        "Cloud & Storage",
        [
            ("S3ReaderTool", "Read objects from Amazon S3"),
            ("S3WriterTool", "Write objects to Amazon S3"),
            ("BedrockInvokeAgentTool", "Invoke an Amazon Bedrock agent"),
            ("BedrockKBRetrieverTool", "Retrieve from Bedrock knowledge bases"),
        ],
    ),
    (
        "Sandbox & Automation",
        [
            ("E2BExecTool", "Run commands in E2B"),
            ("E2BFileTool", "Manage files in E2B"),
            ("E2BPythonTool", "Run Python in E2B"),
            ("DaytonaExecTool", "Run commands in Daytona"),
            ("DaytonaFileTool", "Manage files in Daytona"),
            ("DaytonaPythonTool", "Run Python in Daytona"),
            ("GenerateCrewaiAutomationTool", "Generate CrewAI automations"),
        ],
    ),
    (
        "AI & Vision",
        [
            ("DallETool", "Generate images with DALL-E"),
            ("VisionTool", "Analyze images with vision models"),
            ("AIMindTool", "Connect to MindStudio agents"),
            ("PatronusEvalTool", "Evaluate output with Patronus"),
            ("PatronusLocalEvaluatorTool", "Run local Patronus evaluations"),
        ],
    ),
]

_FLAT_TOOLS: list[tuple[str, str]] = [
    tool for _cat, tools in _TOOL_CATEGORIES for tool in tools
]

_COMMON_TOOL_ORDER = [
    "SerperDevTool",
    "ScrapeWebsiteTool",
    "DirectoryReadTool",
    "FileReadTool",
    "FileWriterTool",
]

_ANSI_SEQUENCE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


# ── Interactive wizard ─────────────────────────────────────────


def _prompt_text(
    label: str,
    default: str = "",
    *,
    spacing_before: bool = True,
) -> str:
    if spacing_before:
        click.echo()

    prompt = click.style(f"  {label}", fg="cyan")
    if default:
        prompt += f" [{default}]"
    prompt += click.style(" > ", fg="bright_white")

    try:
        value = input(_readline_safe_prompt(prompt))
    except (KeyboardInterrupt, EOFError):
        raise click.Abort() from None

    if not value and default:
        value = default
    return value.strip()


def _readline_safe_prompt(prompt: str) -> str:
    if not sys.stdin.isatty():
        return prompt

    try:
        import readline  # noqa: F401
    except ImportError:
        return prompt

    return _ANSI_SEQUENCE_RE.sub(lambda match: f"\001{match.group(0)}\002", prompt)


def _confirm(label: str, default: bool = False) -> bool:
    click.echo()
    return click.confirm(
        click.style(f"  {label}", fg="cyan"),
        default=default,
        prompt_suffix=click.style(" > ", fg="bright_white"),
    )


def _success(message: str, *, bold: bool = False, dim: bool = False) -> None:
    click.echo()
    click.secho(f"  ✔ {message}", fg="green", bold=bold, dim=dim)


def _highlight_placeholders(text: str) -> Text:
    highlighted = Text(text, style="dim")
    highlighted.highlight_regex(r"\{[A-Za-z_][A-Za-z0-9_]*\}", style="bold cyan")
    return highlighted


def _show_interpolation_hint(kind: str) -> None:
    console = Console()
    console.print(
        _highlight_placeholders(
            "  Tip: Use {placeholder} for dynamic values you want to change later."
        )
    )


def _tool_label(name: str, description: str) -> str:
    return f"{description:<48s} {name}"


def _tool_category_label(category: str) -> str:
    return f"── {category} ──"


def _category_row_label(
    category: str, tools: list[tuple[str, str]], selected: set[str], expanded: bool
) -> str:
    """Render an accordion category row with tool/selection counts."""
    marker = "▾" if expanded else "▸"
    sel_count = sum(1 for name, _desc in tools if name in selected)
    suffix = f"{len(tools)} tools"
    if sel_count:
        suffix += f", {sel_count} selected"
    return f"{marker} {category}  ({suffix})"


def _select_tools() -> list[str]:
    """Accordion tool picker.

    Common tools are always visible at the top; every other category shows
    as a single expandable row. Expanding one category collapses the others.
    Selections persist while expanding/collapsing.
    """
    tools_by_name = {name: desc for name, desc in _FLAT_TOOLS}
    common_tools = [
        (name, tools_by_name[name])
        for name in _COMMON_TOOL_ORDER
        if name in tools_by_name
    ]
    common_tool_names = {name for name, _desc in common_tools}

    categories: list[tuple[str, list[tuple[str, str]]]] = []
    for category, category_tools in _TOOL_CATEGORIES:
        remaining_tools = [
            (name, desc)
            for name, desc in category_tools
            if name not in common_tool_names
        ]
        if remaining_tools:
            categories.append((category, remaining_tools))

    selected: set[str] = set()
    expanded: str | None = None
    focus_category: str | None = None

    while True:
        labels: list[str] = []
        tool_by_index: dict[int, str] = {}
        separator_indices: set[int] = set()
        action_indices: set[int] = set()
        category_by_index: dict[int, str] = {}
        preselected: set[int] = set()
        initial_cursor: int | None = None

        separator_indices.add(len(labels))
        labels.append(_tool_category_label("Common tools"))
        for name, desc in common_tools:
            if name in selected:
                preselected.add(len(labels))
            tool_by_index[len(labels)] = name
            labels.append(_tool_label(name, desc))

        for category, category_tools in categories:
            row = len(labels)
            action_indices.add(row)
            category_by_index[row] = category
            is_expanded = category == expanded
            if category == focus_category:
                initial_cursor = row
            labels.append(
                _category_row_label(category, category_tools, selected, is_expanded)
            )
            if is_expanded:
                for name, desc in category_tools:
                    if name in selected:
                        preselected.add(len(labels))
                    tool_by_index[len(labels)] = name
                    labels.append(_tool_label(name, desc))

        indices, action = pick_many(
            "Tools (space to toggle, enter to confirm):",
            labels,
            action_indices=action_indices,
            separator_indices=separator_indices,
            preselected=preselected,
            initial_cursor=initial_cursor,
        )

        # Carry over toggles made on this screen; tools not visible in this
        # render keep their previous state.
        visible = set(tool_by_index.values())
        chosen = {tool_by_index[i] for i in indices if i in tool_by_index}
        selected = (selected - visible) | chosen

        if action is None:
            break
        toggled = category_by_index.get(action)
        focus_category = toggled
        expanded = None if toggled == expanded else toggled

    ordered = [name for name, _desc in common_tools] + [
        name for _cat, cat_tools in categories for name, _desc in cat_tools
    ]
    return [name for name in ordered if name in selected]


def _wizard_agent(
    agent_num: int,
    existing_names: list[str],
    skip_provider: bool = False,
    last_llm: str | None = None,
    preset_llm: str | None = None,
) -> dict[str, Any] | None:
    """Interactive wizard for one agent. Returns agent dict or None if skipped."""
    click.echo()
    click.secho(f"  Agent {agent_num}", fg="cyan", bold=True)

    role = _prompt_text("Role", spacing_before=False)
    if not role:
        return None

    name_default = role.lower().replace(" ", "_")[:30]
    name_default = re.sub(r"[^a-z0-9_]", "", name_default)
    if not name_default:
        # Roles made only of symbols would otherwise produce an empty slug
        # and an invalid agents/.jsonc file name.
        name_default = f"agent_{agent_num}"
    while name_default in existing_names:
        name_default += "_2"

    goal = _prompt_text("Goal", spacing_before=False)

    backstory = _prompt_text("Backstory", spacing_before=False)

    # LLM model
    if preset_llm:
        llm = preset_llm
        _success(llm)
    elif skip_provider:
        llm = last_llm or "openai/gpt-4o"
    elif last_llm:
        reuse_labels = [
            f"Same as before  ({last_llm})",
            "Choose a different model",
        ]
        r_idx = pick_one("LLM:", reuse_labels)
        if r_idx == 1:
            llm = _select_model()
        else:
            llm = last_llm
        _success(llm)
    else:
        llm = _select_model()

    tools = _select_tools()
    if tools:
        _success(f"{len(tools)} tool{'s' if len(tools) != 1 else ''}")
    else:
        _success("No tools", dim=True)

    # Planning
    planning = _confirm("Enable step-by-step planning?", default=False)

    # Allow delegation
    allow_delegation = _confirm("Allow delegation to other agents?", default=False)

    return {
        "name": name_default,
        "role": role,
        "goal": goal,
        "backstory": backstory,
        "llm": llm,
        "tools": tools,
        "planning": planning,
        "allow_delegation": allow_delegation,
    }


def _wizard_task(
    task_num: int,
    agent_names: list[str],
    prior_task_names: list[str],
) -> dict[str, Any] | None:
    """Interactive wizard for one task. Returns task dict or None if skipped."""
    click.echo()
    click.secho(f"  Task {task_num}", fg="cyan", bold=True)

    description = _prompt_text("Description", spacing_before=False)
    if not description:
        return None

    # Auto-generate name from first few words of description
    words = description.lower().split()[:4]
    base = re.sub(r"[^a-z0-9_]", "", "_".join(words))
    name = f"{base}_task" if base else f"task_{task_num}"
    while name in prior_task_names:
        name += "_2"

    expected_output = _prompt_text("Expected output", spacing_before=False)

    # Agent assignment
    if len(agent_names) == 1:
        assigned_agent = agent_names[0]
    else:
        a_idx = pick_one("Assign to agent:", agent_names)
        while a_idx < 0:
            click.secho("  Every task needs an agent — pick one to continue.", dim=True)
            a_idx = pick_one("Assign to agent:", agent_names)
        assigned_agent = agent_names[a_idx]
        _success(f"Agent: {assigned_agent}")

    # Context dependencies
    context: list[str] = []
    if prior_task_names:
        ctx_indices = pick_many(
            "Context from prior tasks (space to toggle):",
            [*prior_task_names, "None"],
        )
        context = [
            prior_task_names[i] for i in ctx_indices if i < len(prior_task_names)
        ]
        if context:
            _success(f"Context: {', '.join(context)}")

    return {
        "name": name,
        "description": description,
        "expected_output": expected_output,
        "agent": assigned_agent,
        "context": context,
    }


def _wizard_agents_and_tasks(
    skip_provider: bool = False,
    default_llm: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the full interactive wizard. Returns (agents, tasks, crew_settings)."""
    agents: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []

    # ── Step 1: Agents ──
    click.echo()
    click.secho("  Step 1/3 — Agents", fg="cyan", bold=True)
    click.secho("  Define the AI agents in your crew.", dim=True)
    _show_interpolation_hint("agents")

    while True:
        last_llm = agents[-1]["llm"] if agents else None
        agent = _wizard_agent(
            agent_num=len(agents) + 1,
            existing_names=[a["name"] for a in agents],
            skip_provider=skip_provider,
            last_llm=last_llm,
            preset_llm=default_llm if not agents else None,
        )
        if agent is None and not agents:
            click.secho("  Need at least one agent.", fg="yellow")
            continue
        if agent is not None:
            agents.append(agent)
            _success(f"{agent['role']} added", bold=True)

        if not _confirm("Add another agent?", default=False):
            break

    # ── Step 2: Tasks ──
    click.echo()
    click.secho("  Step 2/3 — Tasks", fg="cyan", bold=True)
    click.secho("  Define what your agents should do.", dim=True)
    _show_interpolation_hint("tasks")

    agent_names = [a["name"] for a in agents]
    task_names: list[str] = []

    while True:
        task = _wizard_task(
            task_num=len(tasks) + 1,
            agent_names=agent_names,
            prior_task_names=task_names,
        )
        if task is None and not tasks:
            click.secho("  Need at least one task.", fg="yellow")
            continue
        if task is not None:
            tasks.append(task)
            task_names.append(task["name"])
            _success(f"Task {len(tasks)} added", bold=True)

        if not _confirm("Add another task?", default=False):
            break

    # ── Step 3: Settings ──
    click.echo()
    click.secho("  Step 3/3 — Settings", fg="cyan", bold=True)

    process = "sequential"
    memory = _confirm("Enable crew memory?", default=True)

    crew_settings = {
        "process": process,
        "memory": memory,
        "inputs": {},
    }

    return agents, tasks, crew_settings


def _default_agents_and_tasks(
    default_llm: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return deterministic scaffold data for non-interactive project creation."""
    llm = default_llm or "openai/gpt-4o"
    agents = [
        {
            "name": "researcher",
            "role": "Senior Researcher",
            "goal": "Research the requested topic and identify useful findings.",
            "backstory": (
                "You are an experienced researcher who finds relevant information "
                "and presents it clearly."
            ),
            "llm": llm,
            "tools": [],
            "planning": False,
            "allow_delegation": False,
        }
    ]
    tasks = [
        {
            "name": "research_task",
            "description": "Research current AI trends and write a concise summary.",
            "expected_output": "A concise markdown report with key findings.",
            "agent": "researcher",
            "context": [],
        }
    ]
    crew_settings = {
        "process": "sequential",
        "memory": True,
        "inputs": {},
    }
    return agents, tasks, crew_settings


# ── JSONC generation from wizard data ──────────────────────────


def _agent_to_jsonc(agent: dict[str, Any]) -> str:
    """Convert agent wizard data to JSONC string with comments."""
    has_planning = agent["planning"]
    settings_block = _render_json_crew_template(
        "agent_settings.jsonc",
        {
            "allow_delegation": "true" if agent["allow_delegation"] else "false",
            "delegation_comma": "," if has_planning else "",
            "planning_line": '"planning": true'
            if has_planning
            else '// "planning": false',
        },
    )

    return _render_json_crew_template(
        "agent.jsonc",
        {
            "role_json": json.dumps(agent["role"]),
            "goal_json": json.dumps(agent["goal"]),
            "backstory_json": json.dumps(agent["backstory"]),
            "llm_json": json.dumps(agent["llm"]),
            "tools_json": json.dumps(agent["tools"]),
            "settings_block": settings_block,
        },
    )


def _task_to_json_fragment(task: dict[str, Any]) -> str:
    """Convert task wizard data to a JSON-like fragment for embedding in crew JSONC."""
    has_context = bool(task.get("context"))
    has_output_file = bool(task.get("output_file"))
    context_block = ""
    output_file_block = ""

    if has_context:
        context_block = (
            "\n\n"
            "      // Task outputs used as context\n"
            f'      "context": {json.dumps(task["context"])}'
            f"{',' if has_output_file else ''}"
        )

    if has_output_file:
        output_file_block = (
            "\n\n"
            "      // Save output to a file\n"
            f'      "output_file": {json.dumps(task["output_file"])}'
        )

    return _render_json_crew_template(
        "task.jsonc",
        {
            "name_json": json.dumps(task["name"]),
            "description_json": json.dumps(task["description"]),
            "expected_output_json": json.dumps(task["expected_output"]),
            "agent_json": json.dumps(task["agent"]),
            "agent_comma": "," if has_context or has_output_file else "",
            "context_block": context_block,
            "output_file_block": output_file_block,
        },
    )


def _crew_to_jsonc(
    name: str,
    agents: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    settings: dict[str, Any],
) -> str:
    """Generate the full crew.jsonc from wizard data."""
    agent_names_json = json.dumps([a["name"] for a in agents])
    tasks_fragments = ",\n".join(_task_to_json_fragment(t) for t in tasks)
    inputs_json = json.dumps(settings.get("inputs", {}), indent=4)
    # Re-indent inputs to 4-space
    inputs_lines = inputs_json.split("\n")
    if len(inputs_lines) > 1:
        inputs_json = (
            inputs_lines[0] + "\n" + "\n".join("  " + line for line in inputs_lines[1:])
        )

    memory = "true" if settings.get("memory") else "false"

    return _render_json_crew_template(
        "crew.jsonc",
        {
            "name_json": json.dumps(name),
            "agent_names_json": agent_names_json,
            "tasks_fragments": tasks_fragments,
            "process_json": json.dumps(settings.get("process", "sequential")),
            "memory": memory,
            "manager_agent_name": agents[0]["name"],
            "inputs_json": inputs_json,
        },
    )


# ── Model selection ─────────────────────────────────────────────


def _select_model() -> str:
    """Two-step arrow-key selection: provider, then model."""
    provider_labels = [label for _, label in _PROVIDERS]
    provider_labels.append("Other (enter manually)")

    p_idx = pick_one("LLM Provider:", provider_labels)
    if p_idx < 0:
        return "openai/gpt-4o"

    if p_idx == len(_PROVIDERS):
        custom: str = click.prompt(
            click.style("  Enter model (provider/model)", fg="cyan"),
            type=str,
            prompt_suffix=click.style(" > ", fg="bright_white"),
        )
        return custom.strip()

    provider_key, provider_name = _PROVIDERS[p_idx]
    click.secho(f"  → {provider_name}", fg="green")

    models = _PROVIDER_MODELS.get(provider_key, [])
    if not models:
        custom = click.prompt(
            click.style(f"  Enter model name for {provider_key}/", fg="cyan"),
            type=str,
            prompt_suffix=click.style(" > ", fg="bright_white"),
        )
        return f"{provider_key}/{custom.strip()}"

    model_labels = [f"{label}  ({model_id})" for model_id, label in models]
    model_labels.append("Other (enter model name)")

    m_idx = pick_one(f"{provider_name} Model:", model_labels)
    if m_idx < 0:
        return f"{provider_key}/{models[0][0]}"

    if m_idx == len(models):
        custom = click.prompt(
            click.style(f"  Enter model name for {provider_key}/", fg="cyan"),
            type=str,
            prompt_suffix=click.style(" > ", fg="bright_white"),
        )
        result = f"{provider_key}/{custom.strip()}"
    else:
        model_id = models[m_idx][0]
        result = f"{provider_key}/{model_id}"

    click.secho(f"  → {result}", fg="green")
    return result


def _default_model_for_provider(provider: str | None) -> str | None:
    """Return the default provider/model string for a ``--provider`` value."""
    if not provider:
        return None
    normalized = provider.strip().lower()
    if not normalized:
        return None
    if "/" in normalized:
        return normalized
    models = _PROVIDER_MODELS.get(normalized)
    if not models:
        return None
    return f"{normalized}/{models[0][0]}"


# ── Helpers ─────────────────────────────────────────────────────


def _render_json_crew_template(
    template_name: str, replacements: dict[str, str] | None = None
) -> str:
    return render_template(_TEMPLATES_DIR / template_name, replacements or {})


def _write_jsonc(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _setup_env(folder_path: Path, llm_model: str) -> None:
    """Prompt for API keys based on the selected provider."""
    click.echo()
    env_vars = load_env_vars(folder_path)
    env_vars["MODEL"] = llm_model

    provider = llm_model.split("/")[0] if "/" in llm_model else llm_model
    if provider in ENV_VARS:
        for details in ENV_VARS[provider]:
            if details.get("default", False):
                for key, value in details.items():
                    if key not in ["prompt", "key_name", "default"]:
                        env_vars[key] = value
            elif "key_name" in details:
                api_key_value = click.prompt(
                    click.style(f"  {details['prompt']}", fg="cyan"),
                    default="",
                    show_default=False,
                    prompt_suffix=click.style(" > ", fg="bright_white"),
                )
                if api_key_value.strip():
                    env_vars[details["key_name"]] = api_key_value

    if env_vars:
        write_env_file(folder_path, env_vars)
        click.secho("  API keys and model saved to .env file", fg="green")


# ── Main ────────────────────────────────────────────────────────


def create_json_crew(
    name: str,
    provider: str | None = None,
    skip_provider: bool = False,
) -> None:
    """Scaffold a new JSON-first crew project."""
    import keyword
    import shutil

    dmn_mode = is_dmn_mode_enabled()
    if not dmn_mode:
        enable_prompt_line_editing()

    name = name.rstrip("/")
    if not name.strip():
        raise ValueError("Project name cannot be empty")

    folder_name = name.replace(" ", "_").replace("-", "_").lower()
    folder_name = re.sub(r"[^a-zA-Z0-9_]", "", folder_name)

    if not folder_name or folder_name[0].isdigit():
        raise ValueError(
            f"Project name '{name}' produces invalid folder name '{folder_name}'"
        )

    if keyword.iskeyword(folder_name):
        raise ValueError(f"'{folder_name}' is a reserved Python keyword")

    folder_path = Path(folder_name)
    if folder_path.exists():
        if dmn_mode:
            raise click.ClickException(f"Folder {folder_name} already exists.")
        if not click.confirm(f"Folder {folder_name} already exists. Override?"):
            click.secho("Cancelled.", fg="yellow")
            sys.exit(0)
        shutil.rmtree(folder_path)

    click.echo()
    click.secho(f"  Creating crew: {name}", fg="green", bold=True)

    default_llm = _default_model_for_provider(provider)
    if dmn_mode:
        agents, tasks, crew_settings = _default_agents_and_tasks(default_llm)
    else:
        agents, tasks, crew_settings = _wizard_agents_and_tasks(
            skip_provider=skip_provider,
            default_llm=default_llm,
        )

    # Create directories
    folder_path.mkdir(parents=True)
    (folder_path / "agents").mkdir()
    (folder_path / "tools").mkdir()
    (folder_path / "skills").mkdir()
    (folder_path / "knowledge").mkdir()

    for agent in agents:
        _write_jsonc(
            folder_path / "agents" / f"{agent['name']}.jsonc",
            _agent_to_jsonc(agent),
        )

    _write_jsonc(
        folder_path / "crew.jsonc",
        _crew_to_jsonc(name, agents, tasks, crew_settings),
    )

    # Write pyproject.toml
    (folder_path / "pyproject.toml").write_text(
        _render_json_crew_template(
            "pyproject.toml",
            {
                "folder_name": folder_name,
                "name": name,
                "crewai_tools_dependency": get_crewai_tools_dependency(),
            },
        ),
        encoding="utf-8",
    )

    # Write .gitignore
    (folder_path / ".gitignore").write_text(
        _render_json_crew_template(".gitignore"),
        encoding="utf-8",
    )

    # Write README
    (folder_path / "README.md").write_text(
        _render_json_crew_template("README.md", {"name": name}),
        encoding="utf-8",
    )

    # Write knowledge placeholder
    (folder_path / "knowledge" / "user_preference.txt").write_text(
        _render_json_crew_template("knowledge/user_preference.txt"),
        encoding="utf-8",
    )

    # Keep skills dir tracked by git
    (folder_path / "skills" / ".gitkeep").write_text("", encoding="utf-8")

    # Setup .env with API keys
    if not skip_provider and not dmn_mode:
        models = list({a["llm"] for a in agents})
        for model in models:
            _setup_env(folder_path, model)

    click.echo()
    click.secho(f"  ✔ Crew {name} created successfully!", fg="green", bold=True)
    click.echo()
    click.secho("  Next steps:", bold=True)
    click.echo()
    click.echo(f"    cd {folder_name}")
    click.echo()
    click.secho("  Run your crew:", fg="cyan")
    click.echo("    crewai run")
    click.echo()
    click.secho("  Customize your crew:", fg="cyan")
    click.echo("    agents/*.jsonc    Define agent roles, goals, and LLMs")
    click.echo("    crew.jsonc        Configure tasks and optional input defaults")
    click.echo("    tools/            Add custom tools (Python)")
    click.echo()
