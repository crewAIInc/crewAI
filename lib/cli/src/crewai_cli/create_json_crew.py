"""Scaffold a new JSON-first crew project."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import click

from crewai_cli.constants import ENV_VARS
from crewai_cli.tui_picker import pick_many, pick_one
from crewai_cli.utils import load_env_vars, write_env_file


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


# ── Agent / crew templates (JSONC with comments) ──────────────

_RESEARCHER_AGENT_JSONC = """\
{
  // Agent's role title — appears in prompts and logs
  // Supports {placeholder} interpolation from crew inputs
  "role": "{topic} Senior Data Researcher",

  // The agent's primary objective
  "goal": "Uncover cutting-edge developments in {topic}",

  // Background story that shapes the agent's personality and approach
  "backstory": "You're a seasoned researcher with a knack for uncovering the latest developments in {topic}. Known for your ability to find the most relevant information and present it in a clear and concise manner.",

  // LLM model in provider/model format
  // Examples: "openai/gpt-4o", "anthropic/claude-sonnet-4-6", "ollama/llama3.3"
  "llm": "__LLM__",

  // Override LLM used specifically for tool/function calling
  // "function_calling_llm": "openai/gpt-5.4-mini",

  // Tools available to this agent
  // Built-in: "SerperDevTool", "ScrapeWebsiteTool", "FileReadTool", etc.
  // Custom: "custom:my_tool" loads from tools/my_tool.py
  "tools": [],

  // Optional agent-level guardrail — validates this agent's final output.
  // String guardrails are checked by an LLM and can reject/retry output.
  // "guardrail": "Only answer with information supported by retrieved evidence.",
  // "guardrail_max_retries": 2,

  // Advanced agent options:
  // Docs: https://docs.crewai.com/concepts/agents
  // "reasoning": true,
  // "max_reasoning_attempts": 3,
  // "multimodal": false,
  // "allow_code_execution": false,
  // "code_execution_mode": "safe",
  // "knowledge_sources": [],
  // "knowledge_config": {},
  // "inject_date": true,
  // "date_format": "%Y-%m-%d",
  // "security_config": {},

  // Agent behavior settings
  "settings": {
    // Show detailed execution logs
    "verbose": false,

    // Allow this agent to delegate tasks to other agents in the crew
    "allow_delegation": false,

    // Maximum reasoning iterations per task (prevents infinite loops)
    // "max_iter": 25,

    // Maximum tokens for agent's response generation
    // "max_tokens": null,

    // Maximum execution time in seconds
    // "max_execution_time": null,

    // Maximum LLM requests per minute (rate limiting)
    // "max_rpm": null,

    // Enable agent-level memory (persists across tasks)
    // "memory": false,

    // Cache tool results to avoid duplicate calls
    // "cache": true,

    // Auto-summarize context when it exceeds the LLM's context window
    // "respect_context_window": true,

    // Maximum retries on execution errors
    // "max_retry_limit": 2,

    // Enable step-by-step planning before task execution
    // "planning": false,

    // Include system prompt in LLM calls
    // "use_system_prompt": true
  }
}
"""

_REPORTING_ANALYST_JSONC = """\
{
  // Agent's role title — appears in prompts and logs
  "role": "{topic} Reporting Analyst",

  // The agent's primary objective
  "goal": "Create detailed reports based on {topic} data analysis and research findings",

  // Background story that shapes the agent's personality and approach
  "backstory": "You're a meticulous analyst with a keen eye for detail. You're known for your ability to turn complex data into clear and concise reports, making it easy for others to understand and act on the information you provide.",

  // LLM model in provider/model format
  "llm": "__LLM__",

  // "function_calling_llm": "openai/gpt-5.4-mini",

  // Tools available to this agent
  "tools": [],

  // Optional agent-level guardrail — validates this agent's final output.
  // "guardrail": "Only produce reports grounded in the provided task context.",
  // "guardrail_max_retries": 2,

  // Advanced agent options:
  // Docs: https://docs.crewai.com/concepts/agents
  // "reasoning": true,
  // "max_reasoning_attempts": 3,
  // "knowledge_sources": [],
  // "knowledge_config": {},
  // "security_config": {},

  // Agent behavior settings
  "settings": {
    "verbose": false,
    "allow_delegation": false
    // "max_iter": 25,
    // "max_tokens": null,
    // "max_execution_time": null,
    // "max_rpm": null,
    // "memory": false,
    // "cache": true,
    // "respect_context_window": true,
    // "max_retry_limit": 2,
    // "planning": false,
    // "use_system_prompt": true
  }
}
"""

_CREW_JSONC = """\
{
  // Display name for this crew
  "name": "__NAME__",

  // Agents to include — each must have a matching agents/<name>.jsonc file
  "agents": ["researcher", "reporting_analyst"],

  // Task definitions — executed in order for sequential process
  "tasks": [
    {
      // Task identifier — used for context references between tasks
      "name": "research_task",

      // What the task should accomplish
      // Supports {placeholder} interpolation from inputs below
      "description": "Conduct a thorough research about {topic}. Make sure you find any interesting and relevant information given the current year is {current_year}.",

      // Clear definition of what the output should look like
      "expected_output": "A list with 10 bullet points of the most relevant information about {topic}",

      // Optional task guardrail(s) validate output before the task completes.
      // Use either "guardrail" for one rule or "guardrails" for multiple rules.
      // On failure, CrewAI retries up to guardrail_max_retries times.
      // "guardrail": "Every factual claim must be supported by research findings.",
      // "guardrails": [
      //   "Every factual claim must be supported by research findings.",
      //   "The answer must be a numbered or bulleted list."
      // ],
      // "guardrail_max_retries": 2,

      // Advanced task options:
      // Docs: https://docs.crewai.com/concepts/tasks
      // "output_json": null,
      // "output_pydantic": null,
      // "response_model": null,
      // "markdown": false,
      // "input_files": [],
      // "security_config": {},

      // Which agent handles this task (must be in the agents list above)
      "agent": "researcher"

      // List of task names whose outputs become context for this task
      // "context": [],

      // Additional tools available only for this task
      // "tools": [],

      // Write task output to a file
      // "output_file": null,

      // Pause for human review before accepting the output
      // "human_input": false,

      // Run this task in parallel with the next task
      // "async_execution": false
    },
    {
      "name": "reporting_task",
      "description": "Review the context you got and expand each topic into a full section for a report. Make sure the report is detailed and contains any and all relevant information.",
      "expected_output": "A fully fledged report with the main topics, each with a full section of information. Formatted as markdown.",

      // Optional task guardrail(s) validate output before the task completes.
      // "guardrail": "The report must only use facts from the research context.",
      // "guardrail_max_retries": 2,

      // Advanced task options:
      // Docs: https://docs.crewai.com/concepts/tasks
      // "output_json": null,
      // "output_pydantic": null,
      // "markdown": true,
      // "create_directory": true,
      // "security_config": {},

      "agent": "reporting_analyst",

      // This task receives the output of research_task as context
      "context": ["research_task"],

      // Save the final report to a file
      "output_file": "report.md"

      // "human_input": false,
      // "async_execution": false
    }
  ],

  // Execution process
  // "sequential" — tasks run in order, each receiving prior task outputs
  // "hierarchical" — a manager agent delegates tasks (requires manager_llm)
  "process": "sequential",

  // Enable verbose logging during execution
  "verbose": true,

  // Enable crew memory — persists context and learnings across tasks
  "memory": false,

  // Automatically plan the execution strategy before running tasks
  // "planning": false,

  // LLM for the planning step (used when planning is true)
  // "planning_llm": "openai/gpt-4o",

  // LLM for the manager agent (required when process is "hierarchical")
  // "manager_llm": "openai/gpt-4o",

  // Advanced crew options:
  // Docs: https://docs.crewai.com/concepts/crews
  // "manager_agent": "reporting_analyst",
  // "function_calling_llm": "openai/gpt-4o-mini",
  // "max_rpm": null,
  // "cache": true,
  // "knowledge_sources": [],
  // "embedder": {},
  // "output_log_file": "crew.log",
  // "stream": false,
  // "tracing": false,
  // "security_config": {},

  // Default input values — interpolated into {placeholder} strings
  // in agent roles/goals/backstories and task descriptions
  "inputs": {
    "topic": "AI LLMs",
    "current_year": "2025"
  }
}
"""

_CONFIG_JSONC = """\
{
  // Benchmark test configuration (used by `crewai test`)
  "test": {
    // Number of test iterations per case
    // Higher values give more reliable scores but take longer
    "iterations": 3,

    // Minimum average score to pass (0.0 to 1.0)
    "threshold": 0.7,

    // LLM model used to evaluate qualitative test criteria
    "judge_model": "openai/gpt-5.4-mini",

    // Per-case timeout in seconds
    "case_timeout": 90
  }
}
"""

_RESEARCHER_CASES_JSONC = """\
[
  {
    // Input prompt sent to the agent
    "input": "What are the latest developments in AI?",

    // Substring the response must contain (case-insensitive)
    // Set to null to skip substring matching
    "expected": "AI",

    // Qualitative criteria evaluated by the judge LLM
    // Set to null to skip qualitative evaluation
    "criteria": "The response should contain specific, recent AI developments."
  }
]
"""

_REPORTING_ANALYST_CASES_JSONC = """\
[
  {
    "input": "Write a summary report about recent advances in machine learning.",
    "expected": "machine learning",
    "criteria": "The response should be formatted as a report with clear sections."
  }
]
"""

_PYPROJECT_TOML = """\
[project]
name = "{folder_name}"
version = "0.1.0"
description = "{name} using crewAI"
authors = [{{ name = "Your Name", email = "you@example.com" }}]
requires-python = ">=3.10,<3.14"
dependencies = [
    "crewai[tools]>=1.15"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.crewai]
type = "crew"
"""

_GITIGNORE = """\
.env
__pycache__/
.DS_Store
report.md
"""

_README = """\
# {name}

A crewAI project using JSON-first configuration.

## Running

```bash
crewai run
```

## Testing

```bash
crewai test
```

## Project Structure

- `agents/` - Agent definitions (JSONC)
- `crew.jsonc` - Crew definition with tasks and configuration
- `tests/` - Benchmark test cases for each agent
- `tools/` - Custom tools (Python)
- `knowledge/` - Knowledge files for agents
"""


# ── Common tools for picker ────────────────────────────────────

_TOOL_CATEGORIES: list[tuple[str, list[tuple[str, str]]]] = [
    ("Search & Research", [
        ("SerperDevTool", "Google search via Serper API"),
        ("BraveSearchTool", "Web search via Brave Search"),
        ("TavilySearchTool", "Web search via Tavily"),
        ("EXASearchTool", "Semantic web search via Exa"),
        ("GithubSearchTool", "Search GitHub repositories"),
        ("ArxivPaperTool", "Search arXiv academic papers"),
    ]),
    ("Web Scraping", [
        ("ScrapeWebsiteTool", "Extract content from a URL"),
        ("FirecrawlScrapeWebsiteTool", "Scrape with Firecrawl"),
        ("SeleniumScrapingTool", "Browser-based scraping"),
        ("WebsiteSearchTool", "RAG search on a website"),
    ]),
    ("File & Document", [
        ("FileReadTool", "Read local files"),
        ("FileWriterTool", "Write to local files"),
        ("DirectoryReadTool", "List directory contents"),
        ("CSVSearchTool", "Search within CSV files"),
        ("PDFSearchTool", "Search within PDF files"),
        ("DOCXSearchTool", "Search within DOCX files"),
        ("JSONSearchTool", "Search within JSON files"),
        ("TXTSearchTool", "Search within text files"),
    ]),
    ("Code & Data", [
        ("CodeDocsSearchTool", "Search code documentation"),
        ("NL2SQLTool", "Natural language to SQL queries"),
        ("RagTool", "RAG over custom data sources"),
    ]),
    ("AI & Vision", [
        ("DallETool", "Generate images with DALL-E"),
        ("VisionTool", "Analyze images with vision models"),
    ]),
]

_FLAT_TOOLS: list[tuple[str, str]] = [
    tool for _cat, tools in _TOOL_CATEGORIES for tool in tools
]


# ── Interactive wizard ─────────────────────────────────────────


def _prompt_text(label: str, default: str = "") -> str:
    return click.prompt(
        click.style(f"  {label}", fg="cyan"),
        default=default,
        show_default=bool(default),
        prompt_suffix=click.style(" > ", fg="bright_white"),
    ).strip()


def _confirm(label: str, default: bool = False) -> bool:
    return click.confirm(
        click.style(f"  {label}", fg="cyan"),
        default=default,
        prompt_suffix=click.style(" > ", fg="bright_white"),
    )


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
    click.echo()

    role = _prompt_text("Role")
    if not role:
        return None

    name_default = role.lower().replace(" ", "_")[:30]
    for ch in ".,;:!?'\"()[]{}":
        name_default = name_default.replace(ch, "")
    while name_default in existing_names:
        name_default += "_2"

    goal = _prompt_text("Goal")

    backstory = _prompt_text("Backstory")

    # LLM model
    if preset_llm:
        llm = preset_llm
        click.secho(f"  ✔ {llm}", fg="green")
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
        click.secho(f"  ✔ {llm}", fg="green")
    else:
        llm = _select_model()

    # Tools
    tool_labels = [f"{name:<28s} {desc}" for name, desc in _FLAT_TOOLS]
    selected_indices = pick_many("Tools (space to toggle, enter to confirm):", tool_labels)

    tools: list[str] = []
    if selected_indices:
        tools.extend(
            _FLAT_TOOLS[idx][0] for idx in selected_indices if idx < len(_FLAT_TOOLS)
        )
    if tools:
        click.secho(f"  ✔ {len(tools)} tool{'s' if len(tools) != 1 else ''}", fg="green")
    else:
        click.secho("  ✔ No tools", dim=True)

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
    click.echo()

    description = _prompt_text("Description")
    if not description:
        return None

    # Auto-generate name from first few words of description
    words = description.lower().split()[:4]
    name = "_".join(words)
    for ch in ".,;:!?'\"()[]{}":
        name = name.replace(ch, "")
    name = name + "_task"

    expected_output = _prompt_text("Expected output")

    # Agent assignment
    if len(agent_names) == 1:
        assigned_agent = agent_names[0]
        click.secho(f"  ✔ Agent: {assigned_agent}", fg="green")
    else:
        a_idx = pick_one("Assign to agent:", agent_names)
        assigned_agent = agent_names[max(a_idx, 0)]

    # Context dependencies
    context: list[str] = []
    if prior_task_names:
        ctx_indices = pick_many(
            "Context from prior tasks (space to toggle):",
            [*prior_task_names, "None"],
        )
        context = [
            prior_task_names[i]
            for i in ctx_indices
            if i < len(prior_task_names)
        ]
        if context:
            click.secho(f"  ✔ Context: {', '.join(context)}", fg="green")

    # Output file
    output_file = _prompt_text("Output file (leave empty to skip)", default="")

    return {
        "name": name,
        "description": description,
        "expected_output": expected_output,
        "agent": assigned_agent,
        "context": context,
        "output_file": output_file or None,
    }


def _wizard_agents_and_tasks(
    skip_provider: bool = False,
    default_llm: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Run the full interactive wizard. Returns (agents, tasks, crew_settings)."""
    agents: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []

    # ── Step 1: Agents ──
    click.echo()
    click.secho("  Step 1/3 — Agents", fg="cyan", bold=True)
    click.secho("  Define the AI agents in your crew.", dim=True)

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
            click.echo()
            click.secho(f"  ✔ '{agent['role']}' added", fg="green", bold=True)

        if not _confirm("Add another agent?", default=False):
            break

    # ── Step 2: Tasks ──
    click.echo()
    click.secho("  Step 2/3 — Tasks", fg="cyan", bold=True)
    click.secho("  Define what your agents should do.", dim=True)

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
            click.echo()
            click.secho(f"  ✔ '{task['name']}' added", fg="green", bold=True)

        if not _confirm("Add another task?", default=False):
            break

    # ── Step 3: Settings ──
    click.echo()
    click.secho("  Step 3/3 — Settings", fg="cyan", bold=True)

    process_labels = [
        "Sequential  (tasks run in order)",
        "Hierarchical  (manager delegates tasks)",
    ]
    p_idx = pick_one("Execution process:", process_labels)
    process = "hierarchical" if p_idx == 1 else "sequential"

    memory = _confirm("Enable crew memory?", default=False)

    # Default inputs
    click.echo()
    click.secho("  Default inputs (key=value, empty line to finish):", dim=True)
    inputs: dict[str, str] = {}
    while True:
        raw = click.prompt(
            click.style("  ", fg="cyan"),
            default="",
            show_default=False,
            prompt_suffix=click.style("> ", fg="bright_white"),
        ).strip()
        if not raw:
            break
        if "=" in raw:
            k, v = raw.split("=", 1)
            inputs[k.strip()] = v.strip()
        else:
            click.secho("  Format: key=value", fg="yellow")

    crew_settings = {
        "process": process,
        "memory": memory,
        "inputs": inputs,
    }

    return agents, tasks, crew_settings


# ── JSONC generation from wizard data ──────────────────────────


def _agent_to_jsonc(agent: dict[str, Any]) -> str:
    """Convert agent wizard data to JSONC string with comments."""
    has_planning = agent["planning"]
    delegation_val = "true" if agent["allow_delegation"] else "false"
    delegation_comma = "," if has_planning else ""

    settings_lines = []
    settings_lines.append('    // Show detailed execution logs')
    settings_lines.append('    "verbose": false,')
    settings_lines.append('')
    settings_lines.append('    // Allow this agent to delegate tasks to other agents in the crew')
    settings_lines.append(f'    "allow_delegation": {delegation_val}{delegation_comma}')
    settings_lines.append('')
    settings_lines.append('    // Maximum reasoning iterations per task (prevents infinite loops)')
    settings_lines.append('    // "max_iter": 25,')
    settings_lines.append('')
    settings_lines.append('    // Maximum tokens for agent\'s response generation')
    settings_lines.append('    // "max_tokens": null,')
    settings_lines.append('')
    settings_lines.append('    // Maximum execution time in seconds')
    settings_lines.append('    // "max_execution_time": null,')
    settings_lines.append('')
    settings_lines.append('    // Maximum LLM requests per minute (rate limiting)')
    settings_lines.append('    // "max_rpm": null,')
    settings_lines.append('')
    settings_lines.append('    // Enable agent-level memory (persists across tasks)')
    settings_lines.append('    // "memory": false,')
    settings_lines.append('')
    settings_lines.append('    // Cache tool results to avoid duplicate calls')
    settings_lines.append('    // "cache": true,')
    settings_lines.append('')
    settings_lines.append('    // Auto-summarize context when it exceeds the LLM\'s context window')
    settings_lines.append('    // "respect_context_window": true,')
    settings_lines.append('')
    settings_lines.append('    // Maximum retries on execution errors')
    settings_lines.append('    // "max_retry_limit": 2,')
    settings_lines.append('')
    settings_lines.append('    // Enable step-by-step planning before task execution')
    if has_planning:
        settings_lines.append('    "planning": true')
    else:
        settings_lines.append('    // "planning": false')
    settings_lines.append('')
    settings_lines.append('    // Include system prompt in LLM calls')
    settings_lines.append('    // "use_system_prompt": true')

    settings_block = "\n".join(settings_lines)

    return f"""\
{{
  // Agent's role title — appears in prompts and logs
  // Supports {{placeholder}} interpolation from crew inputs
  "role": {json.dumps(agent["role"])},

  // The agent's primary objective
  "goal": {json.dumps(agent["goal"])},

  // Background story that shapes the agent's personality and approach
  "backstory": {json.dumps(agent["backstory"])},

  // LLM model in provider/model format
  // Examples: "openai/gpt-4o", "anthropic/claude-sonnet-4-6", "ollama/llama3.3"
  "llm": {json.dumps(agent["llm"])},

  // Override LLM used specifically for tool/function calling
  // "function_calling_llm": "openai/gpt-5.4-mini",

  // Tools available to this agent
  // Built-in: "SerperDevTool", "ScrapeWebsiteTool", "FileReadTool", etc.
  // Custom: "custom:my_tool" loads from tools/my_tool.py
  "tools": {json.dumps(agent["tools"])},

  // Optional agent-level guardrail — validates this agent's final output.
  // String guardrails are checked by an LLM and can reject/retry output.
  // "guardrail": "Only answer with information supported by retrieved evidence.",
  // "guardrail_max_retries": 2,

  // Advanced agent options:
  // Docs: https://docs.crewai.com/concepts/agents
  // "reasoning": true,
  // "max_reasoning_attempts": 3,
  // "multimodal": false,
  // "allow_code_execution": false,
  // "code_execution_mode": "safe",
  // "knowledge_sources": [],
  // "knowledge_config": {{}},
  // "inject_date": true,
  // "date_format": "%Y-%m-%d",
  // "security_config": {{}},

  // Agent behavior settings
  "settings": {{
{settings_block}
  }}
}}
"""


def _task_to_json_fragment(task: dict[str, Any]) -> str:
    """Convert task wizard data to a JSON-like fragment for embedding in crew JSONC."""
    lines = []
    lines.append("    {")
    lines.append('      // Task identifier')
    lines.append(f'      "name": {json.dumps(task["name"])},')
    lines.append('')
    lines.append('      // What the task should accomplish')
    lines.append(f'      "description": {json.dumps(task["description"])},')
    lines.append('')
    lines.append('      // Clear definition of what the output should look like')
    lines.append(f'      "expected_output": {json.dumps(task["expected_output"])},')
    lines.append('')
    lines.append('      // Optional task guardrail(s) validate output before completion')
    lines.append('      // Use "guardrail" for one rule or "guardrails" for many')
    lines.append('      // Failed guardrails retry up to guardrail_max_retries times')
    lines.append('      // "guardrail": "Every factual claim needs context support.",')
    lines.append('      // "guardrails": [')
    lines.append('      //   "Every factual claim must be supported by context.",')
    lines.append('      //   "The answer must match the expected output format."')
    lines.append('      // ],')
    lines.append('      // "guardrail_max_retries": 2,')
    lines.append('')
    lines.append('      // Advanced task options:')
    lines.append('      // Docs: https://docs.crewai.com/concepts/tasks')
    lines.append('      // "output_json": null,')
    lines.append('      // "output_pydantic": null,')
    lines.append('      // "response_model": null,')
    lines.append('      // "markdown": false,')
    lines.append('      // "input_files": [],')
    lines.append('      // "security_config": {},')
    lines.append('')
    lines.append('      // Which agent handles this task')
    lines.append(f'      "agent": {json.dumps(task["agent"])}')

    if task.get("context"):
        lines[-1] += ","  # add comma to agent line
        lines.append('')
        lines.append('      // Task outputs used as context')
        lines.append(f'      "context": {json.dumps(task["context"])}')

    if task.get("output_file"):
        lines[-1] += ","
        lines.append('')
        lines.append('      // Save output to a file')
        lines.append(f'      "output_file": {json.dumps(task["output_file"])}')

    lines.append('')
    lines.append('      // "tools": [],')
    lines.append('      // "human_input": false,')
    lines.append('      // "async_execution": false')
    lines.append("    }")
    return "\n".join(lines)


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
            inputs_lines[0]
            + "\n"
            + "\n".join("  " + line for line in inputs_lines[1:])
        )

    process = settings.get("process", "sequential")
    memory = "true" if settings.get("memory") else "false"

    return f"""\
{{
  // Display name for this crew
  "name": {json.dumps(name)},

  // Agents to include — each must have a matching agents/<name>.jsonc file
  "agents": {agent_names_json},

  // Task definitions — executed in order for sequential process
  "tasks": [
{tasks_fragments}
  ],

  // Execution process
  // "sequential" — tasks run in order, each receiving prior task outputs
  // "hierarchical" — a manager agent delegates tasks (requires manager_llm)
  "process": "{process}",

  // Enable verbose logging during execution
  "verbose": true,

  // Enable crew memory — persists context and learnings across tasks
  "memory": {memory},

  // Automatically plan the execution strategy before running tasks
  // "planning": false,

  // LLM for the planning step (used when planning is true)
  // "planning_llm": "openai/gpt-4o",

  // LLM for the manager agent (required when process is "hierarchical")
  // "manager_llm": "openai/gpt-4o",

  // Advanced crew options:
  // Docs: https://docs.crewai.com/concepts/crews
  // "manager_agent": "{agents[0]['name']}",
  // "function_calling_llm": "openai/gpt-4o-mini",
  // "max_rpm": null,
  // "cache": true,
  // "knowledge_sources": [],
  // "embedder": {{}},
  // "output_log_file": "crew.log",
  // "stream": false,
  // "tracing": false,
  // "security_config": {{}},

  // Default input values — interpolated into {{placeholder}} strings
  // in agent roles/goals/backstories and task descriptions
  "inputs": {inputs_json}
}}
"""


def _test_case_jsonc(agent: dict[str, Any]) -> str:
    """Generate a default test case JSONC for an agent."""
    role = agent["role"]
    return f"""\
[
  {{
    // Input prompt sent to the agent
    "input": "Perform a sample task as a {role}.",

    // Substring the response must contain (case-insensitive)
    // Set to null to skip substring matching
    "expected": null,

    // Qualitative criteria evaluated by the judge LLM
    // Set to null to skip qualitative evaluation
    "criteria": "The response should demonstrate expertise as a {role}."
  }}
]
"""


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


def _write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


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
    import re
    import shutil

    name = name.rstrip("/")
    if not name.strip():
        raise ValueError("Project name cannot be empty")

    folder_name = name.replace(" ", "_").replace("-", "_").lower()
    folder_name = re.sub(r"[^a-zA-Z0-9_]", "", folder_name)

    if not folder_name or folder_name[0].isdigit():
        raise ValueError(f"Project name '{name}' produces invalid folder name '{folder_name}'")

    if keyword.iskeyword(folder_name):
        raise ValueError(f"'{folder_name}' is a reserved Python keyword")

    folder_path = Path(folder_name)
    if folder_path.exists():
        if not click.confirm(f"Folder {folder_name} already exists. Override?"):
            click.secho("Cancelled.", fg="yellow")
            sys.exit(0)
        shutil.rmtree(folder_path)

    click.echo()
    click.secho(f"  Creating crew: {name}", fg="green", bold=True)

    agents, tasks, crew_settings = _wizard_agents_and_tasks(
        skip_provider=skip_provider,
        default_llm=_default_model_for_provider(provider),
    )

    # Create directories
    folder_path.mkdir(parents=True)
    (folder_path / "agents").mkdir()
    (folder_path / "tools").mkdir()
    (folder_path / "skills").mkdir()
    (folder_path / "tests").mkdir()
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

    for agent in agents:
        _write_jsonc(
            folder_path / "tests" / f"{agent['name']}_cases.jsonc",
            _test_case_jsonc(agent),
        )

    # Write config
    _write_jsonc(folder_path / "config.jsonc", _CONFIG_JSONC)

    # Write pyproject.toml
    (folder_path / "pyproject.toml").write_text(
        _PYPROJECT_TOML.format(folder_name=folder_name, name=name),
        encoding="utf-8",
    )

    # Write .gitignore
    (folder_path / ".gitignore").write_text(_GITIGNORE, encoding="utf-8")

    # Write README
    (folder_path / "README.md").write_text(
        _README.format(name=name),
        encoding="utf-8",
    )

    # Write knowledge placeholder
    (folder_path / "knowledge" / "user_preference.txt").write_text(
        "# Add your knowledge files here\n",
        encoding="utf-8",
    )

    # Keep skills dir tracked by git
    (folder_path / "skills" / ".gitkeep").write_text("", encoding="utf-8")

    # Setup .env with API keys
    if not skip_provider:
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
    click.echo("    crew.jsonc        Configure tasks, process, and inputs")
    click.echo("    tools/            Add custom tools (Python)")
    click.echo()
    click.secho("  Test & benchmark:", fg="cyan")
    click.echo("    crewai test       Run benchmark tests against your agents")
    click.echo("    tests/            Edit test cases and thresholds")
    click.echo()
