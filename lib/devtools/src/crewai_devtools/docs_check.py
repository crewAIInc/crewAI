"""Analyze code changes and generate/update documentation with translations.

Examines a git diff, determines what documentation changes are needed,
and optionally generates English docs + translations for all supported languages.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Final, Literal

import click
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


load_dotenv()

console = Console()

DocLang = Literal["en", "ar", "ko", "pt-BR"]
_TRANSLATION_LANGS: Final[list[DocLang]] = ["ar", "ko", "pt-BR"]

_LANGUAGE_NAMES: Final[dict[DocLang, str]] = {
    "en": "English",
    "ar": "Modern Standard Arabic",
    "ko": "Korean",
    "pt-BR": "Brazilian Portuguese",
}


# --- Structured output models ---


class DocAction(BaseModel):
    """A single documentation action to take."""

    action: Literal["create", "update"] = Field(
        description="Whether to create a new page or update an existing one."
    )
    file: str = Field(
        description="Target docs path relative to docs/en/ (e.g., 'concepts/skills.mdx')."
    )
    reason: str = Field(description="Why this documentation change is needed.")
    section: str | None = Field(
        default=None,
        description="For updates, which section of the existing doc needs changing.",
    )


class DocsAnalysis(BaseModel):
    """Analysis of what documentation changes are needed for a code diff."""

    needs_docs: bool = Field(
        description="Whether any documentation changes are needed."
    )
    summary: str = Field(description="One-line summary of documentation impact.")
    actions: list[DocAction] = Field(
        default_factory=list,
        description="List of documentation actions to take.",
    )


# --- Prompts ---

_ANALYZE_SYSTEM: Final[str] = """\
You are a documentation analyst for the CrewAI open-source framework.

Analyze git diffs and determine what documentation changes are needed.

Consider these categories:
- New features (new classes, decorators, CLI commands) → may need a new doc page or section
- API changes (new parameters, changed signatures) → update existing docs
- Configuration changes (new settings, env vars) → update relevant config docs
- Deprecations or removals → update affected docs
- Bug fixes with user-visible behavior changes → may need doc clarification

Only flag changes that affect the PUBLIC API or user-facing behavior.
Do NOT flag internal refactors, test changes, CI changes, or type annotation fixes."""

_ANALYZE_USER: Final[str] = "Analyze the following git diff:\n\n"

_GENERATE_DOC_PROMPT: Final[str] = """\
You are a technical writer for the CrewAI open-source framework.

Generate documentation in MDX format for the following change.

Rules:
- Use the same style and structure as existing CrewAI docs
- Start with YAML frontmatter: title, description, icon (optional)
- Use MDX components: <Tip>, <Warning>, <Note>, <Info>, <Steps>, <Step>, \
<CodeGroup>, <Card>, <CardGroup>, <Tabs>, <Tab>, <Accordion>, <AccordionGroup>
- Include code examples in Python
- Keep prose concise and technical
- Do not include translator notes or meta-commentary

Context about the change:
{reason}

{existing_content}

{diff_context}

Generate the full MDX file content:"""

_UPDATE_DOC_PROMPT: Final[str] = """\
You are a technical writer for the CrewAI open-source framework.

Update the following existing documentation based on the code changes described below.

Rules:
- Preserve the overall structure and style of the existing document
- Only modify sections that are affected by the changes
- Keep all MDX components, frontmatter structure, and code formatting intact
- Do not remove existing content unless it is now incorrect
- Add new sections where appropriate

Change description:
{reason}

Section to update: {section}

Existing document:
{existing_content}

Code diff context:
{diff_context}

Generate the complete updated MDX file:"""

_TRANSLATE_DOC_PROMPT: Final[str] = """\
Translate the following MDX documentation into {language}.

Rules:
- Translate ALL prose text (headings, descriptions, paragraphs, list items)
- Keep all MDX/JSX syntax, component tags, frontmatter keys, code blocks, \
URLs, and variable names in English
- Translate frontmatter values (title, description, sidebarTitle)
- Keep technical terms like Agent, Crew, Task, Flow, LLM, API, CLI, MCP \
in English as appropriate for {language} technical writing
- Keep code examples exactly as-is
- Do NOT add translator notes or comments
- Internal doc links should use /{lang_code}/ prefix instead of /en/

Document to translate:
{content}"""


def _run_git(args: list[str]) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _get_diff(base: str) -> str:
    """Get the git diff against a base ref."""
    return _run_git(["diff", base, "--", "lib/"])


def _get_openai_client() -> OpenAI:
    """Create an OpenAI client."""
    return OpenAI()


def _analyze_diff(diff: str, client: OpenAI) -> DocsAnalysis:
    """Analyze a git diff and determine what docs are needed.

    Args:
        diff: Git diff output.
        client: OpenAI client.

    Returns:
        Structured analysis result with actions.
    """
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _ANALYZE_SYSTEM},
            {"role": "user", "content": _ANALYZE_USER + diff[:50000]},
        ],
        temperature=0.2,
        response_format=DocsAnalysis,
    )
    return response.choices[0].message.parsed or DocsAnalysis(
        needs_docs=False, summary="Analysis failed."
    )


def _generate_doc(
    reason: str,
    existing_content: str | None,
    diff_context: str,
    client: OpenAI,
) -> str:
    """Generate a new documentation page.

    Args:
        reason: Why this doc is needed.
        existing_content: Existing doc content for style reference, or None.
        diff_context: The code diff to document.
        client: OpenAI client.

    Returns:
        Generated MDX content.
    """
    context = ""
    if existing_content:
        context = f"Reference existing doc for style:\n{existing_content[:5000]}"

    diff_section = ""
    if diff_context:
        diff_section = f"Code changes:\n{diff_context[:10000]}"

    prompt = _GENERATE_DOC_PROMPT.format(
        reason=reason,
        existing_content=context,
        diff_context=diff_section,
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a technical writer. Output only MDX content.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def _update_doc(
    reason: str,
    section: str,
    existing_content: str,
    diff_context: str,
    client: OpenAI,
) -> str:
    """Update an existing documentation page.

    Args:
        reason: Why this update is needed.
        section: Which section to update.
        existing_content: Current doc content.
        diff_context: Relevant portion of the diff.
        client: OpenAI client.

    Returns:
        Updated MDX content.
    """
    prompt = _UPDATE_DOC_PROMPT.format(
        reason=reason,
        section=section,
        existing_content=existing_content,
        diff_context=diff_context[:10000],
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a technical writer. Output only the complete updated MDX file.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def _translate_doc(
    content: str,
    lang: DocLang,
    client: OpenAI,
) -> str:
    """Translate an English doc to another language.

    Args:
        content: English MDX content.
        lang: Target language code.
        client: OpenAI client.

    Returns:
        Translated MDX content.
    """
    language_name = _LANGUAGE_NAMES[lang]
    prompt = _TRANSLATE_DOC_PROMPT.format(
        language=language_name,
        lang_code=lang,
        content=content,
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"You are a professional translator. Translate technical documentation into {language_name}. Output only the translated MDX.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def _print_analysis(analysis: DocsAnalysis) -> None:
    """Print the analysis results."""
    if not analysis.needs_docs:
        console.print("[green]No documentation changes needed.[/green]")
        return

    console.print(
        Panel(analysis.summary, title="Documentation Impact", border_style="yellow")
    )

    table = Table(title="Required Actions")
    table.add_column("Action", style="cyan")
    table.add_column("File", style="white")
    table.add_column("Reason", style="dim")

    for action in analysis.actions:
        table.add_row(action.action, action.file, action.reason)

    console.print(table)


@click.command("docs-check")
@click.option(
    "--base",
    default="main",
    help="Base ref to diff against (default: main).",
)
@click.option(
    "--write",
    is_flag=True,
    help="Generate/update docs and translations (not just analyze).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be written without writing files.",
)
def docs_check(base: str, write: bool, dry_run: bool) -> None:
    """Analyze code changes and determine if documentation is needed.

    Examines the diff between the current branch and --base, classifies
    changes, and reports what documentation should be created or updated.

    With --write, generates English docs and translates to all supported
    languages (ar, ko, pt-BR).

    Args:
        base: Base git ref to diff against.
        write: Whether to generate/update docs.
        dry_run: Show what would be done without writing.
    """
    cwd = Path.cwd()
    docs_dir = cwd / "docs"

    with console.status("[cyan]Getting diff..."):
        diff = _get_diff(base)

    if not diff:
        console.print("[green]No code changes found.[/green]")
        return

    with console.status("[cyan]Analyzing changes..."):
        client = _get_openai_client()
        analysis = _analyze_diff(diff, client)

    _print_analysis(analysis)

    if not analysis.needs_docs or not analysis.actions:
        return

    if not write:
        console.print(
            "\n[dim]Run with --write to generate docs, "
            "or --write --dry-run to preview.[/dim]"
        )
        return

    for action_item in analysis.actions:
        if action_item.action not in ("create", "update") or not action_item.file:
            continue

        rel_path = action_item.file
        en_path = (docs_dir / "en" / rel_path).resolve()
        if not en_path.is_relative_to(docs_dir.resolve()):
            console.print(f"  [red]✗ Skipping unsafe path: {rel_path!r}[/red]")
            continue
        console.print(f"\n[bold]Processing:[/bold] {rel_path}")

        content: str = ""

        if action_item.action == "create":
            if en_path.exists():
                console.print("  [yellow]⚠[/yellow] Already exists, skipping create")
                continue

            with console.status(f"  [cyan]Generating {rel_path}..."):
                ref_content = None
                parent = en_path.parent
                if parent.exists():
                    siblings = list(parent.glob("*.mdx"))
                    if siblings:
                        ref_content = siblings[0].read_text()
                content = _generate_doc(action_item.reason, ref_content, diff, client)

            if dry_run:
                console.print(f"  [dim][DRY RUN] Would create {en_path}[/dim]")
                console.print(f"  [dim]Preview: {content[:200]}...[/dim]")
            else:
                en_path.parent.mkdir(parents=True, exist_ok=True)
                en_path.write_text(content)
                console.print(f"  [green]✓[/green] Created {en_path}")

        elif action_item.action == "update":
            if not en_path.exists():
                console.print("  [yellow]⚠[/yellow] File not found, skipping update")
                continue

            existing = en_path.read_text()
            with console.status(f"  [cyan]Updating {rel_path}..."):
                content = _update_doc(
                    action_item.reason,
                    action_item.section or "",
                    existing,
                    diff,
                    client,
                )

            if not content:
                console.print("  [yellow]⚠[/yellow] Empty response, skipping update")
                continue

            if dry_run:
                console.print(f"  [dim][DRY RUN] Would update {en_path}[/dim]")
            else:
                en_path.write_text(content)
                console.print(f"  [green]✓[/green] Updated {en_path}")

        if not content:
            continue

        resolved_docs = docs_dir.resolve()
        for lang in _TRANSLATION_LANGS:
            lang_path = (docs_dir / lang / rel_path).resolve()
            if not lang_path.is_relative_to(resolved_docs):
                continue

            with console.status(f"  [cyan]Translating to {_LANGUAGE_NAMES[lang]}..."):
                translated = _translate_doc(content, lang, client)

            if dry_run:
                console.print(f"  [dim][DRY RUN] Would write {lang_path}[/dim]")
            else:
                lang_path.parent.mkdir(parents=True, exist_ok=True)
                lang_path.write_text(translated)
                console.print(f"  [green]✓[/green] Translated → {lang_path}")

    console.print("\n[green]✓ Done.[/green]")
