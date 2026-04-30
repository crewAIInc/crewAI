"""Development tools for version bumping and git automation."""

from collections.abc import Mapping
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Final, Literal
from urllib.request import urlopen

import click
from dotenv import load_dotenv
from github import Github
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
import tomlkit

from crewai_devtools.docs_check import docs_check
from crewai_devtools.prompts import RELEASE_NOTES_PROMPT, TRANSLATE_RELEASE_NOTES_PROMPT


load_dotenv()

console = Console()


def _resume_hint(message: str) -> None:
    """Print a boxed resume hint after a failure."""
    console.print()
    console.print(
        Panel(
            message,
            title="[bold yellow]How to resume[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def _print_release_error(e: BaseException) -> None:
    """Print a release error with stderr if available."""
    if isinstance(e, KeyboardInterrupt):
        raise
    if isinstance(e, SystemExit):
        return
    if isinstance(e, subprocess.CalledProcessError):
        console.print(f"[red]Error running command:[/red] {e}")
        if e.stderr:
            console.print(e.stderr)
    else:
        console.print(f"[red]Error:[/red] {e}")


def run_command(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a shell command and return output.

    Args:
        cmd: Command to run as list of strings.
        cwd: Working directory for command.

    Returns:
        Command output as string.

    Raises:
        subprocess.CalledProcessError: If command fails.
    """
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def check_gh_installed() -> None:
    """Check if GitHub CLI is installed and offer to install it.

    Raises:
        SystemExit: If gh is not installed and user declines installation.
    """
    try:
        run_command(["gh", "--version"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[yellow]Warning:[/yellow] GitHub CLI (gh) is not installed")
        import platform

        if platform.system() == "Darwin":
            try:
                run_command(["brew", "--version"])
                from rich.prompt import Confirm

                if Confirm.ask(
                    "\n[bold]Would you like to install GitHub CLI via Homebrew?[/bold]",
                    default=True,
                ):
                    try:
                        console.print("\nInstalling GitHub CLI...")
                        subprocess.run(
                            ["brew", "install", "gh"],  # noqa: S607
                            check=True,
                        )
                        console.print(
                            "[green]✓[/green] GitHub CLI installed successfully"
                        )
                        console.print("\nAuthenticating with GitHub...")
                        subprocess.run(
                            ["gh", "auth", "login"],  # noqa: S607
                            check=True,
                        )
                        console.print("[green]✓[/green] GitHub authentication complete")
                        return
                    except subprocess.CalledProcessError as e:
                        console.print(
                            f"[red]Error:[/red] Failed to install or authenticate gh: {e}"
                        )
                        console.print(
                            "\nYou can try running [bold]gh auth login[/bold] manually"
                        )
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        console.print("\nPlease install GitHub CLI from: https://cli.github.com/")
        console.print("\nInstallation instructions:")
        console.print("  macOS:   brew install gh")
        console.print(
            "  Linux:   https://github.com/cli/cli/blob/trunk/docs/install_linux.md"
        )
        console.print("  Windows: winget install --id GitHub.cli")
        sys.exit(1)


def check_git_clean() -> None:
    """Check if git working directory is clean.

    Raises:
        SystemExit: If there are uncommitted changes.
    """
    try:
        status = run_command(["git", "status", "--porcelain"])
        if status:
            console.print(
                "[red]Error:[/red] You have uncommitted changes. Please commit or stash them first."
            )
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error checking git status:[/red] {e}")
        sys.exit(1)


def _branch_exists_local(branch: str, cwd: Path | None = None) -> bool:
    try:
        subprocess.run(  # noqa: S603
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],  # noqa: S607
            cwd=cwd,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _branch_exists_remote(branch: str, cwd: Path | None = None) -> bool:
    try:
        output = run_command(["git", "ls-remote", "--heads", "origin", branch], cwd=cwd)
        return bool(output.strip())
    except subprocess.CalledProcessError:
        return False


def _open_pr_url_for_branch(branch: str, cwd: Path | None = None) -> str | None:
    """Return URL of open PR for branch, or None if no open PR exists."""
    try:
        url = run_command(
            [
                "gh",
                "pr",
                "list",
                "--head",
                branch,
                "--state",
                "open",
                "--json",
                "url",
                "--jq",
                ".[0].url // empty",
            ],
            cwd=cwd,
        )
        return url or None
    except subprocess.CalledProcessError:
        return None


def create_or_reset_branch(branch: str, cwd: Path | None = None) -> None:
    """Create ``branch`` from current HEAD, resetting any stale copy.

    If the branch exists locally or on origin, prompts the user to
    choose between resetting it or aborting. If an open PR exists on
    the branch, the prompt surfaces the PR URL and includes a
    close-and-reset option so in-flight work isn't silently clobbered.

    Raises:
        SystemExit: If the user declines to reset.
    """
    local_exists = _branch_exists_local(branch, cwd=cwd)
    remote_exists = _branch_exists_remote(branch, cwd=cwd)
    open_pr = _open_pr_url_for_branch(branch, cwd=cwd) if remote_exists else None

    if local_exists or remote_exists:
        if open_pr:
            console.print(
                f"\n[yellow]![/yellow] Branch [bold]{branch}[/bold] already has an open PR: {open_pr}"
            )
            prompt = "Close the PR, reset the branch, and continue?"
        else:
            where = []
            if local_exists:
                where.append("local")
            if remote_exists:
                where.append("remote")
            console.print(
                f"\n[yellow]![/yellow] Branch [bold]{branch}[/bold] already exists ({', '.join(where)}) with no open PR"
            )
            prompt = "Delete it and recreate?"

        if not Confirm.ask(prompt, default=False):
            console.print("[red]Aborted.[/red]")
            sys.exit(1)

        if open_pr:
            console.print(f"Closing PR {open_pr}...")
            run_command(
                ["gh", "pr", "close", branch, "--delete-branch"],
                cwd=cwd,
            )
            # `gh pr close --delete-branch` removes the remote branch
            # and, when checked out, the local branch too.
            local_exists = _branch_exists_local(branch, cwd=cwd)
            remote_exists = False

        if local_exists:
            current = run_command(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd
            ).strip()
            if current == branch:
                console.print(
                    f"[yellow]![/yellow] Currently on {branch}, switching to main before delete"
                )
                run_command(["git", "checkout", "main"], cwd=cwd)
            console.print(f"[yellow]![/yellow] Deleting local branch {branch}")
            run_command(["git", "branch", "-D", branch], cwd=cwd)

        if remote_exists:
            console.print(f"[yellow]![/yellow] Deleting remote branch {branch}")
            run_command(["git", "push", "origin", "--delete", branch], cwd=cwd)

    run_command(["git", "checkout", "-b", branch], cwd=cwd)


def update_version_in_file(file_path: Path, new_version: str) -> bool:
    """Update __version__ attribute in a Python file.

    Args:
        file_path: Path to Python file.
        new_version: New version string.

    Returns:
        True if version was updated, False otherwise.
    """
    if not file_path.exists():
        return False

    content = file_path.read_text()
    lines = content.splitlines()
    updated = False

    for i, line in enumerate(lines):
        if line.strip().startswith("__version__"):
            lines[i] = f'__version__ = "{new_version}"'
            updated = True
            break

    if updated:
        file_path.write_text("\n".join(lines) + "\n")
        return True

    return False


def update_pyproject_version(file_path: Path, new_version: str) -> bool:
    """Update the [project] version field in a pyproject.toml file.

    Args:
        file_path: Path to pyproject.toml file.
        new_version: New version string.

    Returns:
        True if version was updated, False otherwise.
    """
    if not file_path.exists():
        return False

    doc = tomlkit.parse(file_path.read_text())
    project = doc.get("project")
    if project is None:
        return False
    old_version = project.get("version")
    if old_version is None or old_version == new_version:
        return False

    project["version"] = new_version
    file_path.write_text(tomlkit.dumps(doc))
    return True


_DEFAULT_WORKSPACE_PACKAGES: Final[list[str]] = [
    "crewai",
    "crewai-tools",
    "crewai-devtools",
]


def update_pyproject_dependencies(
    file_path: Path,
    new_version: str,
    extra_packages: list[str] | None = None,
) -> bool:
    """Update workspace dependency versions in pyproject.toml.

    Args:
        file_path: Path to pyproject.toml file.
        new_version: New version string.
        extra_packages: Additional package names to update beyond the defaults.

    Returns:
        True if any dependencies were updated, False otherwise.
    """
    if not file_path.exists():
        return False

    content = file_path.read_text()
    lines = content.splitlines()
    updated = False

    workspace_packages = _DEFAULT_WORKSPACE_PACKAGES + (extra_packages or [])

    for i, line in enumerate(lines):
        for pkg in workspace_packages:
            if f"{pkg}==" in line:
                stripped = line.lstrip()
                indent = line[: len(line) - len(stripped)]

                if '"' in line:
                    lines[i] = f'{indent}"{pkg}=={new_version}",'
                elif "'" in line:
                    lines[i] = f"{indent}'{pkg}=={new_version}',"
                else:
                    lines[i] = f"{indent}{pkg}=={new_version},"

                updated = True

    if updated:
        file_path.write_text("\n".join(lines) + "\n")
        return True

    return False


def add_docs_version(docs_json_path: Path, version: str) -> bool:
    """Add a new version to the Mintlify docs.json versioning config.

    Copies the current default version's tabs into a new version entry,
    sets the new version as default, and marks the previous default as
    non-default. Operates on all languages.

    Args:
        docs_json_path: Path to docs/docs.json.
        version: Version string (e.g., "1.10.1b1").

    Returns:
        True if docs.json was updated, False otherwise.
    """
    import json

    if not docs_json_path.exists():
        return False

    data = json.loads(docs_json_path.read_text())
    version_label = f"v{version}"
    updated = False

    for lang in data.get("navigation", {}).get("languages", []):
        versions = lang.get("versions", [])
        if not versions:
            continue

        if any(v.get("version") == version_label for v in versions):
            continue

        default_version = next(
            (v for v in versions if v.get("default")),
            versions[0],
        )

        new_version = {
            "version": version_label,
            "default": True,
            "tabs": default_version.get("tabs", []),
        }

        default_version.pop("default", None)
        versions.insert(0, new_version)
        updated = True

    if not updated:
        return False

    docs_json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return True


ChangelogLang = Literal["en", "pt-BR", "ko", "ar"]

_PT_BR_MONTHS: Final[dict[int, str]] = {
    1: "jan",
    2: "fev",
    3: "mar",
    4: "abr",
    5: "mai",
    6: "jun",
    7: "jul",
    8: "ago",
    9: "set",
    10: "out",
    11: "nov",
    12: "dez",
}

_AR_MONTHS: Final[dict[int, str]] = {
    1: "يناير",
    2: "فبراير",
    3: "مارس",
    4: "أبريل",
    5: "مايو",
    6: "يونيو",
    7: "يوليو",
    8: "أغسطس",
    9: "سبتمبر",
    10: "أكتوبر",
    11: "نوفمبر",
    12: "ديسمبر",
}

_CHANGELOG_LOCALES: Final[
    dict[ChangelogLang, dict[Literal["link_text", "language_name"], str]]
] = {
    "en": {
        "link_text": "View release on GitHub",
        "language_name": "English",
    },
    "pt-BR": {
        "link_text": "Ver release no GitHub",
        "language_name": "Brazilian Portuguese",
    },
    "ko": {
        "link_text": "GitHub 릴리스 보기",
        "language_name": "Korean",
    },
    "ar": {
        "link_text": "عرض الإصدار على GitHub",
        "language_name": "Modern Standard Arabic",
    },
}


def translate_release_notes(
    release_notes: str,
    lang: ChangelogLang,
    client: OpenAI,
) -> str:
    """Translate release notes into the target language using OpenAI.

    Args:
        release_notes: English release notes markdown.
        lang: Language code (e.g., "pt-BR", "ko").
        client: OpenAI client instance.

    Returns:
        Translated release notes, or original on failure.
    """
    locale_cfg = _CHANGELOG_LOCALES.get(lang)
    if not locale_cfg:
        return release_notes

    language_name = locale_cfg["language_name"]
    prompt = TRANSLATE_RELEASE_NOTES_PROMPT.substitute(
        language=language_name,
        release_notes=release_notes,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. Translate technical documentation into {language_name}.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or release_notes
    except Exception as e:
        console.print(
            f"[yellow]Warning:[/yellow] Could not translate to {language_name}: {e}"
        )
        return release_notes


def _format_changelog_date(lang: ChangelogLang) -> str:
    """Format today's date for a changelog entry in the given language."""
    from datetime import datetime

    now = datetime.now()
    if lang == "ko":
        return f"{now.year}년 {now.month}월 {now.day}일"
    if lang == "pt-BR":
        return f"{now.day:02d} {_PT_BR_MONTHS[now.month]} {now.year}"
    if lang == "ar":
        return f"{now.day} {_AR_MONTHS[now.month]} {now.year}"
    return now.strftime("%b %d, %Y")


def update_changelog(
    changelog_path: Path,
    version: str,
    release_notes: str,
    lang: ChangelogLang = "en",
) -> bool:
    """Prepend a new release entry to a docs changelog file.

    Args:
        changelog_path: Path to the changelog.mdx file.
        version: Version string (e.g., "1.9.3").
        release_notes: Markdown release notes content.
        lang: Language code for localized date/link text.

    Returns:
        True if changelog was updated, False otherwise.
    """
    if not changelog_path.exists():
        return False

    locale_cfg = _CHANGELOG_LOCALES.get(lang, _CHANGELOG_LOCALES["en"])
    date_label = _format_changelog_date(lang)
    link_text = locale_cfg["link_text"]

    # Indent each non-empty line with 2 spaces to match <Update> block format
    indented_lines = []
    for line in release_notes.splitlines():
        if line.strip():
            indented_lines.append(f"  {line}")
        else:
            indented_lines.append("")
    indented_notes = "\n".join(indented_lines)

    entry = (
        f'<Update label="{date_label}">\n'
        f"  ## v{version}\n"
        f"\n"
        f"  [{link_text}]"
        f"(https://github.com/crewAIInc/crewAI/releases/tag/{version})\n"
        f"\n"
        f"{indented_notes}\n"
        f"\n"
        f"</Update>"
    )

    content = changelog_path.read_text()

    # Insert after the frontmatter closing ---
    parts = content.split("---", 2)
    if len(parts) >= 3:
        new_content = (
            parts[0]
            + "---"
            + parts[1]
            + "---\n"
            + entry
            + "\n\n"
            + parts[2].lstrip("\n")
        )
    else:
        new_content = entry + "\n\n" + content

    changelog_path.write_text(new_content)
    return True


def _is_crewai_dep(spec: str) -> bool:
    """Return True if *spec* is a ``crewai`` or ``crewai[...]`` dependency."""
    if not spec.startswith("crewai"):
        return False
    rest = spec[6:]
    return len(rest) > 0 and rest[0] in ("[", "=", ">", "<", "~", "!")


def _pin_crewai_deps(content: str, version: str) -> str:
    """Replace crewai dependency version pins in a pyproject.toml string.

    Handles both pinned (==) and minimum (>=) version specifiers,
    as well as extras like [tools].

    Args:
        content: File content to transform.
        version: New version string.

    Returns:
        Transformed content.
    """
    doc = tomlkit.parse(content)
    for key in ("dependencies", "optional-dependencies"):
        deps = doc.get("project", {}).get(key)
        if deps is None:
            continue
        dep_lists = deps.values() if isinstance(deps, Mapping) else [deps]
        for dep_list in dep_lists:
            for i, dep in enumerate(dep_list):
                s = str(dep)
                if not _is_crewai_dep(s) or ("==" not in s and ">=" not in s):
                    continue
                extras = s[6 : s.index("]") + 1] if "[" in s[6:7] else ""
                dep_list[i] = f"crewai{extras}=={version}"
    return tomlkit.dumps(doc)


def update_template_dependencies(templates_dir: Path, new_version: str) -> list[Path]:
    """Update crewai dependency versions in CLI template pyproject.toml files.

    Uses simple string replacement instead of TOML parsing because
    template files contain Jinja placeholders (``{{folder_name}}``)
    that are not valid TOML.

    Args:
        templates_dir: Path to the CLI templates directory.
        new_version: New version string.

    Returns:
        List of paths that were updated.
    """
    import re

    pattern = re.compile(r"(crewai(?:\[[\w,]+\])?)(?:==|>=)[^\s\"']+")
    updated = []
    for pyproject in templates_dir.rglob("pyproject.toml"):
        content = pyproject.read_text()
        new_content = pattern.sub(rf"\1=={new_version}", content)
        if new_content != content:
            pyproject.write_text(new_content)
            updated.append(pyproject)

    return updated


def find_version_files(base_path: Path) -> list[Path]:
    """Find all __init__.py files that contain __version__.

    Args:
        base_path: Base directory to search in.

    Returns:
        List of paths to files containing __version__.
    """
    return [
        init_file
        for init_file in base_path.rglob("__init__.py")
        if "__version__" in init_file.read_text()
    ]


def get_packages(lib_dir: Path) -> list[Path]:
    """Get all packages from lib/ directory.

    Args:
        lib_dir: Path to lib/ directory.

    Returns:
        List of package directory paths.

    Raises:
        SystemExit: If lib/ doesn't exist or no packages found.
    """
    if not lib_dir.exists():
        console.print("[red]Error:[/red] lib/ directory not found")
        sys.exit(1)

    packages = [p for p in lib_dir.iterdir() if p.is_dir()]

    if not packages:
        console.print("[red]Error:[/red] No packages found in lib/")
        sys.exit(1)

    return packages


PrereleaseIndicator = Literal["a", "b", "rc", "alpha", "beta", "dev"]
_PRERELEASE_INDICATORS: Final[tuple[PrereleaseIndicator, ...]] = (
    "a",
    "b",
    "rc",
    "alpha",
    "beta",
    "dev",
)


def _is_prerelease(version: str) -> bool:
    """Check if a version string represents a pre-release."""
    v = version.lower().lstrip("v")
    return any(indicator in v for indicator in _PRERELEASE_INDICATORS)


def get_commits_from_last_tag(tag_name: str, version: str) -> tuple[str, str]:
    """Get commits from the last tag, excluding current version.

    Args:
        tag_name: Current tag name (e.g., "v1.0.0").
        version: Current version (e.g., "1.0.0").

    Returns:
        Tuple of (commit_range, commits) where commits is newline-separated.
    """
    try:
        all_tags = run_command(["git", "tag", "--sort=-version:refname"]).split("\n")
        prev_tags = [t for t in all_tags if t and t != tag_name and t != f"v{version}"]

        if not _is_prerelease(version):
            prev_tags = [t for t in prev_tags if not _is_prerelease(t)]

        if prev_tags:
            last_tag = prev_tags[0]
            commit_range = f"{last_tag}..HEAD"
            commits = run_command(["git", "log", commit_range, "--pretty=format:%s"])
        else:
            commit_range = "HEAD"
            commits = run_command(["git", "log", "--pretty=format:%s"])
    except subprocess.CalledProcessError:
        commit_range = "HEAD"
        commits = run_command(["git", "log", "--pretty=format:%s"])

    return commit_range, commits


def get_github_contributors(commit_range: str) -> list[str]:
    """Get GitHub usernames from commit range using GitHub API.

    Args:
        commit_range: Git commit range (e.g., "abc123..HEAD").

    Returns:
        List of GitHub usernames sorted alphabetically.
    """
    try:
        try:
            gh_token = run_command(["gh", "auth", "token"])
        except subprocess.CalledProcessError:
            gh_token = None

        g = Github(login_or_token=gh_token) if gh_token else Github()
        github_repo = g.get_repo("crewAIInc/crewAI")

        commit_shas = run_command(
            ["git", "log", commit_range, "--pretty=format:%H"]
        ).split("\n")

        contributors = set()
        for sha in commit_shas:
            if not sha:
                continue
            try:
                commit = github_repo.get_commit(sha)
                if commit.author and commit.author.login:
                    contributors.add(commit.author.login)

                if commit.commit.message:
                    for line in commit.commit.message.split("\n"):
                        if line.strip().startswith("Co-authored-by:"):
                            if "<" in line and ">" in line:
                                email_part = line.split("<")[1].split(">")[0]
                                if "@users.noreply.github.com" in email_part:
                                    username = email_part.split("+")[-1].split("@")[0]
                                    contributors.add(username)
            except Exception:  # noqa: S112
                continue

        return sorted(list(contributors))

    except Exception as e:
        console.print(
            f"[yellow]Warning:[/yellow] Could not fetch GitHub contributors: {e}"
        )
        return []


def _poll_pr_until_merged(
    branch_name: str, label: str, repo: str | None = None
) -> None:
    """Poll a GitHub PR until it is merged. Exit if closed without merging.

    Args:
        branch_name: Branch name to look up the PR.
        label: Human-readable label for status messages.
        repo: Optional GitHub repo (owner/name) for cross-repo PRs.
    """
    console.print(f"[cyan]Waiting for {label} to be merged...[/cyan]")
    cmd = ["gh", "pr", "view", branch_name]
    if repo:
        cmd.extend(["--repo", repo])
    cmd.extend(["--json", "state", "--jq", ".state"])

    while True:
        time.sleep(10)
        try:
            state = run_command(cmd)
        except subprocess.CalledProcessError:
            state = ""

        if state == "MERGED":
            break

        if state == "CLOSED":
            console.print(f"[red]✗[/red] {label} was closed without merging")
            sys.exit(1)

        console.print(f"[dim]Still waiting for {label} to merge...[/dim]")

    console.print(f"[green]✓[/green] {label} merged")


def _update_all_versions(
    cwd: Path,
    lib_dir: Path,
    version: str,
    packages: list[Path],
    dry_run: bool,
) -> list[Path]:
    """Bump __version__, pyproject deps, template deps, and run uv sync."""
    updated_files: list[Path] = []

    for pkg in packages:
        version_files = find_version_files(pkg)
        for vfile in version_files:
            if dry_run:
                console.print(
                    f"[dim][DRY RUN][/dim] Would update: {vfile.relative_to(cwd)}"
                )
            else:
                if update_version_in_file(vfile, version):
                    console.print(f"[green]✓[/green] Updated: {vfile.relative_to(cwd)}")
                    updated_files.append(vfile)
                else:
                    console.print(
                        f"[red]✗[/red] Failed to update: {vfile.relative_to(cwd)}"
                    )

        pyproject = pkg / "pyproject.toml"
        if pyproject.exists():
            if dry_run:
                console.print(
                    f"[dim][DRY RUN][/dim] Would update dependencies in: {pyproject.relative_to(cwd)}"
                )
            else:
                if update_pyproject_dependencies(pyproject, version):
                    console.print(
                        f"[green]✓[/green] Updated dependencies in: {pyproject.relative_to(cwd)}"
                    )
                    updated_files.append(pyproject)

    if not updated_files and not dry_run:
        console.print(
            "[yellow]Warning:[/yellow] No __version__ attributes found to update"
        )

    templates_dir = lib_dir / "crewai" / "src" / "crewai" / "cli" / "templates"
    if templates_dir.exists():
        if dry_run:
            for tpl in templates_dir.rglob("pyproject.toml"):
                console.print(
                    f"[dim][DRY RUN][/dim] Would update template: {tpl.relative_to(cwd)}"
                )
        else:
            tpl_updated = update_template_dependencies(templates_dir, version)
            for tpl in tpl_updated:
                console.print(
                    f"[green]✓[/green] Updated template: {tpl.relative_to(cwd)}"
                )
                updated_files.append(tpl)

    if not dry_run:
        console.print("\nSyncing workspace...")
        run_command(["uv", "sync"])
        console.print("[green]✓[/green] Workspace synced")
    else:
        console.print("[dim][DRY RUN][/dim] Would run: uv sync")

    return updated_files


def _generate_release_notes(
    version: str,
    tag_name: str,
    no_edit: bool,
) -> tuple[str, OpenAI, bool]:
    """Generate, display, and optionally edit release notes.

    Returns:
        Tuple of (release_notes, openai_client, is_prerelease).
    """
    release_notes = f"Release {version}"
    commits = ""

    with console.status("[cyan]Generating release notes..."):
        try:
            prev_bump_output = run_command(
                [
                    "git",
                    "log",
                    "--grep=^feat: bump versions to",
                    "--format=%H %s",
                ]
            )
            bump_entries = [
                line for line in prev_bump_output.strip().split("\n") if line.strip()
            ]

            is_stable = not _is_prerelease(version)
            prev_commit = None
            for entry in bump_entries[1:]:
                bump_ver = entry.split("feat: bump versions to", 1)[-1].strip()
                if is_stable and _is_prerelease(bump_ver):
                    continue
                prev_commit = entry.split()[0]
                break

            if prev_commit:
                commit_range = f"{prev_commit}..HEAD"
                commits = run_command(
                    ["git", "log", commit_range, "--pretty=format:%s"]
                )

                commit_lines = [
                    line
                    for line in commits.split("\n")
                    if not line.startswith("feat: bump versions to")
                ]
                commits = "\n".join(commit_lines)
            else:
                commit_range, commits = get_commits_from_last_tag(tag_name, version)

        except subprocess.CalledProcessError:
            commit_range, commits = get_commits_from_last_tag(tag_name, version)

        github_contributors = get_github_contributors(commit_range)

        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if commits.strip():
            contributors_section = ""
            if github_contributors:
                contributors_section = f"\n\n## Contributors\n\n{', '.join([f'@{u}' for u in github_contributors])}"

            prompt = RELEASE_NOTES_PROMPT.substitute(
                version=version,
                commits=commits,
                contributors_section=contributors_section,
            )

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates clear, concise release notes.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            release_notes = response.choices[0].message.content or f"Release {version}"

    console.print("[green]✓[/green] Generated release notes")

    if commits.strip():
        try:
            console.print()
            md = Markdown(release_notes, justify="left")
            console.print(
                Panel(
                    md,
                    title="[bold cyan]Generated Release Notes[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
        except Exception as e:
            console.print(
                f"[yellow]Warning:[/yellow] Could not render release notes: {e}"
            )
            console.print("Using default release notes")

    if not no_edit:
        if Confirm.ask(
            "\n[bold]Would you like to edit the release notes?[/bold]", default=True
        ):
            edited_notes = click.edit(release_notes)
            if edited_notes is not None:
                release_notes = edited_notes.strip()
                console.print("\n[green]✓[/green] Release notes updated")
            else:
                console.print("\n[green]✓[/green] Using original release notes")
        else:
            console.print(
                "\n[green]✓[/green] Using generated release notes without editing"
            )
    else:
        console.print(
            "\n[green]✓[/green] Using generated release notes without editing"
        )

    is_prerelease = _is_prerelease(version)

    return release_notes, openai_client, is_prerelease


def _update_docs_and_create_pr(
    cwd: Path,
    version: str,
    release_notes: str,
    openai_client: OpenAI,
    is_prerelease: bool,
    dry_run: bool,
) -> str | None:
    """Update changelogs and docs version switcher, create PR if needed.

    Returns:
        The docs branch name if a PR was created, None otherwise.
    """
    docs_json_path = cwd / "docs" / "docs.json"
    changelog_langs: list[ChangelogLang] = ["en", "pt-BR", "ko", "ar"]

    if not dry_run:
        docs_files_staged: list[str] = []

        for lang in changelog_langs:
            cl_path = cwd / "docs" / lang / "changelog.mdx"
            if lang == "en":
                notes_for_lang = release_notes
            else:
                console.print(f"[dim]Translating release notes to {lang}...[/dim]")
                notes_for_lang = translate_release_notes(
                    release_notes, lang, openai_client
                )
            if update_changelog(cl_path, version, notes_for_lang, lang=lang):
                console.print(f"[green]✓[/green] Updated {cl_path.relative_to(cwd)}")
                docs_files_staged.append(str(cl_path))
            else:
                console.print(
                    f"[yellow]Warning:[/yellow] Changelog not found at {cl_path.relative_to(cwd)}"
                )

        if not is_prerelease:
            if add_docs_version(docs_json_path, version):
                console.print(
                    f"[green]✓[/green] Added v{version} to docs version switcher"
                )
                docs_files_staged.append(str(docs_json_path))
            else:
                console.print(
                    f"[yellow]Warning:[/yellow] docs.json not found at {docs_json_path.relative_to(cwd)}"
                )

        if docs_files_staged:
            docs_branch = f"docs/changelog-v{version}"
            create_or_reset_branch(docs_branch)
            for f in docs_files_staged:
                run_command(["git", "add", f])
            run_command(
                [
                    "git",
                    "commit",
                    "-m",
                    f"docs: update changelog and version for v{version}",
                ]
            )
            console.print("[green]✓[/green] Committed docs updates")

            run_command(["git", "push", "-u", "origin", docs_branch])
            console.print(f"[green]✓[/green] Pushed branch {docs_branch}")

            pr_url = run_command(
                [
                    "gh",
                    "pr",
                    "create",
                    "--base",
                    "main",
                    "--title",
                    f"docs: update changelog and version for v{version}",
                    "--body",
                    "",
                ]
            )
            console.print("[green]✓[/green] Created docs PR")
            console.print(f"[cyan]PR URL:[/cyan] {pr_url}")
            return docs_branch

        return None
    for lang in changelog_langs:
        cl_path = cwd / "docs" / lang / "changelog.mdx"
        translated = " (translated)" if lang != "en" else ""
        console.print(
            f"[dim][DRY RUN][/dim] Would update {cl_path.relative_to(cwd)}{translated}"
        )
    if not is_prerelease:
        console.print(
            f"[dim][DRY RUN][/dim] Would add v{version} to docs version switcher"
        )
    else:
        console.print("[dim][DRY RUN][/dim] Skipping docs version (pre-release)")
    console.print(
        f"[dim][DRY RUN][/dim] Would create branch docs/changelog-v{version}, PR, and wait for merge"
    )
    return None


def _create_tag_and_release(
    tag_name: str,
    release_notes: str,
    is_prerelease: bool,
) -> None:
    """Create git tag, push it, and create a GitHub release."""
    with console.status(f"[cyan]Creating tag {tag_name}..."):
        try:
            run_command(["git", "tag", "-a", tag_name, "-m", release_notes])
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗[/red] Created tag {tag_name}: {e}")
            sys.exit(1)
    console.print(f"[green]✓[/green] Created tag {tag_name}")

    with console.status(f"[cyan]Pushing tag {tag_name}..."):
        try:
            run_command(["git", "push", "origin", tag_name])
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗[/red] Pushed tag {tag_name}: {e}")
            sys.exit(1)
    console.print(f"[green]✓[/green] Pushed tag {tag_name}")

    with console.status("[cyan]Creating GitHub Release..."):
        try:
            gh_cmd = [
                "gh",
                "release",
                "create",
                tag_name,
                "--title",
                tag_name,
                "--notes",
                release_notes,
            ]
            if is_prerelease:
                gh_cmd.append("--prerelease")

            run_command(gh_cmd)
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗[/red] Created GitHub Release: {e}")
            sys.exit(1)

    release_type = "prerelease" if is_prerelease else "release"
    console.print(f"[green]✓[/green] Created GitHub {release_type} for {tag_name}")


_ENTERPRISE_REPO: Final[str | None] = os.getenv("ENTERPRISE_REPO")
_ENTERPRISE_VERSION_DIRS: Final[tuple[str, ...]] = tuple(
    d.strip() for d in os.getenv("ENTERPRISE_VERSION_DIRS", "").split(",") if d.strip()
)
_ENTERPRISE_CREWAI_DEP_PATH: Final[str | None] = os.getenv("ENTERPRISE_CREWAI_DEP_PATH")
_ENTERPRISE_EXTRA_PACKAGES: Final[tuple[str, ...]] = tuple(
    p.strip()
    for p in os.getenv("ENTERPRISE_EXTRA_PACKAGES", "").split(",")
    if p.strip()
)
_ENTERPRISE_WORKFLOW_PATHS: Final[tuple[str, ...]] = tuple(
    p.strip()
    for p in os.getenv("ENTERPRISE_WORKFLOW_PATHS", "").split(",")
    if p.strip()
)


def _update_enterprise_crewai_dep(pyproject_path: Path, version: str) -> bool:
    """Update the crewai[tools] pin in an enterprise pyproject.toml.

    Args:
        pyproject_path: Path to the pyproject.toml file.
        version: New crewai version string.

    Returns:
        True if the file was modified.
    """
    if not pyproject_path.exists():
        return False

    content = pyproject_path.read_text()
    new_content = _pin_crewai_deps(content, version)
    if new_content != content:
        pyproject_path.write_text(new_content)
        return True
    return False


def _update_enterprise_workflows(repo_dir: Path, version: str) -> list[Path]:
    """Update crewai version pins in enterprise CI workflow files.

    Applies ``_repin_crewai_install`` line-by-line on the raw file so
    only version numbers change and all formatting is preserved.

    Args:
        repo_dir: Root of the cloned enterprise repo.
        version: New crewai version string.

    Returns:
        List of workflow paths that were modified.
    """
    updated: list[Path] = []
    for rel_path in _ENTERPRISE_WORKFLOW_PATHS:
        workflow = repo_dir / rel_path
        if not workflow.exists():
            continue

        raw = workflow.read_text()
        lines = raw.splitlines(keepends=True)
        changed = False
        for i, line in enumerate(lines):
            if "crewai[" not in line:
                continue
            new_line = _repin_crewai_install(line, version)
            if new_line != line:
                lines[i] = new_line
                changed = True

        if changed:
            new_raw = "".join(lines)
        else:
            new_raw = raw

        if new_raw != raw:
            workflow.write_text(new_raw)
            updated.append(workflow)

    return updated


def _repin_crewai_install(run_value: str, version: str) -> str:
    """Rewrite ``crewai[extras]==old`` pins in a shell command string.

    Splits on the known ``crewai[`` prefix and reconstructs the pin
    with the new version, avoiding regex.

    Args:
        run_value: The ``run:`` string from a workflow step.
        version: New version to pin to.

    Returns:
        The updated string.
    """
    result: list[str] = []
    remainder = run_value
    marker = "crewai["
    while marker in remainder:
        before, _, after = remainder.partition(marker)
        result.append(before)
        bracket_end = after.index("]")
        extras = after[:bracket_end]
        rest = after[bracket_end + 1 :]
        if rest.startswith("=="):
            ver_start = 2
            ver_end = ver_start
            while ver_end < len(rest) and rest[ver_end] not in ('"', "'", " ", "\n"):
                ver_end += 1
            result.append(f"crewai[{extras}]=={version}")
            remainder = rest[ver_end:]
        else:
            result.append(f"crewai[{extras}]")
            remainder = rest
    result.append(remainder)
    return "".join(result)


_DEPLOYMENT_TEST_REPO: Final[str] = "crewAIInc/crew_deployment_test"

_PYPI_POLL_INTERVAL: Final[int] = 15
_PYPI_POLL_TIMEOUT: Final[int] = 600


def _update_deployment_test_repo(version: str, is_prerelease: bool) -> None:
    """Update the deployment test repo to pin the new crewai version.

    Clones the repo, updates the crewai[tools] pin in pyproject.toml,
    regenerates the lockfile, commits, and pushes directly to main.

    Args:
        version: New crewai version string.
        is_prerelease: Whether this is a pre-release version.
    """
    console.print(
        f"\n[bold cyan]Updating {_DEPLOYMENT_TEST_REPO} to {version}[/bold cyan]"
    )

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(tmp) / "crew_deployment_test"
        run_command(["gh", "repo", "clone", _DEPLOYMENT_TEST_REPO, str(repo_dir)])
        console.print(f"[green]✓[/green] Cloned {_DEPLOYMENT_TEST_REPO}")

        pyproject = repo_dir / "pyproject.toml"
        content = pyproject.read_text()
        new_content = _pin_crewai_deps(content, version)
        if new_content == content:
            console.print(
                "[yellow]Warning:[/yellow] No crewai[tools] pin found to update"
            )
            return
        pyproject.write_text(new_content)
        console.print(f"[green]✓[/green] Updated crewai[tools] pin to {version}")

        lock_cmd = [
            "uv",
            "lock",
            "--refresh-package",
            "crewai",
            "--refresh-package",
            "crewai-tools",
        ]
        if is_prerelease:
            lock_cmd.append("--prerelease=allow")

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                run_command(lock_cmd, cwd=repo_dir)
                break
            except subprocess.CalledProcessError:
                if attempt == max_retries:
                    console.print(
                        f"[red]Error:[/red] uv lock failed after {max_retries} attempts"
                    )
                    raise
                console.print(
                    f"[yellow]uv lock failed (attempt {attempt}/{max_retries}),"
                    f" retrying in {_PYPI_POLL_INTERVAL}s...[/yellow]"
                )
                time.sleep(_PYPI_POLL_INTERVAL)
        console.print("[green]✓[/green] Lockfile updated")

        run_command(["git", "add", "pyproject.toml", "uv.lock"], cwd=repo_dir)
        run_command(
            ["git", "commit", "-m", f"chore: bump crewai to {version}"],
            cwd=repo_dir,
        )
        run_command(["git", "push"], cwd=repo_dir)
        console.print(f"[green]✓[/green] Pushed to {_DEPLOYMENT_TEST_REPO}")


def _wait_for_pypi(package: str, version: str) -> None:
    """Poll PyPI until a specific package version is available.

    Args:
        package: PyPI package name.
        version: Version string to wait for.
    """
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    deadline = time.monotonic() + _PYPI_POLL_TIMEOUT

    console.print(f"[cyan]Waiting for {package}=={version} to appear on PyPI...[/cyan]")
    while time.monotonic() < deadline:
        try:
            with urlopen(url) as resp:  # noqa: S310
                if resp.status == 200:
                    console.print(
                        f"[green]✓[/green] {package}=={version} is available on PyPI"
                    )
                    return
        except Exception:  # noqa: S110
            pass
        time.sleep(_PYPI_POLL_INTERVAL)

    console.print(
        f"[red]Error:[/red] Timed out waiting for {package}=={version} on PyPI"
    )
    sys.exit(1)


def _release_enterprise(version: str, is_prerelease: bool, dry_run: bool) -> None:
    """Clone the enterprise repo, bump versions, and create a release PR.

    Expects ENTERPRISE_REPO, ENTERPRISE_VERSION_DIRS, and
    ENTERPRISE_CREWAI_DEP_PATH to be validated before calling.

    Args:
        version: New version string.
        is_prerelease: Whether this is a pre-release version.
        dry_run: Show what would be done without making changes.
    """
    if (
        not _ENTERPRISE_REPO
        or not _ENTERPRISE_VERSION_DIRS
        or not _ENTERPRISE_CREWAI_DEP_PATH
    ):
        console.print("[red]Error:[/red] Enterprise env vars not configured")
        sys.exit(1)

    enterprise_repo: str = _ENTERPRISE_REPO
    enterprise_dep_path: str = _ENTERPRISE_CREWAI_DEP_PATH

    console.print(
        f"\n[bold cyan]Phase 3: Releasing {enterprise_repo} {version}[/bold cyan]"
    )

    if dry_run:
        console.print(f"[dim][DRY RUN][/dim] Would clone {enterprise_repo}")
        for d in _ENTERPRISE_VERSION_DIRS:
            console.print(f"[dim][DRY RUN][/dim] Would update versions in {d}")
        console.print(
            f"[dim][DRY RUN][/dim] Would update crewai[tools] dep in "
            f"{enterprise_dep_path}"
        )
        console.print(
            "[dim][DRY RUN][/dim] Would create bump PR, wait for merge, "
            "then tag and release"
        )
        return

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(tmp) / enterprise_repo.split("/")[-1]
        console.print(f"Cloning {enterprise_repo}...")
        run_command(["gh", "repo", "clone", enterprise_repo, str(repo_dir)])
        console.print(f"[green]✓[/green] Cloned {enterprise_repo}")

        for rel_dir in _ENTERPRISE_VERSION_DIRS:
            pkg_dir = repo_dir / rel_dir
            if not pkg_dir.exists():
                console.print(
                    f"[yellow]Warning:[/yellow] {rel_dir} not found, skipping"
                )
                continue

            for vfile in find_version_files(pkg_dir):
                if update_version_in_file(vfile, version):
                    console.print(
                        f"[green]✓[/green] Updated: {vfile.relative_to(repo_dir)}"
                    )

            pyproject = pkg_dir / "pyproject.toml"
            if pyproject.exists():
                if update_pyproject_version(pyproject, version):
                    console.print(
                        f"[green]✓[/green] Updated version in: "
                        f"{pyproject.relative_to(repo_dir)}"
                    )
                if update_pyproject_dependencies(
                    pyproject, version, extra_packages=list(_ENTERPRISE_EXTRA_PACKAGES)
                ):
                    console.print(
                        f"[green]✓[/green] Updated deps in: "
                        f"{pyproject.relative_to(repo_dir)}"
                    )

        enterprise_pyproject = repo_dir / enterprise_dep_path
        if _update_enterprise_crewai_dep(enterprise_pyproject, version):
            console.print(
                f"[green]✓[/green] Updated crewai[tools] dep in {enterprise_dep_path}"
            )

        for wf in _update_enterprise_workflows(repo_dir, version):
            console.print(
                f"[green]✓[/green] Updated crewai pin in {wf.relative_to(repo_dir)}"
            )

        _wait_for_pypi("crewai", version)

        console.print("\nSyncing workspace...")
        sync_cmd = [
            "uv",
            "sync",
            "--refresh-package",
            "crewai",
            "--refresh-package",
            "crewai-tools",
            "--refresh-package",
            "crewai-files",
        ]
        if is_prerelease:
            sync_cmd.append("--prerelease=allow")

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                run_command(sync_cmd, cwd=repo_dir)
                break
            except subprocess.CalledProcessError:
                if attempt == max_retries:
                    console.print(
                        f"[red]Error:[/red] uv sync failed after {max_retries} attempts"
                    )
                    raise
                console.print(
                    f"[yellow]uv sync failed (attempt {attempt}/{max_retries}),"
                    f" retrying in {_PYPI_POLL_INTERVAL}s...[/yellow]"
                )
                time.sleep(_PYPI_POLL_INTERVAL)
        console.print("[green]✓[/green] Workspace synced")

        branch_name = f"feat/bump-version-{version}"
        create_or_reset_branch(branch_name, cwd=repo_dir)
        run_command(["git", "add", "."], cwd=repo_dir)
        run_command(
            ["git", "commit", "-m", f"feat: bump versions to {version}"],
            cwd=repo_dir,
        )
        console.print("[green]✓[/green] Changes committed")

        run_command(["git", "push", "-u", "origin", branch_name], cwd=repo_dir)
        console.print("[green]✓[/green] Branch pushed")

        pr_url = run_command(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                enterprise_repo,
                "--base",
                "main",
                "--title",
                f"feat: bump versions to {version}",
                "--body",
                "",
            ],
            cwd=repo_dir,
        )
        console.print("[green]✓[/green] Enterprise bump PR created")
        console.print(f"[cyan]PR URL:[/cyan] {pr_url}")

        _poll_pr_until_merged(branch_name, "enterprise bump PR", repo=enterprise_repo)

        run_command(["git", "checkout", "main"], cwd=repo_dir)
        run_command(["git", "pull"], cwd=repo_dir)

        tag_name = version
        run_command(
            ["git", "tag", "-a", tag_name, "-m", f"Release {version}"],
            cwd=repo_dir,
        )
        run_command(["git", "push", "origin", tag_name], cwd=repo_dir)
        console.print(f"[green]✓[/green] Pushed tag {tag_name}")

        gh_cmd = [
            "gh",
            "release",
            "create",
            tag_name,
            "--repo",
            enterprise_repo,
            "--title",
            tag_name,
            "--notes",
            f"Release {version}",
        ]
        if is_prerelease:
            gh_cmd.append("--prerelease")

        run_command(gh_cmd)
        release_type = "prerelease" if is_prerelease else "release"
        console.print(
            f"[green]✓[/green] Created GitHub {release_type} for "
            f"{enterprise_repo} {tag_name}"
        )


def _trigger_pypi_publish(tag_name: str, wait: bool = False) -> None:
    """Trigger the PyPI publish GitHub Actions workflow.

    Args:
        tag_name: The release tag to publish.
        wait: Block until the workflow run completes.
    """
    prev_run_id = ""
    if wait:
        try:
            prev_run_id = run_command(
                [
                    "gh",
                    "run",
                    "list",
                    "--workflow=publish.yml",
                    "--limit=1",
                    "--json=databaseId",
                    "--jq=.[0].databaseId",
                ]
            )
        except subprocess.CalledProcessError:
            console.print(
                "[yellow]Note:[/yellow] Could not determine previous workflow run; "
                "continuing without previous run ID"
            )

    with console.status("[cyan]Triggering PyPI publish workflow..."):
        try:
            run_command(
                [
                    "gh",
                    "workflow",
                    "run",
                    "publish.yml",
                    "-f",
                    f"release_tag={tag_name}",
                ]
            )
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗[/red] Triggered PyPI publish workflow: {e}")
            sys.exit(1)
    console.print("[green]✓[/green] Triggered PyPI publish workflow")

    if wait:
        console.print("[cyan]Waiting for PyPI publish workflow to complete...[/cyan]")
        run_id = ""
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            time.sleep(5)
            try:
                run_id = run_command(
                    [
                        "gh",
                        "run",
                        "list",
                        "--workflow=publish.yml",
                        "--limit=1",
                        "--json=databaseId",
                        "--jq=.[0].databaseId",
                    ]
                )
            except subprocess.CalledProcessError:
                continue
            if run_id and run_id != prev_run_id:
                break

        if not run_id or run_id == prev_run_id:
            console.print(
                "[red]Error:[/red] Could not find the PyPI publish workflow run"
            )
            sys.exit(1)

        try:
            run_command(["gh", "run", "watch", run_id, "--exit-status"])
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗[/red] PyPI publish workflow failed: {e}")
            sys.exit(1)
        console.print("[green]✓[/green] PyPI publish workflow completed")


@click.group()
def cli() -> None:
    """Development tools for version bumping and git automation."""


@click.command()
@click.argument("version")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be done without making changes"
)
@click.option("--no-push", is_flag=True, help="Don't push changes to remote")
@click.option(
    "--no-commit", is_flag=True, help="Don't commit changes (just update files)"
)
def bump(version: str, dry_run: bool, no_push: bool, no_commit: bool) -> None:
    """Bump version across all packages in lib/.

    Args:
        version: New version to set (e.g., 1.0.0, 1.0.0a1).
        dry_run: Show what would be done without making changes.
        no_push: Don't push changes to remote.
        no_commit: Don't commit changes (just update files).
    """
    console.print(
        f"\n[yellow]Note:[/yellow] [bold]devtools bump[/bold] only bumps versions "
        f"in this repo. It will not tag, publish to PyPI, or release enterprise.\n"
        f"If you want a full end-to-end release, run "
        f"[bold]devtools release {version}[/bold] instead."
    )
    if not Confirm.ask("Continue with bump only?", default=True):
        sys.exit(0)

    try:
        check_gh_installed()

        cwd = Path.cwd()
        lib_dir = cwd / "lib"

        if not dry_run:
            console.print("Checking git status...")
            check_git_clean()
            console.print("[green]✓[/green] Working directory is clean")
        else:
            console.print("[dim][DRY RUN][/dim] Would check git status")

        packages = get_packages(lib_dir)

        console.print(f"\nFound {len(packages)} package(s) to update:")
        for pkg in packages:
            console.print(f"  - {pkg.name}")

        if no_commit:
            console.print(f"\nUpdating version to {version}...")
            _update_all_versions(cwd, lib_dir, version, packages, dry_run)
            console.print("\nSkipping git operations (--no-commit flag set)")
        else:
            branch_name = f"feat/bump-version-{version}"
            if not dry_run:
                console.print(f"\nCreating branch {branch_name}...")
                create_or_reset_branch(branch_name)
                console.print("[green]✓[/green] Branch created")

                console.print(f"\nUpdating version to {version}...")
                _update_all_versions(cwd, lib_dir, version, packages, dry_run)

                console.print("\nCommitting changes...")
                run_command(["git", "add", "."])
                run_command(
                    ["git", "commit", "-m", f"feat: bump versions to {version}"]
                )
                console.print("[green]✓[/green] Changes committed")

                if not no_push:
                    console.print("\nPushing branch...")
                    run_command(["git", "push", "-u", "origin", branch_name])
                    console.print("[green]✓[/green] Branch pushed")
            else:
                console.print(
                    f"[dim][DRY RUN][/dim] Would create branch: {branch_name}"
                )
                console.print(f"\nUpdating version to {version}...")
                _update_all_versions(cwd, lib_dir, version, packages, dry_run)
                console.print(
                    f"[dim][DRY RUN][/dim] Would commit: feat: bump versions to {version}"
                )
                if not no_push:
                    console.print(
                        f"[dim][DRY RUN][/dim] Would push branch: {branch_name}"
                    )

            if not dry_run and not no_push:
                console.print("\nCreating pull request...")
                run_command(
                    [
                        "gh",
                        "pr",
                        "create",
                        "--base",
                        "main",
                        "--title",
                        f"feat: bump versions to {version}",
                        "--body",
                        "",
                    ]
                )
                console.print("[green]✓[/green] Pull request created")
            elif dry_run:
                console.print(
                    f"[dim][DRY RUN][/dim] Would create PR: feat: bump versions to {version}"
                )
            else:
                console.print("\nSkipping PR creation (--no-push flag set)")

        console.print(f"\n[green]✓[/green] Version bump to {version} complete!")

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running command:[/red] {e}")
        if e.stderr:
            console.print(e.stderr)
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@click.command()
@click.option(
    "--dry-run", is_flag=True, help="Show what would be done without making changes"
)
@click.option("--no-edit", is_flag=True, help="Skip editing release notes")
def tag(dry_run: bool, no_edit: bool) -> None:
    """Create and push a version tag on main branch.

    Run this after the version bump PR has been merged.
    Automatically detects version from __version__ in packages.

    Args:
        dry_run: Show what would be done without making changes.
        no_edit: Skip editing release notes.
    """
    console.print(
        "\n[yellow]Note:[/yellow] [bold]devtools tag[/bold] only tags and creates "
        "a GitHub release for this repo. It will not bump versions, publish to "
        "PyPI, or release enterprise.\n"
        "If you want a full end-to-end release, run "
        "[bold]devtools release <version>[/bold] instead."
    )
    if not Confirm.ask("Continue with tag only?", default=True):
        sys.exit(0)

    try:
        cwd = Path.cwd()
        lib_dir = cwd / "lib"

        packages = get_packages(lib_dir)

        with console.status("[cyan]Validating package versions..."):
            versions = {}
            for pkg in packages:
                version_files = find_version_files(pkg)
                for vfile in version_files:
                    content = vfile.read_text()
                    for line in content.splitlines():
                        if line.strip().startswith("__version__"):
                            ver = line.split("=")[1].strip().strip('"').strip("'")
                            versions[vfile.relative_to(cwd)] = ver
                            break

        if not versions:
            console.print(
                "[red]✗[/red] Validated package versions: Could not find __version__ in any package"
            )
            sys.exit(1)

        unique_versions = set(versions.values())
        if len(unique_versions) > 1:
            console.print(
                "[red]✗[/red] Validated package versions: Version mismatch detected"
            )
            for file, ver in versions.items():
                console.print(f"  {file}: {ver}")
            sys.exit(1)

        version = unique_versions.pop()
        console.print(f"[green]✓[/green] Validated packages @ [bold]{version}[/bold]")
        tag_name = version

        if not dry_run:
            with console.status("[cyan]Checking out main branch..."):
                try:
                    run_command(["git", "checkout", "main"])
                except subprocess.CalledProcessError as e:
                    console.print(f"[red]✗[/red] Checked out main branch: {e}")
                    sys.exit(1)
            console.print("[green]✓[/green] On main branch")

            with console.status("[cyan]Pulling latest changes..."):
                try:
                    run_command(["git", "pull"])
                except subprocess.CalledProcessError as e:
                    console.print(f"[red]✗[/red] Pulled latest changes: {e}")
                    sys.exit(1)
            console.print("[green]✓[/green] main branch up to date")

        release_notes, openai_client, is_prerelease = _generate_release_notes(
            version, tag_name, no_edit
        )

        docs_branch = _update_docs_and_create_pr(
            cwd, version, release_notes, openai_client, is_prerelease, dry_run
        )
        if docs_branch:
            _poll_pr_until_merged(docs_branch, "docs PR")
            run_command(["git", "checkout", "main"])
            run_command(["git", "pull"])
            console.print("[green]✓[/green] main branch updated with docs changes")

        if not dry_run:
            _create_tag_and_release(tag_name, release_notes, is_prerelease)

        console.print(
            f"\n[green]✓[/green] Packages @ [bold]{version}[/bold] tagged successfully!"
        )

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running command:[/red] {e}")
        if e.stderr:
            console.print(e.stderr)
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@click.command()
@click.argument("version")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be done without making changes"
)
@click.option("--no-edit", is_flag=True, help="Skip editing release notes")
@click.option(
    "--skip-enterprise",
    is_flag=True,
    help="Skip the enterprise release phase",
)
@click.option(
    "--skip-to-enterprise",
    is_flag=True,
    help="Skip phases 1 & 2, run only the enterprise release phase",
)
def release(
    version: str,
    dry_run: bool,
    no_edit: bool,
    skip_enterprise: bool,
    skip_to_enterprise: bool,
) -> None:
    """Full release: bump versions, tag, and publish a GitHub release.

    Combines bump and tag into a single workflow. Creates a version bump PR,
    waits for it to be merged, then generates release notes, updates docs,
    creates the tag, and publishes a GitHub release. Then bumps versions and
    releases the enterprise repo.

    Args:
        version: New version to set (e.g., 1.0.0, 1.0.0a1).
        dry_run: Show what would be done without making changes.
        no_edit: Skip editing release notes.
        skip_enterprise: Skip the enterprise release phase.
        skip_to_enterprise: Skip phases 1 & 2, run only the enterprise release phase.
    """
    flags: list[str] = []
    if no_edit:
        flags.append("--no-edit")
    if skip_enterprise:
        flags.append("--skip-enterprise")
    flag_suffix = (" " + " ".join(flags)) if flags else ""
    enterprise_hint = (
        ""
        if skip_enterprise
        else f"\n\nThen release enterprise:\n\n"
        f"  devtools release {version} --skip-to-enterprise"
    )

    check_gh_installed()

    if skip_enterprise and skip_to_enterprise:
        console.print(
            "[red]Error:[/red] Cannot use both --skip-enterprise "
            "and --skip-to-enterprise"
        )
        sys.exit(1)

    if not skip_enterprise or skip_to_enterprise:
        missing: list[str] = []
        if not _ENTERPRISE_REPO:
            missing.append("ENTERPRISE_REPO")
        if not _ENTERPRISE_VERSION_DIRS:
            missing.append("ENTERPRISE_VERSION_DIRS")
        if not _ENTERPRISE_CREWAI_DEP_PATH:
            missing.append("ENTERPRISE_CREWAI_DEP_PATH")
        if missing:
            console.print(
                f"[red]Error:[/red] Missing required environment variable(s): "
                f"{', '.join(missing)}\n"
                f"Set them or pass --skip-enterprise to skip the enterprise release."
            )
            sys.exit(1)

    cwd = Path.cwd()
    lib_dir = cwd / "lib"

    is_prerelease = _is_prerelease(version)

    if skip_to_enterprise:
        try:
            _release_enterprise(version, is_prerelease, dry_run)
        except BaseException as e:
            _print_release_error(e)
            _resume_hint(
                f"Fix the issue, then re-run:\n\n"
                f"  devtools release {version} --skip-to-enterprise"
            )
            sys.exit(1)
        console.print(
            f"\n[green]✓[/green] Enterprise release [bold]{version}[/bold] complete!"
        )
        return

    if not dry_run:
        console.print("Checking git status...")
        check_git_clean()
        console.print("[green]✓[/green] Working directory is clean")
    else:
        console.print("[dim][DRY RUN][/dim] Would check git status")

    packages = get_packages(lib_dir)

    console.print(f"\nFound {len(packages)} package(s) to update:")
    for pkg in packages:
        console.print(f"  - {pkg.name}")

    console.print(f"\n[bold cyan]Phase 1: Bumping versions to {version}[/bold cyan]")

    try:
        branch_name = f"feat/bump-version-{version}"
        if not dry_run:
            console.print(f"\nCreating branch {branch_name}...")
            create_or_reset_branch(branch_name)
            console.print("[green]✓[/green] Branch created")

            _update_all_versions(cwd, lib_dir, version, packages, dry_run)

            console.print("\nCommitting changes...")
            run_command(["git", "add", "."])
            run_command(["git", "commit", "-m", f"feat: bump versions to {version}"])
            console.print("[green]✓[/green] Changes committed")

            console.print("\nPushing branch...")
            run_command(["git", "push", "-u", "origin", branch_name])
            console.print("[green]✓[/green] Branch pushed")

            console.print("\nCreating pull request...")
            bump_pr_url = run_command(
                [
                    "gh",
                    "pr",
                    "create",
                    "--base",
                    "main",
                    "--title",
                    f"feat: bump versions to {version}",
                    "--body",
                    "",
                ]
            )
            console.print("[green]✓[/green] Pull request created")
            console.print(f"[cyan]PR URL:[/cyan] {bump_pr_url}")

            _poll_pr_until_merged(branch_name, "bump PR")
        else:
            console.print(f"[dim][DRY RUN][/dim] Would create branch: {branch_name}")
            _update_all_versions(cwd, lib_dir, version, packages, dry_run)
            console.print(
                f"[dim][DRY RUN][/dim] Would commit: feat: bump versions to {version}"
            )
            console.print(
                "[dim][DRY RUN][/dim] Would push branch, create PR, and wait for merge"
            )
    except BaseException as e:
        _print_release_error(e)
        _resume_hint(
            f"Phase 1 failed. Fix the issue, then re-run:\n\n"
            f"  devtools release {version}{flag_suffix}"
        )
        sys.exit(1)

    console.print(f"\n[bold cyan]Phase 2: Tagging and releasing {version}[/bold cyan]")

    try:
        tag_name = version

        if not dry_run:
            with console.status("[cyan]Checking out main branch..."):
                run_command(["git", "checkout", "main"])
            console.print("[green]✓[/green] On main branch")

            with console.status("[cyan]Pulling latest changes..."):
                run_command(["git", "pull"])
            console.print("[green]✓[/green] main branch up to date")

        release_notes, openai_client, is_prerelease = _generate_release_notes(
            version, tag_name, no_edit
        )

        docs_branch = _update_docs_and_create_pr(
            cwd, version, release_notes, openai_client, is_prerelease, dry_run
        )
        if docs_branch:
            _poll_pr_until_merged(docs_branch, "docs PR")
            run_command(["git", "checkout", "main"])
            run_command(["git", "pull"])
            console.print("[green]✓[/green] main branch updated with docs changes")

        if not dry_run:
            _create_tag_and_release(tag_name, release_notes, is_prerelease)
    except BaseException as e:
        _print_release_error(e)
        _resume_hint(
            "Phase 2 failed before PyPI publish. The bump PR is already merged.\n"
            "Fix the issue, then resume with:\n\n"
            "  devtools tag"
            f"\n\nAfter tagging, publish to PyPI and update deployment test:\n\n"
            f"  gh workflow run publish.yml -f release_tag={version}"
            f"{enterprise_hint}"
        )
        sys.exit(1)

    try:
        if not dry_run:
            _trigger_pypi_publish(tag_name, wait=True)
    except BaseException as e:
        _print_release_error(e)
        _resume_hint(
            f"Phase 2 failed at PyPI publish. Tag and GitHub release already exist.\n"
            f"Retry PyPI publish manually:\n\n"
            f"  gh workflow run publish.yml -f release_tag={version}"
            f"{enterprise_hint}"
        )
        sys.exit(1)

    try:
        if not dry_run:
            _update_deployment_test_repo(version, is_prerelease)
    except BaseException as e:
        _print_release_error(e)
        _resume_hint(
            f"Phase 2 failed updating deployment test repo. "
            f"Tag, release, and PyPI are done.\n"
            f"Fix the issue and update {_DEPLOYMENT_TEST_REPO} manually."
            f"{enterprise_hint}"
        )
        sys.exit(1)

    if not skip_enterprise:
        try:
            _release_enterprise(version, is_prerelease, dry_run)
        except BaseException as e:
            _print_release_error(e)
            _resume_hint(
                f"Phase 3 (enterprise) failed. Phases 1 & 2 completed successfully.\n"
                f"Fix the issue, then resume:\n\n"
                f"  devtools release {version} --skip-to-enterprise"
            )
            sys.exit(1)

    console.print(f"\n[green]✓[/green] Release [bold]{version}[/bold] complete!")


cli.add_command(bump)
cli.add_command(tag)
cli.add_command(release)
cli.add_command(docs_check)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
