"""Development tools for version bumping and git automation."""

import os
from pathlib import Path
import subprocess
import sys

import click
from dotenv import load_dotenv
from github import Github
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm

from crewai_devtools.prompts import RELEASE_NOTES_PROMPT


load_dotenv()

console = Console()


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


def update_pyproject_dependencies(file_path: Path, new_version: str) -> bool:
    """Update workspace dependency versions in pyproject.toml.

    Args:
        file_path: Path to pyproject.toml file.
        new_version: New version string.

    Returns:
        True if any dependencies were updated, False otherwise.
    """
    if not file_path.exists():
        return False

    content = file_path.read_text()
    lines = content.splitlines()
    updated = False

    workspace_packages = ["crewai", "crewai-tools", "crewai-devtools"]

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
        # Get GitHub token from gh CLI
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


@click.group()
def cli() -> None:
    """Development tools for version bumping and git automation."""


@click.command()
@click.argument("version")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be done without making changes"
)
@click.option("--no-push", is_flag=True, help="Don't push changes to remote")
def bump(version: str, dry_run: bool, no_push: bool) -> None:
    """Bump version across all packages in lib/.

    Args:
        version: New version to set (e.g., 1.0.0, 1.0.0a1).
        dry_run: Show what would be done without making changes.
        no_push: Don't push changes to remote.
    """
    try:
        # Check prerequisites
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

        console.print(f"\nUpdating version to {version}...")
        updated_files = []

        for pkg in packages:
            version_files = find_version_files(pkg)
            for vfile in version_files:
                if dry_run:
                    console.print(
                        f"[dim][DRY RUN][/dim] Would update: {vfile.relative_to(cwd)}"
                    )
                else:
                    if update_version_in_file(vfile, version):
                        console.print(
                            f"[green]✓[/green] Updated: {vfile.relative_to(cwd)}"
                        )
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

        if not dry_run:
            console.print("\nSyncing workspace...")
            run_command(["uv", "sync"])
            console.print("[green]✓[/green] Workspace synced")
        else:
            console.print("[dim][DRY RUN][/dim] Would run: uv sync")

        branch_name = f"feat/bump-version-{version}"
        if not dry_run:
            console.print(f"\nCreating branch {branch_name}...")
            run_command(["git", "checkout", "-b", branch_name])
            console.print("[green]✓[/green] Branch created")

            console.print("\nCommitting changes...")
            run_command(["git", "add", "."])
            run_command(["git", "commit", "-m", f"feat: bump versions to {version}"])
            console.print("[green]✓[/green] Changes committed")

            if not no_push:
                console.print("\nPushing branch...")
                run_command(["git", "push", "-u", "origin", branch_name])
                console.print("[green]✓[/green] Branch pushed")
        else:
            console.print(f"[dim][DRY RUN][/dim] Would create branch: {branch_name}")
            console.print(
                f"[dim][DRY RUN][/dim] Would commit: feat: bump versions to {version}"
            )
            if not no_push:
                console.print(f"[dim][DRY RUN][/dim] Would push branch: {branch_name}")

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

        release_notes = f"Release {version}"
        commits = ""

        with console.status("[cyan]Generating release notes..."):
            try:
                prev_bump_commit = run_command(
                    [
                        "git",
                        "log",
                        "--grep=^feat: bump versions to",
                        "--format=%H",
                        "-n",
                        "2",
                    ]
                )
                commits_list = prev_bump_commit.strip().split("\n")

                if len(commits_list) > 1:
                    prev_commit = commits_list[1]
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

            if commits.strip():
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                contributors_section = ""
                if github_contributors:
                    contributors_section = f"\n\n## Contributors\n\n{', '.join([f'@{u}' for u in github_contributors])}"

                prompt = RELEASE_NOTES_PROMPT.substitute(
                    version=version,
                    commits=commits,
                    contributors_section=contributors_section,
                )

                response = client.chat.completions.create(
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

                release_notes = (
                    response.choices[0].message.content or f"Release {version}"
                )

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
                    f"[yellow]Warning:[/yellow] Could not generate release notes with OpenAI: {e}"
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

        if not dry_run:
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

            is_prerelease = any(
                indicator in version.lower()
                for indicator in ["a", "b", "rc", "alpha", "beta", "dev"]
            )

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
            console.print(
                f"[green]✓[/green] Created GitHub {release_type} for {tag_name}"
            )

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


cli.add_command(bump)
cli.add_command(tag)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
