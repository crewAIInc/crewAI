"""Skill Repository CLI commands for CrewAI."""

from __future__ import annotations

import base64
import io
import json
import os
import tarfile
from pathlib import Path
import zipfile

from rich.console import Console
from rich.table import Table

from crewai_cli.command import BaseCommand, PlusAPIMixin
from crewai_cli.config import Settings
from crewai_cli.constants import DEFAULT_CREWAI_ENTERPRISE_URL


console = Console()

_SKILL_MD_TEMPLATE = """\
---
name: {name}
version: 0.1.0
description: |
  A short description of what this skill does.
---

## Instructions

Describe the skill behaviour here. This section is shown to the agent at activation time.
"""


class SkillCommand(BaseCommand, PlusAPIMixin):
    """Skill Repository related operations for CrewAI projects."""

    def __init__(self) -> None:
        BaseCommand.__init__(self)
        PlusAPIMixin.__init__(self, telemetry=self._telemetry)

    # ------------------------------------------------------------------
    # create
    # ------------------------------------------------------------------

    def create(self, name: str, in_project: bool = True) -> None:
        """Scaffold a new skill directory.

        If pyproject.toml is present (crew project), creates ./skills/{name}/.
        Otherwise creates ./{name}/.
        """
        if in_project and os.path.isfile("pyproject.toml"):
            skill_dir = Path("skills") / name
        else:
            skill_dir = Path(name)

        if skill_dir.exists():
            console.print(f"[red]Directory {skill_dir} already exists.[/red]")
            raise SystemExit(1)

        skill_dir.mkdir(parents=True)
        (skill_dir / "scripts").mkdir()
        (skill_dir / "references").mkdir()
        (skill_dir / "assets").mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(_SKILL_MD_TEMPLATE.format(name=name))

        console.print(
            f"[green]Created skill [bold]{name}[/bold] at [bold]{skill_dir}[/bold].[/green]"
        )
        console.print(
            f"Edit [bold]{skill_md}[/bold] to define the skill instructions."
        )

    # ------------------------------------------------------------------
    # install
    # ------------------------------------------------------------------

    def install(self, ref: str) -> None:
        """Download and install a registry skill.

        Format: @org/name

        Inside a crew project (pyproject.toml present): installs to ./skills/{name}/
        Outside a project: installs to ~/.crewai/skills/{org}/{name}/
        """
        if not ref.startswith("@"):
            console.print(
                "[red]Invalid skill reference. Use the format @org/name.[/red]"
            )
            raise SystemExit(1)

        without_at = ref[1:]
        if "/" not in without_at:
            console.print(
                "[red]Invalid skill reference. Use the format @org/name.[/red]"
            )
            raise SystemExit(1)

        org, _, name = without_at.partition("/")
        if not org or not name:
            console.print(
                "[red]Invalid skill reference: org and name must be non-empty.[/red]"
            )
            raise SystemExit(1)

        console.print(f"[bold blue]Downloading skill {ref}...[/bold blue]")

        get_response = self.plus_api_client.get_skill(org, name)

        if get_response.status_code == 404:
            console.print(
                f"[red]Skill {ref} not found. Ensure it has been published and you have access.[/red]"
            )
            raise SystemExit(1)
        if get_response.status_code != 200:
            console.print(
                f"[red]Failed to download skill {ref}: {get_response.status_code}[/red]"
            )
            raise SystemExit(1)

        data = get_response.json()
        encoded = data.get("file", "")
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        archive_bytes = base64.b64decode(encoded)
        version = data.get("version")

        in_project = os.path.isfile("pyproject.toml")
        if in_project:
            dest = Path("skills") / name
            dest.mkdir(parents=True, exist_ok=True)
            self._unpack_archive(archive_bytes, dest)
            console.print(
                f"[green]Installed [bold]{ref}[/bold]{' (' + version + ')' if version else ''} to [bold]{dest}[/bold].[/green]"
            )
        else:
            from crewai_core.plus_api import PlusAPI as _  # noqa: F401 (ensure crewai_core present)
            cache_dir = Path.home() / ".crewai" / "skills" / org / name
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._unpack_archive(archive_bytes, cache_dir)
            import json as _json
            from datetime import datetime, timezone
            meta = {
                "org": org,
                "name": name,
                "version": version,
                "installed_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            (cache_dir / ".crewai_meta.json").write_text(_json.dumps(meta, indent=2))
            console.print(
                f"[green]Installed [bold]{ref}[/bold]{' (' + version + ')' if version else ''} to global cache.[/green]"
            )

    # ------------------------------------------------------------------
    # publish
    # ------------------------------------------------------------------

    def publish(self, is_public: bool, org: str | None, force: bool = False) -> None:
        """Publish the skill in the current directory to the registry."""
        skill_md = Path("SKILL.md")
        if not skill_md.exists():
            console.print(
                "[red]No SKILL.md found in current directory. "
                "Run this command from inside a skill directory.[/red]"
            )
            raise SystemExit(1)

        # Parse frontmatter to extract name + version
        try:
            frontmatter = self._parse_frontmatter(skill_md.read_text())
        except ValueError as exc:
            console.print(f"[red]Failed to parse SKILL.md frontmatter: {exc}[/red]")
            raise SystemExit(1) from exc

        name = frontmatter.get("name")
        version = frontmatter.get("version")
        description = frontmatter.get("description")

        if not name:
            console.print("[red]SKILL.md frontmatter must include a 'name' field.[/red]")
            raise SystemExit(1)

        if not version:
            console.print(
                "[red]SKILL.md frontmatter must include a 'version' field before publishing.[/red]"
            )
            raise SystemExit(1)

        settings = Settings()
        effective_org = org or settings.org_name
        if not effective_org:
            console.print(
                "[red]No organisation set. Run `crewai org switch <org_id>` first, "
                "or pass --org.[/red]"
            )
            raise SystemExit(1)

        console.print(
            f"[bold blue]Publishing skill [bold]{name}[/bold] v{version} to {effective_org}...[/bold blue]"
        )

        archive_bytes = self._build_skill_zip()
        encoded_file = (
            "data:application/zip;base64,"
            + base64.b64encode(archive_bytes).decode("utf-8")
        )

        response = self.plus_api_client.publish_skill(
            org=effective_org,
            name=name,
            version=version,
            is_public=is_public,
            description=description,
            encoded_file=encoded_file,
        )

        self._validate_response(response)

        base_url = settings.enterprise_base_url or DEFAULT_CREWAI_ENTERPRISE_URL
        console.print(
            f"[green]Published [bold]{effective_org}/{name}[/bold] v{version}.\n\n"
            "Security checks are running in the background. "
            "Your skill will be available once checks complete.\n"
            f"Monitor status at: {base_url}/crewai_plus/skills/{effective_org}/{name}[/green]"
        )

    # ------------------------------------------------------------------
    # list_cached
    # ------------------------------------------------------------------

    def list_cached(self) -> None:
        """Show locally installed skills."""
        table = Table(title="Installed Skills", show_lines=True)
        table.add_column("Source", style="dim")
        table.add_column("Ref")
        table.add_column("Version")
        table.add_column("Path")

        # Project-local ./skills/
        local_skills_dir = Path("skills")
        if local_skills_dir.is_dir():
            for skill_dir in sorted(local_skills_dir.iterdir()):
                if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                    version = self._read_version(skill_dir / "SKILL.md")
                    table.add_row(
                        "project",
                        skill_dir.name,
                        version or "-",
                        str(skill_dir),
                    )

        # Global cache
        cache_root = Path.home() / ".crewai" / "skills"
        if cache_root.exists():
            for org_dir in sorted(cache_root.iterdir()):
                if not org_dir.is_dir():
                    continue
                for skill_dir in sorted(org_dir.iterdir()):
                    meta_file = skill_dir / ".crewai_meta.json"
                    if meta_file.exists():
                        try:
                            meta = json.loads(meta_file.read_text())
                            table.add_row(
                                "cache",
                                f"@{meta['org']}/{meta['name']}",
                                meta.get("version") or "-",
                                str(skill_dir),
                            )
                        except (json.JSONDecodeError, KeyError):
                            console.print(f"[yellow]Warning: skipping malformed cache entry at {meta_file}[/yellow]")

        console.print(table)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _unpack_archive(self, archive_bytes: bytes, dest: Path) -> None:
        """Unpack a .tar.gz or .zip archive into dest."""
        # Try tar first, then zip
        try:
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tf:
                try:
                    tf.extractall(dest, filter="data")
                except TypeError:
                    _safe_extractall(tf, dest)
            return
        except tarfile.TarError:
            pass

        # Fallback: zip
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            _safe_extract_zip(zf, dest)

    def _build_skill_zip(self) -> bytes:
        """Build an in-memory ZIP of SKILL.md + scripts/ + references/ + assets/."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write("SKILL.md")
            for folder in ("scripts", "references", "assets"):
                folder_path = Path(folder)
                if folder_path.is_dir():
                    for fpath in sorted(folder_path.rglob("*")):
                        if fpath.is_file():
                            zf.write(fpath)
        return buf.getvalue()

    def _parse_frontmatter(self, content: str) -> dict[str, str]:
        """Extract YAML frontmatter fields from a SKILL.md string."""
        import re
        match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not match:
            raise ValueError("No YAML frontmatter block found")
        try:
            import yaml
            return yaml.safe_load(match.group(1)) or {}
        except ImportError:
            # Minimal fallback: parse key: value lines
            result: dict[str, str] = {}
            for line in match.group(1).splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    result[key.strip()] = value.strip()
            return result

    def _read_version(self, skill_md: Path) -> str | None:
        """Read the version field from a SKILL.md file, or None."""
        try:
            fm = self._parse_frontmatter(skill_md.read_text())
            return fm.get("version")
        except Exception:
            return None


def _safe_extractall(tf: tarfile.TarFile, dest: Path) -> None:
    """Path-traversal-safe extraction for Python < 3.12."""
    dest_resolved = dest.resolve()
    for member in tf.getmembers():
        member_path = (dest / member.name).resolve()
        if not member_path.is_relative_to(dest_resolved):
            raise ValueError(f"Blocked path traversal attempt: {member.name!r}")
    tf.extractall(dest)  # noqa: S202


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    """Path-traversal-safe ZIP extraction."""
    dest_resolved = dest.resolve()
    for member in zf.namelist():
        member_path = (dest / member).resolve()
        if not member_path.is_relative_to(dest_resolved):
            raise ValueError(f"Blocked path traversal attempt: {member!r}")
    zf.extractall(dest)  # noqa: S202
