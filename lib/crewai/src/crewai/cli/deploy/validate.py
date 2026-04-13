"""Pre-deploy validation for CrewAI projects.

Catches locally what a deploy would reject at build or runtime so users
don't burn deployment attempts on fixable project-structure problems.

Each check is grouped into one of:
- ERROR: will block a deployment; validator exits non-zero.
- WARNING: may still deploy but is almost always a deployment bug; printed
  but does not block.

The individual checks mirror the categories observed in production
deployment-failure logs:

1. pyproject.toml present with ``[project].name``
2. lockfile (``uv.lock`` or ``poetry.lock``) present and not stale
3. package directory at ``src/<package>/`` exists (no empty name, no egg-info)
4. standard crew files: ``crew.py``, ``config/agents.yaml``, ``config/tasks.yaml``
5. flow entrypoint: ``main.py`` with a Flow subclass
6. hatch wheel target resolves (packages = [...] or default dir matches name)
7. crew/flow module imports cleanly (catches ``@CrewBase not found``,
   ``No Flow subclass found``, provider import errors)
8. environment variables referenced in code vs ``.env`` / deployment env
9. installed crewai vs lockfile pin (catches missing-attribute failures from
   stale pins)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

from rich.console import Console

from crewai.cli.utils import parse_toml


console = Console()
logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Severity of a validation finding."""

    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationResult:
    """A single finding from a validation check.

    Attributes:
        severity: whether this blocks deploy or is advisory.
        code: stable short identifier, used in tests and docs
            (e.g. ``missing_pyproject``, ``stale_lockfile``).
        title: one-line summary shown to the user.
        detail: optional multi-line explanation.
        hint: optional remediation suggestion.
    """

    severity: Severity
    code: str
    title: str
    detail: str = ""
    hint: str = ""


# Maps known provider env var names → label used in hint messages.
_KNOWN_API_KEY_HINTS: dict[str, str] = {
    "OPENAI_API_KEY": "OpenAI",
    "ANTHROPIC_API_KEY": "Anthropic",
    "GOOGLE_API_KEY": "Google",
    "GEMINI_API_KEY": "Gemini",
    "AZURE_OPENAI_API_KEY": "Azure OpenAI",
    "AZURE_API_KEY": "Azure",
    "AWS_ACCESS_KEY_ID": "AWS",
    "AWS_SECRET_ACCESS_KEY": "AWS",
    "COHERE_API_KEY": "Cohere",
    "GROQ_API_KEY": "Groq",
    "MISTRAL_API_KEY": "Mistral",
    "TAVILY_API_KEY": "Tavily",
    "SERPER_API_KEY": "Serper",
    "SERPLY_API_KEY": "Serply",
    "PERPLEXITY_API_KEY": "Perplexity",
    "DEEPSEEK_API_KEY": "DeepSeek",
    "OPENROUTER_API_KEY": "OpenRouter",
    "FIRECRAWL_API_KEY": "Firecrawl",
    "EXA_API_KEY": "Exa",
    "BROWSERBASE_API_KEY": "Browserbase",
}


def normalize_package_name(project_name: str) -> str:
    """Normalize a pyproject project.name into a Python package directory name.

    Mirrors the rules in ``crewai.cli.create_crew.create_crew`` so the
    validator agrees with the scaffolder about where ``src/<pkg>/`` should
    live.
    """
    folder = project_name.replace(" ", "_").replace("-", "_").lower()
    return re.sub(r"[^a-zA-Z0-9_]", "", folder)


class DeployValidator:
    """Runs the full pre-deploy validation suite against a project directory."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root: Path = (project_root or Path.cwd()).resolve()
        self.results: list[ValidationResult] = []
        self._pyproject: dict[str, Any] | None = None
        self._project_name: str | None = None
        self._package_name: str | None = None
        self._package_dir: Path | None = None
        self._is_flow: bool = False

    def _add(
        self,
        severity: Severity,
        code: str,
        title: str,
        detail: str = "",
        hint: str = "",
    ) -> None:
        self.results.append(
            ValidationResult(
                severity=severity,
                code=code,
                title=title,
                detail=detail,
                hint=hint,
            )
        )

    @property
    def errors(self) -> list[ValidationResult]:
        return [r for r in self.results if r.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationResult]:
        return [r for r in self.results if r.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def run(self) -> list[ValidationResult]:
        """Run all checks. Later checks are skipped when earlier ones make
        them impossible (e.g. no pyproject.toml → no lockfile check)."""
        if not self._check_pyproject():
            return self.results

        self._check_lockfile()

        if not self._check_package_dir():
            self._check_hatch_wheel_target()
            return self.results

        if self._is_flow:
            self._check_flow_entrypoint()
        else:
            self._check_crew_entrypoint()
            self._check_config_yamls()

        self._check_hatch_wheel_target()
        self._check_module_imports()
        self._check_env_vars()
        self._check_version_vs_lockfile()

        return self.results

    def _check_pyproject(self) -> bool:
        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists():
            self._add(
                Severity.ERROR,
                "missing_pyproject",
                "Cannot find pyproject.toml",
                detail=(
                    f"Expected pyproject.toml at {pyproject_path}. "
                    "CrewAI projects must be installable Python packages."
                ),
                hint="Run `crewai create crew <name>` to scaffold a valid project layout.",
            )
            return False

        try:
            self._pyproject = parse_toml(pyproject_path.read_text())
        except Exception as e:
            self._add(
                Severity.ERROR,
                "invalid_pyproject",
                "pyproject.toml is not valid TOML",
                detail=str(e),
            )
            return False

        project = self._pyproject.get("project") or {}
        name = project.get("name")
        if not isinstance(name, str) or not name.strip():
            self._add(
                Severity.ERROR,
                "missing_project_name",
                "pyproject.toml is missing [project].name",
                detail=(
                    "Without a project name the platform cannot resolve your "
                    "package directory (this produces errors like "
                    "'Cannot find src//crew.py')."
                ),
                hint='Set a `name = "..."` field under `[project]` in pyproject.toml.',
            )
            return False

        self._project_name = name
        self._package_name = normalize_package_name(name)
        self._is_flow = (self._pyproject.get("tool") or {}).get("crewai", {}).get(
            "type"
        ) == "flow"
        return True

    def _check_lockfile(self) -> None:
        uv_lock = self.project_root / "uv.lock"
        poetry_lock = self.project_root / "poetry.lock"
        pyproject = self.project_root / "pyproject.toml"

        if not uv_lock.exists() and not poetry_lock.exists():
            self._add(
                Severity.ERROR,
                "missing_lockfile",
                "Expected to find at least one of these files: uv.lock or poetry.lock",
                hint=(
                    "Run `uv lock` (recommended) or `poetry lock` in your project "
                    "directory, commit the lockfile, then redeploy."
                ),
            )
            return

        lockfile = uv_lock if uv_lock.exists() else poetry_lock
        try:
            if lockfile.stat().st_mtime < pyproject.stat().st_mtime:
                self._add(
                    Severity.WARNING,
                    "stale_lockfile",
                    f"{lockfile.name} is older than pyproject.toml",
                    detail=(
                        "Your lockfile may not reflect recent dependency changes. "
                        "The platform resolves from the lockfile, so deployed "
                        "dependencies may differ from local."
                    ),
                    hint="Run `uv lock` (or `poetry lock`) and commit the result.",
                )
        except OSError:
            pass

    def _check_package_dir(self) -> bool:
        if self._package_name is None:
            return False

        src_dir = self.project_root / "src"
        if not src_dir.is_dir():
            self._add(
                Severity.ERROR,
                "missing_src_dir",
                "Missing src/ directory",
                detail=(
                    "CrewAI deployments expect a src-layout project: "
                    f"src/{self._package_name}/crew.py (or main.py for flows)."
                ),
                hint="Run `crewai create crew <name>` to see the expected layout.",
            )
            return False

        package_dir = src_dir / self._package_name
        if not package_dir.is_dir():
            siblings = [
                p.name
                for p in src_dir.iterdir()
                if p.is_dir() and not p.name.endswith(".egg-info")
            ]
            egg_info = [
                p.name for p in src_dir.iterdir() if p.name.endswith(".egg-info")
            ]

            hint_parts = [
                f'Create src/{self._package_name}/ to match [project].name = "{self._project_name}".'
            ]
            if siblings:
                hint_parts.append(
                    f"Found other package directories: {', '.join(siblings)}. "
                    f"Either rename one to '{self._package_name}' or update [project].name."
                )
            if egg_info:
                hint_parts.append(
                    f"Delete stale build artifacts: {', '.join(egg_info)} "
                    "(these confuse the platform's package discovery)."
                )

            self._add(
                Severity.ERROR,
                "missing_package_dir",
                f"Cannot find src/{self._package_name}/",
                detail=(
                    "The platform looks for your crew source under "
                    "src/<package_name>/, derived from [project].name."
                ),
                hint=" ".join(hint_parts),
            )
            return False

        for p in src_dir.iterdir():
            if p.name.endswith(".egg-info"):
                self._add(
                    Severity.WARNING,
                    "stale_egg_info",
                    f"Stale build artifact in src/: {p.name}",
                    detail=(
                        ".egg-info directories can be mistaken for your package "
                        "and cause 'Cannot find src/<name>.egg-info/crew.py' errors."
                    ),
                    hint=f"Delete {p} and add `*.egg-info/` to .gitignore.",
                )

        self._package_dir = package_dir
        return True

    def _check_crew_entrypoint(self) -> None:
        if self._package_dir is None:
            return
        crew_py = self._package_dir / "crew.py"
        if not crew_py.is_file():
            self._add(
                Severity.ERROR,
                "missing_crew_py",
                f"Cannot find {crew_py.relative_to(self.project_root)}",
                detail=(
                    "Standard crew projects must define a Crew class decorated "
                    "with @CrewBase inside crew.py."
                ),
                hint=(
                    "Create crew.py with an @CrewBase-annotated class, or set "
                    '`[tool.crewai] type = "flow"` in pyproject.toml if this is a flow.'
                ),
            )

    def _check_config_yamls(self) -> None:
        if self._package_dir is None:
            return
        config_dir = self._package_dir / "config"
        if not config_dir.is_dir():
            self._add(
                Severity.ERROR,
                "missing_config_dir",
                f"Cannot find {config_dir.relative_to(self.project_root)}",
                hint="Create a config/ directory with agents.yaml and tasks.yaml.",
            )
            return

        for yaml_name in ("agents.yaml", "tasks.yaml"):
            yaml_path = config_dir / yaml_name
            if not yaml_path.is_file():
                self._add(
                    Severity.ERROR,
                    f"missing_{yaml_name.replace('.', '_')}",
                    f"Cannot find {yaml_path.relative_to(self.project_root)}",
                    detail=(
                        "CrewAI loads agent and task config from these files; "
                        "missing them causes empty-config warnings and runtime crashes."
                    ),
                )

    def _check_flow_entrypoint(self) -> None:
        if self._package_dir is None:
            return
        main_py = self._package_dir / "main.py"
        if not main_py.is_file():
            self._add(
                Severity.ERROR,
                "missing_flow_main",
                f"Cannot find {main_py.relative_to(self.project_root)}",
                detail=(
                    "Flow projects must define a Flow subclass in main.py. "
                    'This project has `[tool.crewai] type = "flow"` set.'
                ),
                hint="Create main.py with a `class MyFlow(Flow[...])`.",
            )

    def _check_hatch_wheel_target(self) -> None:
        if not self._pyproject:
            return

        build_system = self._pyproject.get("build-system") or {}
        backend = build_system.get("build-backend", "")
        if "hatchling" not in backend:
            return

        hatch_wheel = (
            (self._pyproject.get("tool") or {})
            .get("hatch", {})
            .get("build", {})
            .get("targets", {})
            .get("wheel", {})
        )
        if hatch_wheel.get("packages") or hatch_wheel.get("only-include"):
            return

        if self._package_dir and self._package_dir.is_dir():
            return

        self._add(
            Severity.ERROR,
            "hatch_wheel_target_missing",
            "Hatchling cannot determine which files to ship",
            detail=(
                "Your pyproject uses hatchling but has no "
                "[tool.hatch.build.targets.wheel] configuration and no "
                "directory matching your project name."
            ),
            hint=(
                "Add:\n"
                "  [tool.hatch.build.targets.wheel]\n"
                f'  packages = ["src/{self._package_name}"]'
            ),
        )

    def _check_module_imports(self) -> None:
        """Import the user's crew/flow via `uv run` so the check sees the same
        package versions as `crewai run` would. Result is reported as JSON on
        the subprocess's stdout."""
        script = (
            "import json, sys, traceback, os\n"
            "os.chdir(sys.argv[1])\n"
            "try:\n"
            "    from crewai.cli.utils import get_crews, get_flows\n"
            "    is_flow = sys.argv[2] == 'flow'\n"
            "    if is_flow:\n"
            "        instances = get_flows()\n"
            "        kind = 'flow'\n"
            "    else:\n"
            "        instances = get_crews()\n"
            "        kind = 'crew'\n"
            "    print(json.dumps({'ok': True, 'kind': kind, 'count': len(instances)}))\n"
            "except BaseException as e:\n"
            "    print(json.dumps({\n"
            "        'ok': False,\n"
            "        'error_type': type(e).__name__,\n"
            "        'error': str(e),\n"
            "        'traceback': traceback.format_exc(),\n"
            "    }))\n"
        )

        uv_path = shutil.which("uv")
        if uv_path is None:
            self._add(
                Severity.WARNING,
                "uv_not_found",
                "Skipping import check: `uv` not installed",
                hint="Install uv: https://docs.astral.sh/uv/",
            )
            return

        try:
            proc = subprocess.run(  # noqa: S603 - args constructed from trusted inputs
                [
                    uv_path,
                    "run",
                    "python",
                    "-c",
                    script,
                    str(self.project_root),
                    "flow" if self._is_flow else "crew",
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self._add(
                Severity.ERROR,
                "import_timeout",
                "Importing your crew/flow module timed out after 120s",
                detail=(
                    "User code may be making network calls or doing heavy work "
                    "at import time. Move that work into agent methods."
                ),
            )
            return

        # The payload is the last JSON object on stdout; user code may print
        # other lines before it.
        payload: dict[str, Any] | None = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        if payload is None:
            self._add(
                Severity.ERROR,
                "import_failed",
                "Could not import your crew/flow module",
                detail=(proc.stderr or proc.stdout or "").strip()[:1500],
                hint="Run `crewai run` locally first to reproduce the error.",
            )
            return

        if payload.get("ok"):
            if payload.get("count", 0) == 0:
                kind = payload.get("kind", "crew")
                if kind == "flow":
                    self._add(
                        Severity.ERROR,
                        "no_flow_subclass",
                        "No Flow subclass found in the module",
                        hint=(
                            "main.py must define a class extending "
                            "`crewai.flow.Flow`, instantiable with no arguments."
                        ),
                    )
                else:
                    self._add(
                        Severity.ERROR,
                        "no_crewbase_class",
                        "Crew class annotated with @CrewBase not found",
                        hint=(
                            "Decorate your crew class with @CrewBase from "
                            "crewai.project (see `crewai create crew` template)."
                        ),
                    )
            return

        err_msg = str(payload.get("error", ""))
        err_type = str(payload.get("error_type", "Exception"))
        tb = str(payload.get("traceback", ""))
        self._classify_import_error(err_type, err_msg, tb)

    def _classify_import_error(self, err_type: str, err_msg: str, tb: str) -> None:
        """Turn a raw import-time exception into a user-actionable finding."""
        # Must be checked before the generic "native provider" branch below:
        # the extras-missing message contains the same phrase. Providers
        # format the install command as plain text (`to install: uv add
        # "crewai[extra]"`); also tolerate backtick-delimited variants.
        m = re.search(
            r"(?P<pkg>[A-Za-z0-9_ -]+?)\s+native provider not available"
            r".*?to install:\s*`?(?P<cmd>uv add [\"']crewai\[[^\]]+\][\"'])`?",
            err_msg,
        )
        if m:
            self._add(
                Severity.ERROR,
                "missing_provider_extra",
                f"{m.group('pkg').strip()} provider extra not installed",
                hint=f"Run: {m.group('cmd')}",
            )
            return

        # crewai.llm.LLM.__new__ wraps provider init errors as
        # ImportError("Error importing native provider: ...").
        if "Error importing native provider" in err_msg or "native provider" in err_msg:
            missing_key = self._extract_missing_api_key(err_msg)
            if missing_key:
                provider = _KNOWN_API_KEY_HINTS.get(missing_key, missing_key)
                self._add(
                    Severity.WARNING,
                    "llm_init_missing_key",
                    f"LLM is constructed at import time but {missing_key} is not set",
                    detail=(
                        f"Your crew instantiates a {provider} LLM during module "
                        "load (e.g. in a class field default or @crew method). "
                        f"The {provider} provider currently requires {missing_key} "
                        "at construction time, so this will fail on the platform "
                        "unless the key is set in your deployment environment."
                    ),
                    hint=(
                        f"Add {missing_key} to your deployment's Environment "
                        "Variables before deploying, or move LLM construction "
                        "inside agent methods so it runs lazily."
                    ),
                )
                return
            self._add(
                Severity.ERROR,
                "llm_provider_init_failed",
                "LLM native provider failed to initialize",
                detail=err_msg,
                hint=(
                    "Check your LLM(model=...) configuration and provider-specific "
                    "extras (e.g. `uv add 'crewai[azure-ai-inference]'` for Azure)."
                ),
            )
            return

        if err_type == "KeyError":
            key = err_msg.strip("'\"")
            if key in _KNOWN_API_KEY_HINTS or key.endswith("_API_KEY"):
                self._add(
                    Severity.WARNING,
                    "env_var_read_at_import",
                    f"{key} is read at import time via os.environ[...]",
                    detail=(
                        "Using os.environ[...] (rather than os.getenv(...)) "
                        "at module scope crashes the build if the key isn't set."
                    ),
                    hint=(
                        f"Either add {key} as a deployment env var, or switch "
                        "to os.getenv() and move the access inside agent methods."
                    ),
                )
                return

        if "Crew class annotated with @CrewBase not found" in err_msg:
            self._add(
                Severity.ERROR,
                "no_crewbase_class",
                "Crew class annotated with @CrewBase not found",
                detail=err_msg,
            )
            return
        if "No Flow subclass found" in err_msg:
            self._add(
                Severity.ERROR,
                "no_flow_subclass",
                "No Flow subclass found in the module",
                detail=err_msg,
            )
            return

        if (
            err_type == "AttributeError"
            and "has no attribute '_load_response_format'" in err_msg
        ):
            self._add(
                Severity.ERROR,
                "stale_crewai_pin",
                "Your lockfile pins a crewai version missing `_load_response_format`",
                detail=err_msg,
                hint=(
                    "Run `uv lock --upgrade-package crewai` (or `poetry update crewai`) "
                    "to pin a newer release."
                ),
            )
            return

        if "pydantic" in tb.lower() or "validation error" in err_msg.lower():
            self._add(
                Severity.ERROR,
                "pydantic_validation_error",
                "Pydantic validation failed while loading your crew",
                detail=err_msg[:800],
                hint=(
                    "Check agent/task configuration fields. `crewai run` locally "
                    "will show the full traceback."
                ),
            )
            return

        self._add(
            Severity.ERROR,
            "import_failed",
            f"Importing your crew failed: {err_type}",
            detail=err_msg[:800],
            hint="Run `crewai run` locally to see the full traceback.",
        )

    @staticmethod
    def _extract_missing_api_key(err_msg: str) -> str | None:
        """Pull 'FOO_API_KEY' out of '... FOO_API_KEY is required ...'."""
        m = re.search(r"([A-Z][A-Z0-9_]*_API_KEY)\s+is required", err_msg)
        if m:
            return m.group(1)
        m = re.search(r"['\"]([A-Z][A-Z0-9_]*_API_KEY)['\"]", err_msg)
        if m:
            return m.group(1)
        return None

    def _check_env_vars(self) -> None:
        """Warn about env vars referenced in user code but missing locally.
        Best-effort only — the platform sets vars server-side, so we never error.
        """
        if not self._package_dir:
            return

        referenced: set[str] = set()
        pattern = re.compile(
            r"""(?x)
            (?:os\.environ\s*(?:\[\s*|\.get\s*\(\s*)
              |os\.getenv\s*\(\s*
              |getenv\s*\(\s*)
            ['"]([A-Z][A-Z0-9_]*)['"]
            """
        )

        for path in self._package_dir.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            referenced.update(pattern.findall(text))

        for path in self._package_dir.rglob("*.yaml"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            referenced.update(re.findall(r"\$\{?([A-Z][A-Z0-9_]+)\}?", text))

        env_file = self.project_root / ".env"
        env_keys: set[str] = set()
        if env_file.exists():
            for line in env_file.read_text(errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                env_keys.add(line.split("=", 1)[0].strip())

        missing_known: list[str] = sorted(
            var
            for var in referenced
            if var in _KNOWN_API_KEY_HINTS
            and var not in env_keys
            and var not in os.environ
        )
        if missing_known:
            self._add(
                Severity.WARNING,
                "env_vars_not_in_dotenv",
                f"{len(missing_known)} referenced API key(s) not in .env",
                detail=(
                    "These env vars are referenced in your source but not set "
                    f"locally: {', '.join(missing_known)}. Deploys will fail "
                    "unless they are added to the deployment's Environment "
                    "Variables in the CrewAI dashboard."
                ),
            )

    def _check_version_vs_lockfile(self) -> None:
        """Warn when the lockfile pins a crewai release older than 1.13.0,
        which is where ``_load_response_format`` was introduced.
        """
        uv_lock = self.project_root / "uv.lock"
        poetry_lock = self.project_root / "poetry.lock"
        lockfile = (
            uv_lock
            if uv_lock.exists()
            else poetry_lock
            if poetry_lock.exists()
            else None
        )
        if lockfile is None:
            return

        try:
            text = lockfile.read_text(errors="ignore")
        except OSError:
            return

        m = re.search(
            r'name\s*=\s*"crewai"\s*\nversion\s*=\s*"([^"]+)"',
            text,
        )
        if not m:
            return
        locked = m.group(1)

        try:
            from packaging.version import Version

            if Version(locked) < Version("1.13.0"):
                self._add(
                    Severity.WARNING,
                    "old_crewai_pin",
                    f"Lockfile pins crewai=={locked} (older than 1.13.0)",
                    detail=(
                        "Older pinned versions are missing API surface the "
                        "platform builder expects (e.g. `_load_response_format`)."
                    ),
                    hint="Run `uv lock --upgrade-package crewai` and redeploy.",
                )
        except Exception as e:
            logger.debug("Could not parse crewai pin from lockfile: %s", e)


def render_report(results: list[ValidationResult]) -> None:
    """Pretty-print results to the shared rich console."""
    if not results:
        console.print("[bold green]Pre-deploy validation passed.[/bold green]")
        return

    errors = [r for r in results if r.severity is Severity.ERROR]
    warnings = [r for r in results if r.severity is Severity.WARNING]

    for result in errors:
        console.print(f"[bold red]ERROR[/bold red] [{result.code}] {result.title}")
        if result.detail:
            console.print(f"  {result.detail}")
        if result.hint:
            console.print(f"  [dim]hint:[/dim] {result.hint}")

    for result in warnings:
        console.print(
            f"[bold yellow]WARNING[/bold yellow] [{result.code}] {result.title}"
        )
        if result.detail:
            console.print(f"  {result.detail}")
        if result.hint:
            console.print(f"  [dim]hint:[/dim] {result.hint}")

    summary_parts: list[str] = []
    if errors:
        summary_parts.append(f"[bold red]{len(errors)} error(s)[/bold red]")
    if warnings:
        summary_parts.append(f"[bold yellow]{len(warnings)} warning(s)[/bold yellow]")
    console.print(f"\n{' / '.join(summary_parts)}")


def validate_project(project_root: Path | None = None) -> DeployValidator:
    """Entrypoint: run validation, render results, return the validator.

    The caller inspects ``validator.ok`` to decide whether to proceed with a
    deploy.
    """
    validator = DeployValidator(project_root=project_root)
    validator.run()
    render_report(validator.results)
    return validator


def run_validate_command() -> None:
    """Implementation of `crewai deploy validate`."""
    validator = validate_project()
    if not validator.ok:
        sys.exit(1)
