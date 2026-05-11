"""Tests for `crewai.cli.deploy.validate`.

The fixtures here correspond 1:1 to the deployment-failure patterns observed
in the #crewai-deployment-failures Slack channel that motivated this work.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Iterable
from unittest.mock import patch

import pytest

from crewai_cli.deploy.validate import (
    DeployValidator,
    Severity,
    normalize_package_name,
)


def _make_pyproject(
    name: str = "my_crew",
    dependencies: Iterable[str] = ("crewai>=1.14.0",),
    *,
    hatchling: bool = False,
    flow: bool = False,
    extra: str = "",
) -> str:
    deps = ", ".join(f'"{d}"' for d in dependencies)
    lines = [
        "[project]",
        f'name = "{name}"',
        'version = "0.1.0"',
        f"dependencies = [{deps}]",
    ]
    if hatchling:
        lines += [
            "",
            "[build-system]",
            'requires = ["hatchling"]',
            'build-backend = "hatchling.build"',
        ]
    if flow:
        lines += ["", "[tool.crewai]", 'type = "flow"']
    if extra:
        lines += ["", extra]
    return "\n".join(lines) + "\n"


def _scaffold_standard_crew(
    root: Path,
    *,
    name: str = "my_crew",
    include_crew_py: bool = True,
    include_agents_yaml: bool = True,
    include_tasks_yaml: bool = True,
    include_lockfile: bool = True,
    pyproject: str | None = None,
) -> Path:
    (root / "pyproject.toml").write_text(pyproject or _make_pyproject(name=name))
    if include_lockfile:
        (root / "uv.lock").write_text("# dummy uv lockfile\n")

    pkg_dir = root / "src" / normalize_package_name(name)
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")

    if include_crew_py:
        (pkg_dir / "crew.py").write_text(
            dedent(
                """
                from crewai.project import CrewBase, crew

                @CrewBase
                class MyCrew:
                    agents_config = "config/agents.yaml"
                    tasks_config = "config/tasks.yaml"

                    @crew
                    def crew(self):
                        from crewai import Crew
                        return Crew(agents=[], tasks=[])
                """
            ).strip()
            + "\n"
        )

    config_dir = pkg_dir / "config"
    config_dir.mkdir()
    if include_agents_yaml:
        (config_dir / "agents.yaml").write_text("{}\n")
    if include_tasks_yaml:
        (config_dir / "tasks.yaml").write_text("{}\n")

    return pkg_dir


def _codes(validator: DeployValidator) -> set[str]:
    return {r.code for r in validator.results}


def _run_without_import_check(root: Path) -> DeployValidator:
    """Run validation with the subprocess-based import check stubbed out;
    the classifier is exercised directly in its own tests below."""
    with patch.object(DeployValidator, "_check_module_imports", lambda self: None):
        v = DeployValidator(project_root=root)
        v.run()
    return v


@pytest.mark.parametrize(
    "project_name, expected",
    [
        ("my-crew", "my_crew"),
        ("My Cool-Project", "my_cool_project"),
        ("crew123", "crew123"),
        ("crew.name!with$chars", "crewnamewithchars"),
    ],
)
def test_normalize_package_name(project_name: str, expected: str) -> None:
    assert normalize_package_name(project_name) == expected


def test_valid_standard_crew_project_passes(tmp_path: Path) -> None:
    _scaffold_standard_crew(tmp_path)
    v = _run_without_import_check(tmp_path)
    assert v.ok, f"expected clean run, got {v.results}"


def test_missing_pyproject_errors(tmp_path: Path) -> None:
    v = _run_without_import_check(tmp_path)
    assert "missing_pyproject" in _codes(v)
    assert not v.ok


def test_invalid_pyproject_errors(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("this is not valid toml ====\n")
    v = _run_without_import_check(tmp_path)
    assert "invalid_pyproject" in _codes(v)


def test_missing_project_name_errors(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "0.1.0"\ndependencies = ["crewai>=1.14.0"]\n'
    )
    v = _run_without_import_check(tmp_path)
    assert "missing_project_name" in _codes(v)


def test_missing_lockfile_errors(tmp_path: Path) -> None:
    _scaffold_standard_crew(tmp_path, include_lockfile=False)
    v = _run_without_import_check(tmp_path)
    assert "missing_lockfile" in _codes(v)


def test_poetry_lock_is_accepted(tmp_path: Path) -> None:
    _scaffold_standard_crew(tmp_path, include_lockfile=False)
    (tmp_path / "poetry.lock").write_text("# poetry lockfile\n")
    v = _run_without_import_check(tmp_path)
    assert "missing_lockfile" not in _codes(v)


def test_stale_lockfile_warns(tmp_path: Path) -> None:
    _scaffold_standard_crew(tmp_path)
    # Make lockfile older than pyproject.
    lock = tmp_path / "uv.lock"
    pyproject = tmp_path / "pyproject.toml"
    old_time = pyproject.stat().st_mtime - 60
    import os

    os.utime(lock, (old_time, old_time))
    v = _run_without_import_check(tmp_path)
    assert "stale_lockfile" in _codes(v)
    # Stale is a warning, so the run can still be ok (no errors).
    assert v.ok


def test_missing_package_dir_errors(tmp_path: Path) -> None:
    # pyproject says name=my_crew but we only create src/other_pkg/
    (tmp_path / "pyproject.toml").write_text(_make_pyproject(name="my_crew"))
    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "src" / "other_pkg").mkdir(parents=True)
    v = _run_without_import_check(tmp_path)
    codes = _codes(v)
    assert "missing_package_dir" in codes
    finding = next(r for r in v.results if r.code == "missing_package_dir")
    assert "other_pkg" in finding.hint


def test_egg_info_only_errors_with_targeted_hint(tmp_path: Path) -> None:
    """Regression for the case where only src/<name>.egg-info/ exists."""
    (tmp_path / "pyproject.toml").write_text(_make_pyproject(name="odoo_pm_agents"))
    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "src" / "odoo_pm_agents.egg-info").mkdir(parents=True)
    v = _run_without_import_check(tmp_path)
    finding = next(r for r in v.results if r.code == "missing_package_dir")
    assert "egg-info" in finding.hint


def test_stale_egg_info_sibling_warns(tmp_path: Path) -> None:
    _scaffold_standard_crew(tmp_path)
    (tmp_path / "src" / "my_crew.egg-info").mkdir()
    v = _run_without_import_check(tmp_path)
    assert "stale_egg_info" in _codes(v)


def test_missing_crew_py_errors(tmp_path: Path) -> None:
    _scaffold_standard_crew(tmp_path, include_crew_py=False)
    v = _run_without_import_check(tmp_path)
    assert "missing_crew_py" in _codes(v)


def test_missing_agents_yaml_errors(tmp_path: Path) -> None:
    _scaffold_standard_crew(tmp_path, include_agents_yaml=False)
    v = _run_without_import_check(tmp_path)
    assert "missing_agents_yaml" in _codes(v)


def test_missing_tasks_yaml_errors(tmp_path: Path) -> None:
    _scaffold_standard_crew(tmp_path, include_tasks_yaml=False)
    v = _run_without_import_check(tmp_path)
    assert "missing_tasks_yaml" in _codes(v)


def test_flow_project_requires_main_py(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        _make_pyproject(name="my_flow", flow=True)
    )
    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "src" / "my_flow").mkdir(parents=True)
    v = _run_without_import_check(tmp_path)
    assert "missing_flow_main" in _codes(v)


def test_flow_project_with_main_py_passes(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        _make_pyproject(name="my_flow", flow=True)
    )
    (tmp_path / "uv.lock").write_text("")
    pkg = tmp_path / "src" / "my_flow"
    pkg.mkdir(parents=True)
    (pkg / "main.py").write_text("# flow entrypoint\n")
    v = _run_without_import_check(tmp_path)
    assert "missing_flow_main" not in _codes(v)


def test_hatchling_without_wheel_config_passes_when_pkg_dir_matches(
    tmp_path: Path,
) -> None:
    _scaffold_standard_crew(
        tmp_path, pyproject=_make_pyproject(name="my_crew", hatchling=True)
    )
    v = _run_without_import_check(tmp_path)
    # src/my_crew/ exists, so hatch default should find it — no wheel error.
    assert "hatch_wheel_target_missing" not in _codes(v)


def test_hatchling_with_explicit_wheel_config_passes(tmp_path: Path) -> None:
    extra = (
        "[tool.hatch.build.targets.wheel]\n"
        'packages = ["src/my_crew"]'
    )
    _scaffold_standard_crew(
        tmp_path,
        pyproject=_make_pyproject(name="my_crew", hatchling=True, extra=extra),
    )
    v = _run_without_import_check(tmp_path)
    assert "hatch_wheel_target_missing" not in _codes(v)


def test_classify_missing_openai_key_is_warning(tmp_path: Path) -> None:
    v = DeployValidator(project_root=tmp_path)
    v._classify_import_error(
        "ImportError",
        "Error importing native provider: 1 validation error for OpenAICompletion\n"
        "  Value error, OPENAI_API_KEY is required",
        tb="",
    )
    assert len(v.results) == 1
    result = v.results[0]
    assert result.code == "llm_init_missing_key"
    assert result.severity is Severity.WARNING
    assert "OPENAI_API_KEY" in result.title


def test_classify_azure_extra_missing_is_error(tmp_path: Path) -> None:
    """The real message raised by the Azure provider module uses plain
    double quotes around the install command (no backticks). Match the
    exact string that ships in the provider source so this test actually
    guards the regex used in production."""
    v = DeployValidator(project_root=tmp_path)
    v._classify_import_error(
        "ImportError",
        'Azure AI Inference native provider not available, to install: uv add "crewai[azure-ai-inference]"',
        tb="",
    )
    assert "missing_provider_extra" in _codes(v)
    finding = next(r for r in v.results if r.code == "missing_provider_extra")
    assert finding.title.startswith("Azure AI Inference")
    assert 'uv add "crewai[azure-ai-inference]"' in finding.hint


@pytest.mark.parametrize(
    "pkg_label, install_cmd",
    [
        ("Anthropic", 'uv add "crewai[anthropic]"'),
        ("AWS Bedrock", 'uv add "crewai[bedrock]"'),
        ("Google Gen AI", 'uv add "crewai[google-genai]"'),
    ],
)
def test_classify_missing_provider_extra_matches_real_messages(
    tmp_path: Path, pkg_label: str, install_cmd: str
) -> None:
    """Regression for the four provider error strings verbatim."""
    v = DeployValidator(project_root=tmp_path)
    v._classify_import_error(
        "ImportError",
        f"{pkg_label} native provider not available, to install: {install_cmd}",
        tb="",
    )
    assert "missing_provider_extra" in _codes(v)
    finding = next(r for r in v.results if r.code == "missing_provider_extra")
    assert install_cmd in finding.hint


def test_classify_keyerror_at_import_is_warning(tmp_path: Path) -> None:
    """Regression for `KeyError: 'SERPLY_API_KEY'` raised at import time."""
    v = DeployValidator(project_root=tmp_path)
    v._classify_import_error("KeyError", "'SERPLY_API_KEY'", tb="")
    codes = _codes(v)
    assert "env_var_read_at_import" in codes


def test_classify_no_crewbase_class_is_error(tmp_path: Path) -> None:
    v = DeployValidator(project_root=tmp_path)
    v._classify_import_error(
        "ValueError",
        "Crew class annotated with @CrewBase not found.",
        tb="",
    )
    assert "no_crewbase_class" in _codes(v)


def test_classify_no_flow_subclass_is_error(tmp_path: Path) -> None:
    v = DeployValidator(project_root=tmp_path)
    v._classify_import_error("ValueError", "No Flow subclass found in the module.", tb="")
    assert "no_flow_subclass" in _codes(v)


def test_classify_stale_crewai_pin_attribute_error(tmp_path: Path) -> None:
    """Regression for a stale crewai pin missing `_load_response_format`."""
    v = DeployValidator(project_root=tmp_path)
    v._classify_import_error(
        "AttributeError",
        "'EmploymentServiceDecisionSupportSystemCrew' object has no attribute '_load_response_format'",
        tb="",
    )
    assert "stale_crewai_pin" in _codes(v)


def test_classify_unknown_error_is_fallback(tmp_path: Path) -> None:
    v = DeployValidator(project_root=tmp_path)
    v._classify_import_error("RuntimeError", "something weird happened", tb="")
    assert "import_failed" in _codes(v)


def test_env_var_referenced_but_missing_warns(tmp_path: Path) -> None:
    pkg = _scaffold_standard_crew(tmp_path)
    (pkg / "tools.py").write_text(
        'import os\nkey = os.getenv("TAVILY_API_KEY")\n'
    )
    import os

    # Make sure the test doesn't inherit the key from the host environment.
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TAVILY_API_KEY", None)
        v = _run_without_import_check(tmp_path)
    codes = _codes(v)
    assert "env_vars_not_in_dotenv" in codes


def test_env_var_in_dotenv_does_not_warn(tmp_path: Path) -> None:
    pkg = _scaffold_standard_crew(tmp_path)
    (pkg / "tools.py").write_text(
        'import os\nkey = os.getenv("TAVILY_API_KEY")\n'
    )
    (tmp_path / ".env").write_text("TAVILY_API_KEY=abc\n")
    v = _run_without_import_check(tmp_path)
    assert "env_vars_not_in_dotenv" not in _codes(v)


def test_old_crewai_pin_in_uv_lock_warns(tmp_path: Path) -> None:
    _scaffold_standard_crew(tmp_path)
    (tmp_path / "uv.lock").write_text(
        'name = "crewai"\nversion = "1.10.0"\nsource = { registry = "..." }\n'
    )
    v = _run_without_import_check(tmp_path)
    assert "old_crewai_pin" in _codes(v)


def test_modern_crewai_pin_does_not_warn(tmp_path: Path) -> None:
    _scaffold_standard_crew(tmp_path)
    (tmp_path / "uv.lock").write_text(
        'name = "crewai"\nversion = "1.14.1"\nsource = { registry = "..." }\n'
    )
    v = _run_without_import_check(tmp_path)
    assert "old_crewai_pin" not in _codes(v)


def test_create_crew_aborts_on_validation_error(tmp_path: Path) -> None:
    """`crewai deploy create` must not contact the API when validation fails."""
    from unittest.mock import MagicMock, patch as mock_patch

    from crewai_cli.deploy.main import DeployCommand

    with (
        mock_patch("crewai_cli.command.get_auth_token", return_value="tok"),
        mock_patch("crewai_cli.deploy.main.get_project_name", return_value="p"),
        mock_patch("crewai_cli.command.PlusAPI") as mock_api,
        mock_patch(
            "crewai_cli.deploy.main.validate_project"
        ) as mock_validate,
    ):
        mock_validate.return_value = MagicMock(ok=False)
        cmd = DeployCommand()
        cmd.create_crew()
        assert not cmd.plus_api_client.create_crew.called
        del mock_api  # silence unused-var lint