"""Tests for TOML-based version and dependency update functions."""

from pathlib import Path
from textwrap import dedent

from crewai_devtools.cli import (
    _pin_crewai_deps,
    _repin_crewai_install,
    update_pyproject_version,
    update_template_dependencies,
)


# --- update_pyproject_version ---


class TestUpdatePyprojectVersion:
    def test_updates_version(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            dedent("""\
            [project]
            name = "my-pkg"
            version = "1.0.0"
        """)
        )

        assert update_pyproject_version(pyproject, "2.0.0") is True
        assert 'version = "2.0.0"' in pyproject.read_text()

    def test_returns_false_when_already_current(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            dedent("""\
            [project]
            name = "my-pkg"
            version = "1.0.0"
        """)
        )

        assert update_pyproject_version(pyproject, "1.0.0") is False

    def test_returns_false_when_no_project_section(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.ruff]\nline-length = 88\n")

        assert update_pyproject_version(pyproject, "1.0.0") is False

    def test_returns_false_when_version_is_dynamic(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            dedent("""\
            [project]
            name = "my-pkg"
            dynamic = ["version"]
        """)
        )

        assert update_pyproject_version(pyproject, "1.0.0") is False
        assert 'version = "1.0.0"' not in pyproject.read_text()

    def test_returns_false_for_missing_file(self, tmp_path: Path) -> None:
        assert update_pyproject_version(tmp_path / "nope.toml", "1.0.0") is False

    def test_preserves_comments_and_formatting(self, tmp_path: Path) -> None:
        content = dedent("""\
            # This is important
            [project]
            name = "my-pkg"
            version = "1.0.0"  # current version
            description = "A package"
        """)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(content)

        update_pyproject_version(pyproject, "2.0.0")
        result = pyproject.read_text()

        assert "# This is important" in result
        assert 'description = "A package"' in result


# --- _pin_crewai_deps ---


class TestPinCrewaiDeps:
    def test_pins_exact_version(self) -> None:
        content = dedent("""\
            [project]
            dependencies = [
                "crewai==1.0.0",
            ]
        """)
        result = _pin_crewai_deps(content, "2.0.0")
        assert '"crewai==2.0.0"' in result

    def test_pins_minimum_version(self) -> None:
        content = dedent("""\
            [project]
            dependencies = [
                "crewai>=1.0.0",
            ]
        """)
        result = _pin_crewai_deps(content, "2.0.0")
        assert '"crewai==2.0.0"' in result
        assert ">=" not in result

    def test_pins_with_tools_extra(self) -> None:
        content = dedent("""\
            [project]
            dependencies = [
                "crewai[tools]==1.0.0",
            ]
        """)
        result = _pin_crewai_deps(content, "2.0.0")
        assert '"crewai[tools]==2.0.0"' in result

    def test_leaves_unrelated_deps_alone(self) -> None:
        content = dedent("""\
            [project]
            dependencies = [
                "requests>=2.0",
                "crewai==1.0.0",
                "click~=8.1",
            ]
        """)
        result = _pin_crewai_deps(content, "2.0.0")
        assert '"requests>=2.0"' in result
        assert '"click~=8.1"' in result

    def test_handles_optional_dependencies(self) -> None:
        content = dedent("""\
            [project]
            dependencies = []

            [project.optional-dependencies]
            tools = [
                "crewai[tools]>=1.0.0",
            ]
        """)
        result = _pin_crewai_deps(content, "3.0.0")
        assert '"crewai[tools]==3.0.0"' in result

    def test_handles_multiple_crewai_entries(self) -> None:
        content = dedent("""\
            [project]
            dependencies = [
                "crewai==1.0.0",
                "crewai[tools]==1.0.0",
            ]
        """)
        result = _pin_crewai_deps(content, "2.0.0")
        assert '"crewai==2.0.0"' in result
        assert '"crewai[tools]==2.0.0"' in result

    def test_preserves_arbitrary_extras(self) -> None:
        content = dedent("""\
            [project]
            dependencies = [
                "crewai[a2a]==1.0.0",
            ]
        """)
        result = _pin_crewai_deps(content, "2.0.0")
        assert '"crewai[a2a]==2.0.0"' in result

    def test_no_deps_returns_unchanged(self) -> None:
        content = dedent("""\
            [project]
            name = "empty"
        """)
        result = _pin_crewai_deps(content, "2.0.0")
        assert "empty" in result

    def test_skips_crewai_without_version_specifier(self) -> None:
        content = dedent("""\
            [project]
            dependencies = [
                "crewai-tools~=1.0",
            ]
        """)
        result = _pin_crewai_deps(content, "2.0.0")
        assert '"crewai-tools~=1.0"' in result

    def test_skips_crewai_extras_without_pin(self) -> None:
        content = dedent("""\
            [project]
            dependencies = [
                "crewai[tools]",
            ]
        """)
        result = _pin_crewai_deps(content, "2.0.0")
        assert '"crewai[tools]"' in result
        assert "==" not in result


# --- _repin_crewai_install ---


class TestRepinCrewaiInstall:
    def test_repins_a2a_extra(self) -> None:
        result = _repin_crewai_install('uv pip install "crewai[a2a]==1.14.0"', "2.0.0")
        assert result == 'uv pip install "crewai[a2a]==2.0.0"'

    def test_repins_tools_extra(self) -> None:
        result = _repin_crewai_install('uv pip install "crewai[tools]==1.0.0"', "3.0.0")
        assert result == 'uv pip install "crewai[tools]==3.0.0"'

    def test_leaves_unrelated_commands_alone(self) -> None:
        cmd = "uv pip install requests"
        assert _repin_crewai_install(cmd, "2.0.0") == cmd

    def test_handles_multiple_pins(self) -> None:
        cmd = 'pip install "crewai[a2a]==1.0.0" "crewai[tools]==1.0.0"'
        result = _repin_crewai_install(cmd, "2.0.0")
        assert result == 'pip install "crewai[a2a]==2.0.0" "crewai[tools]==2.0.0"'

    def test_preserves_surrounding_text(self) -> None:
        cmd = 'echo hello && uv pip install "crewai[a2a]==1.14.0" && echo done'
        result = _repin_crewai_install(cmd, "2.0.0")
        assert (
            result == 'echo hello && uv pip install "crewai[a2a]==2.0.0" && echo done'
        )

    def test_no_version_specifier_unchanged(self) -> None:
        cmd = 'pip install "crewai[tools]>=1.0"'
        assert _repin_crewai_install(cmd, "2.0.0") == cmd


# --- update_template_dependencies ---


class TestUpdateTemplateDependencies:
    def test_updates_jinja_template(self, tmp_path: Path) -> None:
        """Template pyproject.toml files with Jinja placeholders should not break."""
        tpl = tmp_path / "crew" / "pyproject.toml"
        tpl.parent.mkdir()
        tpl.write_text(
            dedent("""\
            [project]
            name = "{{folder_name}}"
            version = "0.1.0"
            dependencies = [
                "crewai[tools]==1.14.0"
            ]

            [project.scripts]
            {{folder_name}} = "{{folder_name}}.main:run"
        """)
        )

        updated = update_template_dependencies(tmp_path, "2.0.0")

        assert len(updated) == 1
        content = tpl.read_text()
        assert '"crewai[tools]==2.0.0"' in content
        assert "{{folder_name}}" in content

    def test_updates_bare_crewai(self, tmp_path: Path) -> None:
        tpl = tmp_path / "pyproject.toml"
        tpl.write_text('dependencies = [\n    "crewai==1.0.0"\n]\n')

        updated = update_template_dependencies(tmp_path, "3.0.0")

        assert len(updated) == 1
        assert '"crewai==3.0.0"' in tpl.read_text()

    def test_skips_unrelated_deps(self, tmp_path: Path) -> None:
        tpl = tmp_path / "pyproject.toml"
        tpl.write_text('dependencies = [\n    "requests>=2.0"\n]\n')

        updated = update_template_dependencies(tmp_path, "2.0.0")

        assert len(updated) == 0
        assert '"requests>=2.0"' in tpl.read_text()
