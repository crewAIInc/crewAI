"""Tests for TOML-based version and dependency update functions."""

from pathlib import Path
from textwrap import dedent

from crewai_devtools.cli import _pin_crewai_deps, update_pyproject_version


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
