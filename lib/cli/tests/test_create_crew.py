import ast
import asyncio
import builtins
import io
import keyword
import itertools
import json
import os
import re
import shlex
import shutil
import tempfile
import threading
import warnings
from pathlib import Path
from unittest import mock
from uuid import UUID

import click
import httpx
import pytest
import tomli
from click.testing import CliRunner
from crewai_core.platform_apps import PLATFORM_APP_CATEGORIES, PLATFORM_APPS
from packaging.requirements import Requirement
from packaging.version import Version
import crewai_cli.create_json_crew as json_crew
import crewai_cli.tui_picker as tui_picker
from crewai_cli import __version__ as crewai_cli_version
from crewai_cli.cli import crewai
from crewai_cli.constants import MODELS, PROVIDERS
from crewai_cli.create_crew import create_crew, create_folder_structure
from crewai_cli.utils import copy_assistant_instructions, render_template
from crewai_cli.version import (
    get_crewai_dependency_range,
    get_crewai_tools_dependency,
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def temp_dir():
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


def test_create_folder_structure_strips_single_trailing_slash():
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_path, folder_name, class_name = create_folder_structure(
            "hello/", parent_folder=temp_dir
        )

        assert folder_name == "hello"
        assert class_name == "Hello"
        assert folder_path.name == "hello"
        assert folder_path.exists()
        assert folder_path.parent == Path(temp_dir)


def test_create_folder_structure_strips_multiple_trailing_slashes():
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_path, folder_name, class_name = create_folder_structure(
            "hello///", parent_folder=temp_dir
        )

        assert folder_name == "hello"
        assert class_name == "Hello"
        assert folder_path.name == "hello"
        assert folder_path.exists()
        assert folder_path.parent == Path(temp_dir)


def test_create_folder_structure_handles_complex_name_with_trailing_slash():
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_path, folder_name, class_name = create_folder_structure(
            "my-awesome_project/", parent_folder=temp_dir
        )

        assert folder_name == "my_awesome_project"
        assert class_name == "MyAwesomeProject"
        assert folder_path.name == "my_awesome_project"
        assert folder_path.exists()
        assert folder_path.parent == Path(temp_dir)


def test_create_folder_structure_normal_name_unchanged():
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_path, folder_name, class_name = create_folder_structure(
            "hello", parent_folder=temp_dir
        )

        assert folder_name == "hello"
        assert class_name == "Hello"
        assert folder_path.name == "hello"
        assert folder_path.exists()
        assert folder_path.parent == Path(temp_dir)


def test_create_folder_structure_with_parent_folder():
    with tempfile.TemporaryDirectory() as temp_dir:
        parent_path = Path(temp_dir) / "parent"
        parent_path.mkdir()

        folder_path, folder_name, class_name = create_folder_structure(
            "child/", parent_folder=parent_path
        )

        assert folder_name == "child"
        assert class_name == "Child"
        assert folder_path.name == "child"
        assert folder_path.parent == parent_path
        assert folder_path.exists()


@mock.patch("crewai_cli.create_crew.copy_template")
@mock.patch("crewai_cli.create_crew.write_env_file")
@mock.patch("crewai_cli.create_crew.load_env_vars")
def test_create_crew_with_trailing_slash_creates_valid_project(
    mock_load_env, mock_write_env, mock_copy_template, temp_dir
):
    mock_load_env.return_value = {}

    with tempfile.TemporaryDirectory() as work_dir:
        with mock.patch(
            "crewai_cli.create_crew.create_folder_structure"
        ) as mock_create_folder:
            mock_folder_path = Path(work_dir) / "test_project"
            mock_folder_path.mkdir()
            mock_create_folder.return_value = (
                mock_folder_path,
                "test_project",
                "TestProject",
            )

            create_crew("test-project/", skip_provider=True)

            mock_create_folder.assert_called_once_with("test-project/", None)
            mock_copy_template.assert_called()
            copy_calls = mock_copy_template.call_args_list

            for call in copy_calls:
                args = call[0]
                if len(args) >= 5:
                    folder_name_arg = args[4]
                    assert not folder_name_arg.endswith("/"), (
                        f"folder_name should not end with slash: {folder_name_arg}"
                    )


@mock.patch("crewai_cli.create_crew.copy_template")
@mock.patch("crewai_cli.create_crew.write_env_file")
@mock.patch("crewai_cli.create_crew.load_env_vars")
def test_create_crew_with_multiple_trailing_slashes(
    mock_load_env, mock_write_env, mock_copy_template, temp_dir
):
    mock_load_env.return_value = {}

    with tempfile.TemporaryDirectory() as work_dir:
        with mock.patch(
            "crewai_cli.create_crew.create_folder_structure"
        ) as mock_create_folder:
            mock_folder_path = Path(work_dir) / "test_project"
            mock_folder_path.mkdir()
            mock_create_folder.return_value = (
                mock_folder_path,
                "test_project",
                "TestProject",
            )

            create_crew("test-project///", skip_provider=True)

            mock_create_folder.assert_called_once_with("test-project///", None)


@mock.patch("crewai_cli.create_crew.copy_template")
@mock.patch("crewai_cli.create_crew.write_env_file")
@mock.patch("crewai_cli.create_crew.load_env_vars")
def test_create_crew_normal_name_still_works(
    mock_load_env, mock_write_env, mock_copy_template, temp_dir
):
    mock_load_env.return_value = {}

    with tempfile.TemporaryDirectory() as work_dir:
        with mock.patch(
            "crewai_cli.create_crew.create_folder_structure"
        ) as mock_create_folder:
            mock_folder_path = Path(work_dir) / "normal_project"
            mock_folder_path.mkdir()
            mock_create_folder.return_value = (
                mock_folder_path,
                "normal_project",
                "NormalProject",
            )

            create_crew("normal-project", skip_provider=True)

            mock_create_folder.assert_called_once_with("normal-project", None)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
@pytest.mark.parametrize(
    ("args", "project_root"),
    [
        (["create", "crew", "Git Crew"], "git_crew"),
        (["create", "flow", "Git Flow"], "git_flow"),
    ],
)
def test_create_initializes_git_repo_when_git_is_available(
    args, project_root, tmp_path, monkeypatch, runner
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREWAI_DMN", "True")

    result = runner.invoke(crewai, args)

    assert result.exit_code == 0, result.output
    assert (tmp_path / project_root / ".git").is_dir()


def test_create_folder_structure_handles_spaces_and_dashes_with_slash():
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_path, folder_name, class_name = create_folder_structure(
            "My Cool-Project/", parent_folder=temp_dir
        )

        assert folder_name == "my_cool_project"
        assert class_name == "MyCoolProject"
        assert folder_path.name == "my_cool_project"
        assert folder_path.exists()
        assert folder_path.parent == Path(temp_dir)


def test_create_folder_structure_raises_error_for_invalid_names():
    with tempfile.TemporaryDirectory() as temp_dir:
        invalid_cases = [
            ("123project/", "cannot start with a digit"),
            ("True/", "reserved Python keyword"),
            ("False/", "reserved Python keyword"),
            ("None/", "reserved Python keyword"),
            ("class/", "reserved Python keyword"),
            ("def/", "reserved Python keyword"),
            ("   /", "empty or contain only whitespace"),
            ("", "empty or contain only whitespace"),
            ("@#$/", "contains no valid characters"),
        ]

        for invalid_name, expected_error in invalid_cases:
            with pytest.raises(ValueError, match=expected_error):
                create_folder_structure(invalid_name, parent_folder=temp_dir)


def test_create_folder_structure_validates_names():
    with tempfile.TemporaryDirectory() as temp_dir:
        valid_cases = [
            ("hello/", "hello", "Hello"),
            ("my-project/", "my_project", "MyProject"),
            ("hello_world/", "hello_world", "HelloWorld"),
            ("valid123/", "valid123", "Valid123"),
            ("hello.world/", "helloworld", "HelloWorld"),
            ("hello@world/", "helloworld", "HelloWorld"),
        ]

        for valid_name, expected_folder, expected_class in valid_cases:
            folder_path, folder_name, class_name = create_folder_structure(
                valid_name, parent_folder=temp_dir
            )
            assert folder_name == expected_folder
            assert class_name == expected_class

            assert folder_name.isidentifier(), (
                f"folder_name '{folder_name}' should be valid Python identifier"
            )
            assert not keyword.iskeyword(folder_name), (
                f"folder_name '{folder_name}' should not be Python keyword"
            )
            assert not folder_name[0].isdigit(), (
                f"folder_name '{folder_name}' should not start with digit"
            )

            assert class_name.isidentifier(), (
                f"class_name '{class_name}' should be valid Python identifier"
            )
            assert not keyword.iskeyword(class_name), (
                f"class_name '{class_name}' should not be Python keyword"
            )
            assert folder_path.parent == Path(temp_dir)

            if folder_path.exists():
                shutil.rmtree(folder_path)


@mock.patch("crewai_cli.create_crew.copy_template")
@mock.patch("crewai_cli.create_crew.write_env_file")
@mock.patch("crewai_cli.create_crew.load_env_vars")
def test_create_crew_with_parent_folder_and_trailing_slash(
    mock_load_env, mock_write_env, mock_copy_template, temp_dir
):
    mock_load_env.return_value = {}

    with tempfile.TemporaryDirectory() as work_dir:
        parent_path = Path(work_dir) / "parent"
        parent_path.mkdir()

        create_crew("child-crew/", skip_provider=True, parent_folder=parent_path)

        crew_path = parent_path / "child_crew"
        assert crew_path.exists()
        assert not (crew_path / "src").exists()


def test_create_folder_structure_folder_name_validation():
    """Test that folder names are validated as valid Python module names"""
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_invalid_cases = [
            ("123invalid/", "cannot start with a digit.*invalid Python module name"),
            ("import/", "reserved Python keyword"),
            ("class/", "reserved Python keyword"),
            ("for/", "reserved Python keyword"),
            ("@#$invalid/", "contains no valid characters.*Python module name"),
        ]

        for invalid_name, expected_error in folder_invalid_cases:
            with pytest.raises(ValueError, match=expected_error):
                create_folder_structure(invalid_name, parent_folder=temp_dir)

        valid_cases = [
            ("hello-world/", "hello_world"),
            ("my.project/", "myproject"),
            ("test@123/", "test123"),
            ("valid_name/", "valid_name"),
        ]

        for valid_name, expected_folder in valid_cases:
            folder_path, folder_name, class_name = create_folder_structure(
                valid_name, parent_folder=temp_dir
            )
            assert folder_name == expected_folder
            assert folder_name.isidentifier()
            assert not keyword.iskeyword(folder_name)

            if folder_path.exists():
                shutil.rmtree(folder_path)


def test_create_folder_structure_rejects_reserved_names():
    """Test that reserved script names are rejected to prevent pyproject.toml conflicts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reserved_names = ["test", "train", "replay", "run_crew", "run_with_trigger"]

        for reserved_name in reserved_names:
            with pytest.raises(ValueError, match="which is reserved"):
                create_folder_structure(reserved_name, parent_folder=temp_dir)

            with pytest.raises(ValueError, match="which is reserved"):
                create_folder_structure(f"{reserved_name}/", parent_folder=temp_dir)

            capitalized = reserved_name.capitalize()
            with pytest.raises(ValueError, match="which is reserved"):
                create_folder_structure(capitalized, parent_folder=temp_dir)


@mock.patch("crewai_cli.create_crew.create_folder_structure")
@mock.patch("crewai_cli.create_crew.copy_template")
@mock.patch("crewai_cli.create_crew.load_env_vars")
@mock.patch("crewai_cli.create_crew.get_provider_data")
@mock.patch("crewai_cli.create_crew.select_provider")
@mock.patch("crewai_cli.create_crew.select_model")
@mock.patch("click.prompt")
def test_env_vars_are_uppercased_in_env_file(
    mock_prompt,
    mock_select_model,
    mock_select_provider,
    mock_get_provider_data,
    mock_load_env_vars,
    mock_copy_template,
    mock_create_folder_structure,
    tmp_path,
):
    crew_path = tmp_path / "test_crew"
    crew_path.mkdir()
    mock_create_folder_structure.return_value = (crew_path, "test_crew", "TestCrew")

    mock_load_env_vars.return_value = {}
    mock_get_provider_data.return_value = {"openai": ["gpt-4"]}
    mock_select_provider.return_value = "azure"
    mock_select_model.return_value = "azure/openai"
    mock_prompt.return_value = "fake-api-key"

    create_crew("Test Crew")

    env_file_path = crew_path / ".env"
    content = env_file_path.read_text()
    assert "MODEL=" in content


def test_json_wizard_defaults_to_sequential_and_memory_enabled(monkeypatch):
    monkeypatch.setattr(
        json_crew,
        "_wizard_agent",
        lambda **_: {
            "name": "researcher",
            "role": "Researcher",
            "goal": "Research",
            "backstory": "Researcher",
            "llm": "openai/gpt-5.5",
            "tools": [],
            "planning": False,
            "allow_delegation": False,
        },
    )
    monkeypatch.setattr(
        json_crew,
        "_wizard_task",
        lambda **_: {
            "name": "research_task",
            "description": "Research",
            "expected_output": "Findings",
            "agent": "researcher",
            "context": [],
        },
    )

    def confirm(label: str, default: bool = False) -> bool:
        if label == "Enable crew memory?":
            return default
        return False

    monkeypatch.setattr(json_crew, "_confirm", confirm)
    monkeypatch.setattr(json_crew.click, "prompt", lambda *_, **__: "")
    monkeypatch.setattr(
        json_crew,
        "pick_one",
        lambda *_args, **_kwargs: pytest.fail("process should not be prompted"),
    )

    _agents, _tasks, settings = json_crew._wizard_agents_and_tasks(
        skip_provider=True,
        default_llm="openai/gpt-5.5",
    )

    assert settings == {"process": "sequential", "memory": True, "inputs": {}}


def test_json_wizard_shows_interpolation_hint(capsys):
    json_crew._show_interpolation_hint("tasks")

    output = capsys.readouterr().out
    assert "{placeholder}" in output
    assert "dynamic values" in output
    assert "{topic}" not in output
    assert "Description >" not in output
    assert '"description"' not in output


def test_json_wizard_text_prompt_uses_full_prompt_for_readline(monkeypatch):
    prompts: list[str] = []

    monkeypatch.setattr(
        json_crew, "_readline_safe_prompt", lambda prompt: f"safe:{prompt}"
    )
    monkeypatch.setattr(
        "builtins.input", lambda prompt: prompts.append(prompt) or "Draft content"
    )

    assert json_crew._prompt_text("Goal", spacing_before=False) == "Draft content"
    assert len(prompts) == 1
    assert prompts[0].startswith("safe:")
    assert "Goal" in prompts[0]
    assert " > " in prompts[0]


def test_json_wizard_tool_picker_prioritizes_common_tools(monkeypatch):
    picker_calls: list[tuple[str, list[str], dict[str, object]]] = []

    def pick_many(title: str, labels: list[str], **kwargs):
        picker_calls.append((title, labels, kwargs))
        return [1, 3], None

    monkeypatch.setattr(json_crew, "pick_many", pick_many)

    tools = json_crew._select_tools()

    assert tools == ["SerperDevTool", "FileReadTool"]
    assert len(picker_calls) == 1
    labels = picker_calls[0][1]
    assert 0 in picker_calls[0][2]["separator_indices"]
    assert labels[0] == "── Common tools ──"
    assert labels[1].strip().endswith("SerperDevTool")
    assert labels[2].strip().endswith("ScrapeWebsiteTool")
    assert labels[3].strip().endswith("FileReadTool")
    assert labels[4].startswith("Gmail Integration")
    assert labels[4].endswith("Platform: GmailIntegration (66 tools)")
    assert labels[5].startswith("WhatsApp Integration")
    assert labels[5].endswith("Platform: WhatsAppIntegration (58 tools)")
    assert labels[1].index("Google search") < labels[1].index("SerperDevTool")
    assert "More tools" not in labels


def test_platform_action_picker_is_searchable(monkeypatch):
    picker_calls: list[dict[str, object]] = []

    def pick_many(title: str, labels: list[str], **kwargs):
        picker_calls.append(kwargs)
        return []

    monkeypatch.setattr(json_crew, "pick_many", pick_many)

    assert json_crew._select_platform_actions(["platform:github"]) == []
    assert picker_calls[0]["searchable"] is True


def test_json_wizard_tool_picker_collapses_categories_by_default(monkeypatch):
    picker_calls: list[tuple[str, list[str], dict[str, object]]] = []

    def pick_many(title: str, labels: list[str], **kwargs):
        picker_calls.append((title, labels, kwargs))
        return [], None

    monkeypatch.setattr(json_crew, "pick_many", pick_many)

    json_crew._select_tools()

    labels = picker_calls[0][1]
    action_indices = picker_calls[0][2]["action_indices"]
    # Categories show as collapsed action rows, not separators with tools
    assert any(label.startswith("▸ Search & Research") for label in labels)
    assert any(label.startswith("▸ Web Scraping") for label in labels)
    assert not any(label.strip().endswith("BraveSearchTool") for label in labels)
    assert len(action_indices) >= 4
    # Only the common tools section is visible beyond the category rows
    assert len(labels) == 1 + 5 + len(action_indices)
    assert f"▸ CrewAI Platform  ({len(PLATFORM_APPS)} applications)" in labels


def test_json_wizard_tool_picker_expands_one_category_at_a_time(monkeypatch):
    picker_calls: list[tuple[str, list[str], dict[str, object]]] = []

    def find_category_row(labels: list[str], category: str) -> int:
        return next(
            idx for idx, label in enumerate(labels) if category in label
        )

    def pick_many(title: str, labels: list[str], **kwargs):
        picker_calls.append((title, labels, kwargs))
        call_num = len(picker_calls)
        if call_num == 1:
            return [], find_category_row(labels, "Search & Research")
        if call_num == 2:
            # Search & Research is expanded; select BraveSearchTool and
            # expand Web Scraping instead
            brave = next(
                idx
                for idx, label in enumerate(labels)
                if label.strip().endswith("BraveSearchTool")
            )
            return [brave], find_category_row(labels, "Web Scraping")
        return [], None

    monkeypatch.setattr(json_crew, "pick_many", pick_many)

    tools = json_crew._select_tools()

    assert tools == ["BraveSearchTool"]
    assert len(picker_calls) == 3
    # Second render: Search & Research expanded, others collapsed
    labels2 = picker_calls[1][1]
    assert any(label.startswith("▾ Search & Research") for label in labels2)
    assert any(label.strip().endswith("BraveSearchTool") for label in labels2)
    assert any(label.startswith("▸ Web Scraping") for label in labels2)
    # Third render: Web Scraping expanded, Search & Research collapsed again
    labels3 = picker_calls[2][1]
    assert any(label.startswith("▸ Search & Research") for label in labels3)
    assert any(label.startswith("▾ Web Scraping") for label in labels3)
    assert not any(label.strip().endswith("BraveSearchTool") for label in labels3)
    # The collapsed Search & Research row reports its selection count
    assert any(
        "Search & Research" in label and "1 selected" in label for label in labels3
    )
    # Cursor returns to the toggled category row
    assert picker_calls[2][2]["initial_cursor"] == next(
        idx for idx, label in enumerate(labels3) if "Web Scraping" in label
    )


def test_json_wizard_tool_picker_preserves_selection_across_renders(monkeypatch):
    picker_calls: list[tuple[str, list[str], dict[str, object]]] = []

    def pick_many(title: str, labels: list[str], **kwargs):
        picker_calls.append((title, labels, kwargs))
        call_num = len(picker_calls)
        if call_num == 1:
            # Select a common tool, then expand a category
            category_row = next(
                idx for idx, label in enumerate(labels) if "Web Scraping" in label
            )
            return [1], category_row
        # Confirm without touching anything else
        return sorted(kwargs["preselected"]), None

    monkeypatch.setattr(json_crew, "pick_many", pick_many)

    tools = json_crew._select_tools()

    # The common-tool selection survived the expand re-render via preselected
    assert tools == ["SerperDevTool"]
    assert 1 in picker_calls[1][2]["preselected"]


def test_json_wizard_tool_picker_lists_builtin_tools_across_categories(monkeypatch):
    picker_calls: list[tuple[str, list[str], dict[str, object]]] = []
    expanded_labels: list[str] = []

    categories = [
        "Search & Research",
        "Web Scraping",
        "File & Document",
        "Code & Data",
        "Cloud & Storage",
        "Sandbox & Automation",
        "AI & Vision",
    ]

    def pick_many(title: str, labels: list[str], **kwargs):
        picker_calls.append((title, labels, kwargs))
        expanded_labels.extend(labels)
        call_num = len(picker_calls)
        if call_num <= len(categories):
            category = categories[call_num - 1]
            category_row = next(
                idx for idx, label in enumerate(labels) if category in label
            )
            return [], category_row
        return [], None

    monkeypatch.setattr(json_crew, "pick_many", pick_many)

    json_crew._select_tools()

    tool_names = {
        label.rsplit(maxsplit=1)[-1]
        for label in expanded_labels
        if not label.startswith(("▸", "▾", "──"))
    }

    assert {
        "DirectorySearchTool",
        "MDXSearchTool",
        "XMLSearchTool",
        "YoutubeVideoSearchTool",
        "S3ReaderTool",
        "E2BExecTool",
        "TavilyResearchTool",
        "SerplyNewsSearchTool",
        "BrowserbaseLoadTool",
        "PatronusEvalTool",
    }.issubset(tool_names)
    assert {
        "MCPServerAdapter",
        "MongoDBVectorSearchConfig",
        "ScrapegraphScrapeToolSchema",
        "SnowflakeConfig",
    }.isdisjoint(tool_names)


def test_json_wizard_platform_tool_selection_stays_in_agent_tools(monkeypatch):
    picker_calls = 0

    def pick_many(title: str, labels: list[str], **kwargs):
        nonlocal picker_calls
        picker_calls += 1
        if picker_calls == 1:
            platform_row = next(
                idx for idx, label in enumerate(labels) if "CrewAI Platform" in label
            )
            return [], platform_row

        if picker_calls == 2:
            engineering = next(
                idx
                for idx, label in enumerate(labels)
                if "Engineering, data & infrastructure" in label
            )
            return [], engineering

        if picker_calls == 3:
            github = next(
                idx
                for idx, label in enumerate(labels)
                if label.strip().startswith("GitHub Integration")
            )
            return [github], None

        create_repository = next(
            idx
            for idx, label in enumerate(labels)
            if label == "Create repository (create_repository)"
        )
        return [create_repository]

    monkeypatch.setattr(json_crew, "pick_many", pick_many)
    monkeypatch.setattr(
        json_crew, "_prompt_text", lambda label, **kwargs: label.lower()
    )
    monkeypatch.setattr(json_crew, "_select_model", lambda: "openai/gpt-5.5")
    monkeypatch.setattr(json_crew, "_confirm", lambda *_args, **_kwargs: False)

    agent = json_crew._wizard_agent(agent_num=1, existing_names=[])

    assert agent is not None
    assert agent["tools"] == ["platform:github/create_repository"]
    assert (
        '"tools": ["platform:github/create_repository"]'
        in json_crew._agent_to_jsonc(agent)
    )


def test_platform_applications_expand_in_the_main_tool_picker(monkeypatch):
    picker_calls: list[tuple[str, list[str], dict[str, object]]] = []

    def pick_many(title: str, labels: list[str], **kwargs):
        picker_calls.append((title, labels, kwargs))
        if len(picker_calls) == 1:
            platform = next(
                index
                for index, label in enumerate(labels)
                if "CrewAI Platform" in label
            )
            return [], platform
        if len(picker_calls) == 2:
            group = next(
                index
                for index, label in enumerate(labels)
                if "Engineering, data & infrastructure" in label
            )
            return [], group
        github = next(
            index
            for index, label in enumerate(labels)
            if label.strip().startswith("GitHub Integration")
        )
        return [github], None

    monkeypatch.setattr(json_crew, "pick_many", pick_many)

    assert json_crew._select_tools() == ["platform:github"]
    labels = picker_calls[1][1]
    assert labels[0] == "── Common tools ──"
    for category in PLATFORM_APP_CATEGORIES:
        assert f"  ▸ {category.name}  ({len(category.apps)} applications)" in labels
    assert picker_calls[1][2]["initial_cursor"] == next(
        index for index, label in enumerate(labels) if "CrewAI Platform" in label
    )
    assert picker_calls[1][2]["searchable"] is True
    github_index = next(
        index
        for index, label in enumerate(labels)
        if label.strip().startswith("GitHub Integration")
    )
    assert github_index in picker_calls[1][2]["collapsed_indices"]
    assert all(title != "Platform applications (space to toggle, enter to return):" for title, *_ in picker_calls)


def test_json_wizard_platform_catalog_contains_every_supported_app():
    platform_category = next(
        tools
        for category, tools in json_crew._TOOL_CATEGORIES
        if category == "CrewAI Platform"
    )

    assert [name for name, _description in platform_category] == [
        f"platform:{app}" for app in PLATFORM_APPS
    ]


def test_platform_validation_deduplicates_actions_by_application() -> None:
    apps = json_crew._platform_apps_from_agents(
        [
            {
                "tools": [
                    "platform:github/create_repository",
                    "platform:github/get_repository_content",
                    "platform:gmail/send_email",
                ]
            },
            {"tools": ["platform:github/list_repository_issues"]},
        ]
    )

    assert apps == ["github", "gmail"]


def test_platform_revalidation_requires_confirmation(monkeypatch) -> None:
    confirm_options: list[dict[str, object]] = []

    def confirm(_label: str, **kwargs: object) -> bool:
        confirm_options.append(kwargs)
        return True

    monkeypatch.setattr(json_crew.click, "confirm", confirm)

    assert json_crew._confirm_platform_revalidation()
    assert confirm_options == [{"default": True}]


def test_platform_validation_guidance_links_to_amp(capsys) -> None:
    json_crew._show_platform_validation_guidance(
        ["whatsapp"], False, "https://app.crewai.com/"
    )

    output = capsys.readouterr().out
    assert "Connect the WhatsApp integration in CrewAI AMP" in output
    assert "https://app.crewai.com" in output


def test_get_logged_in_platform_token_uses_saved_login_and_organization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = mock.Mock()
    client.get_integration_token.return_value = "integration-token"
    monkeypatch.setattr(json_crew, "get_auth_token", lambda: "login-token")
    monkeypatch.setenv("CREWAI_PLUS_URL", "https://fake.crewai.com")
    monkeypatch.setattr(
        json_crew,
        "Settings",
        lambda: mock.Mock(
            org_uuid="organization-id",
            enterprise_base_url="https://app.crewai.com",
        ),
    )
    plus_api = mock.Mock(return_value=client)
    monkeypatch.setattr(json_crew, "PlusAPI", plus_api)

    assert json_crew._get_logged_in_platform_token() == "integration-token"

    plus_api.assert_called_once_with(
        api_key="login-token",
        organization_id="organization-id",
        base_url="https://app.crewai.com",
    )
    client.get_integration_token.assert_called_once_with()


def test_get_logged_in_platform_token_requires_selected_organization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(json_crew, "get_auth_token", lambda: "login-token")
    monkeypatch.setattr(json_crew, "Settings", lambda: mock.Mock(org_uuid=None))

    with pytest.raises(RuntimeError, match="No CrewAI organization is selected"):
        json_crew._get_logged_in_platform_token()


def test_get_platform_token_from_login_signs_in_then_retries(monkeypatch) -> None:
    tokens = iter([json_crew.AuthError("No token found"), "integration-token"])

    def get_token() -> str:
        token = next(tokens)
        if isinstance(token, Exception):
            raise token
        return token

    monkeypatch.setattr(
        json_crew,
        "_get_logged_in_platform_token",
        get_token,
    )

    authentication_command = mock.Mock()
    monkeypatch.setattr(
        json_crew, "AuthenticationCommand", mock.Mock(return_value=authentication_command)
    )

    assert json_crew._get_platform_token_from_login() == "integration-token"

    authentication_command.login.assert_called_once_with()


def test_get_platform_token_from_login_stops_when_the_login_times_out(
    monkeypatch,
) -> None:
    """A timed-out login exits cleanly instead of retrying into an AuthError."""
    attempts: list[bool] = []

    def get_token() -> str:
        attempts.append(True)
        raise json_crew.AuthError("No token found")

    monkeypatch.setattr(json_crew, "_get_logged_in_platform_token", get_token)
    authentication_command = mock.Mock()
    authentication_command.login.side_effect = SystemExit(1)
    monkeypatch.setattr(
        json_crew, "AuthenticationCommand", mock.Mock(return_value=authentication_command)
    )

    with pytest.raises(SystemExit) as exc_info:
        json_crew._get_platform_token_from_login()

    assert exc_info.value.code == 1
    assert attempts == [True]


def test_get_platform_token_from_login_reauthenticates_an_expired_login(
    monkeypatch,
) -> None:
    response = mock.Mock(status_code=401)
    tokens = iter(
        [
            httpx.HTTPStatusError(
                "Unauthorized", request=mock.Mock(), response=response
            ),
            "integration-token",
        ]
    )

    def get_token() -> str:
        token = next(tokens)
        if isinstance(token, Exception):
            raise token
        return token

    monkeypatch.setattr(json_crew, "_get_logged_in_platform_token", get_token)
    authentication_command = mock.Mock()
    monkeypatch.setattr(
        json_crew, "AuthenticationCommand", mock.Mock(return_value=authentication_command)
    )

    assert json_crew._get_platform_token_from_login() == "integration-token"

    authentication_command.login.assert_called_once_with()


def test_get_platform_token_from_login_reports_forbidden_organization(monkeypatch) -> None:
    response = mock.Mock(status_code=403)
    error = httpx.HTTPStatusError("Forbidden", request=mock.Mock(), response=response)
    monkeypatch.setattr(
        json_crew,
        "_get_logged_in_platform_token",
        mock.Mock(side_effect=error),
    )
    authentication_command = mock.Mock()
    monkeypatch.setattr(
        json_crew, "AuthenticationCommand", mock.Mock(return_value=authentication_command)
    )

    with pytest.raises(click.ClickException, match="does not have access"):
        json_crew._get_platform_token_from_login()

    authentication_command.login.assert_not_called()


def test_get_platform_token_from_login_selects_missing_organization(monkeypatch) -> None:
    tokens = iter(
        [json_crew._NoOrganizationSelectedError(), "integration-token"]
    )

    def get_token() -> str:
        token = next(tokens)
        if isinstance(token, Exception):
            raise token
        return token

    monkeypatch.setattr(json_crew, "_get_logged_in_platform_token", get_token)
    authentication_command = mock.Mock()
    monkeypatch.setattr(
        json_crew, "AuthenticationCommand", mock.Mock(return_value=authentication_command)
    )

    assert json_crew._get_platform_token_from_login() == "integration-token"

    authentication_command._select_organization.assert_called_once_with()


def test_get_platform_token_from_login_requires_organization_selection(monkeypatch) -> None:
    monkeypatch.setattr(json_crew, "get_auth_token", lambda: "login-token")
    monkeypatch.setattr(json_crew, "Settings", lambda: mock.Mock(org_uuid=None))
    authentication_command = mock.Mock()
    monkeypatch.setattr(
        json_crew, "AuthenticationCommand", mock.Mock(return_value=authentication_command)
    )

    with pytest.raises(
        click.ClickException,
        match="A CrewAI organization must be selected",
    ):
        json_crew._get_platform_token_from_login()

    authentication_command._select_organization.assert_called_once_with()


def test_platform_auth_suppresses_warnings_only_while_importing_tools(
    monkeypatch, capsys
):
    class FakeApplicationSelector:
        @classmethod
        def from_string(cls, value: str) -> str:
            return value

    fake_client = mock.Mock()
    fake_client.get_actions.return_value = [object()]
    fake_integrations_client = mock.Mock(
        ApplicationSelector=FakeApplicationSelector,
        client_for_selector=lambda _selector, **_kwargs: fake_client,
    )
    original_import = builtins.__import__

    def import_with_warning(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "crewai_tools.tools.crewai_platform_tools.integrations_client":
            warnings.warn("optional dependency import warning", UserWarning)
            return fake_integrations_client
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_warning)
    monkeypatch.setenv("CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN", "test-token")
    get_token = mock.Mock()
    monkeypatch.setattr(json_crew, "_get_platform_token_from_login", get_token)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        json_crew._setup_platform_auth([{"tools": ["platform:github"]}])
        warnings.warn("warning after import", UserWarning)

    assert [str(warning.message) for warning in caught_warnings] == [
        "warning after import"
    ]
    output = capsys.readouterr().out
    assert "Checking for CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN" in output
    assert (
        "Enterprise Action Auth Token found via "
        "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN environment variable" in output
    )
    get_token.assert_not_called()


def test_platform_auth_fetches_token_from_login_when_env_is_missing(monkeypatch) -> None:
    class FakeApplicationSelector:
        @classmethod
        def from_string(cls, value: str) -> str:
            return value

    fake_client = mock.Mock()
    fake_client.get_actions.return_value = [object()]
    fake_integrations_client = mock.Mock(
        ApplicationSelector=FakeApplicationSelector,
        client_for_selector=lambda _selector, **_kwargs: fake_client,
    )
    original_import = builtins.__import__

    def import_integrations(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "crewai_tools.tools.crewai_platform_tools.integrations_client":
            return fake_integrations_client
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_integrations)
    monkeypatch.delenv("CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN", raising=False)
    get_token = mock.Mock(return_value="login-token")
    monkeypatch.setattr(json_crew, "_get_platform_token_from_login", get_token)

    assert json_crew._setup_platform_auth([{"tools": ["platform:github"]}]) == "login-token"

    get_token.assert_called_once_with()
    assert os.environ["CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN"] == "login-token"


def test_platform_validation_checks_apps_concurrently():
    class FakeApplicationSelector:
        @classmethod
        def from_string(cls, value: str) -> str:
            return value

    barrier = threading.Barrier(2)

    class FakeClient:
        def get_actions(self, _selectors):
            barrier.wait(timeout=1)
            return [object()]

    results = asyncio.run(
        json_crew._check_platform_apps_concurrently(
            ["github", "gmail"],
            FakeApplicationSelector,
            lambda _selector, **_kwargs: FakeClient(),
        )
    )

    assert all(actions and error is None for actions, error in results)


def test_platform_validation_announces_concurrent_apps_together(capsys):
    class FakeApplicationSelector:
        @classmethod
        def from_string(cls, value: str) -> str:
            return value

    class FakeClient:
        def get_actions(self, _selectors):
            return [object()]

    failed_apps, token_invalid = json_crew._validate_platform_apps(
        ["github", "gmail", "google_calendar"],
        FakeApplicationSelector,
        lambda _selector, **_kwargs: FakeClient(),
    )

    output = capsys.readouterr().out
    assert (failed_apps, token_invalid) == ([], False)
    assert "Checking GitHub, Gmail, Google Calendar integrations together on AMP" in output
    assert "Checking Enterprise Action Auth Token and GitHub" not in output


def test_platform_validation_passes_configured_amp_url_to_client_factory() -> None:
    class FakeApplicationSelector:
        @classmethod
        def from_string(cls, value: str) -> str:
            return value

    received_base_urls: list[str | None] = []

    class FakeClient:
        def get_actions(self, _selectors):
            return [object()]

    def client_factory(_selector, *, base_url=None):
        received_base_urls.append(base_url)
        return FakeClient()

    assert json_crew._validate_platform_apps(
        ["github"],
        FakeApplicationSelector,
        client_factory,
        "https://app.crewai.com",
    ) == ([], False)
    assert received_base_urls == ["https://app.crewai.com"]


def test_platform_validation_falls_back_to_sequential_checks(monkeypatch, capsys):
    class FakeApplicationSelector:
        @classmethod
        def from_string(cls, value: str) -> str:
            return value

    checked_apps: list[str] = []

    class FakeClient:
        def __init__(self, app: str):
            self.app = app

        def get_actions(self, _selectors):
            checked_apps.append(self.app)
            return [object()]

    monkeypatch.setattr(json_crew.asyncio, "get_running_loop", object)

    failed_apps, token_invalid = json_crew._validate_platform_apps(
        ["github", "gmail"],
        FakeApplicationSelector,
        lambda app, **_kwargs: FakeClient(app),
    )

    assert (failed_apps, token_invalid) == ([], False)
    assert checked_apps == ["github", "gmail"]
    output = capsys.readouterr().out
    assert "Checking Enterprise Action Auth Token and GitHub integration" in output
    assert "Checking Enterprise Action Auth Token and Gmail integration" in output


def test_multi_picker_skips_separator_on_initial_cursor(monkeypatch):
    cursors: list[int] = []

    monkeypatch.setattr(tui_picker, "_read_key", lambda: "enter")
    monkeypatch.setattr(
        tui_picker,
        "_draw_multi",
        lambda _labels, cursor, *_args, **_kwargs: cursors.append(cursor),
    )
    monkeypatch.setattr(tui_picker, "_clear_lines", lambda *_args, **_kwargs: None)

    assert tui_picker._arrow_select_multi(
        ["── Common tools ──", "Google search via Serper API SerperDevTool"],
        separator_indices={0},
    ) == ([], None)
    assert cursors == [1]


def test_json_wizard_agent_attribute_prompts_are_compact(monkeypatch):
    prompt_calls: list[tuple[str, bool]] = []
    prompt_values = {
        "Role": "Senior Dev Rel",
        "Goal": "Draft content",
        "Backstory": "Knows developer communities",
    }

    def prompt_text(
        label: str,
        default: str = "",
        *,
        spacing_before: bool = True,
    ) -> str:
        prompt_calls.append((label, spacing_before))
        return prompt_values[label]

    monkeypatch.setattr(json_crew, "_prompt_text", prompt_text)
    monkeypatch.setattr(json_crew, "_select_model", lambda: "openai/gpt-5.5")
    monkeypatch.setattr(json_crew, "pick_many", lambda *_args, **_kwargs: ([], None))
    monkeypatch.setattr(json_crew, "_confirm", lambda *_args, **_kwargs: False)

    agent = json_crew._wizard_agent(agent_num=1, existing_names=[])

    assert agent is not None
    assert prompt_calls == [
        ("Role", False),
        ("Goal", False),
        ("Backstory", False),
    ]


def test_json_wizard_task_attribute_prompts_are_compact(monkeypatch):
    prompt_calls: list[tuple[str, bool]] = []
    prompt_values = {
        "Description": "Research latest release",
        "Expected output": "Release summary",
    }

    def prompt_text(
        label: str,
        default: str = "",
        *,
        spacing_before: bool = True,
    ) -> str:
        prompt_calls.append((label, spacing_before))
        return prompt_values[label]

    monkeypatch.setattr(json_crew, "_prompt_text", prompt_text)

    task = json_crew._wizard_task(
        task_num=1,
        agent_names=["senior_dev_rel"],
        prior_task_names=[],
    )

    assert task is not None
    assert prompt_calls == [
        ("Description", False),
        ("Expected output", False),
    ]


def test_json_create_provider_preselects_default_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with mock.patch(
        "crewai_cli.create_json_crew._wizard_agents_and_tasks"
    ) as mock_wizard:
        mock_wizard.return_value = (
            [
                {
                    "name": "researcher",
                    "role": "Researcher",
                    "goal": "Research",
                    "backstory": "Researcher",
                    "llm": "openai/gpt-5.5",
                    "tools": [],
                    "planning": False,
                    "allow_delegation": False,
                }
            ],
            [
                {
                    "name": "research_task",
                    "description": "Research",
                    "expected_output": "Findings",
                    "agent": "researcher",
                    "context": [],
                }
            ],
            {"process": "sequential", "memory": False, "inputs": {}},
        )

        json_crew.create_json_crew("JSON Crew", provider="openai", skip_provider=True)

    mock_wizard.assert_called_once_with(
        skip_provider=True,
        default_llm="openai/gpt-6-luna",
    )
    assert (tmp_path / "json_crew" / "crew.jsonc").exists()
    assert not (tmp_path / "json_crew" / "src").exists()
    assert not (tmp_path / "json_crew" / "tests").exists()
    assert not (tmp_path / "json_crew" / "config.jsonc").exists()
    generated_paths = {
        path.relative_to(tmp_path / "json_crew").as_posix()
        for path in (tmp_path / "json_crew").rglob("*")
        if path.is_file()
    }
    assert not any(
        path.endswith("/crew.py") or path == "crew.py" for path in generated_paths
    )
    assert not any(
        path.endswith("/agents.yaml") or path == "agents.yaml"
        for path in generated_paths
    )
    assert not any(
        path.endswith("/tasks.yaml") or path == "tasks.yaml"
        for path in generated_paths
    )
    assert not any(path.startswith("src/") for path in generated_paths)

    pyproject = tomli.loads((tmp_path / "json_crew" / "pyproject.toml").read_text())
    dependency = pyproject["project"]["dependencies"][0]
    assert dependency == get_crewai_tools_dependency()
    current = Version(crewai_cli_version)
    assert current in Requirement(dependency).specifier
    assert Version(f"{current.major + 1}.0.0") not in Requirement(dependency).specifier
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"][
        "only-include"
    ] == ["agents", "crew.jsonc", "tools", "knowledge", "skills"]
    crewai_table = pyproject["tool"]["crewai"]
    UUID(crewai_table.pop("project_id"))
    assert crewai_table == {
        "type": "crew",
        "definition": "crew.jsonc",
    }

    crew_template = (tmp_path / "json_crew" / "crew.jsonc").read_text()
    assert (
        '"guardrail": "Every factual claim needs context support."'
        in crew_template
    )
    assert '"guardrails": [' in crew_template
    assert '"guardrail_max_retries": 2' in crew_template
    assert "Docs: https://docs.crewai.com/concepts/tasks" in crew_template
    assert '"output_pydantic": null' in crew_template
    assert '"type": "ConditionalTask"' in crew_template
    assert '"condition": { "python": "my_project.conditions.should_run" }' in (
        crew_template
    )
    assert '"output_json": { "python": "my_project.models.ReportOutput" }' in (
        crew_template
    )
    assert (
        '"converter_cls": { "python": "my_project.converters.CustomConverter" }'
        in crew_template
    )
    assert '"markdown": false' in crew_template
    assert '"input_files": { "brief": "data/brief.txt" }' in crew_template
    assert "Docs: https://docs.crewai.com/concepts/crews" in crew_template
    assert "manager_agent can reference an agents/<name>.jsonc file" in crew_template
    assert '"manager_agent": "researcher"' in crew_template
    assert (
        '"before_kickoff_callbacks": [{"python": '
        '"my_project.callbacks.before_kickoff"}]'
    ) in crew_template
    assert (
        '"after_kickoff_callbacks": [{"python": '
        '"my_project.callbacks.after_kickoff"}]'
    ) in crew_template
    assert '"output_log_file": "crew.log"' in crew_template
    assert "Crew-level LLM fields also accept object form" in crew_template
    assert '"chat_llm": {"model": "llama3", "provider": "ollama"' in (
        crew_template
    )
    assert "Use {placeholder} in agent or task text" in crew_template
    assert "`crewai run` prompts for any placeholders" in crew_template
    assert "Use {placeholder} inputs here" in crew_template

    agent_template = (
        tmp_path / "json_crew" / "agents" / "researcher.jsonc"
    ).read_text()
    assert "You can use {placeholder} inputs in role, goal, or backstory" in (
        agent_template
    )
    assert '"role": "Senior {industry} Researcher"' in agent_template
    assert '"type": {"python": "my_project.agents.CustomAgent"}' in agent_template
    assert "Optional agent-level guardrail" in agent_template
    assert "Python refs must point to module-level functions/classes" in agent_template
    assert (
        '"step_callback": {"python": "my_project.callbacks.on_agent_step"}'
        in agent_template
    )
    assert '"guardrail_max_retries": 2' in agent_template
    assert "Docs: https://docs.crewai.com/concepts/agents" in agent_template
    assert '"reasoning": true' in agent_template
    assert "For custom endpoints or deployment-based providers" in agent_template
    assert '"deployment_name": "my-deployment", "provider": "azure"' in (
        agent_template
    )
    assert '"planning_config": {' in agent_template
    assert '"llm": {"model": "deepseek-chat", "provider": "deepseek"}' in (
        agent_template
    )
    assert '"knowledge_sources": []' in agent_template


def test_json_create_saves_platform_token_to_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        json_crew,
        "_wizard_agents_and_tasks",
        lambda **_: (
            [
                {
                    "name": "researcher",
                    "role": "Researcher",
                    "goal": "Research",
                    "backstory": "Researcher",
                    "llm": "openai/gpt-5.5",
                    "tools": ["platform:github"],
                    "planning": False,
                    "allow_delegation": False,
                }
            ],
            [
                {
                    "name": "research_task",
                    "description": "Research",
                    "expected_output": "Findings",
                    "agent": "researcher",
                    "context": [],
                }
            ],
            {"process": "sequential", "memory": False, "inputs": {}},
        ),
    )
    monkeypatch.setattr(json_crew, "_setup_platform_auth", lambda _agents: "token")

    json_crew.create_json_crew("Platform Crew", skip_provider=True)

    env_file = tmp_path / "platform_crew" / ".env"
    assert "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN=token" in env_file.read_text()


def test_json_crew_uses_template_files():
    template_names = {
        "pyproject.toml",
        "README.md",
        ".gitignore",
        "agent.jsonc",
        "agent_settings.jsonc",
        "task.jsonc",
        "crew.jsonc",
        "knowledge/user_preference.txt",
    }

    for template_name in template_names:
        assert (json_crew._TEMPLATES_DIR / template_name).is_file()


def test_render_template_does_not_replace_tokens_inside_replacement_values(tmp_path):
    template = tmp_path / "template.txt"
    template.write_text("{{first}} {{second}}", encoding="utf-8")

    rendered = render_template(
        template,
        {
            "first": "{{second}}",
            "second": "done",
        },
    )

    assert rendered == "{{second}} done"


def test_json_provider_default_model_helper():
    assert json_crew._default_model_for_provider("openai") == "openai/gpt-6-luna"
    assert json_crew._default_model_for_provider("anthropic") == (
        "anthropic/claude-sonnet-5-5"
    )
    assert json_crew._default_model_for_provider("anthropic/claude-custom") == (
        "anthropic/claude-custom"
    )
    assert json_crew._default_model_for_provider("unknown") is None


def test_json_wizard_task_reprompts_on_cancelled_agent_pick(monkeypatch):
    """Esc on the agent picker must reprompt, not silently assign agent 0."""
    prompts = iter(["Do the research", "A report"])
    monkeypatch.setattr(json_crew, "_prompt_text", lambda *a, **k: next(prompts))

    pick_calls: list[str] = []
    picks = iter([-1, 1])

    def fake_pick_one(title: str, labels: list[str]) -> int:
        pick_calls.append(title)
        return next(picks)

    monkeypatch.setattr(json_crew, "pick_one", fake_pick_one)

    task = json_crew._wizard_task(
        task_num=1,
        agent_names=["first_agent", "second_agent"],
        prior_task_names=[],
    )

    assert len(pick_calls) == 2
    assert task["agent"] == "second_agent"


def test_json_create_dmn_mode_uses_non_interactive_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREWAI_DMN", "True")
    monkeypatch.setattr(
        json_crew,
        "_wizard_agents_and_tasks",
        lambda **_: pytest.fail("DMN mode must not run the wizard"),
    )
    monkeypatch.setattr(
        json_crew,
        "_setup_env",
        lambda *_args, **_kwargs: pytest.fail("DMN mode must not prompt for env vars"),
    )

    json_crew.create_json_crew("DMN Crew", provider="anthropic", skip_provider=False)

    project_root = tmp_path / "dmn_crew"
    assert (project_root / "crew.jsonc").exists()
    assert (project_root / "agents" / "researcher.jsonc").exists()
    assert not (project_root / ".env").exists()

    crew_template = (project_root / "crew.jsonc").read_text()
    agent_template = (project_root / "agents" / "researcher.jsonc").read_text()

    assert '"memory": true' in crew_template
    assert '"description": "Research current AI trends and write a concise summary."' in (
        crew_template
    )
    default_model = json_crew._default_model_for_provider("anthropic")
    assert f'"llm": "{default_model}"' in agent_template


FALLBACK_LLM = "openai/gpt-6-luna"


def _agent(name: str, llm: str) -> dict[str, object]:
    return {
        "name": name,
        "role": name.title(),
        "goal": "Research",
        "backstory": "Researcher",
        "llm": llm,
        "tools": [],
        "planning": False,
        "allow_delegation": False,
    }


def _record_key_prompts(monkeypatch, answers: list[str]) -> list[dict[str, object]]:
    """Answer API-key prompts in order and record each prompt's text and options."""
    calls: list[dict[str, object]] = []
    replies = iter(answers)

    def fake_prompt(text: str, **kwargs: object) -> str:
        calls.append({"text": click.unstyle(text), **kwargs})
        return next(replies)

    monkeypatch.setattr(json_crew.click, "prompt", fake_prompt)
    return calls


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def terminal_stdin(monkeypatch):
    """Prompt as a person at a terminal would be prompted."""
    monkeypatch.setattr("sys.stdin", _Terminal())


def _pipe_to_stdin(monkeypatch, text: str) -> None:
    """Pipe ``text`` to the real ``click.prompt``; a hidden prompt fails the test,
    because outside pytest it reads the terminal and never sees the pipe."""
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    monkeypatch.setattr(
        click.termui,
        "hidden_prompt_func",
        lambda _prompt: pytest.fail("a hidden prompt does not read piped stdin"),
    )


def _wizard_result(*agents: dict[str, object]):
    return (
        list(agents),
        [
            {
                "name": "research_task",
                "description": "Research",
                "expected_output": "Findings",
                "agent": agents[0]["name"],
                "context": [],
            }
        ],
        {"process": "sequential", "memory": False, "inputs": {}},
    )


def test_json_wizard_agent_falls_back_to_current_small_model_when_provider_is_skipped(
    monkeypatch,
):
    monkeypatch.setattr(json_crew, "_prompt_text", lambda label, **_: label)
    monkeypatch.setattr(json_crew, "pick_many", lambda *_args, **_kwargs: ([], None))
    monkeypatch.setattr(json_crew, "_confirm", lambda *_args, **_kwargs: False)

    agent = json_crew._wizard_agent(agent_num=1, existing_names=[], skip_provider=True)

    assert agent is not None
    assert agent["llm"] == FALLBACK_LLM


def test_json_model_picker_falls_back_to_current_small_model_on_escape(monkeypatch):
    monkeypatch.setattr(json_crew, "pick_one", lambda *_args, **_kwargs: -1)

    assert json_crew._select_model() == FALLBACK_LLM


@pytest.mark.parametrize(
    ("provider", "model", "key_name"),
    [
        (None, FALLBACK_LLM, "OPENAI_API_KEY"),
        ("anthropic", "anthropic/claude-sonnet-5-5", "ANTHROPIC_API_KEY"),
    ],
)
def test_json_create_dmn_mode_next_steps_name_the_key_the_model_needs(
    tmp_path, monkeypatch, capsys, no_provider_keys, provider, model, key_name
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREWAI_DMN", "True")

    json_crew.create_json_crew("DMN Crew", provider=provider, skip_provider=True)

    agent_file = tmp_path / "dmn_crew" / "agents" / "researcher.jsonc"
    assert f'"llm": "{model}"' in agent_file.read_text()
    output = capsys.readouterr().out
    assert f"  Set {key_name} in .env before `crewai run`\n" in output
    assert output.index("cd dmn_crew") < output.index(key_name)
    assert output.index(key_name) < output.index("Run your crew:")
    assert not (tmp_path / "dmn_crew" / ".env").exists()


def test_json_create_dmn_mode_next_steps_name_each_key_of_a_multi_key_provider(
    tmp_path, monkeypatch, capsys, no_provider_keys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREWAI_DMN", "True")

    json_crew.create_json_crew(
        "DMN Crew", provider="bedrock/anthropic.claude-sonnet-5", skip_provider=True
    )

    output = capsys.readouterr().out
    assert (
        "  Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY in .env before `crewai run`\n"
    ) in output
    assert "AWS_DEFAULT_REGION" not in output


def test_json_create_dmn_mode_with_ollama_names_no_key(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREWAI_DMN", "True")

    json_crew.create_json_crew("DMN Crew", provider="ollama", skip_provider=True)

    output = capsys.readouterr().out
    assert "Run your crew:" in output
    assert "in .env before `crewai run`" not in output


def test_json_create_dmn_mode_next_steps_skip_a_key_already_exported(
    tmp_path, monkeypatch, capsys, no_provider_keys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREWAI_DMN", "True")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")

    json_crew.create_json_crew("DMN Crew", skip_provider=True)

    output = capsys.readouterr().out
    assert "Run your crew:" in output
    assert "OPENAI_API_KEY" not in output
    assert not (tmp_path / "dmn_crew" / ".env").exists()


def test_json_create_skip_provider_names_the_key_in_next_steps_without_dmn(
    tmp_path, monkeypatch, capsys, no_provider_keys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    monkeypatch.setattr(
        json_crew,
        "_wizard_agents_and_tasks",
        lambda **_: _wizard_result(_agent("researcher", FALLBACK_LLM)),
    )
    monkeypatch.setattr(
        json_crew.click,
        "prompt",
        lambda *_args, **_kwargs: pytest.fail("--skip-provider asks for no key"),
    )

    json_crew.create_json_crew("Skip Crew", skip_provider=True)

    output = capsys.readouterr().out
    assert "  Set OPENAI_API_KEY in .env before `crewai run`\n" in output
    assert output.index("cd skip_crew") < output.index("Set OPENAI_API_KEY")
    assert not (tmp_path / "skip_crew" / ".env").exists()


def test_json_setup_env_blank_key_prints_the_line_to_add_and_claims_no_save(
    tmp_path, monkeypatch, capsys, no_provider_keys, terminal_stdin
):
    calls = _record_key_prompts(monkeypatch, [""])

    json_crew._setup_env(tmp_path, [FALLBACK_LLM])

    output = capsys.readouterr().out
    assert (
        f"  API key not saved. Add this line to {tmp_path / '.env'} "
        "before `crewai run`:\n    OPENAI_API_KEY=<your key>\n"
    ) in output
    assert "saved to .env file" not in output
    env_file = (tmp_path / ".env").read_text()
    assert f"MODEL={FALLBACK_LLM}" in env_file
    assert "OPENAI_API_KEY" not in env_file
    assert [call["hide_input"] for call in calls] == [True]


def test_json_setup_env_saved_key_keeps_the_success_message(
    tmp_path, monkeypatch, capsys, terminal_stdin
):
    calls = _record_key_prompts(monkeypatch, ["sk-test"])

    json_crew._setup_env(tmp_path, [FALLBACK_LLM])

    output = capsys.readouterr().out
    assert "  API keys and model saved to .env file" in output
    assert "not saved" not in output
    assert "OPENAI_API_KEY=sk-test" in (tmp_path / ".env").read_text()
    assert [call["hide_input"] for call in calls] == [True]


def test_json_setup_env_lists_each_missing_key_of_a_multi_key_provider(
    tmp_path, monkeypatch, capsys, no_provider_keys
):
    _record_key_prompts(monkeypatch, ["https://us-south.ml.cloud.ibm.com", "", ""])

    json_crew._setup_env(tmp_path, ["watsonx/ibm/granite-4-h-small"])

    output = capsys.readouterr().out
    assert (
        f"  API keys not saved. Add these lines to {tmp_path / '.env'} "
        "before `crewai run`:\n"
        "    WATSONX_APIKEY=<your key>\n"
        "    WATSONX_PROJECT_ID=<your key>\n"
    ) in output
    assert "WATSONX_URL=<your key>" not in output
    assert "saved to .env file" not in output
    env_file = (tmp_path / ".env").read_text()
    assert "WATSONX_URL=https://us-south.ml.cloud.ibm.com" in env_file


def test_json_watsonx_models_use_the_prefix_crewai_can_run(
    monkeypatch, no_provider_keys
):
    """LiteLLM rejects `watson/`; it knows the provider only as `watsonx`."""
    watson = [key for key, _ in json_crew._PROVIDERS].index("watson")
    picks = iter([watson, 0])
    monkeypatch.setattr(json_crew, "pick_one", lambda *_a, **_k: next(picks))

    assert json_crew._select_model() == "watsonx/ibm/granite-4-h-small"
    for provider in ("watson", "watsonx"):
        assert json_crew._default_model_for_provider(provider) == (
            "watsonx/ibm/granite-4-h-small"
        )


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [("claude/claude-sonnet-5-5", "ANTHROPIC_API_KEY"), ("google/gemini-3.8-flash", "GEMINI_API_KEY")],
)
def test_json_dmn_next_steps_name_the_key_of_an_aliased_prefix(
    tmp_path, monkeypatch, capsys, no_provider_keys, provider, key_name
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREWAI_DMN", "True")

    json_crew.create_json_crew("DMN Crew", provider=provider, skip_provider=True)

    assert f"Set {key_name}" in capsys.readouterr().out


def test_json_dmn_next_steps_name_the_watsonx_keys(
    tmp_path, monkeypatch, capsys, no_provider_keys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREWAI_DMN", "True")

    json_crew.create_json_crew("DMN Crew", provider="watson", skip_provider=True)

    assert (
        "Set WATSONX_URL, WATSONX_APIKEY, WATSONX_PROJECT_ID in .env before "
        "`crewai run`"
    ) in capsys.readouterr().out


def test_json_setup_env_lists_neither_azure_deployment_name_nor_api_version(
    tmp_path, monkeypatch, capsys, no_provider_keys
):
    _record_key_prompts(monkeypatch, ["", "", "", ""])

    json_crew._setup_env(tmp_path, ["azure/my-deployment"])

    output = capsys.readouterr().out
    assert (
        "    AZURE_API_KEY=<your key>\n    AZURE_API_BASE=<your key>\n"
    ) in output
    assert "AZURE_API_VERSION" not in output
    assert "MODEL=<your key>" not in output
    assert "MODEL=azure/my-deployment" in (tmp_path / ".env").read_text()


def test_json_setup_env_blank_region_still_reports_the_keys_saved(
    tmp_path, monkeypatch, capsys, no_provider_keys
):
    _record_key_prompts(monkeypatch, ["AKIATEST", "aws-secret", ""])

    json_crew._setup_env(tmp_path, ["bedrock/anthropic.claude-sonnet-5"])

    output = capsys.readouterr().out
    assert "  API keys and model saved to .env file" in output
    assert "not saved" not in output
    assert "AWS_DEFAULT_REGION" not in output
    env_file = (tmp_path / ".env").read_text()
    assert "AWS_SECRET_ACCESS_KEY=aws-secret" in env_file
    assert "AWS_DEFAULT_REGION" not in env_file


def test_json_setup_env_blank_key_already_exported_prints_no_line_to_add(
    tmp_path, monkeypatch, capsys, no_provider_keys
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    _record_key_prompts(monkeypatch, [""])

    json_crew._setup_env(tmp_path, [FALLBACK_LLM])

    output = capsys.readouterr().out
    assert "<your key>" not in output
    assert "not saved" not in output
    assert "OPENAI_API_KEY" not in (tmp_path / ".env").read_text()


@pytest.mark.parametrize(
    ("model", "prompts"),
    [
        (
            "azure/my-deployment",
            [
                ("Enter your Azure deployment name (must start with 'azure/')", False),
                ("Enter your AZURE API key (press Enter to skip)", True),
                ("Enter your AZURE API base URL (press Enter to skip)", False),
                ("Enter your AZURE API version (press Enter to skip)", False),
            ],
        ),
        (
            "bedrock/anthropic.claude-sonnet-5",
            [
                ("Enter your AWS Access Key ID (press Enter to skip)", True),
                ("Enter your AWS Secret Access Key (press Enter to skip)", True),
                ("Enter your AWS Region Name (press Enter to skip)", False),
            ],
        ),
        (
            "watsonx/ibm/granite-4-h-small",
            [
                ("Enter your WATSONX URL (press Enter to skip)", False),
                ("Enter your WATSONX API Key (press Enter to skip)", True),
                ("Enter your WATSONX Project Id (press Enter to skip)", False),
            ],
        ),
        (
            "cerebras/llama3.1-8b",
            [
                ("Enter your Cerebras model name (must start with 'cerebras/')", False),
                ("Enter your Cerebras API version (press Enter to skip)", True),
            ],
        ),
    ],
)
def test_json_setup_env_hides_only_secrets_at_a_terminal(
    tmp_path, monkeypatch, terminal_stdin, model, prompts
):
    calls = _record_key_prompts(monkeypatch, [""] * len(prompts))

    json_crew._setup_env(tmp_path, [model])

    assert [(str(call["text"]).strip(), call["hide_input"]) for call in calls] == (
        prompts
    )


@pytest.mark.parametrize(
    ("model", "prompt", "key_name"),
    [
        (
            "claude/claude-sonnet-5",
            "Enter your ANTHROPIC API key (press Enter to skip)",
            "ANTHROPIC_API_KEY",
        ),
        (
            "google/gemini-3.8-flash",
            "Enter your GEMINI API key from https://ai.dev/apikey (press Enter to skip)",
            "GEMINI_API_KEY",
        ),
        (
            "claude-sonnet-5",
            "Enter your ANTHROPIC API key (press Enter to skip)",
            "ANTHROPIC_API_KEY",
        ),
    ],
    ids=["claude-prefix", "google-prefix", "bare-name"],
)
def test_json_setup_env_asks_for_the_key_of_the_provider_crewai_runs_the_model_on(
    tmp_path, monkeypatch, capsys, no_provider_keys, terminal_stdin, model, prompt, key_name
):
    calls = _record_key_prompts(monkeypatch, [""])

    json_crew._setup_env(tmp_path, [model])

    assert [str(call["text"]).strip() for call in calls] == [prompt]
    assert f"    {key_name}=<your key>\n" in capsys.readouterr().out
    assert (tmp_path / ".env").read_text() == f"MODEL={model}\n"


def test_json_setup_env_reads_a_piped_key_from_stdin(tmp_path, monkeypatch, capsys):
    _pipe_to_stdin(monkeypatch, "sk-piped\n")

    json_crew._setup_env(tmp_path, [FALLBACK_LLM])

    assert "OPENAI_API_KEY=sk-piped" in (tmp_path / ".env").read_text()
    assert "  API keys and model saved to .env file" in capsys.readouterr().out


def test_json_setup_env_ollama_needs_no_key_and_prints_nothing_extra(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        json_crew.click,
        "prompt",
        lambda *_args, **_kwargs: pytest.fail("ollama has no API key to prompt for"),
    )

    json_crew._setup_env(tmp_path, ["ollama/llama3.3"])

    output = capsys.readouterr().out
    assert "  API keys and model saved to .env file" in output
    assert "not saved" not in output
    assert "<your key>" not in output
    assert "API_BASE=http://localhost:11434" in (tmp_path / ".env").read_text()


def test_json_create_prompts_once_per_provider_and_first_agent_sets_model(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    monkeypatch.setattr(
        json_crew,
        "_wizard_agents_and_tasks",
        lambda **_: _wizard_result(
            _agent("researcher", FALLBACK_LLM),
            _agent("writer", "anthropic/claude-sonnet-5"),
            _agent("editor", "openai/gpt-5.5"),
        ),
    )
    calls = _record_key_prompts(monkeypatch, ["sk-openai", "sk-anthropic"])

    json_crew.create_json_crew("Mixed Crew", skip_provider=False)

    assert [call["text"] for call in calls] == [
        "  Enter your OPENAI API key (press Enter to skip)",
        "  Enter your ANTHROPIC API key (press Enter to skip)",
    ]
    env_lines = (tmp_path / "mixed_crew" / ".env").read_text().splitlines()
    assert env_lines == [
        f"MODEL={FALLBACK_LLM}",
        "OPENAI_API_KEY=sk-openai",
        "ANTHROPIC_API_KEY=sk-anthropic",
    ]
    output = capsys.readouterr().out
    assert output.count("API keys and model saved to .env file") == 1
    assert "in .env before `crewai run`" not in output


@mock.patch("crewai_cli.create_crew.create_folder_structure")
@mock.patch("crewai_cli.create_crew.copy_template")
@mock.patch("crewai_cli.create_crew.get_provider_data")
@mock.patch("crewai_cli.create_crew.select_provider")
@mock.patch("crewai_cli.create_crew.select_model")
@mock.patch("click.prompt")
def test_classic_create_blank_key_prints_the_line_to_add_and_claims_no_save(
    mock_prompt,
    mock_select_model,
    mock_select_provider,
    mock_get_provider_data,
    mock_copy_template,
    mock_create_folder_structure,
    tmp_path,
    capsys,
    no_provider_keys,
    terminal_stdin,
):
    crew_path = tmp_path / "test_crew"
    crew_path.mkdir()
    mock_create_folder_structure.return_value = (crew_path, "test_crew", "TestCrew")
    mock_get_provider_data.return_value = {"openai": ["gpt-5.6-luna"]}
    mock_select_provider.return_value = "openai"
    mock_select_model.return_value = "gpt-5.6-luna"
    mock_prompt.return_value = ""

    create_crew("Test Crew")

    output = capsys.readouterr().out
    assert (
        f"  API key not saved. Add this line to {crew_path / '.env'} "
        "before `crewai run`:\n    OPENAI_API_KEY=<your key>\n"
    ) in output
    assert "saved to .env file" not in output
    env_file = (crew_path / ".env").read_text()
    assert "MODEL=gpt-5.6-luna" in env_file
    assert "OPENAI_API_KEY" not in env_file
    assert mock_prompt.call_args.kwargs["hide_input"] is True


@mock.patch("crewai_cli.create_crew.create_folder_structure")
@mock.patch("crewai_cli.create_crew.copy_template")
@mock.patch("crewai_cli.create_crew.get_provider_data")
@mock.patch("crewai_cli.create_crew.select_provider")
@mock.patch("crewai_cli.create_crew.select_model")
@mock.patch("click.prompt")
def test_classic_create_saved_key_keeps_the_success_message(
    mock_prompt,
    mock_select_model,
    mock_select_provider,
    mock_get_provider_data,
    mock_copy_template,
    mock_create_folder_structure,
    tmp_path,
    capsys,
    terminal_stdin,
):
    crew_path = tmp_path / "test_crew"
    crew_path.mkdir()
    mock_create_folder_structure.return_value = (crew_path, "test_crew", "TestCrew")
    mock_get_provider_data.return_value = {"anthropic": ["claude-sonnet-5"]}
    mock_select_provider.return_value = "anthropic"
    mock_select_model.return_value = "claude-sonnet-5"
    mock_prompt.return_value = "sk-ant-test"

    create_crew("Test Crew")

    output = capsys.readouterr().out
    assert "API keys and model saved to .env file" in output
    assert "not saved" not in output
    assert "ANTHROPIC_API_KEY=sk-ant-test" in (crew_path / ".env").read_text()
    assert mock_prompt.call_args.kwargs["hide_input"] is True


@mock.patch("crewai_cli.create_crew.create_folder_structure")
@mock.patch("crewai_cli.create_crew.copy_template")
@mock.patch("crewai_cli.create_crew.get_provider_data")
@mock.patch("crewai_cli.create_crew.select_provider")
@mock.patch("crewai_cli.create_crew.select_model")
@mock.patch("click.prompt")
def test_classic_create_blank_region_still_reports_the_keys_saved(
    mock_prompt,
    mock_select_model,
    mock_select_provider,
    mock_get_provider_data,
    mock_copy_template,
    mock_create_folder_structure,
    tmp_path,
    capsys,
    no_provider_keys,
    terminal_stdin,
):
    crew_path = tmp_path / "test_crew"
    crew_path.mkdir()
    mock_create_folder_structure.return_value = (crew_path, "test_crew", "TestCrew")
    mock_get_provider_data.return_value = {"bedrock": ["bedrock/amazon.nova-pro"]}
    mock_select_provider.return_value = "bedrock"
    mock_select_model.return_value = "bedrock/amazon.nova-pro"
    mock_prompt.side_effect = ["AKIATEST", "aws-secret", ""]

    create_crew("Test Crew")

    output = capsys.readouterr().out
    assert "API keys and model saved to .env file" in output
    assert "AWS_DEFAULT_REGION" not in output
    assert [call.kwargs["hide_input"] for call in mock_prompt.call_args_list] == [
        True,
        True,
        False,
    ]


@mock.patch("crewai_cli.create_crew.create_folder_structure")
@mock.patch("crewai_cli.create_crew.copy_template")
@mock.patch("crewai_cli.create_crew.get_provider_data")
@mock.patch("crewai_cli.create_crew.select_provider")
@mock.patch("crewai_cli.create_crew.select_model")
@mock.patch("click.prompt")
def test_classic_create_blank_key_already_exported_prints_no_line_to_add(
    mock_prompt,
    mock_select_model,
    mock_select_provider,
    mock_get_provider_data,
    mock_copy_template,
    mock_create_folder_structure,
    tmp_path,
    monkeypatch,
    capsys,
    no_provider_keys,
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    crew_path = tmp_path / "test_crew"
    crew_path.mkdir()
    mock_create_folder_structure.return_value = (crew_path, "test_crew", "TestCrew")
    mock_get_provider_data.return_value = {"openai": ["gpt-5.6-luna"]}
    mock_select_provider.return_value = "openai"
    mock_select_model.return_value = "gpt-5.6-luna"
    mock_prompt.return_value = ""

    create_crew("Test Crew")

    output = capsys.readouterr().out
    assert "<your key>" not in output
    assert "not saved" not in output
    assert "OPENAI_API_KEY" not in (crew_path / ".env").read_text()


@mock.patch("crewai_cli.create_crew.create_folder_structure")
@mock.patch("crewai_cli.create_crew.copy_template")
@mock.patch("crewai_cli.create_crew.get_provider_data")
@mock.patch("crewai_cli.create_crew.select_provider")
@mock.patch("crewai_cli.create_crew.select_model")
def test_classic_create_reads_a_piped_key_from_stdin(
    mock_select_model,
    mock_select_provider,
    mock_get_provider_data,
    mock_copy_template,
    mock_create_folder_structure,
    tmp_path,
    monkeypatch,
    capsys,
):
    _pipe_to_stdin(monkeypatch, "sk-piped\n")
    crew_path = tmp_path / "test_crew"
    crew_path.mkdir()
    mock_create_folder_structure.return_value = (crew_path, "test_crew", "TestCrew")
    mock_get_provider_data.return_value = {"openai": ["gpt-5.6-luna"]}
    mock_select_provider.return_value = "openai"
    mock_select_model.return_value = "gpt-5.6-luna"

    create_crew("Test Crew")

    assert "OPENAI_API_KEY=sk-piped" in (crew_path / ".env").read_text()
    assert "API keys and model saved to .env file" in capsys.readouterr().out


def _crewai_dependency(project_root: Path) -> str:
    pyproject = tomli.loads((project_root / "pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]
    assert len(dependencies) == 1
    return dependencies[0]


@pytest.mark.parametrize(
    ("llm", "crewai_with_extras"),
    [
        ("anthropic/claude-sonnet-5", "crewai[tools,anthropic]"),
        ("claude/claude-sonnet-5", "crewai[tools,anthropic]"),
        ("gemini/gemini-3.8-flash", "crewai[tools,google-genai]"),
        ("bedrock/us.amazon.nova-pro-v1:0", "crewai[tools,bedrock]"),
        ("azure/my-deployment", "crewai[tools,azure-ai-inference]"),
        ("groq/llama-3.3-70b-versatile", "crewai[tools,litellm]"),
        ("anthropic/my-finetune", "crewai[tools,anthropic,litellm]"),
        ("openai/chatgpt-4o-latest", "crewai[tools,litellm]"),
        ("mistral/mistral-large", "crewai[tools,litellm]"),
        ("custom/my-model", "crewai[tools,litellm]"),
    ],
)
def test_json_create_adds_the_provider_extra_to_the_crewai_dependency(
    tmp_path, monkeypatch, llm, crewai_with_extras
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREWAI_DMN", "True")

    json_crew.create_json_crew("Provider Crew", provider=llm, skip_provider=True)

    project_root = tmp_path / "provider_crew"
    agent_file = project_root / "agents" / "researcher.jsonc"
    assert f'"llm": "{llm}"' in agent_file.read_text()
    assert _crewai_dependency(project_root) == (
        f"{crewai_with_extras}{get_crewai_dependency_range()}"
    )


@pytest.mark.parametrize("provider", [None, "openai", "ollama", "cerebras/llama3.1-8b"])
def test_json_create_keeps_the_tools_only_dependency_when_no_extra_is_needed(
    tmp_path, monkeypatch, provider
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREWAI_DMN", "True")

    json_crew.create_json_crew("Plain Crew", provider=provider, skip_provider=True)

    assert _crewai_dependency(tmp_path / "plain_crew") == (
        f"crewai[tools]{get_crewai_dependency_range()}"
    )


def test_json_create_with_a_bare_unknown_model_adds_no_extra(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    monkeypatch.setattr(
        json_crew,
        "_wizard_agents_and_tasks",
        lambda **_: _wizard_result(_agent("researcher", "my-model")),
    )

    json_crew.create_json_crew("Bare Crew", skip_provider=True)

    assert _crewai_dependency(tmp_path / "bare_crew") == get_crewai_tools_dependency()


def test_json_create_lists_each_agent_provider_extra_once_and_sorted(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    monkeypatch.setattr(
        json_crew,
        "_wizard_agents_and_tasks",
        lambda **_: _wizard_result(
            _agent("researcher", "openai/gpt-5.6-luna"),
            _agent("writer", "gemini/gemini-3.8-flash"),
            _agent("editor", "anthropic/claude-sonnet-5"),
            _agent("reviewer", "claude-haiku-4-5"),
            _agent("checker", "my-model"),
        ),
    )

    json_crew.create_json_crew("Mixed Crew", skip_provider=True)

    assert _crewai_dependency(tmp_path / "mixed_crew") == (
        f"crewai[tools,anthropic,google-genai]{get_crewai_dependency_range()}"
    )


def test_json_create_adds_litellm_for_a_listed_bedrock_model_crewai_runs_through_it(
    tmp_path, monkeypatch
):
    """Bedrock's picker lists LiteLLM feed ids; crewAI runs one with no "." in
    its name through LiteLLM, so the project needs that extra as well."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    monkeypatch.setattr(
        json_crew,
        "get_provider_models",
        lambda _provider, _fallback: [
            ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "Claude Haiku 4.5"),
            ("claude-sonnet-4-5-20250929-v1:0", "Claude Sonnet 4.5"),
        ],
    )
    bedrock = [key for key, _ in json_crew._PROVIDERS].index("bedrock")
    picks = iter([bedrock, 1])
    monkeypatch.setattr(json_crew, "pick_one", lambda *_args, **_kwargs: next(picks))
    monkeypatch.setattr(
        json_crew,
        "_wizard_agents_and_tasks",
        lambda **_: _wizard_result(_agent("researcher", json_crew._select_model())),
    )

    json_crew.create_json_crew("Bedrock Crew", skip_provider=True)

    project_root = tmp_path / "bedrock_crew"
    agent_file = project_root / "agents" / "researcher.jsonc"
    assert '"llm": "bedrock/claude-sonnet-4-5-20250929-v1:0"' in agent_file.read_text()
    assert _crewai_dependency(project_root) == (
        f"crewai[tools,bedrock,litellm]{get_crewai_dependency_range()}"
    )


def test_json_create_adds_the_extra_when_the_key_is_entered_at_the_prompt(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    monkeypatch.setattr(
        json_crew,
        "_wizard_agents_and_tasks",
        lambda **_: _wizard_result(_agent("researcher", "anthropic/claude-sonnet-5")),
    )
    calls = _record_key_prompts(monkeypatch, ["sk-ant-test"])

    json_crew.create_json_crew("Claude Crew", skip_provider=False)

    project_root = tmp_path / "claude_crew"
    assert [call["text"] for call in calls] == [
        "  Enter your ANTHROPIC API key (press Enter to skip)"
    ]
    assert "ANTHROPIC_API_KEY=sk-ant-test" in (project_root / ".env").read_text()
    assert _crewai_dependency(project_root) == (
        f"crewai[tools,anthropic]{get_crewai_dependency_range()}"
    )
    assert "API keys and model saved to .env file" in capsys.readouterr().out


def _pick_classic_provider(
    monkeypatch, provider: str, model: str | None, answers: list[str]
) -> None:
    """Choose ``provider`` and ``model`` in the classic pickers and answer its
    key prompts in order."""
    monkeypatch.setattr(
        "crewai_cli.create_crew.get_provider_data",
        lambda: {provider: [model] if model else []},
    )
    monkeypatch.setattr(
        "crewai_cli.create_crew.select_provider", lambda _provider_models: provider
    )
    monkeypatch.setattr(
        "crewai_cli.create_crew.select_model",
        lambda _provider, _provider_models: model,
    )
    replies = iter(answers)
    monkeypatch.setattr(click, "prompt", lambda *_args, **_kwargs: next(replies))


@pytest.mark.parametrize(
    ("provider", "model", "answers", "crewai_with_extras"),
    [
        ("anthropic", "claude-sonnet-5", ["sk-ant"], "crewai[tools,anthropic]"),
        (
            "gemini",
            "gemini/gemini-3.8-flash",
            ["gemini-key"],
            "crewai[tools,google-genai]",
        ),
        (
            "bedrock",
            "bedrock/us.amazon.nova-pro-v1:0",
            ["AKIATEST", "aws-secret", ""],
            "crewai[tools,bedrock]",
        ),
        (
            "azure",
            None,
            ["azure/my-deployment", "azure-key", "https://example.azure.com", ""],
            "crewai[tools,azure-ai-inference]",
        ),
        ("groq", "groq/llama-3.1-8b-instant", ["gsk"], "crewai[tools,litellm]"),
        ("openai", "gpt-5.6-luna", ["sk-openai"], "crewai[tools]"),
        ("ollama", "ollama/llama3.1", [], "crewai[tools]"),
        ("cerebras", None, ["cerebras/llama3.1-8b", "csk"], "crewai[tools]"),
    ],
)
def test_classic_create_adds_the_selected_provider_extra_to_the_crewai_dependency(
    tmp_path, monkeypatch, provider, model, answers, crewai_with_extras
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    _pick_classic_provider(monkeypatch, provider, model, answers)

    create_crew("Provider Crew")

    project_root = tmp_path / "provider_crew"
    assert _crewai_dependency(project_root) == (
        f"{crewai_with_extras}{get_crewai_dependency_range()}"
    )
    assert (project_root / ".env").exists()


def test_classic_create_with_skip_provider_keeps_the_tools_only_dependency(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    monkeypatch.setattr(
        "crewai_cli.create_crew.select_provider",
        lambda _provider_models: pytest.fail("--skip-provider picks no provider"),
    )

    create_crew("Plain Crew", skip_provider=True)

    assert _crewai_dependency(tmp_path / "plain_crew") == get_crewai_tools_dependency()


def test_json_create_returns_the_folder_it_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREWAI_DMN", "True")

    project = json_crew.create_json_crew("News v2.0", skip_provider=True)

    # A crew folder drops the ".", which a flow folder would keep.
    assert project == Path("news_v20")
    assert (tmp_path / "news_v20" / "crew.jsonc").is_file()


def test_classic_create_returns_the_folder_it_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CREWAI_DMN", raising=False)

    project = create_crew("News v2.0", skip_provider=True)

    assert project == Path("news_v20")
    assert (tmp_path / "news_v20" / "pyproject.toml").is_file()


def test_classic_create_returns_none_when_it_adds_a_crew_to_a_project(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    project = create_crew("Writer", skip_provider=True, parent_folder="crews")

    assert project is None
    assert (tmp_path / "crews" / "writer" / "crew.py").is_file()


def test_classic_create_returns_the_folder_when_the_provider_list_is_unavailable(
    tmp_path, monkeypatch
):
    """The curated lists stand in for the catalog, so the crew is still created."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    monkeypatch.setattr("crewai_cli.create_crew.get_provider_data", lambda: None)
    monkeypatch.setattr("crewai_cli.create_crew.select_provider", lambda *_: "openai")
    monkeypatch.setattr(
        "crewai_cli.create_crew.select_model", lambda *_: "gpt-6-luna"
    )
    monkeypatch.setattr("crewai_cli.create_crew.click.prompt", lambda *_a, **_k: "")

    project = create_crew("Offline Crew")

    assert project == Path("offline_crew")
    assert (tmp_path / "offline_crew" / "pyproject.toml").is_file()


@pytest.fixture
def classic_create(tmp_path, monkeypatch, runner):
    """Run `crewai create crew my_crew --classic` in ``tmp_path`` with
    ``tmp_path`` as home, so the LiteLLM cache starts cold."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    def run(answers: list[str]):
        return runner.invoke(
            crewai,
            ["create", "crew", "my_crew", "--classic"],
            input="".join(f"{answer}\n" for answer in answers),
        )

    return run


def _catalog_unreachable(*_args, **_kwargs):
    raise httpx.ConnectError("[Errno 8] nodename nor servname provided")


@pytest.mark.parametrize("provider", list(MODELS))
def test_classic_create_offers_curated_models_when_the_catalog_is_unreachable(
    provider, classic_create, tmp_path, monkeypatch
):
    monkeypatch.setattr("crewai_cli.provider.httpx.stream", _catalog_unreachable)

    result = classic_create([str(PROVIDERS.index(provider) + 1), "1", "", "", ""])

    assert result.exit_code == 0, result.output
    assert "Error fetching provider data: [Errno 8]" in result.output
    for number, model in enumerate(MODELS[provider], start=1):
        assert f"{number}. {model}\n" in result.output
    assert "Crew my_crew created successfully!" in result.output
    project = tmp_path / "my_crew"
    assert f"MODEL={MODELS[provider][0]}" in (project / ".env").read_text()
    assert (project / "src" / "my_crew" / "crew.py").is_file()


def test_classic_create_other_lists_catalog_providers_when_the_catalog_loads(
    classic_create, tmp_path, monkeypatch
):
    cache = tmp_path / ".crewai" / "provider_cache.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps({"xai/grok-5": {"litellm_provider": "xai"}}))
    monkeypatch.setattr(
        "crewai_cli.provider.httpx.stream",
        lambda *_a, **_k: pytest.fail("a fresh cache must not be refetched"),
    )
    xai_number = sorted({*PROVIDERS, "xai"}).index("xai") + 1

    result = classic_create([str(len(PROVIDERS) + 1), str(xai_number)])

    assert result.exit_code == 0, result.output
    _, full_list = result.output.split("Select a provider from the full list:\n")
    assert f"{xai_number}. xai\n" in full_list
    assert "Crew my_crew created successfully!" in result.output


def test_create_crew_scaffolds_assistant_instructions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_crew("my-crew", skip_provider=True)

    project_root = tmp_path / "my_crew"
    agents_md = (project_root / "AGENTS.md").read_text(encoding="utf-8")
    assert agents_md.index("## Working in this project") < agents_md.index("## Reference")
    claude_md = (project_root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in claude_md.splitlines()
    cursor_md = (project_root / "CURSOR.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in cursor_md.splitlines()
    gemini_md = (project_root / "GEMINI.md").read_text(encoding="utf-8")
    assert "@./AGENTS.md" in gemini_md.splitlines()


def test_scaffolded_agents_md_tells_assistants_to_keep_observability_on(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_crew("my-crew", skip_provider=True)

    agents_md = (tmp_path / "my_crew" / "AGENTS.md").read_text(encoding="utf-8")
    [keep_on] = [
        line
        for line in agents_md.splitlines()
        if line.startswith("Never turn off tracing or telemetry")
    ]
    assert "to silence errors, speed things up or tidy the setup" in keep_on
    assert "check DNS and proxy settings" in keep_on
    assert "it's their decision" in keep_on
    assert "free" not in agents_md.lower()
    assert "pricing" not in agents_md.lower()


def test_json_create_scaffolds_assistant_instructions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with mock.patch(
        "crewai_cli.create_json_crew._wizard_agents_and_tasks",
        return_value=(
            [
                {
                    "name": "researcher",
                    "role": "Researcher",
                    "goal": "Research",
                    "backstory": "Researcher",
                    "llm": "openai/gpt-4o",
                    "tools": [],
                    "planning": False,
                    "allow_delegation": False,
                }
            ],
            [
                {
                    "name": "research_task",
                    "description": "Research",
                    "expected_output": "Findings",
                    "agent": "researcher",
                    "context": [],
                }
            ],
            {"process": "sequential", "memory": False, "inputs": {}},
        ),
    ):
        json_crew.create_json_crew("JSON Crew", provider="openai", skip_provider=True)

    project_root = tmp_path / "json_crew"
    agents_md = (project_root / "AGENTS.md").read_text(encoding="utf-8")
    assert "## Working in this project" in agents_md
    assert "### Check every change: evaluate" in agents_md
    assert "crewai-eval:" not in agents_md
    assert "crew.jsonc" in agents_md
    claude_md = (project_root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in claude_md.splitlines()
    cursor_md = (project_root / "CURSOR.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in cursor_md.splitlines()
    gemini_md = (project_root / "GEMINI.md").read_text(encoding="utf-8")
    assert "@./AGENTS.md" in gemini_md.splitlines()


def test_scaffolded_agents_md_tells_assistants_to_evaluate_with_crewai_eval(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_crew("my-crew", skip_provider=True)

    agents_md = (tmp_path / "my_crew" / "AGENTS.md").read_text(encoding="utf-8")
    assert "is done when a traced run passes `crewai eval`, not when the code runs" in agents_md
    assert "prints a markdown brief of what failed and the change to make" in agents_md
    assert "It exits 0 only when the goal gate passed" in agents_md
    assert "`crewai eval --models " in agents_md
    assert "crewai-eval:" not in agents_md  # the markers never reach a project


def test_scaffolded_agents_md_evaluates_logged_out_without_skipping_the_login(
    tmp_path, monkeypatch
):
    """Logged out, agents evaluate right away and still start the login; measured 10-19,
    "don't wait on a login" alone made agents stop offering to log in."""
    monkeypatch.chdir(tmp_path)
    create_crew("my-crew", skip_provider=True)

    agents_md = (tmp_path / "my_crew" / "AGENTS.md").read_text(encoding="utf-8")
    assert "`crewai eval`, logged in or out, with or without a terminal" in agents_md
    assert "start the login hand-off" in agents_md
    assert "evaluate without waiting for it" in agents_md
    assert "Stop when the goal gate passes or after three rounds" in agents_md
    assert "report the first and latest grades with both report links" in agents_md


def _cli_args(line: str) -> list[str]:
    """Arguments of a documented `crewai …` line, with `<placeholder>` values filled."""
    words = shlex.split(line.split("#", 1)[0])[1:]
    return ["value" if "<" in word else word for word in words]


def _parse_cli_args(args: list[str]) -> None:
    """Parse ARGS against the real CLI without running the command."""
    command: click.Command = crewai
    name = "crewai"
    # A group can run without a subcommand, options and all (`crewai eval --models …`,
    # beside `crewai eval export ID`).
    while isinstance(command, click.Group) and args and not args[0].startswith("-"):
        name, *args = args
        subcommand = command.get_command(click.Context(command), name)
        assert subcommand is not None, f"unknown command {name!r}"
        command = subcommand
    command.make_context(name, args)


def test_agents_md_command_parser_accepts_a_group_run_without_a_subcommand():
    _parse_cli_args(_cli_args("crewai eval"))
    _parse_cli_args(_cli_args('crewai eval --models "openai/gpt-4o-mini,anthropic/claude-haiku-4-5"'))


def test_agents_md_command_parser_rejects_a_group_that_needs_a_subcommand():
    with pytest.raises(click.UsageError):
        _parse_cli_args(_cli_args("crewai deploy"))


def test_parse_cli_args_rejects_a_positional_deployment_id():
    with pytest.raises(click.UsageError, match="unexpected extra argument"):
        _parse_cli_args(_cli_args("crewai deploy remove <id>"))


def _documented_commands(agents_md: str) -> list[str]:
    """Every inline `crewai …` command in AGENTS.md, one per `a|b` alternative."""
    commands = []
    for span in re.findall(r"`(crewai [^`]+)`", agents_md):
        words = shlex.split(span)
        alternatives = [word.split("|") if "|" in word and "<" not in word else [word] for word in words]
        for choice in itertools.product(*alternatives):
            commands.append(" ".join(choice))
    return commands


def test_scaffolded_agents_md_commands_parse_against_the_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_crew("my-crew", skip_provider=True)

    agents_md = (tmp_path / "my_crew" / "AGENTS.md").read_text(encoding="utf-8")
    commands = _documented_commands(agents_md)

    for documented in (
        "crewai deploy remove -u <uuid>",
        "crewai deploy create -y",
        "crewai org list",
        "crewai org switch <uuid>",
        "crewai eval",
    ):
        assert documented in commands, documented
    for command in commands:
        try:
            _parse_cli_args(_cli_args(command))
        except click.MissingParameter:
            pass  # named without its arguments in prose, e.g. "the old `crewai tool create` style"


def test_scaffolded_agents_md_deploy_prerequisites_allow_zip_deploys(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_crew("my-crew", skip_provider=True)

    agents_md = (tmp_path / "my_crew" / "AGENTS.md").read_text(encoding="utf-8")
    assert "Code is in a GitHub repository" not in agents_md
    # `crewai deploy create` sends any `origin` URL; AMP builds from GitHub or another git host.
    assert "AMP builds from GitHub" not in agents_md
    assert "AMP builds from that remote, on GitHub or another git host" in agents_md
    assert "With no git `origin` it uploads a ZIP" in agents_md
    assert "There's no flag to upload a ZIP" not in agents_md
    assert "`crewai deploy create -y --zip`" in agents_md
    assert "uploads `.env` as its environment variables" in agents_md


def test_scaffolded_agents_md_platform_tools_guidance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_crew("my-crew", skip_provider=True)

    agents_md = (tmp_path / "my_crew" / "AGENTS.md").read_text(encoding="utf-8")
    # PLATFORM_APPS lists the apps, but some of its action names resolve to no tool.
    assert "source of truth" not in agents_md
    assert "copy selectors from those; a selector must load at least one tool" in agents_md
    assert "`CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN` in `.env`" in agents_md
    assert "CREWAI_PLATFORM_INTEGRATION_TOKEN" not in agents_md
    assert "`app[/action][@connection]`" in agents_md
    assert "`apps: [slack/send_message]` in YAML" in agents_md


@pytest.mark.parametrize("evaluation", [True, False])
def test_agents_md_numbered_lists_have_no_gaps(tmp_path, evaluation):
    # The evaluation blocks are cut from tool projects; what remains must still read in order.
    copy_assistant_instructions(tmp_path, evaluation=evaluation)

    agents_md = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    numbers: list[int] = []
    for line in [*agents_md.splitlines(), ""]:
        item = re.match(r"(\d+)\. ", line)
        if item:
            numbers.append(int(item.group(1)))
        elif not line.startswith((" ", "\t")):  # an indented line continues the item
            assert numbers == list(range(1, len(numbers) + 1)), numbers
            numbers = []
    assert not re.search(r"^### \d", agents_md, re.M)
    assert ("crewai eval" in agents_md) is evaluation


# Guards for the AGENTS.md every crew and flow project receives: references nobody
# meant to delete, a size budget, and code and links that work. Whether an edit makes
# coding agents better or worse is measured separately, with models in the loop.

# Every capability the file must at least name. Removing one is a deliberate edit
# here, in the same diff, so a reviewer sees it.
AGENTS_MD_REFERENCES = {
    # Core API: models outside the Claude family don't know these without the file.
    "classic crew class": r"@CrewBase",
    "task context": r"context=",
    "crew output": r"CrewOutput",
    "custom tools": r"BaseTool",
    "flow decorators": r"@start\(\)",
    "flow state": r"Flow\[",
    "flow router": r"@router",
    # Facts V6 lost to today's file until they were put back.
    "context window": r"respect_context_window",
    "current date": r"inject_date",
    "human review of a task": r"human_input",
    "LiteLLM for other providers": r"crewai\[litellm\]",
    "test targets before writes": r"test (channel|target)",
    # The features telemetry counts most, named the way telemetry counts them.
    "interception hooks": r"InterceptionPoint",
    "agent planning": r"PlanningConfig",
    "OpenAI Responses API": r'api="responses"',
    "answering human feedback through AMP": r"human_feedback_requests",
    "workflow: trace and evaluate every change": r"traced run passes `crewai eval`",
    "workflow: ask before deploying": r"Ask before the first `crewai deploy create`",
    "flow memory": r"self\.remember",
    "AMP integrations": r"CrewaiPlatformTools",
    "AMP skills": r"skills:",
    "tool failure policy": r"tool_failure_policy",
    "guardrails": r"guardrail",
    "structured output": r"output_pydantic|response_format",
    "knowledge sources": r"knowledge_sources",
    "hierarchical process": r"Process\.hierarchical|manager_llm",
    "delegation": r"allow_delegation",
    "planning": r"planning_config",
    "async and batch kickoff": r"kickoff_async|akickoff|kickoff_for_each",
    "streaming": r"stream=True",
    "event listeners": r"BaseEventListener",
    "lifecycle hooks": r"@before_kickoff|@after_kickoff",
    "MCP": r"mcps=",
    "human feedback in flows": r"@human_feedback",
    "conversational flows": r"conversational",
    "declarative flows": r"--declarative|declarative flow",
    "checkpointing": r"checkpoint",
    "flow conditions": r"or_\(|and_\(",
    "crewai_tools catalog": r"crewai_tools|crewai-tools",
    "A2A": r"A2AClientConfig",
    "file inputs": r"input_files",
}

# Always-loaded tokens (characters / 4). Raise it on purpose, with the eval's results.
AGENTS_MD_TOKEN_BUDGET = 5_000

DOCS = Path(__file__).resolve().parents[3] / "docs"

# Patterns for text the file once carried that is wrong on main. Naming a
# deprecated option to say what replaces it is allowed.
AGENTS_MD_NEVER = (
    r"CREWAI_PLATFORM_INTEGRATION_TOKEN",  # renamed to CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN
    r"from crewai\.flow\.flow import persist",  # ImportError on crewai<=1.15.30; use crewai.flow
    r"(?<!replaces `)reasoning=True",  # deprecated for planning_config
    r"multimodal=True",  # deprecated: pass files natively
    r"allow_code_execution=True",  # deprecated: CodeInterpreterTool is gone
)


@pytest.fixture
def shipped_agents_md(tmp_path) -> str:
    copy_assistant_instructions(tmp_path, evaluation=True)
    return (tmp_path / "AGENTS.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("feature", sorted(AGENTS_MD_REFERENCES))
def test_agents_md_references_the_feature(shipped_agents_md, feature):
    assert re.search(AGENTS_MD_REFERENCES[feature], shipped_agents_md), (
        f"AGENTS.md no longer references {feature}; if that is intended, "
        "remove it from AGENTS_MD_REFERENCES in the same change"
    )


def test_agents_md_stays_within_its_token_budget(shipped_agents_md):
    tokens = len(shipped_agents_md) // 4
    assert tokens <= AGENTS_MD_TOKEN_BUDGET, (
        f"AGENTS.md is ~{tokens} tokens, over the {AGENTS_MD_TOKEN_BUDGET} budget"
    )


@pytest.mark.parametrize("text", AGENTS_MD_NEVER)
def test_agents_md_never_teaches_what_main_removed(shipped_agents_md, text):
    match = re.search(text, shipped_agents_md)
    assert match is None, f"AGENTS.md teaches {match.group(0)!r}, which is wrong on main"


def test_agents_md_never_list_still_catches_advice():
    assert re.search(AGENTS_MD_NEVER[2], "Agent(role=..., reasoning=True)")
    assert not re.search(AGENTS_MD_NEVER[2], "it replaces `reasoning=True`")


def test_agents_md_imports_resolve(shipped_agents_md):
    imports = re.findall(r"`((?:from|import) crewai[\w.]*(?: import [\w, ]+)?)`", shipped_agents_md)
    imports += [
        line
        for block in re.findall(r"```python\n(.*?)```", shipped_agents_md, re.S)
        for line in block.splitlines()
        if re.match(r"(from|import) crewai", line)
    ]
    assert imports, "AGENTS.md shows no crewai import to check"
    for statement in imports:
        exec(statement, {})  # noqa: S102 - our own template's import lines


def test_agents_md_python_examples_compile(shipped_agents_md):
    for block in re.findall(r"```python\n(.*?)```", shipped_agents_md, re.S):
        ast.parse(block)


def test_agents_md_docs_paths_exist(shipped_agents_md):
    # Index lines end with a docs path relative to https://docs.crewai.com/en/.
    links = re.findall(
        r"`([a-z]+(?:/[a-z0-9-]+)+)(?:#([a-z0-9-]+))?`\s*$", shipped_agents_md, re.M
    )
    assert links, "AGENTS.md links no docs pages to check"
    nav = (DOCS / "docs.json").read_text(encoding="utf-8")
    for page, anchor in links:
        assert f'"edge/en/{page}"' in nav, f"{page} is not in the docs nav"
        if anchor:
            mdx = (DOCS / "edge/en" / f"{page}.mdx").read_text(encoding="utf-8")
            headings = re.findall(r"^#+ (.+)$", mdx, re.M)
            slugs = {re.sub(r"[^a-z0-9]+", "-", h.lower()).strip("-") for h in headings}
            assert anchor in slugs, f"{page} has no #{anchor} heading"
