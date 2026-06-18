import keyword
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import tomli
from click.testing import CliRunner
from packaging.requirements import Requirement
from packaging.version import Version
import crewai_cli.create_json_crew as json_crew
import crewai_cli.tui_picker as tui_picker
from crewai_cli.create_crew import create_crew, create_folder_structure


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
            mock_create_folder.return_value = (
                mock_folder_path,
                "normal_project",
                "NormalProject",
            )

            create_crew("normal-project", skip_provider=True)

            mock_create_folder.assert_called_once_with("normal-project", None)


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

    assert tools == ["SerperDevTool", "DirectoryReadTool"]
    assert len(picker_calls) == 1
    labels = picker_calls[0][1]
    assert 0 in picker_calls[0][2]["separator_indices"]
    assert labels[0] == "── Common tools ──"
    assert labels[1].strip().endswith("SerperDevTool")
    assert labels[2].strip().endswith("ScrapeWebsiteTool")
    assert labels[3].strip().endswith("DirectoryReadTool")
    assert labels[4].strip().endswith("FileReadTool")
    assert labels[5].strip().endswith("FileWriterTool")
    assert labels[1].index("Google search") < labels[1].index("SerperDevTool")
    assert "More tools" not in labels


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

    def pick_many(title: str, labels: list[str], **kwargs):
        picker_calls.append((title, labels, kwargs))
        expanded_labels.extend(labels)
        action_indices = sorted(kwargs["action_indices"])
        call_num = len(picker_calls)
        if call_num <= len(action_indices):
            # Expand the n-th category (indices shift between renders, so
            # recompute from this render's action rows)
            return [], action_indices[call_num - 1]
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
        default_llm="openai/gpt-5.5",
    )
    assert (tmp_path / "json_crew" / "crew.jsonc").exists()
    assert not (tmp_path / "json_crew" / "tests").exists()
    assert not (tmp_path / "json_crew" / "config.jsonc").exists()

    pyproject = tomli.loads((tmp_path / "json_crew" / "pyproject.toml").read_text())
    dependency = pyproject["project"]["dependencies"][0]
    assert dependency == "crewai[tools]==1.14.8a2"
    assert Version("1.14.8a2") in Requirement(dependency).specifier
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"][
        "only-include"
    ] == ["agents", "crew.jsonc", "tools", "knowledge", "skills"]

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


def test_json_provider_default_model_helper():
    assert json_crew._default_model_for_provider("openai") == "openai/gpt-5.5"
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

    assert '"memory": false' in crew_template
    assert '"description": "Research current AI trends and write a concise summary."' in (
        crew_template
    )
    assert '"llm": "anthropic/claude-opus-4-6"' in agent_template
