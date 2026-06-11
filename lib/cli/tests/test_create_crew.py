import keyword
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner
import crewai_cli.create_json_crew as json_crew
import crewai_cli.tui_picker as tui_picker
from crewai_cli.create_crew import create_crew, create_folder_structure
from crewai_cli.create_json_crew import _default_model_for_provider, create_json_crew


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

    def pick_many(title: str, labels: list[str], **kwargs) -> list[int]:
        picker_calls.append((title, labels, kwargs))
        return [0, 2]

    monkeypatch.setattr(json_crew, "pick_many", pick_many)

    tools = json_crew._select_tools()

    assert tools == ["SerperDevTool", "DirectoryReadTool"]
    assert len(picker_calls) == 1
    labels = picker_calls[0][1]
    assert picker_calls[0][2]["action_indices"] == {5}
    assert labels[0].strip().endswith("SerperDevTool")
    assert labels[1].strip().endswith("ScrapeWebsiteTool")
    assert labels[2].strip().endswith("DirectoryReadTool")
    assert labels[3].strip().endswith("FileReadTool")
    assert labels[4].strip().endswith("FileWriterTool")
    assert labels[0].index("Google search") < labels[0].index("SerperDevTool")
    assert "More tools" in labels[5]


def test_json_wizard_tool_picker_expands_more_tools(monkeypatch):
    picker_calls: list[tuple[str, list[str]]] = []

    def pick_many(title: str, labels: list[str], **_kwargs) -> tuple[list[int], int | None] | list[int]:
        picker_calls.append((title, labels))
        if title == "Tools (space to toggle, enter to confirm):":
            return [0], 5
        return [
            idx
            for idx, label in enumerate(labels)
            if label.strip().endswith("BraveSearchTool")
        ]

    monkeypatch.setattr(json_crew, "pick_many", pick_many)

    tools = json_crew._select_tools()

    assert tools == ["SerperDevTool", "BraveSearchTool"]
    assert len(picker_calls) == 2
    assert picker_calls[1][0] == "More tools:"
    assert any("Search & Research:" in label for label in picker_calls[1][1])


def test_multi_picker_enter_activates_action_row(monkeypatch):
    monkeypatch.setattr(tui_picker, "_read_key", lambda: "enter")
    monkeypatch.setattr(tui_picker, "_draw_multi", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tui_picker, "_clear_lines", lambda *_args, **_kwargs: None)

    assert tui_picker._arrow_select_multi(
        ["More tools..."],
        action_indices={0},
    ) == ([], 0)


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
    monkeypatch.setattr(json_crew, "pick_many", lambda *_args, **_kwargs: [])
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

        create_json_crew("JSON Crew", provider="openai", skip_provider=True)

    mock_wizard.assert_called_once_with(
        skip_provider=True,
        default_llm="openai/gpt-5.5",
    )
    assert (tmp_path / "json_crew" / "crew.jsonc").exists()
    assert not (tmp_path / "json_crew" / "tests").exists()
    assert not (tmp_path / "json_crew" / "config.jsonc").exists()

    crew_template = (tmp_path / "json_crew" / "crew.jsonc").read_text()
    assert (
        '"guardrail": "Every factual claim needs context support."'
        in crew_template
    )
    assert '"guardrails": [' in crew_template
    assert '"guardrail_max_retries": 2' in crew_template
    assert "Docs: https://docs.crewai.com/concepts/tasks" in crew_template
    assert '"output_pydantic": null' in crew_template
    assert '"markdown": false' in crew_template
    assert "Docs: https://docs.crewai.com/concepts/crews" in crew_template
    assert '"manager_agent": "researcher"' in crew_template
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
    assert "Optional agent-level guardrail" in agent_template
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
    assert _default_model_for_provider("openai") == "openai/gpt-5.5"
    assert _default_model_for_provider("anthropic/claude-custom") == (
        "anthropic/claude-custom"
    )
    assert _default_model_for_provider("unknown") is None
