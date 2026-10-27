from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from crewai.project import load_agent_project
from crewai.project.json_loader import JSONProjectError, load_agent
from crewai_core.project import ProjectDefinitionError
import yaml
import pytest


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "support-agent"\n')
    package = tmp_path / "src" / "support_agent"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('ROLE = "Support agent"\n')
    (package / "tools.py").write_text(
        "from crewai.tools import BaseTool\n"
        "class CounterTool(BaseTool):\n"
        '    name: str = "counter"\n'
        '    description: str = "Count calls"\n'
        "    calls: int = 0\n"
        "    def _run(self) -> str:\n"
        "        self.calls += 1\n"
        "        return str(self.calls)\n"
        "counter = CounterTool()\n"
    )
    (package / "agent.py").write_text(
        "from crewai import Agent\n"
        "from . import ROLE\n"
        "from .tools import counter\n"
        'agent = Agent(role=ROLE, goal="Help", backstory="Helpful", '
        'llm="openai/gpt-4o-mini", tools=[counter])\n'
    )
    return tmp_path


def test_loads_fresh_agents_and_relative_module_tools(project):
    first = load_agent_project(project)
    assert first.role == "Support agent"
    assert first.tools[0].run() == "1"
    second = load_agent_project(project, project_name="support_agent")
    assert second is not first
    assert second.tools[0] is not first.tools[0]
    assert second.tools[0].run() == "1"
    assert first.tools[0].run() == "2"


def test_concurrent_loads_do_not_share_tools(project):
    with ThreadPoolExecutor(max_workers=3) as executor:
        agents = list(executor.map(load_agent_project, [project] * 3))
    assert len({id(agent.tools[0]) for agent in agents}) == 3


def test_project_names_do_not_collide(project, tmp_path):
    import shutil

    other = tmp_path / "other"
    shutil.copytree(project / "src", other / "src")
    (other / "src/support_agent/__init__.py").write_text('ROLE = "Other"\n')
    first = load_agent_project(project)
    second = load_agent_project(other, "support_agent")
    assert first.role == "Support agent"
    assert second.role == "Other"


@pytest.mark.parametrize("source", ["agent = object()", "other = 42"])
def test_invalid_export_has_actionable_error(project, source):
    (project / "src/support_agent/agent.py").write_text(source)
    with pytest.raises(ValueError, match=r"must export `agent = Agent"):
        load_agent_project(project)


def test_missing_entrypoint_and_project_import_errors(project):
    path = project / "src/support_agent/agent.py"
    path.unlink()
    with pytest.raises(ProjectDefinitionError, match="No agent definition found"):
        load_agent_project(project)
    path.write_text('raise RuntimeError("project import failed")')
    with pytest.raises(RuntimeError, match="project import failed"):
        load_agent_project(project)


def test_rejects_non_package_name(project):
    with pytest.raises(ValueError, match="Python package name"):
        load_agent_project(project, "../escape")


@pytest.mark.parametrize(
    "project_name,package_name",
    [("Support-Agent", "support_agent"), ("Foo.Bar", "foobar")],
)
def test_loader_uses_scaffold_package_normalization(
    project, project_name, package_name
):
    package = project / "src/support_agent"
    package.rename(project / "src" / package_name)
    (project / "pyproject.toml").write_text(f'[project]\nname = "{project_name}"\n')
    assert load_agent_project(project).role == "Support agent"
    assert load_agent_project(project, project_name).role == "Support agent"


@pytest.fixture
def data_project(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "support-agent"\n[tool.crewai]\ntype = "agent"\n'
    )
    return tmp_path


def write_agent_data(root, filename, **extra):
    definition = dict(
        role="Support agent",
        goal="Help",
        backstory="Helpful",
        llm="openai/gpt-4o-mini",
        **extra,
    )
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(definition) if path.suffix == ".json" else yaml.safe_dump(definition)
    )
    return path


@pytest.mark.parametrize("extension", ["json", "yml", "yaml"])
def test_loads_config_only_agent_without_static_input(data_project, extension):
    path = write_agent_data(data_project, f"agent.{extension}")
    agent = load_agent_project(data_project)
    assert agent.role == "Support agent"
    assert agent.goal == "Help"
    assert load_agent(path).backstory == "Helpful"


@pytest.mark.parametrize("extension", ["json", "yml", "yaml"])
def test_explicit_data_definition_uses_project_root_for_custom_tools(
    data_project, extension
):
    path = write_agent_data(
        data_project, f"config/support.{extension}", tools=["custom:counter"]
    )
    with (data_project / "pyproject.toml").open("a") as config:
        config.write(f'definition = "config/support.{extension}"\n')
    tools = data_project / "tools"
    tools.mkdir()
    (tools / "counter.py").write_text(
        "from crewai.tools import BaseTool\n"
        "class CounterTool(BaseTool):\n"
        '    name: str = "counter"\n'
        '    description: str = "Count calls"\n'
        "    calls: int = 0\n"
        "    def _run(self) -> str:\n"
        "        self.calls += 1\n"
        "        return str(self.calls)\n"
    )
    first = load_agent_project(data_project)
    second = load_agent_project(data_project)
    assert first is not second
    assert first.tools[0] is not second.tools[0]
    assert first.tools[0].run() == "1"
    assert second.tools[0].run() == "1"
    assert first.tools[0].run() == "2"


def test_explicit_data_definition_wins_over_default_python(project):
    write_agent_data(project, "agent.json")
    with (project / "pyproject.toml").open("a") as config:
        config.write('[tool.crewai]\ntype = "agent"\ndefinition = "agent.json"\n')
    assert load_agent_project(project).tools == []


def test_explicit_python_filename_supports_relative_imports(project):
    package = project / "src/support_agent"
    (package / "agent.py").rename(package / "support.py")
    with (project / "pyproject.toml").open("a") as config:
        config.write(
            '[tool.crewai]\ntype = "agent"\ndefinition = "src/support_agent/support.py"\n'
        )
    assert load_agent_project(project).tools[0].run() == "1"
    assert load_agent_project(project).tools[0].run() == "1"


def test_root_python_agent_without_package_directory(data_project):
    (data_project / "agent.py").write_text(
        "from crewai import Agent\n"
        "from .context import ROLE\n"
        'agent = Agent(role=ROLE, goal="Help", backstory="Helpful", llm="openai/gpt-4o-mini")\n'
    )
    (data_project / "context.py").write_text('ROLE = "Root support"\n')
    assert load_agent_project(data_project).role == "Root support"


def test_root_config_without_pyproject(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    write_agent_data(tmp_path, "agent.json")
    assert load_agent_project(tmp_path).role == "Support agent"


def test_ambiguous_default_entrypoints_require_definition(data_project):
    write_agent_data(data_project, "agent.json")
    write_agent_data(data_project, "agent.yaml")
    with pytest.raises(ProjectDefinitionError, match="[Aa]mbiguous|[Mm]ultiple"):
        load_agent_project(data_project)


@pytest.mark.parametrize("extension", ["json", "yml", "yaml"])
def test_definition_must_contain_mapping(data_project, extension):
    (data_project / f"agent.{extension}").write_text('["not an agent"]')
    with pytest.raises(JSONProjectError, match="expected a JSON object"):
        load_agent_project(data_project)


def test_yaml_uses_safe_loader(data_project):
    marker = data_project / "must_not_exist"
    (data_project / "agent.yaml").write_text(
        f'!!python/object/apply:os.system ["touch {marker}"]'
    )
    with pytest.raises(JSONProjectError, match="invalid YAML"):
        load_agent_project(data_project)
    assert not marker.exists()


def test_invalid_json_and_yaml_keep_source_path(data_project):
    (data_project / "agent.json").write_text('{"role":')
    with pytest.raises(JSONProjectError, match="agent.json: invalid JSON"):
        load_agent_project(data_project)
    (data_project / "agent.json").unlink()
    (data_project / "agent.yml").write_text("role: [unclosed")
    with pytest.raises(JSONProjectError, match="agent.yml: invalid YAML"):
        load_agent_project(data_project)


def test_explicit_definition_cannot_escape_project(data_project, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-external.json"
    outside.write_text('{"role":"outside"}')
    with (data_project / "pyproject.toml").open("a") as config:
        config.write(f'definition = "../{outside.name}"\n')
    with pytest.raises(ProjectDefinitionError, match="inside the project root"):
        load_agent_project(data_project)


@pytest.mark.parametrize("filename", ["agent.v1.py", "AGENT.PY"])
def test_explicit_python_definition_loads_selected_filename(data_project, filename):
    if filename == "agent.v1.py":
        (data_project / "__init__.py").write_text("from . import _agent_entrypoint\n")
        (data_project / "_agent_entrypoint.py").write_text("agent = object()\n")
    (data_project / "context.py").write_text('ROLE = "Selected support"\n')
    (data_project / filename).write_text(
        "from crewai import Agent\n"
        "from .context import ROLE\n"
        'agent = Agent(role=ROLE, goal="Help", backstory="Helpful", llm="openai/gpt-4o-mini")\n'
    )
    with (data_project / "pyproject.toml").open("a") as config:
        config.write(f'definition = "{filename}"\n')
    first = load_agent_project(data_project)
    second = load_agent_project(data_project)
    assert first.role == "Selected support"
    assert second.role == "Selected support"
    assert first is not second


def test_nested_python_definition_retains_package_parent_imports(project):
    package = project / "src/support_agent"
    subpackage = package / "sub"
    subpackage.mkdir()
    (subpackage / "__init__.py").write_text("")
    (subpackage / "agent.py").write_text(
        (package / "agent.py")
        .read_text()
        .replace("from . import ROLE", "from .. import ROLE")
        .replace("from .tools import counter", "from ..tools import counter")
    )
    with (project / "pyproject.toml").open("a") as config:
        config.write(
            '[tool.crewai]\ntype = "agent"\ndefinition = "src/support_agent/sub/agent.py"\n'
        )
    first = load_agent_project(project)
    second = load_agent_project(project)
    assert first.role == "Support agent"
    assert first.tools[0].run() == "1"
    assert second.tools[0].run() == "1"
    assert first.tools[0].run() == "2"


@pytest.mark.parametrize(
    "invalid_fields",
    ["1: bad\nunknown: bad\n", "settings:\n  1: bad\n  unknown: bad\n"],
)
def test_yaml_constructor_keys_must_be_strings(data_project, invalid_fields):
    (data_project / "agent.yaml").write_text(
        "role: Support\ngoal: Help\nbackstory: Helpful\nllm: openai/gpt-4o-mini\n"
        + invalid_fields
    )
    with pytest.raises(
        JSONProjectError, match="agent.yaml.*constructor field names must be strings"
    ):
        load_agent_project(data_project)
