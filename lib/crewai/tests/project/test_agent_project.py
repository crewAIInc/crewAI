from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from crewai.project import load_agent_project
import pytest


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "support-agent"\n')
    package = tmp_path / "src" / "support_agent"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('ROLE = "Support agent"\n')
    (package / "tools.py").write_text(
        'from crewai.tools import BaseTool\n'
        'class CounterTool(BaseTool):\n'
        '    name: str = "counter"\n'
        '    description: str = "Count calls"\n'
        '    calls: int = 0\n'
        '    def _run(self) -> str:\n'
        '        self.calls += 1\n'
        '        return str(self.calls)\n'
        'counter = CounterTool()\n'
    )
    (package / "agent.py").write_text(
        'from crewai import Agent\n'
        'from . import ROLE\n'
        'from .tools import counter\n'
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
    with pytest.raises(FileNotFoundError, match="Agent entrypoint not found"):
        load_agent_project(project)
    path.write_text('raise RuntimeError("project import failed")')
    with pytest.raises(RuntimeError, match="project import failed"):
        load_agent_project(project)


def test_rejects_non_package_name(project):
    with pytest.raises(ValueError, match="Python package name"):
        load_agent_project(project, "../escape")



@pytest.mark.parametrize("project_name,package_name", [("Support-Agent", "support_agent"), ("Foo.Bar", "foobar")])
def test_loader_uses_scaffold_package_normalization(project, project_name, package_name):
    package = project / "src/support_agent"
    package.rename(project / "src" / package_name)
    (project / "pyproject.toml").write_text(f'[project]\nname = "{project_name}"\n')
    assert load_agent_project(project).role == "Support agent"
    assert load_agent_project(project, project_name).role == "Support agent"
