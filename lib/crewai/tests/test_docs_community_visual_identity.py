"""Documentation checks for community visual identity references."""

from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "docs").exists():
            return parent
    raise RuntimeError("Could not locate repository root from test path.")


def test_agentavatar_links_are_present_in_examples_docs() -> None:
    doc_path = _repo_root() / "docs/en/examples/example.mdx"
    content = doc_path.read_text(encoding="utf-8")

    assert "https://agentavatar.dev/#/gallery" in content
    assert "https://agentavatar.dev/#/agent" in content
    assert "CrewAI - Crew Orbit" in content
