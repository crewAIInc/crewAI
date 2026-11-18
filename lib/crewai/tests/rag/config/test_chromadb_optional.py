"""crewai must import, and explain what to install, when chromadb is unavailable."""

import subprocess
import sys
import textwrap

import pytest


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_crewai_imports_when_chromadb_is_unavailable():
    result = _run(
        """
        import sys
        sys.modules["chromadb"] = None  # makes `import chromadb` fail
        import crewai
        from crewai.rag.config.utils import get_rag_provider
        from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
        print(get_rag_provider())
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("qdrant")


def test_importing_crewai_does_not_load_chromadb():
    pytest.importorskip("chromadb")

    result = _run(
        """
        import sys
        import crewai
        print("chromadb" in sys.modules)
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("False")


@pytest.mark.parametrize(
    "module",
    ["crewai.rag.chromadb.config", "crewai.rag.chromadb.factory"],
)
def test_chromadb_provider_modules_explain_the_missing_extra(module):
    result = _run(
        f"""
        import sys
        sys.modules["chromadb"] = None
        import {module}
        """
    )

    assert result.returncode != 0
    assert "pip install crewai[chromadb]" in result.stderr
