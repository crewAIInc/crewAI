"""Test that all public API classes are properly importable."""


def test_task_output_import():
    """Test that TaskOutput can be imported from crewai."""
    from crewai import TaskOutput

    assert TaskOutput is not None


def test_crew_output_import():
    """Test that CrewOutput can be imported from crewai."""
    from crewai import CrewOutput

    assert CrewOutput is not None


def test_import_crewai_does_not_load_qdrant():
    """Importing crewai must not eagerly load the qdrant_client dependency chain.

    Regression test for lazy RAG config imports: the provider config union
    (crewai.rag.config.types -> qdrant_client/chromadb) must only load when
    the RAG config surface is actually used, not at `import crewai` time.
    """
    import subprocess
    import sys

    code = (
        "import sys; import crewai; "
        "sys.exit(1 if 'qdrant_client' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, (
        "qdrant_client was imported eagerly by `import crewai`"
        f"\nstderr: {result.stderr}"
    )


def test_rag_module_setattr_contract():
    """crewai.rag still routes `config` assignment and rejects other attributes."""
    from types import ModuleType

    import crewai.rag as rag_mod
    import crewai.rag.config.utils as rag_config_utils

    # Non-config attributes are rejected.
    try:
        rag_mod.some_random_attribute = 1
    except AttributeError:
        pass
    else:
        raise AssertionError("crewai.rag accepted an arbitrary attribute")

    # `config` assignment routes to set_rag_config.
    recorded = []
    original = rag_config_utils.set_rag_config
    rag_config_utils.set_rag_config = recorded.append
    try:
        sentinel = object()
        rag_mod.config = sentinel
        unrelated_module = ModuleType("unrelated")
        rag_mod.config = unrelated_module
        assert recorded == [sentinel, unrelated_module]
    finally:
        rag_config_utils.set_rag_config = original


def test_lazy_import_annotations_resolve():
    """Lazy imports must not leave unresolved runtime annotations."""
    import typing

    import crewai.rag as rag_mod
    from crewai.knowledge.storage import knowledge_storage

    assert typing.get_type_hints(type(rag_mod).__setattr__)["value"] is typing.Any
    assert typing.get_type_hints(knowledge_storage.create_client)["config"] is typing.Any


def test_knowledge_storage_instantiates():
    """KnowledgeStorage still builds its pydantic model with lazy rag imports."""
    from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage

    storage = KnowledgeStorage(collection_name="import-regression")
    assert storage.collection_name == "import-regression"
